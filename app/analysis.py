"""Persistence and workflow for water-connectivity surveys.

A *run* fixes one published version and one black/white survey mask; it owns
the per-block connectivity summaries.  A field re-survey (CSV of ``col,row``
reclassifications) recomputes only the blocks whose pixels actually change.
The block rows and the patch record are written in one SQLite transaction, so
a failure leaves the previous summaries (and therefore previous query results)
intact.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .mosaic import utc_now_iso
from .water import (
    BLOCK_SIZE,
    LAND,
    UNKNOWN,
    WATER,
    BlockSummary,
    BlockView,
    StateGrid,
    WaterAnalysisError,
    WaterBounds,
    block_bounds,
    block_columns,
    block_rows,
    block_view_from_row,
    query_bounds,
    summarize_block,
    summary_columns,
)


class AnalysisConflict(WaterAnalysisError):
    """The run moved on (another patch landed) while this one was computed."""


@dataclass(frozen=True)
class RunRecord:
    id: str
    version_id: str
    created_at: str
    canvas_width: int
    canvas_height: int
    block_size: int
    mask_sha256: str
    water_pixels: int
    land_pixels: int
    unknown_pixels: int
    last_patch_id: str | None

    @property
    def block_count(self) -> int:
        return block_columns(self.canvas_width) * block_rows(self.canvas_height)


def run_from_row(row: Mapping) -> RunRecord:
    return RunRecord(
        id=row["id"],
        version_id=row["version_id"],
        created_at=row["created_at"],
        canvas_width=int(row["canvas_width"]),
        canvas_height=int(row["canvas_height"]),
        block_size=int(row["block_size"]),
        mask_sha256=row["mask_sha256"],
        water_pixels=int(row["water_pixels"]),
        land_pixels=int(row["land_pixels"]),
        unknown_pixels=int(row["unknown_pixels"]),
        last_patch_id=row["last_patch_id"],
    )


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #


def get_run(conn: sqlite3.Connection, run_id: str) -> RunRecord | None:
    row = conn.execute(
        "SELECT * FROM analysis_runs WHERE id = ?", (run_id,)
    ).fetchone()
    return None if row is None else run_from_row(row)


def find_run(
    conn: sqlite3.Connection, version_id: str, mask_sha256: str
) -> RunRecord | None:
    row = conn.execute(
        "SELECT * FROM analysis_runs WHERE version_id = ? AND mask_sha256 = ?",
        (version_id, mask_sha256),
    ).fetchone()
    return None if row is None else run_from_row(row)


def list_runs(conn: sqlite3.Connection, limit: int = 20) -> list[RunRecord]:
    rows = conn.execute(
        "SELECT * FROM analysis_runs ORDER BY rowid DESC LIMIT ?", (limit,)
    ).fetchall()
    return [run_from_row(row) for row in rows]


def summarize_grid(
    grid: StateGrid, canvas_width: int, canvas_height: int
) -> list[BlockSummary]:
    """Cut the canvas pixel states into blocks and summarise each."""
    pixels = grid.pixels
    summaries: list[BlockSummary] = []
    for by in range(block_rows(canvas_height)):
        for bx in range(block_columns(canvas_width)):
            x0, y0, width, height = block_bounds(
                bx, by, canvas_width, canvas_height
            )
            block = b"".join(
                pixels[
                    (y0 + y) * canvas_width + x0:
                    (y0 + y) * canvas_width + x0 + width
                ]
                for y in range(height)
            )
            summaries.append(summarize_block(bx, by, width, height, block))
    return summaries


def insert_run(
    conn: sqlite3.Connection,
    *,
    version_id: str,
    grid: StateGrid,
    canvas_width: int,
    canvas_height: int,
    mask_sha256: str,
    summaries: Sequence[BlockSummary],
) -> RunRecord:
    """Insert a complete run and all block summaries in one transaction.

    If an identical (version, mask) run appeared concurrently, the existing
    run wins and the freshly computed summaries are discarded.
    """
    existing = find_run(conn, version_id, mask_sha256)
    if existing is not None:
        return existing
    run_id = secrets.token_hex(12)
    conn.execute(
        "INSERT INTO analysis_runs(id, version_id, created_at, canvas_width, "
        "canvas_height, block_size, mask_sha256, water_pixels, land_pixels, "
        "unknown_pixels, last_patch_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
        (
            run_id,
            version_id,
            utc_now_iso(),
            canvas_width,
            canvas_height,
            BLOCK_SIZE,
            mask_sha256,
            grid.water_pixels,
            grid.land_pixels,
            grid.unknown_pixels,
        ),
    )
    for summary in summaries:
        columns = summary_columns(summary)
        conn.execute(
            "INSERT INTO analysis_blocks(run_id, bx, by, width, height, pixels, "
            "wu_labels, w_labels, wu_weights, w_weights, wu_ports, w_ports) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                columns["bx"],
                columns["by"],
                columns["width"],
                columns["height"],
                columns["pixels"],
                columns["wu_labels"],
                columns["w_labels"],
                columns["wu_weights"],
                columns["w_weights"],
                columns["wu_ports"],
                columns["w_ports"],
            ),
        )
    run = get_run(conn, run_id)
    assert run is not None
    return run


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #


def load_block(
    conn: sqlite3.Connection,
    run_id: str,
    bx: int,
    by: int,
    include_labels: bool,
    include_pixels: bool = False,
) -> BlockView:
    row = conn.execute(
        "SELECT * FROM analysis_blocks WHERE run_id = ? AND bx = ? AND by = ?",
        (run_id, bx, by),
    ).fetchone()
    if row is None:
        raise WaterAnalysisError("缺少区块摘要，调查数据不完整。")
    return block_view_from_row(row, include_labels, include_pixels)


def run_bounds(
    conn: sqlite3.Connection, run: RunRecord, col: int, row: int
) -> WaterBounds:
    return query_bounds(
        lambda bx, by, labels: load_block(conn, run.id, bx, by, labels),
        run.canvas_width,
        run.canvas_height,
        col,
        row,
    )


def run_fingerprint(conn: sqlite3.Connection, run: RunRecord) -> str:
    """Digest of the run's source mask and every applied patch, in order."""
    digest = hashlib.sha256()
    digest.update(b"water-analysis-v1\0")
    digest.update(f"{run.version_id}\0{run.mask_sha256}\0".encode())
    rows = conn.execute(
        "SELECT id, csv_sha256 FROM analysis_patches WHERE run_id = ? "
        "ORDER BY rowid",
        (run.id,),
    ).fetchall()
    for row in rows:
        digest.update(f"{row['id']}\0{row['csv_sha256']}\0".encode())
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Patches
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PatchPlan:
    """Blocks touched by a re-survey at the time it was planned."""

    run_id: str
    base_patch_id: str | None
    current_pixels: Mapping[tuple[int, int], bytes]


@dataclass(frozen=True)
class PatchComputation:
    summaries: dict[tuple[int, int], BlockSummary]
    changed_pixels: int
    delta_water: int
    delta_land: int
    delta_unknown: int

    @property
    def touched(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(self.summaries))


@dataclass(frozen=True)
class PatchOutcome:
    patch_id: str
    row_count: int
    changed_pixels: int
    changed_blocks: tuple[tuple[int, int], ...]


def plan_patch(
    conn: sqlite3.Connection,
    run: RunRecord,
    changes: Iterable[tuple[int, int, int]],
) -> PatchPlan:
    """Read the pixel states of every block a re-survey touches.

    Only these blocks are read; their summaries are the only ones recomputed
    later.  The run's current patch id is remembered so a concurrent patch
    can be detected at commit time.
    """
    touched: set[tuple[int, int]] = set()
    for col, row_index, _state in changes:
        touched.add((col // BLOCK_SIZE, row_index // BLOCK_SIZE))
    current: dict[tuple[int, int], bytes] = {}
    for bx, by in touched:
        view = load_block(conn, run.id, bx, by, False, include_pixels=True)
        if view.pixels is None:
            raise WaterAnalysisError("区块摘要不完整。")
        current[(bx, by)] = view.pixels
    return PatchPlan(
        run_id=run.id,
        base_patch_id=run.last_patch_id,
        current_pixels=current,
    )


def compute_patch(
    plan: PatchPlan,
    changes: Iterable[tuple[int, int, int]],
    canvas_width: int,
    canvas_height: int,
) -> PatchComputation:
    """Relabel only the blocks whose pixels the re-survey changes.

    Blocks are relabelled independently; the aligned block-level stitch in
    :mod:`app.water` makes this equivalent to relabelling the whole canvas.
    """
    by_block: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for col, row_index, state in changes:
        by_block.setdefault((col // BLOCK_SIZE, row_index // BLOCK_SIZE), []).append(
            (col, row_index, state)
        )

    summaries: dict[tuple[int, int], BlockSummary] = {}
    changed_pixels = 0
    delta = {WATER: 0, LAND: 0, UNKNOWN: 0}
    for (bx, by), edits in by_block.items():
        x0, y0, width, height = block_bounds(bx, by, canvas_width, canvas_height)
        pixels = bytearray(plan.current_pixels[(bx, by)])
        block_changed = 0
        for col, row_index, state in edits:
            index = (row_index - y0) * width + (col - x0)
            previous = pixels[index]
            if previous != state:
                pixels[index] = state
                delta[previous] -= 1
                delta[state] += 1
                block_changed += 1
        if not block_changed:
            continue
        changed_pixels += block_changed
        summaries[(bx, by)] = summarize_block(bx, by, width, height, bytes(pixels))
    return PatchComputation(
        summaries=summaries,
        changed_pixels=changed_pixels,
        delta_water=delta[WATER],
        delta_land=delta[LAND],
        delta_unknown=delta[UNKNOWN],
    )


def commit_patch(
    conn: sqlite3.Connection,
    run: RunRecord,
    plan: PatchPlan,
    computation: PatchComputation,
    *,
    csv_sha256: str,
    row_count: int,
) -> PatchOutcome:
    """Write the recomputed summaries and the patch record atomically.

    The caller must own the transaction: on any exception the connection is
    rolled back and neither the block summaries nor the patch record change.
    A stale computation (another patch landed meanwhile) or a re-uploaded CSV
    is rejected before anything is written.
    """
    fresh = get_run(conn, run.id)
    if fresh is None:
        raise WaterAnalysisError("调查记录不存在。")
    if fresh.last_patch_id != plan.base_patch_id:
        raise AnalysisConflict(
            "补测期间已有其他补测提交；请刷新后重试（原结果未受影响）。"
        )
    if conn.execute(
        "SELECT 1 FROM analysis_patches WHERE run_id = ? AND csv_sha256 = ?",
        (run.id, csv_sha256),
    ).fetchone():
        raise WaterAnalysisError("这份补测 CSV 已经应用过（同一文件不可重复应用）。")

    for (bx, by), summary in computation.summaries.items():
        columns = summary_columns(summary)
        cursor = conn.execute(
            "UPDATE analysis_blocks SET width = ?, height = ?, pixels = ?, "
            "wu_labels = ?, w_labels = ?, wu_weights = ?, w_weights = ?, "
            "wu_ports = ?, w_ports = ? WHERE run_id = ? AND bx = ? AND by = ?",
            (
                columns["width"],
                columns["height"],
                columns["pixels"],
                columns["wu_labels"],
                columns["w_labels"],
                columns["wu_weights"],
                columns["w_weights"],
                columns["wu_ports"],
                columns["w_ports"],
                run.id,
                bx,
                by,
            ),
        )
        if cursor.rowcount != 1:
            raise WaterAnalysisError("区块摘要更新失败。")

    patch_id = secrets.token_hex(12)
    conn.execute(
        "INSERT INTO analysis_patches(id, run_id, created_at, csv_sha256, "
        "row_count, changed_pixels, changed_blocks) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            patch_id,
            run.id,
            utc_now_iso(),
            csv_sha256,
            row_count,
            computation.changed_pixels,
            ",".join(f"{bx},{by}" for bx, by in computation.touched),
        ),
    )
    conn.execute(
        "UPDATE analysis_runs SET last_patch_id = ?, water_pixels = ?, "
        "land_pixels = ?, unknown_pixels = ? WHERE id = ?",
        (
            patch_id,
            run.water_pixels + computation.delta_water,
            run.land_pixels + computation.delta_land,
            run.unknown_pixels + computation.delta_unknown,
            run.id,
        ),
    )
    return PatchOutcome(
        patch_id=patch_id,
        row_count=row_count,
        changed_pixels=computation.changed_pixels,
        changed_blocks=computation.touched,
    )


def list_patches(
    conn: sqlite3.Connection, run_id: str, limit: int | None = 10
) -> list[Mapping]:
    sql = "SELECT * FROM analysis_patches WHERE run_id = ? ORDER BY rowid DESC"
    params: list[object] = [run_id]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


def patch_count(conn: sqlite3.Connection, run_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM analysis_patches WHERE run_id = ?", (run_id,)
    ).fetchone()
    return int(row["n"])
