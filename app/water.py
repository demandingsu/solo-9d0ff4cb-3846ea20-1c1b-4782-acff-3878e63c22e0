"""Water-body connectivity analysis over a published mosaic.

Every pixel of a published canvas is classified as *land* (the surveyor's
mask is black), *water* (mask is white) or *unknown* (the published image is
transparent there, e.g. clouds covered it).  For a seed known-water pixel two
areas are reported:

* **lower bound** - the 4-connected component of the seed over known water
  only;
* **upper bound** - the 4-connected component of the seed when unknown pixels
  are also treated as water (clouds may hide a land bridge).

The canvas is cut into :data:`BLOCK_SIZE` x :data:`BLOCK_SIZE` blocks.  Each
block stores its pixel states plus a compact connectivity summary: local
component ids, per-component pixel weights and the component id on every
boundary cell.  Queries only walk that block-level graph; re-survey a CSV of
coordinates and only the blocks whose pixels changed are relabelled.  Since
block boundaries align exactly, the block graph reproduces the full-canvas
flood fill exactly.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zlib
from array import array
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from PIL import Image

BLOCK_SIZE = 128

LAND = 0
WATER = 1
UNKNOWN = 2

STATE_NAMES = {LAND: "陆地", WATER: "水面", UNKNOWN: "未知"}

MAX_RECLASS_BYTES = 64 * 1024 * 1024
MAX_RECLASS_ROWS = 2_000_000

SIDES = ("top", "bottom", "left", "right")

WATER_LAYER = "w"            # passable: known water only
WATER_UNKNOWN_LAYER = "wu"   # passable: known water and unknown

_PASSABLE_W = bytes(1 if state == WATER else 0 for state in range(256))
_PASSABLE_WU = bytes(
    1 if state in (WATER, UNKNOWN) else 0 for state in range(256)
)
_WATER_ONLY = bytes(1 if state == WATER else 0 for state in range(256))


class WaterAnalysisError(ValueError):
    """Invalid mask, seed, CSV or a corrupt block summary."""


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def block_columns(width: int) -> int:
    return (width + BLOCK_SIZE - 1) // BLOCK_SIZE


def block_rows(height: int) -> int:
    return (height + BLOCK_SIZE - 1) // BLOCK_SIZE


def block_bounds(
    bx: int, by: int, canvas_width: int, canvas_height: int
) -> tuple[int, int, int, int]:
    """Return ``(x0, y0, width, height)`` for block ``(bx, by)``."""
    x0 = bx * BLOCK_SIZE
    y0 = by * BLOCK_SIZE
    width = min(BLOCK_SIZE, canvas_width - x0)
    height = min(BLOCK_SIZE, canvas_height - y0)
    if width <= 0 or height <= 0:
        raise WaterAnalysisError("区块坐标超出画幅。")
    return x0, y0, width, height


# --------------------------------------------------------------------------- #
# Component labelling of one block
# --------------------------------------------------------------------------- #


def _label_layer(
    pixels: bytes, width: int, height: int, *, allow_unknown: bool
) -> tuple[array, tuple[int, ...]]:
    """Run-based 4-connected labelling of one passability layer.

    Returns ``(labels, weight)`` where ``labels[i]`` is the 1-based component
    id of pixel ``i`` (0 for non-passable pixels) and ``weight[k-1]`` counts
    the passable pixels of component ``k``.  Runs are scanned with ``bytes``
    primitives, so the Python loop scales with the number of runs rather than
    the number of pixels.
    """
    total = width * height
    labels = array("H", [0]) * total
    table = _PASSABLE_WU if allow_unknown else _PASSABLE_W
    parent: list[int] = [0]  # 1-based union-find; slot 0 is unused

    def find(label: int) -> int:
        root = label
        while parent[root] != root:
            root = parent[root]
        while parent[label] != root:
            parent[label], label = root, parent[label]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    runs_by_row: list[list[tuple[int, int, int]]] = []
    for y in range(height):
        row = pixels[y * width:(y + 1) * width].translate(table)
        runs: list[tuple[int, int, int]] = []
        position = row.find(b"\x01")
        while position != -1:
            end = row.find(b"\x00", position)
            if end == -1:
                end = width
            label = len(parent)
            parent.append(label)
            runs.append((position, end, label))
            position = row.find(b"\x01", end)
        runs_by_row.append(runs)

    # Union runs that touch vertically between adjacent rows.
    for y in range(1, height):
        previous = runs_by_row[y - 1]
        current = runs_by_row[y]
        i = j = 0
        while i < len(previous) and j < len(current):
            p_start, p_end, p_label = previous[i]
            c_start, c_end, c_label = current[j]
            if p_start < c_end and c_start < p_end:
                union(p_label, c_label)
            if p_end <= c_end:
                i += 1
            else:
                j += 1

    # Canonical ids, then paint runs and accumulate weights.
    canonical: dict[int, int] = {}
    weights: list[int] = []
    for y, runs in enumerate(runs_by_row):
        row_offset = y * width
        for start, end, label in runs:
            root = find(label)
            component = canonical.get(root)
            if component is None:
                component = len(canonical) + 1
                canonical[root] = component
                weights.append(0)
            length = end - start
            weights[component - 1] += length
            segment = array("H", [component]) * length
            base = row_offset + start
            labels[base:base + length] = segment
    return labels, tuple(weights)


@dataclass(frozen=True)
class BlockSummary:
    """Connectivity of one block for both passability layers."""

    bx: int
    by: int
    width: int
    height: int
    pixels: bytes
    wu_labels: array
    w_labels: array
    wu_weights: tuple[tuple[int, int], ...]  # (water, unknown) per component
    w_weights: tuple[int, ...]               # water pixels per component
    wu_ports: dict[str, tuple[int, ...]]
    w_ports: dict[str, tuple[int, ...]]


def _ports(labels: array, width: int, height: int) -> dict[str, tuple[int, ...]]:
    return {
        "top": tuple(labels[0:width]),
        "bottom": tuple(labels[(height - 1) * width:height * width]),
        "left": tuple(labels[y * width] for y in range(height)),
        "right": tuple(labels[y * width + width - 1] for y in range(height)),
    }


def summarize_block(
    bx: int,
    by: int,
    width: int,
    height: int,
    pixels: bytes,
) -> BlockSummary:
    """Compute the connectivity summary for one block's pixel states."""
    if width < 1 or height < 1:
        raise WaterAnalysisError("区块尺寸必须为正。")
    if len(pixels) != width * height:
        raise WaterAnalysisError("区块像元数与尺寸不一致。")
    if not set(pixels) <= {LAND, WATER, UNKNOWN}:
        raise WaterAnalysisError("区块包含未知的像元状态。")
    wu_labels, wu_lengths = _label_layer(
        pixels, width, height, allow_unknown=True
    )
    w_labels, w_lengths = _label_layer(
        pixels, width, height, allow_unknown=False
    )
    # Split each component's weight into known water and unknown pixels by
    # visiting known-water runs only (not every pixel).
    wu_water = [0] * len(wu_lengths)
    for y in range(height):
        row = pixels[y * width:(y + 1) * width].translate(_WATER_ONLY)
        position = row.find(b"\x01")
        while position != -1:
            end = row.find(b"\x00", position)
            if end == -1:
                end = width
            wu_water[wu_labels[y * width + position] - 1] += end - position
            position = row.find(b"\x01", end)
    wu_weights = tuple(
        (wu_water[k], wu_lengths[k] - wu_water[k]) for k in range(len(wu_lengths))
    )
    return BlockSummary(
        bx=bx,
        by=by,
        width=width,
        height=height,
        pixels=pixels,
        wu_labels=wu_labels,
        w_labels=w_labels,
        wu_weights=wu_weights,
        w_weights=w_lengths,
        wu_ports=_ports(wu_labels, width, height),
        w_ports=_ports(w_labels, width, height),
    )


# --------------------------------------------------------------------------- #
# Storage encoding
# --------------------------------------------------------------------------- #


def _pack(labels: array) -> bytes:
    return zlib.compress(labels.tobytes(), 6)


def _unpack(blob: bytes, expected: int) -> array:
    labels = array("H")
    labels.frombytes(zlib.decompress(blob))
    if len(labels) != expected:
        raise WaterAnalysisError("区块摘要已损坏。")
    return labels


def summary_columns(summary: BlockSummary) -> dict[str, object]:
    """SQLite column values for one ``analysis_blocks`` row."""
    return {
        "bx": summary.bx,
        "by": summary.by,
        "width": summary.width,
        "height": summary.height,
        "pixels": zlib.compress(summary.pixels, 6),
        "wu_labels": _pack(summary.wu_labels),
        "w_labels": _pack(summary.w_labels),
        "wu_weights": json.dumps([list(pair) for pair in summary.wu_weights]),
        "w_weights": json.dumps(list(summary.w_weights)),
        "wu_ports": json.dumps({side: list(summary.wu_ports[side]) for side in SIDES}),
        "w_ports": json.dumps({side: list(summary.w_ports[side]) for side in SIDES}),
    }


@dataclass(frozen=True)
class BlockView:
    """A block summary as read back from storage.

    Label arrays / pixel states are only decoded when ``include_labels`` is
    requested (seed block and re-surveyed blocks); a query over the rest of
    the canvas reads weights and ports only.
    """

    bx: int
    by: int
    width: int
    height: int
    wu_weights: tuple[tuple[int, int], ...]
    w_weights: tuple[int, ...]
    wu_ports: Mapping[str, Sequence[int]]
    w_ports: Mapping[str, Sequence[int]]
    pixels: bytes | None = None
    wu_labels: Sequence[int] | None = None
    w_labels: Sequence[int] | None = None


def block_view_from_row(
    row: Mapping, include_labels: bool, include_pixels: bool = False
) -> BlockView:
    """Rebuild a :class:`BlockView` from a database row (or row-like map).

    Queries need only the weights/ports; the seed block additionally needs the
    local component labels, and re-surveying a block also needs its pixels.
    """
    try:
        summary = BlockView(
            bx=int(row["bx"]),
            by=int(row["by"]),
            width=int(row["width"]),
            height=int(row["height"]),
            wu_weights=tuple(
                (int(pair[0]), int(pair[1]))
                for pair in json.loads(row["wu_weights"])
            ),
            w_weights=tuple(int(v) for v in json.loads(row["w_weights"])),
            wu_ports={
                side: tuple(int(v) for v in values)
                for side, values in json.loads(row["wu_ports"]).items()
            },
            w_ports={
                side: tuple(int(v) for v in values)
                for side, values in json.loads(row["w_ports"]).items()
            },
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise WaterAnalysisError("区块摘要已损坏。") from exc
    if include_labels or include_pixels:
        size = summary.width * summary.height
        try:
            labels = (
                {
                    "wu_labels": _unpack(bytes(row["wu_labels"]), size),
                    "w_labels": _unpack(bytes(row["w_labels"]), size),
                }
                if include_labels
                else {}
            )
            pixels = (
                zlib.decompress(bytes(row["pixels"])) if include_pixels else None
            )
        except (KeyError, TypeError, zlib.error) as exc:
            raise WaterAnalysisError("区块摘要已损坏。") from exc
        summary = BlockView(**{**summary.__dict__, **labels, "pixels": pixels})
    return summary


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WaterBounds:
    lower: int
    upper: int
    blocks_visited: int
    lower_components: int
    upper_components: int

    @property
    def uncertain(self) -> int:
        return self.upper - self.lower


BlockLoader = Callable[[int, int, bool], BlockView]


def query_bounds(
    load_block: BlockLoader,
    canvas_width: int,
    canvas_height: int,
    col: int,
    row: int,
) -> WaterBounds:
    """Return the lower/upper water area for the given seed pixel.

    ``load_block(bx, by, include_labels)`` must return the stored summary;
    label arrays are only requested for the seed block, so a query never
    needs block pixels.
    """
    if not (0 <= col < canvas_width and 0 <= row < canvas_height):
        raise WaterAnalysisError(
            f"坐标超出画幅范围（列 0–{canvas_width - 1}，行 0–{canvas_height - 1}）。"
        )
    seed_bx = col // BLOCK_SIZE
    seed_by = row // BLOCK_SIZE
    cache: dict[tuple[int, int], BlockView] = {}

    def cached(bx: int, by: int, need_labels: bool = False) -> BlockView:
        key = (bx, by)
        view = cache.get(key)
        if view is not None and (not need_labels or view.w_labels is not None):
            return view
        view = load_block(bx, by, need_labels)
        cache[key] = view
        return view

    seed = cached(seed_bx, seed_by, True)
    x0, y0, _width, _height = block_bounds(
        seed_bx, seed_by, canvas_width, canvas_height
    )
    local = (row - y0) * seed.width + (col - x0)
    if seed.w_labels is None or seed.wu_labels is None:
        raise WaterAnalysisError("种子区块的摘要不完整。")
    # A pixel is known water iff it has a component in the water-only layer.
    if not seed.w_labels[local]:
        if seed.wu_labels[local]:
            raise WaterAnalysisError(
                "该坐标不是已知水面像元（当前为未知/被云遮挡）。"
            )
        raise WaterAnalysisError("该坐标不是已知水面像元（当前为陆地）。")

    lower, lower_components = _walk_layer(
        cached, canvas_width, canvas_height,
        seed_bx, seed_by, seed.w_labels[local], WATER_LAYER,
    )
    upper, upper_components = _walk_layer(
        cached, canvas_width, canvas_height,
        seed_bx, seed_by, seed.wu_labels[local], WATER_UNKNOWN_LAYER,
    )
    return WaterBounds(
        lower=lower,
        upper=upper,
        blocks_visited=len(cache),
        lower_components=lower_components,
        upper_components=upper_components,
    )


def _walk_layer(
    load_block: BlockLoader,
    canvas_width: int,
    canvas_height: int,
    start_bx: int,
    start_by: int,
    start_component: int,
    layer: str,
) -> tuple[int, int]:
    """Size of the global component of ``start_component`` in one layer."""
    if layer == WATER_LAYER:
        weight_attr, ports_attr = "w_weights", "w_ports"
    else:
        weight_attr, ports_attr = "wu_weights", "wu_ports"

    stack = [(start_bx, start_by, start_component)]
    seen: set[tuple[int, int, int]] = set()
    total = 0
    while stack:
        bx, by, component = stack.pop()
        key = (bx, by, component)
        if key in seen:
            continue
        seen.add(key)
        block = load_block(bx, by, False)
        weights = getattr(block, weight_attr)
        if not (1 <= component <= len(weights)):
            raise WaterAnalysisError("区块摘要中的连通分量编号无效。")
        if layer == WATER_LAYER:
            total += int(weights[component - 1])
        else:
            water, unknown = weights[component - 1]
            total += int(water) + int(unknown)
        ports = getattr(block, ports_attr)

        if (bx + 1) * BLOCK_SIZE < canvas_width:
            _stitch(
                stack, component, ports["right"],
                load_block(bx + 1, by, False), ports_attr, "left", bx + 1, by,
            )
        if bx > 0:
            _stitch(
                stack, component, ports["left"],
                load_block(bx - 1, by, False), ports_attr, "right", bx - 1, by,
            )
        if (by + 1) * BLOCK_SIZE < canvas_height:
            _stitch(
                stack, component, ports["bottom"],
                load_block(bx, by + 1, False), ports_attr, "top", bx, by + 1,
            )
        if by > 0:
            _stitch(
                stack, component, ports["top"],
                load_block(bx, by - 1, False), ports_attr, "bottom", bx, by - 1,
            )
    return total, len(seen)


def _stitch(
    stack: list[tuple[int, int, int]],
    component: int,
    own_edge: Sequence[int],
    neighbour: BlockView,
    ports_attr: str,
    neighbour_side: str,
    neighbour_bx: int,
    neighbour_by: int,
) -> None:
    other_edge = getattr(neighbour, ports_attr)[neighbour_side]
    if len(own_edge) != len(other_edge):
        raise WaterAnalysisError("相邻区块的边界长度不一致。")
    for index in range(len(own_edge)):
        if own_edge[index] == component:
            other = int(other_edge[index])
            if other:
                stack.append((neighbour_bx, neighbour_by, other))


# --------------------------------------------------------------------------- #
# Reference implementation (full-canvas flood fill)
# --------------------------------------------------------------------------- #


def full_bounds(
    pixels: bytes | bytearray,
    canvas_width: int,
    canvas_height: int,
    col: int,
    row: int,
) -> tuple[int, int]:
    """Brute-force lower/upper bounds, used to verify the block summaries."""
    if len(pixels) != canvas_width * canvas_height:
        raise WaterAnalysisError("像元数与画幅尺寸不一致。")
    if not (0 <= col < canvas_width and 0 <= row < canvas_height):
        raise WaterAnalysisError("坐标超出画幅范围。")
    seed = row * canvas_width + col
    if pixels[seed] != WATER:
        raise WaterAnalysisError("该坐标不是已知水面像元。")
    lower = _flood(pixels, canvas_width, canvas_height, seed, allow_unknown=False)
    upper = _flood(pixels, canvas_width, canvas_height, seed, allow_unknown=True)
    return lower, upper


def _flood(
    pixels: bytes | bytearray,
    width: int,
    height: int,
    seed: int,
    *,
    allow_unknown: bool,
) -> int:
    total = width * height
    visited = bytearray(total)
    visited[seed] = 1
    queue = [seed]
    head = 0
    count = 0
    while head < len(queue):
        index = queue[head]
        head += 1
        count += 1
        x = index % width
        if x and not visited[index - 1] and _passable(pixels[index - 1], allow_unknown):
            visited[index - 1] = 1
            queue.append(index - 1)
        if x + 1 < width and not visited[index + 1] and _passable(
            pixels[index + 1], allow_unknown
        ):
            visited[index + 1] = 1
            queue.append(index + 1)
        if index >= width and not visited[index - width] and _passable(
            pixels[index - width], allow_unknown
        ):
            visited[index - width] = 1
            queue.append(index - width)
        if index + width < total and not visited[index + width] and _passable(
            pixels[index + width], allow_unknown
        ):
            visited[index + width] = 1
            queue.append(index + width)
    return count


def _passable(state: int, allow_unknown: bool) -> bool:
    return state == WATER or (allow_unknown and state == UNKNOWN)


# --------------------------------------------------------------------------- #
# Mask ingest
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StateGrid:
    pixels: bytes
    water_pixels: int
    land_pixels: int
    unknown_pixels: int


def _decode_png(raw: bytes, label: str) -> Image.Image:
    if not raw:
        raise WaterAnalysisError(f"{label}为空。")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception:
        raise WaterAnalysisError(f"{label}不是可读取的 PNG 图像。") from None
    if image.format != "PNG":
        raise WaterAnalysisError(f"{label}必须是 PNG 格式。")
    return image


def build_state_grid(
    mask_bytes: bytes,
    mosaic_png_bytes: bytes,
    canvas_width: int,
    canvas_height: int,
) -> StateGrid:
    """Combine the black/white survey mask with the mosaic's alpha channel.

    Transparent mosaic pixels are *unknown* regardless of the mask; opaque
    pixels are land (mask 0) or water (mask 255).
    """
    mask = _decode_png(mask_bytes, "水陆掩膜")
    if mask.size != (canvas_width, canvas_height):
        raise WaterAnalysisError(
            "水陆掩膜尺寸必须与所选已发布画幅一致"
            f"（画幅 {canvas_width}x{canvas_height}，"
            f"掩膜 {mask.width}x{mask.height}）。"
        )
    mask_l = mask.convert("L")
    histogram = mask_l.histogram()
    if sum(histogram[1:255]) > 0:
        raise WaterAnalysisError("水陆掩膜必须是纯黑白 PNG（只能含 0 和 255）。")
    mask_pixels = mask_l.tobytes()

    mosaic = _decode_png(mosaic_png_bytes, "已发布画幅")
    if mosaic.size != (canvas_width, canvas_height):
        raise WaterAnalysisError("已发布画幅的尺寸与版本记录不一致。")
    alpha = mosaic.convert("RGBA").getchannel("A").tobytes()

    # 0 (black) -> land, 255 (white) -> water; intermediate values were
    # rejected above.  Transparent mosaic pixels are unknown regardless.
    mask_states = mask_pixels.translate(
        bytes([LAND] + [UNKNOWN] * 254 + [WATER])
    )
    states = bytearray(
        UNKNOWN if a == 0 else m for m, a in zip(mask_states, alpha)
    )
    water = states.count(WATER)
    land = states.count(LAND)
    return StateGrid(
        pixels=bytes(states),
        water_pixels=water,
        land_pixels=land,
        unknown_pixels=canvas_width * canvas_height - water - land,
    )


def mask_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------- #
# Re-survey CSV
# --------------------------------------------------------------------------- #

_INT_RE = re.compile(r"[+-]?\d+")
_WATER_WORDS = {"water", "w", "水", "水面", "1", "true", "yes", "y"}
_LAND_WORDS = {"land", "l", "陆", "陆地", "地", "0", "false", "no", "n"}
_UNKNOWN_WORDS = {"unknown", "u", "未知", "?"}
_HEADER_WORDS = {"col", "column", "x", "列"}


def _parse_state(raw: str, line_no: int) -> int:
    value = (raw or "").strip().lower()
    if value in _WATER_WORDS:
        return WATER
    if value in _LAND_WORDS:
        return LAND
    if value in _UNKNOWN_WORDS:
        raise WaterAnalysisError(
            f"CSV 第 {line_no} 行：补测只能改判为水面或陆地，不能改回未知。"
        )
    raise WaterAnalysisError(
        f"CSV 第 {line_no} 行：无法识别状态 “{raw.strip()}”"
        "（可用 water/水/1 或 land/陆/0）。"
    )


def parse_reclassification_csv(
    raw: bytes, canvas_width: int, canvas_height: int
) -> list[tuple[int, int, int]]:
    """Parse ``col,row,state`` rows; an optional header row is skipped."""
    if len(raw) > MAX_RECLASS_BYTES:
        raise WaterAnalysisError("CSV 文件不能超过 64 MB。")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise WaterAnalysisError("CSV 必须是 UTF-8 编码。") from None
    reader = csv.reader(io.StringIO(text))
    changes: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    first_row = True
    for line_no, row in enumerate(reader, start=1):
        if not row or all(not field.strip() for field in row):
            continue
        if row[0].lstrip().startswith("#"):
            continue
        if first_row:
            first_row = False
            if row[0].strip().lower() in _HEADER_WORDS:
                continue
        if len(row) != 3:
            raise WaterAnalysisError(
                f"CSV 第 {line_no} 行：应为 3 列（列,行,状态），实际 {len(row)} 列。"
            )
        col_text, row_text, state_text = (field.strip() for field in row)
        if not _INT_RE.fullmatch(col_text) or not _INT_RE.fullmatch(row_text):
            raise WaterAnalysisError(
                f"CSV 第 {line_no} 行：列和行必须是从 0 开始的整数。"
            )
        col = int(col_text)
        row_index = int(row_text)
        if not (0 <= col < canvas_width and 0 <= row_index < canvas_height):
            raise WaterAnalysisError(
                f"CSV 第 {line_no} 行：坐标 ({col},{row_index}) 超出画幅 "
                f"{canvas_width}x{canvas_height}。"
            )
        if (col, row_index) in seen:
            raise WaterAnalysisError(
                f"CSV 第 {line_no} 行：坐标 ({col},{row_index}) 重复出现。"
            )
        seen.add((col, row_index))
        changes.append((col, row_index, _parse_state(state_text, line_no)))
        if len(changes) > MAX_RECLASS_ROWS:
            raise WaterAnalysisError("CSV 改判行数超过上限（200 万行）。")
    if not changes:
        raise WaterAnalysisError("CSV 中没有可应用的改判行。")
    return changes
