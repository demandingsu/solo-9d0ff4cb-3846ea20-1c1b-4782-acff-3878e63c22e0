"""Water-area lower/upper bounds over a published mosaic, in 128x128 blocks."""

from __future__ import annotations

import io
import random

import pytest
from PIL import Image

from app import analysis, water
from app.water import (
    LAND,
    UNKNOWN,
    WATER,
    WaterAnalysisError,
    block_bounds,
    block_columns,
    block_rows,
    build_state_grid,
    full_bounds,
    parse_reclassification_csv,
    query_bounds,
    summarize_block,
    summary_columns,
)
from tests.conftest import mask_bytes


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _grid_from_states(width, height, states):
    """Build a mosaic PNG + survey mask that encodes these pixel states."""
    mosaic = Image.new("RGBA", (width, height), (20, 40, 60, 255))
    mask = Image.new("L", (width, height), 0)
    mosaic_px = mosaic.load()
    mask_px = mask.load()
    for index, state in enumerate(states):
        x, y = index % width, index // width
        if state == UNKNOWN:
            mosaic_px[x, y] = (0, 0, 0, 0)
        elif state == WATER:
            mask_px[x, y] = 255
        else:
            mask_px[x, y] = 0

    def encode(image, mode):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    return encode(mask, "L"), encode(mosaic, "RGBA")


def _random_states(width, height, weights=(5, 4, 1), seed=0):
    random.seed(seed)
    return random.choices([LAND, WATER, UNKNOWN], weights=weights, k=width * height)


def _build_views(states, width, height):
    """Summaries in the same form the database stores them."""
    plain, labelled = {}, {}
    for by in range(block_rows(height)):
        for bx in range(block_columns(width)):
            x0, y0, w, h = block_bounds(bx, by, width, height)
            block = bytes(
                states[(y0 + y) * width + x0 + x] for y in range(h) for x in range(w)
            )
            columns = summary_columns(summarize_block(bx, by, w, h, block))
            plain[(bx, by)] = water.block_view_from_row(columns, False)
            labelled[(bx, by)] = water.block_view_from_row(columns, True)
    return plain, labelled


def _loader(plain, labelled):
    def load(bx, by, include_labels):
        return labelled[(bx, by)] if include_labels else plain[(bx, by)]

    return load


# --------------------------------------------------------------------------- #
# Pure geometry / summary equivalence
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "width,height",
    [(1, 2), (3, 5), (128, 128), (129, 127), (256, 129), (300, 250), (37, 201)],
)
def test_block_summaries_match_full_flood_fill(width, height):
    states = _random_states(width, height, seed=width * 31 + height)
    if WATER not in states:  # tiny canvases can miss the random draw
        states[1] = WATER
    plain, labelled = _build_views(states, width, height)
    loader = _loader(plain, labelled)
    pixels = bytes(states)
    checked = 0
    for index, state in enumerate(states):
        if state != WATER:
            continue
        col, row = index % width, index // width
        result = query_bounds(loader, width, height, col, row)
        assert (result.lower, result.upper) == full_bounds(
            pixels, width, height, col, row
        )
        checked += 1
    assert checked > 0


def test_cloud_bridge_changes_upper_bound_only():
    """A cloud band separates two waters in the known layer but bridges them."""
    width, height = 6, 5
    states = [LAND] * (width * height)
    # Left lagoon and right lagoon, separated by a column of unknown pixels.
    for y in range(1, 4):
        states[y * width + 1] = WATER
        states[y * width + 3] = WATER
        states[y * width + 2] = UNKNOWN
    plain, labelled = _build_views(states, width, height)
    result = query_bounds(_loader(plain, labelled), width, height, 1, 2)
    assert result.lower == 3
    assert result.upper == 9  # both lagoons plus the three unknown pixels
    assert result.uncertain == 6

    # Filling the cloud column with land (a field re-survey) disconnects them.
    changes = [(2, y, LAND) for y in range(1, 4)]
    plan = analysis.PatchPlan(
        run_id="x",
        base_patch_id=None,
        current_pixels={(0, 0): bytes(states)},
    )
    computation = analysis.compute_patch(plan, changes, width, height)
    assert computation.changed_pixels == 3
    after_columns = summary_columns(computation.summaries[(0, 0)])
    after = query_bounds(
        _loader(
            {(0, 0): water.block_view_from_row(after_columns, False)},
            {(0, 0): water.block_view_from_row(after_columns, True)},
        ),
        width,
        height,
        1,
        2,
    )
    assert (after.lower, after.upper) == (3, 3)


def test_summary_equivalence_across_block_boundaries():
    """One lake spanning four blocks: global weight equals the pixel count."""
    width, height = 300, 300
    states = [LAND] * (width * height)
    for y in range(60, 240):
        for x in range(60, 240):
            states[y * width + x] = WATER
    # Stripes of unknown inside it, plus a land wall with a single gap.
    for y in range(120, 180):
        for x in range(150, 155):
            states[y * width + x] = UNKNOWN
    plain, labelled = _build_views(states, width, height)
    loader = _loader(plain, labelled)
    result = query_bounds(loader, width, height, 70, 70)
    expected = full_bounds(bytes(states), width, height, 70, 70)
    assert (result.lower, result.upper) == expected
    assert result.lower == 180 * 180 - 60 * 5
    assert result.upper == 180 * 180
    assert result.lower_components == 4  # straddles block boundaries


def test_seed_must_be_known_water():
    states = [LAND, UNKNOWN, WATER, WATER]
    plain, labelled = _build_views(states, 2, 2)
    loader = _loader(plain, labelled)
    with pytest.raises(WaterAnalysisError):
        query_bounds(loader, 2, 2, 0, 0)  # land
    with pytest.raises(WaterAnalysisError):
        query_bounds(loader, 2, 2, 1, 0)  # unknown
    with pytest.raises(WaterAnalysisError):
        query_bounds(loader, 2, 2, 2, 0)  # out of range
    assert query_bounds(loader, 2, 2, 0, 1).lower == 2


# --------------------------------------------------------------------------- #
# Mask ingest
# --------------------------------------------------------------------------- #


def test_mask_transparency_is_unknown_and_mask_must_be_binary():
    w, h = 3, 2
    mosaic = Image.new("RGBA", (w, h), (10, 10, 10, 255))
    mosaic.putpixel((2, 0), (0, 0, 0, 0))
    buffer = io.BytesIO()
    mosaic.save(buffer, format="PNG")
    mask = Image.new("L", (w, h), 0)
    mask.putpixel((0, 0), 255)
    mask.putpixel((2, 0), 255)
    mask_buffer = io.BytesIO()
    mask.save(mask_buffer, format="PNG")
    grid = build_state_grid(mask_buffer.getvalue(), buffer.getvalue(), w, h)
    assert grid.pixels[0] == WATER
    assert grid.pixels[2] == UNKNOWN  # transparency wins
    assert (grid.water_pixels, grid.land_pixels, grid.unknown_pixels) == (1, 4, 1)

    with pytest.raises(WaterAnalysisError):
        build_state_grid(mask_bytes(2, 2), buffer.getvalue(), w, h)  # wrong size
    grey = Image.new("L", (w, h), 128)
    grey_buffer = io.BytesIO()
    grey.save(grey_buffer, format="PNG")
    with pytest.raises(WaterAnalysisError):
        build_state_grid(grey_buffer.getvalue(), buffer.getvalue(), w, h)


def test_reclassification_csv_parsing():
    changes = parse_reclassification_csv(
        b"col,row,state\n0,0,water\n1,0,land\n2,1,\xe6\xb0\xb4\n", 3, 2
    )
    assert changes == [(0, 0, WATER), (1, 0, LAND), (2, 1, WATER)]
    with pytest.raises(WaterAnalysisError):
        parse_reclassification_csv(b"0,0,water\n0,0,land\n", 3, 2)  # duplicate
    with pytest.raises(WaterAnalysisError):
        parse_reclassification_csv(b"0,5,water\n", 3, 2)  # out of range
    with pytest.raises(WaterAnalysisError):
        parse_reclassification_csv(b"0,0,unknown\n", 3, 2)  # cannot go unknown
    with pytest.raises(WaterAnalysisError):
        parse_reclassification_csv(b"\n", 3, 2)  # nothing to do


# --------------------------------------------------------------------------- #
# Web flow
# --------------------------------------------------------------------------- #

MOSAIC_W, MOSAIC_H = 8, 6


def _publish_small_canvas(client, upload, canvas, *, cloud=(), colour=(30, 60, 90)):
    canvas(MOSAIC_W, MOSAIC_H)
    upload(
        tile_id="T", revision=1, x=0, y=0, colour=colour,
        width=MOSAIC_W, height=MOSAIC_H, cloud=cloud,
    )
    assert client.post("/publish", follow_redirects=False).status_code == 303
    return client.get("/versions").json()[0]


def _analysis_form(client, version_id, mask):
    return client.post(
        "/analysis",
        data={"version_id": version_id},
        files={"mask": ("mask.png", mask, "image/png")},
        follow_redirects=False,
    )


def _query(client, run_url, col, row):
    return client.get(f"{run_url}/query", params={"col": str(col), "row": str(row)})


def _runs(client):
    with client.app.state.db_lock as conn:
        return analysis.list_runs(conn)


def test_web_analysis_lower_upper_and_cloud_bridge(client, upload, canvas):
    # Columns 2 and 4 are cloudy, so the published mosaic is transparent
    # there: two lakes (columns 1 and 3) split by a potential land bridge.
    published = _publish_small_canvas(
        client, upload, canvas,
        cloud={(x, y) for x in (2, 4) for y in range(MOSAIC_H)},
    )
    w, h = MOSAIC_W, MOSAIC_H
    mask_image = Image.new("L", (w, h), 0)
    for y in range(h):
        mask_image.putpixel((1, y), 255)
        mask_image.putpixel((3, y), 255)
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")

    response = _analysis_form(client, published["id"], buffer.getvalue())
    assert response.status_code == 303
    run = _runs(client)[0]
    page = client.get(f"/analysis/{run.id}")
    assert page.status_code == 200
    assert "水面面积范围" in page.text

    result = _query(client, f"/analysis/{run.id}", 1, 2)
    assert result.status_code == 200
    assert result.context["query_result"].lower == h
    # Transparent columns 2 and 4 are unknown; treating them as water reaches
    # the second lake (col 3) and the unknown column 4, but not the land at 5.
    assert result.context["query_result"].upper == 4 * h

    # A land point is not a valid seed.
    bad = _query(client, f"/analysis/{run.id}", 5, 2)
    assert "不是已知水面" in bad.context["query_error"]


def test_web_patch_reevaluates_only_changed_blocks(client, upload, canvas):
    # Make columns 4-5 cloudy so both lakes in the mask have no known path.
    cloud = {(x, y) for x in (4, 5) for y in range(MOSAIC_H)}
    published = _publish_small_canvas(client, upload, canvas, cloud=cloud)
    w, h = MOSAIC_W, MOSAIC_H
    mask = Image.new("L", (w, h), 0)
    for y in range(h):
        for x in range(w):
            if 1 <= x <= 3 or 6 <= x <= 8:
                mask.putpixel((x, y), 255)
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    _analysis_form(client, published["id"], buffer.getvalue())
    run = _runs(client)[0]

    result = _query(client, f"/analysis/{run.id}", 1, 1)
    before = result.context["query_result"]
    assert before.lower == 3 * h
    # The unknown (transparent) columns 4-5 plus the far lake (cols 6-7) are
    # all reachable when unknown counts as water.
    assert before.upper == 3 * h + 2 * h + 2 * h

    # Field survey: prove column 4 is water and column 5 is land at row 0,
    # which joins the two lakes for the lower bound.
    csv = b"col,row,state\n4,0,water\n5,0,land\n"
    response = client.post(
        f"/analysis/{run.id}/patch",
        files={"csv_file": ("patch.csv", csv, "text/csv")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "msg=" in response.headers["location"]
    result = _query(client, f"/analysis/{run.id}", 1, 1)
    after = result.context["query_result"]
    assert after.lower == 3 * h + 1
    assert after.blocks_visited == before.blocks_visited  # one block only

    # Recomputing from the full canvas gives the same answer.
    with client.app.state.db_lock as conn:
        stored = conn.execute(
            "SELECT * FROM analysis_blocks WHERE run_id = ?", (run.id,)
        ).fetchone()
        view = water.block_view_from_row(stored, True, include_pixels=True)
    full = full_bounds(view.pixels, view.width, view.height, 1, 1)
    assert (after.lower, after.upper) == full


def _all_block_pixels(client, run_id):
    """Reassemble the canvas pixel states from every stored block."""
    with client.app.state.db_lock as conn:
        run = analysis.get_run(conn, run_id)
        rows = conn.execute(
            "SELECT * FROM analysis_blocks WHERE run_id = ? ORDER BY by, bx",
            (run_id,),
        ).fetchall()
    canvas = bytearray(run.canvas_width * run.canvas_height)
    for row in rows:
        view = water.block_view_from_row(row, True, include_pixels=True)
        x0, y0, w, h = block_bounds(view.bx, view.by, run.canvas_width, run.canvas_height)
        for y in range(h):
            start = (y0 + y) * run.canvas_width + x0
            canvas[start:start + w] = view.pixels[y * w:(y + 1) * w]
    return run, bytes(canvas)


def test_web_query_matches_full_recompute_across_blocks(client, upload, canvas):
    width, height = 140, 130  # spans two blocks in each direction
    canvas(width, height)
    # A rim of land, a lake with a cloudy (transparent) corridor through it.
    cloud = {(x, 40) for x in range(width)}
    upload(tile_id="T", revision=1, x=0, y=0, colour=(40, 80, 120),
           width=width, height=height, cloud=cloud)
    assert client.post("/publish", follow_redirects=False).status_code == 303
    published = client.get("/versions").json()[0]

    mask_image = Image.new("L", (width, height), 0)
    for y in range(10, 120):
        for x in range(10, 130):
            mask_image.putpixel((x, y), 255)
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    assert _analysis_form(client, published["id"], buffer.getvalue()).status_code == 303
    run = _runs(client)[0]

    seeds = [(20, 20), (100, 20), (20, 100), (129, 119), (11, 41)]
    result = _query(client, f"/analysis/{run.id}", *seeds[0])
    before = result.context["query_result"]
    # Row 40 is unknown: it splits the lake for the lower bound but bridges it
    # for the upper bound (the whole lake).
    assert before.lower == 120 * 30  # rows 10..39, 120 columns each
    assert before.upper == 110 * 120 + 20  # lake plus the unknown row inside it

    _, pixels = _all_block_pixels(client, run.id)
    for col, row in seeds:
        queried = _query(client, f"/analysis/{run.id}", col, row).context["query_result"]
        assert (queried.lower, queried.upper) == full_bounds(
            pixels, width, height, col, row
        )

    # Field patch: prove one pixel of the cloudy row is land, then water.
    patch = b"col,row,state\n70,40,land\n"
    response = client.post(
        f"/analysis/{run.id}/patch",
        files={"csv_file": ("p.csv", patch, "text/csv")},
        follow_redirects=False,
    )
    assert "msg=" in response.headers["location"]
    _, pixels = _all_block_pixels(client, run.id)
    for col, row in seeds:
        queried = _query(client, f"/analysis/{run.id}", col, row).context["query_result"]
        assert (queried.lower, queried.upper) == full_bounds(
            pixels, width, height, col, row
        )


def test_web_patch_can_disconnect_water(client, upload, canvas):
    width, height = 9, 5
    canvas(width, height)
    upload(tile_id="T", revision=1, x=0, y=0, colour=(0, 90, 90),
           width=width, height=height)
    assert client.post("/publish", follow_redirects=False).status_code == 303
    published = client.get("/versions").json()[0]

    mask_image = Image.new("L", (width, height), 255)  # all water
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    _analysis_form(client, published["id"], buffer.getvalue())
    run = _runs(client)[0]

    whole = _query(client, f"/analysis/{run.id}", 4, 2).context["query_result"]
    assert (whole.lower, whole.upper) == (width * height, width * height)

    # Reclassify column 4 as land: the seed's lake is now just the left half.
    csv = "col,row,state\n" + "".join(f"4,{y},land\n" for y in range(height))
    response = client.post(
        f"/analysis/{run.id}/patch",
        files={"csv_file": ("p.csv", csv.encode(), "text/csv")},
        follow_redirects=False,
    )
    assert "msg=" in response.headers["location"]
    left = _query(client, f"/analysis/{run.id}", 3, 2).context["query_result"]
    right = _query(client, f"/analysis/{run.id}", 5, 2).context["query_result"]
    assert (left.lower, left.upper) == (4 * height, 4 * height)
    assert (right.lower, right.upper) == (4 * height, 4 * height)
    _, pixels = _all_block_pixels(client, run.id)
    assert (left.lower, left.upper) == full_bounds(pixels, width, height, 3, 2)


def test_failed_patch_keeps_previous_results(client, upload, canvas, monkeypatch):
    width, height = 6, 4
    canvas(width, height)
    upload(tile_id="T", revision=1, x=0, y=0, colour=(0, 0, 0),
           width=width, height=height)
    assert client.post("/publish", follow_redirects=False).status_code == 303
    published = client.get("/versions").json()[0]
    mask_image = Image.new("L", (width, height), 0)
    for y in range(height):
        for x in range(3):
            mask_image.putpixel((x, y), 255)
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    _analysis_form(client, published["id"], buffer.getvalue())
    run = _runs(client)[0]

    before = _query(client, f"/analysis/{run.id}", 0, 0).context["query_result"]
    assert before.lower == 3 * height

    real_commit = analysis.commit_patch

    def commit_then_fail(*args, **kwargs):
        real_commit(*args, **kwargs)
        raise RuntimeError("simulated disk failure after writing summaries")

    monkeypatch.setattr(analysis, "commit_patch", commit_then_fail)
    csv = b"col,row,state\n3,0,water\n"
    with pytest.raises(RuntimeError):
        client.post(
            f"/analysis/{run.id}/patch",
            files={"csv_file": ("p.csv", csv, "text/csv")},
            follow_redirects=False,
        )
    monkeypatch.setattr(analysis, "commit_patch", real_commit)

    # The rolled-back transaction left both the summaries and the patch log
    # untouched: the original result is still queryable.
    after = _query(client, f"/analysis/{run.id}", 0, 0).context["query_result"]
    assert (after.lower, after.upper) == (before.lower, before.upper)
    with client.app.state.db_lock as conn:
        assert analysis.patch_count(conn, run.id) == 0

    # The same CSV can be applied once the failure is resolved.
    ok = client.post(
        f"/analysis/{run.id}/patch",
        files={"csv_file": ("p.csv", csv, "text/csv")},
        follow_redirects=False,
    )
    assert "msg=" in ok.headers["location"]
    fixed = _query(client, f"/analysis/{run.id}", 0, 0).context["query_result"]
    assert (fixed.lower, fixed.upper) == (3 * height + 1, 3 * height + 1)


def test_patch_relabels_only_changed_blocks(client, upload, canvas, monkeypatch):
    """Only the changed block is relabelled, never the whole canvas."""
    width, height = 300, 130
    canvas(width, height)
    upload(tile_id="T", revision=1, x=0, y=0, colour=(0, 0, 0),
           width=width, height=height)
    assert client.post("/publish", follow_redirects=False).status_code == 303
    published = client.get("/versions").json()[0]
    mask_image = Image.new("L", (width, height), 0)
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    _analysis_form(client, published["id"], buffer.getvalue())
    run = _runs(client)[0]

    calls = []
    real = analysis.summarize_block

    def counting(*args, **kwargs):
        calls.append((args[0], args[1]))
        return real(*args, **kwargs)

    monkeypatch.setattr(analysis, "summarize_block", counting)
    # Two coordinates, both inside block (1, 0).
    csv = b"col,row,state\n200,10,water\n201,10,water\n"
    client.post(
        f"/analysis/{run.id}/patch",
        files={"csv_file": ("p.csv", csv, "text/csv")},
        follow_redirects=False,
    )
    assert calls == [(1, 0)]
    result = _query(client, f"/analysis/{run.id}", 200, 10).context["query_result"]
    assert (result.lower, result.upper) == (2, 2)
