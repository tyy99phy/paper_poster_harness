from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from poster_harness.replace import (
    audit_figure_containment,
    audit_generated_placeholder_geometry,
    normalize_placeholder_geometry,
    replace_placeholders,
)


def test_normalize_placeholder_geometry_matches_declared_aspects(tmp_path: Path):
    base = tmp_path / "base.png"
    Image.new("RGB", (1024, 1536), "white").save(base)
    spec = {
        "placeholders": [
            {"id": "FIG 01", "label": "Square result", "aspect": "1:1 square"},
            {"id": "FIG 02", "label": "Wide distribution", "aspect": "2.5:1 wide"},
        ],
        "placements": {
            "FIG 01": [100, 100, 400, 250],
            "FIG 02": [100, 400, 500, 500],
        },
    }
    _, updated = normalize_placeholder_geometry(base_image=base, spec=spec, out_path=tmp_path / "out.png")
    sq = updated["placements"]["FIG 01"]
    wide = updated["placements"]["FIG 02"]
    sq_ratio = (sq[2] - sq[0]) / (sq[3] - sq[1])
    wide_ratio = (wide[2] - wide[0]) / (wide[3] - wide[1])
    assert abs(sq_ratio - 1.0) < 0.03
    assert abs(wide_ratio - 2.5) < 0.05
    assert sq[2] - sq[0] >= 280
    assert wide[2] - wide[0] >= 500
    for box in (sq, wide):
        assert box[0] >= 0 and box[1] >= 0
        assert box[2] <= 1024 and box[3] <= 1536


def test_square_result_in_landscape_seed_is_centered_not_nudged(tmp_path: Path):
    base = tmp_path / "base.png"
    Image.new("RGB", (500, 500), "white").save(base)
    spec = {
        "placeholders": [
            {"id": "FIG 01", "label": "95% CL upper limit result", "aspect": "1:1 square"},
        ],
        # The generated placeholder/mat is slightly landscape.  Replacement
        # should fit a centered square inside it, not shrink and push the real
        # limit plot up-left leaving a large right/bottom white border.
        "placements": {
            "FIG 01": [100, 100, 260, 220],
        },
    }
    _, updated = normalize_placeholder_geometry(
        base_image=base,
        spec=spec,
        out_path=tmp_path / "out.png",
        redraw=False,
    )
    assert updated["placements"]["FIG 01"] == [130, 110, 230, 210]
    assert updated["_replacement_clear_boxes"]["FIG 01"] == [100, 100, 260, 220]


def test_square_result_single_figure_repair_caps_downward_erase(tmp_path: Path):
    base = tmp_path / "base.png"
    Image.new("RGB", (500, 1000), "white").save(base)
    spec = {
        "placeholders": [
            {
                "id": "FIG 01",
                "label": "95% CL upper limit result",
                "aspect": "1:1 square",
                "asset": "fig.png",
            },
        ],
        # Lower-poster square result: hidden normalization nudges it upward.  The
        # old placeholder area may still need erasing, but the white erase mat
        # must not extend below the repaired square/result envelope.
        "placements": {
            "FIG 01": [150, 760, 350, 960],
        },
    }
    _, updated = normalize_placeholder_geometry(
        base_image=base,
        spec=spec,
        out_path=tmp_path / "out.png",
        redraw=False,
    )
    box = updated["placements"]["FIG 01"]
    clear = updated["_replacement_clear_boxes"]["FIG 01"]
    erase = updated["_replacement_erase_boxes"]["FIG 01"]
    frame = updated["_replacement_frame_boxes"]["FIG 01"]
    assert box[3] <= 960
    assert erase[0] >= clear[0]
    assert erase[1] >= clear[1]
    assert erase[2] <= clear[2]
    assert erase[3] <= clear[3]
    assert frame[0] >= clear[0]
    assert frame[1] >= clear[1]
    assert frame[2] <= clear[2]
    assert frame[3] <= clear[3]
    assert not audit_figure_containment(spec=updated)


def test_audit_flags_square_result_visual_envelope_protrusion():
    spec = {
        "placeholders": [
            {
                "id": "FIG 01",
                "label": "95% CL upper limit result",
                "aspect": "1:1 square",
                "asset": "fig.png",
            },
        ],
        "placements": {"FIG 01": [100, 100, 200, 200]},
        "_replacement_clear_boxes": {"FIG 01": [100, 100, 200, 200]},
        "_replacement_erase_boxes": {"FIG 01": [94, 88, 206, 248]},
        "_replacement_frame_boxes": {"FIG 01": [100, 100, 200, 200]},
    }
    issues = audit_figure_containment(spec=spec)
    assert any(issue.get("category") == "figure_visual_envelope" for issue in issues)


def test_audit_flags_square_result_without_enough_inner_margin():
    spec = {
        "placeholders": [
            {
                "id": "FIG 01",
                "label": "95% CL upper limit result",
                "aspect": "1:1 square",
                "asset": "fig.png",
            },
        ],
        "placements": {"FIG 01": [140, 140, 1060, 1060]},
        "_replacement_clear_boxes": {"FIG 01": [100, 100, 1100, 1100]},
        "_replacement_erase_boxes": {"FIG 01": [100, 100, 1100, 1100]},
        "_replacement_frame_boxes": {"FIG 01": [100, 100, 1100, 1100]},
    }
    issues = audit_figure_containment(spec=spec)
    assert any(issue.get("category") == "figure_inner_margin" for issue in issues)


def test_audit_flags_square_result_white_board_that_is_too_large():
    spec = {
        "placeholders": [
            {
                "id": "FIG 01",
                "label": "95% CL upper limit result",
                "aspect": "1:1 square",
                "asset": "fig.png",
            },
        ],
        "placements": {"FIG 01": [250, 250, 850, 850]},
        "_replacement_clear_boxes": {"FIG 01": [100, 100, 1000, 1000]},
        "_replacement_erase_boxes": {"FIG 01": [100, 100, 1000, 1000]},
        "_replacement_frame_boxes": {"FIG 01": [100, 100, 1000, 1000]},
    }
    issues = audit_figure_containment(spec=spec)
    assert any(issue.get("category") == "figure_frame_oversized" for issue in issues)


def test_hidden_normalize_uses_inner_dashed_result_placeholder_not_outer_card(tmp_path: Path):
    base = tmp_path / "base.png"
    im = Image.new("RGB", (640, 640), (246, 244, 238))
    draw = ImageDraw.Draw(im)
    # Outer light result card returned by a simulated vision detection.
    draw.rounded_rectangle([80, 80, 560, 560], radius=24, fill=(250, 248, 242), outline=(210, 170, 70), width=3)
    # The actual initial image_generation placeholder is the inner dashed box.
    for x in range(120, 520, 30):
        draw.line([(x, 150), (min(x + 14, 520), 150)], fill=(95, 85, 75), width=3)
        draw.line([(x, 510), (min(x + 14, 520), 510)], fill=(95, 85, 75), width=3)
    for y in range(150, 510, 30):
        draw.line([(120, y), (120, min(y + 14, 510))], fill=(95, 85, 75), width=3)
        draw.line([(520, y), (520, min(y + 14, 510))], fill=(95, 85, 75), width=3)
    im.save(base)
    spec = {
        "placeholders": [
            {
                "id": "FIG 01",
                "label": "95% CL upper limit result",
                "aspect": "1:1 square",
                "asset": "fig.png",
            }
        ],
        "placements": {"FIG 01": [80, 80, 560, 560]},
    }
    _, updated = normalize_placeholder_geometry(
        base_image=base,
        spec=spec,
        out_path=tmp_path / "planned.png",
        redraw=False,
    )
    clear = updated["_replacement_clear_boxes"]["FIG 01"]
    box = updated["placements"]["FIG 01"]
    assert clear[0] <= 125 and clear[0] >= 110
    assert clear[1] <= 155 and clear[1] >= 140
    assert clear[2] >= 515 and clear[2] <= 525
    assert clear[3] >= 505 and clear[3] <= 515
    assert box[0] > clear[0]
    assert box[1] > clear[1]
    assert box[2] < clear[2]
    assert box[3] < clear[3]


def test_lower_hero_moderate_landscape_is_kept_above_conclusion_zone(tmp_path: Path):
    base = tmp_path / "base.png"
    Image.new("RGB", (500, 1000), "white").save(base)
    spec = {
        "placeholders": [
            {
                "id": "FIG 01",
                "label": "Headline comparison of W mass measurements",
                "aspect": "1.49:1 wide",
                "asset": "fig.png",
                "group": "hero_result",
            },
        ],
        "placements": {
            "FIG 01": [100, 700, 400, 920],
        },
    }
    _, updated = normalize_placeholder_geometry(
        base_image=base,
        spec=spec,
        out_path=tmp_path / "out.png",
        redraw=False,
    )
    box = updated["placements"]["FIG 01"]
    clear = updated["_replacement_clear_boxes"]["FIG 01"]
    assert box[3] <= 890
    assert clear[3] <= 890
    assert abs(((box[2] - box[0]) / (box[3] - box[1])) - 1.49) < 0.03


def test_lower_hero_frame_does_not_extend_downward_past_target(tmp_path: Path):
    base = tmp_path / "base.png"
    Image.new("RGB", (500, 1000), "white").save(base)
    spec = {
        "placeholders": [
            {
                "id": "FIG 01",
                "label": "Headline comparison of W mass measurements",
                "aspect": "1.49:1 wide",
                "asset": "fig.png",
                "group": "hero_result",
            },
        ],
        "placements": {
            "FIG 01": [100, 700, 400, 920],
        },
    }
    _, updated = normalize_placeholder_geometry(
        base_image=base,
        spec=spec,
        out_path=tmp_path / "out.png",
        redraw=False,
    )
    box = updated["placements"]["FIG 01"]
    frame = updated["_replacement_frame_boxes"]["FIG 01"]
    erase = updated["_replacement_erase_boxes"]["FIG 01"]
    assert frame[3] == box[3]
    assert erase[3] <= 890


def test_normalize_placeholder_geometry_expands_from_inner_text_box_to_visible_panel(tmp_path: Path):
    base = tmp_path / "base.png"
    im = Image.new("RGB", (1024, 1536), "white")
    draw = ImageDraw.Draw(im)
    # Simulate a generated wide placeholder whose LLM detection only captured
    # the central text cluster.
    draw.rectangle([40, 520, 984, 615], outline=(160, 60, 140), width=2)
    im.save(base)
    spec = {
        "placeholders": [
            {"id": "FIG 03", "label": "Post-fit distributions", "aspect": "2.5:1 wide"},
        ],
        "placements": {
            "FIG 03": [455, 535, 570, 595],
        },
    }
    _, updated = normalize_placeholder_geometry(base_image=base, spec=spec, out_path=tmp_path / "out.png")
    box = updated["placements"]["FIG 03"]
    ratio = (box[2] - box[0]) / (box[3] - box[1])
    assert abs(ratio - 2.5) < 0.05
    assert box[2] - box[0] >= 500
    assert box[3] - box[1] >= 190


def test_normalize_adjacent_placeholders_do_not_become_insets_or_overlap(tmp_path: Path):
    base = tmp_path / "base.png"
    im = Image.new("RGB", (540, 360), "white")
    draw = ImageDraw.Draw(im)
    draw.rectangle([20, 120, 250, 310], outline=(120, 65, 150), width=2)
    draw.rectangle([260, 120, 490, 310], outline=(120, 65, 150), width=2)
    im.save(base)
    spec = {
        "placeholders": [
            {"id": "FIG 01", "label": "Left VBF diagram", "aspect": "1.2:1"},
            {"id": "FIG 02", "label": "Right VBF diagram", "aspect": "1.2:1"},
        ],
        "placements": {
            "FIG 01": [45, 140, 225, 290],
            "FIG 02": [285, 140, 465, 290],
        },
    }
    _, updated = normalize_placeholder_geometry(base_image=base, spec=spec, out_path=tmp_path / "out.png")
    left = updated["placements"]["FIG 01"]
    right = updated["placements"]["FIG 02"]
    assert left[2] <= right[0]
    assert abs(((left[2] - left[0]) / (left[3] - left[1])) - 1.2) < 0.05
    assert abs(((right[2] - right[0]) / (right[3] - right[1])) - 1.2) < 0.05
    clear = updated["_replacement_clear_boxes"]
    assert clear["FIG 01"][0] <= 20 and clear["FIG 01"][2] >= 250
    assert clear["FIG 02"][0] <= 260 and clear["FIG 02"][2] >= 490


def test_replace_clears_full_original_placeholder_region(tmp_path: Path):
    base = tmp_path / "base.png"
    im = Image.new("RGB", (220, 160), "white")
    draw = ImageDraw.Draw(im)
    draw.line([(30, 120), (190, 120)], fill=(120, 65, 150), width=2)
    im.save(base)
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    Image.new("RGB", (80, 80), "red").save(asset_dir / "fig.png")
    spec = {
        "placeholders": [{"id": "FIG 01", "label": "Square diagnostic", "aspect": "1:1 square", "asset": "fig.png"}],
        "placements": {"FIG 01": [70, 35, 150, 115]},
        "_replacement_clear_boxes": {"FIG 01": [30, 30, 190, 125]},
    }
    out = tmp_path / "out.png"
    replace_placeholders(base_image=base, spec=spec, asset_dir=asset_dir, out_path=out)
    result = Image.open(out).convert("RGB")
    assert result.getpixel((40, 120)) == (255, 255, 255)


def test_replace_never_pastes_real_figure_outside_placeholder_box(tmp_path: Path):
    base = tmp_path / "base.png"
    Image.new("RGB", (180, 140), "white").save(base)
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    Image.new("RGB", (80, 80), "red").save(asset_dir / "fig.png")
    spec = {
        "placeholders": [{"id": "FIG 01", "label": "Square diagnostic", "aspect": "1:1 square", "asset": "fig.png"}],
        "placements": {"FIG 01": [60, 30, 120, 90]},
        # Clear a larger visible placeholder panel, but the actual real figure
        # still must stay within the placement box.
        "_replacement_clear_boxes": {"FIG 01": [40, 20, 140, 100]},
    }
    out = tmp_path / "out.png"
    replace_placeholders(base_image=base, spec=spec, asset_dir=asset_dir, out_path=out)
    result = Image.open(out).convert("RGB")
    red_pixels = []
    for y in range(result.height):
        for x in range(result.width):
            if result.getpixel((x, y)) == (255, 0, 0):
                red_pixels.append((x, y))
    assert red_pixels
    assert min(x for x, _ in red_pixels) >= 60
    assert max(x for x, _ in red_pixels) < 120
    assert min(y for _, y in red_pixels) >= 30
    assert max(y for _, y in red_pixels) < 90
    # With no replacement padding, a same-aspect asset should maximize overlap.
    assert result.getpixel((60, 30)) == (255, 0, 0)
    assert result.getpixel((119, 89)) == (255, 0, 0)


def test_replace_cleans_placeholder_dashes_with_final_frame(tmp_path: Path):
    base = tmp_path / "base.png"
    im = Image.new("RGB", (220, 170), "white")
    draw = ImageDraw.Draw(im)
    # Simulate a generated placeholder dashed border around the placement.
    for x in range(40, 180, 16):
        draw.line([(x, 30), (min(x + 7, 180), 30)], fill=(170, 120, 25), width=2)
        draw.line([(x, 130), (min(x + 7, 180), 130)], fill=(170, 120, 25), width=2)
    for y in range(30, 130, 16):
        draw.line([(40, y), (40, min(y + 7, 130))], fill=(170, 120, 25), width=2)
        draw.line([(180, y), (180, min(y + 7, 130))], fill=(170, 120, 25), width=2)
    im.save(base)
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    Image.new("RGB", (80, 80), "red").save(asset_dir / "fig.png")
    spec = {
        "placeholders": [{"id": "FIG 01", "label": "Result", "aspect": "1:1 square", "asset": "fig.png"}],
        "placements": {"FIG 01": [70, 45, 150, 125]},
        "_replacement_clear_boxes": {"FIG 01": [40, 30, 180, 130]},
    }
    out = tmp_path / "out.png"
    replace_placeholders(base_image=base, spec=spec, asset_dir=asset_dir, out_path=out)
    result = Image.open(out).convert("RGB")
    assert result.getpixel((40, 30)) != (170, 120, 25)
    assert result.getpixel((179, 129)) != (170, 120, 25)
    assert result.getpixel((70, 45)) == (255, 0, 0)
    assert result.getpixel((149, 124)) == (255, 0, 0)


def test_replace_rectangular_erase_covers_antialiased_dashes_on_detected_edge(tmp_path: Path):
    base = tmp_path / "base.png"
    im = Image.new("RGB", (220, 170), "white")
    draw = ImageDraw.Draw(im)
    # Border strokes can sit exactly at the detector's exclusive x1/y1 edge.
    # The final eraser must cover them; a rounded-only clear leaves these
    # strokes visible around pasted figures.
    draw.line([(40, 130), (180, 130)], fill=(170, 120, 25), width=2)
    draw.line([(180, 30), (180, 130)], fill=(170, 120, 25), width=2)
    im.save(base)
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    Image.new("RGB", (80, 80), "red").save(asset_dir / "fig.png")
    spec = {
        "placeholders": [{"id": "FIG 01", "label": "Square diagnostic", "aspect": "1:1 square", "asset": "fig.png"}],
        "placements": {"FIG 01": [70, 45, 150, 125]},
        "_replacement_clear_boxes": {"FIG 01": [40, 30, 180, 130]},
        "_replacement_erase_boxes": {"FIG 01": [40, 30, 182, 132]},
        "_replacement_frame_boxes": {"FIG 01": [40, 30, 180, 130]},
    }
    out = tmp_path / "out.png"
    replace_placeholders(base_image=base, spec=spec, asset_dir=asset_dir, out_path=out)
    result = Image.open(out).convert("RGB")
    assert result.getpixel((180, 130)) != (170, 120, 25)
    assert result.getpixel((181, 131)) == (255, 255, 255)


def test_square_result_erase_does_not_create_large_uniform_rectangle(tmp_path: Path):
    base = tmp_path / "base.png"
    im = Image.new("RGB", (320, 260), "white")
    draw = ImageDraw.Draw(im)
    # A gentle cream gradient in the generated result card.  Pixels away from
    # placeholder glyphs should survive replacement; otherwise the eraser has
    # created a visible same-colored rectangular slab.
    for y in range(260):
        for x in range(320):
            im.putpixel((x, y), (246 + min(6, x // 70), 240 + min(8, y // 45), 228))
    for x in range(50, 270, 24):
        draw.line([(x, 40), (min(x + 12, 270), 40)], fill=(80, 80, 80), width=2)
        draw.line([(x, 220), (min(x + 12, 270), 220)], fill=(80, 80, 80), width=2)
    for y in range(40, 220, 24):
        draw.line([(50, y), (50, min(y + 12, 220))], fill=(80, 80, 80), width=2)
        draw.line([(270, y), (270, min(y + 12, 220))], fill=(80, 80, 80), width=2)
    draw.text((130, 120), "[FIG 01]", fill=(25, 25, 25))
    untouched = im.getpixel((65, 90))
    im.save(base)
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    Image.new("RGB", (80, 80), "red").save(asset_dir / "fig.png")
    spec = {
        "placeholders": [
            {"id": "FIG 01", "label": "95% CL upper limit result", "aspect": "1:1 square", "asset": "fig.png"}
        ],
        "placements": {"FIG 01": [115, 85, 205, 175]},
        "_replacement_clear_boxes": {"FIG 01": [50, 40, 270, 220]},
        "_replacement_erase_boxes": {"FIG 01": [50, 40, 270, 220]},
        "_replacement_frame_boxes": {"FIG 01": [105, 75, 215, 185]},
    }
    out = tmp_path / "out.png"
    replace_placeholders(base_image=base, spec=spec, asset_dir=asset_dir, out_path=out)
    result = Image.open(out).convert("RGB")
    assert result.getpixel((65, 90)) == untouched
    # But a dashed artifact pixel should be repaired.
    assert result.getpixel((50, 40)) != (80, 80, 80)


def test_normalize_can_plan_geometry_without_redrawing_visible_placeholders(tmp_path: Path):
    base = tmp_path / "base.png"
    Image.new("RGB", (320, 240), "white").save(base)
    planned_path = tmp_path / "planned.png"
    spec = {
        "placeholders": [{"id": "FIG 01", "label": "Result", "aspect": "1:1 square"}],
        "placements": {"FIG 01": [60, 40, 220, 120]},
    }
    returned_path, updated = normalize_placeholder_geometry(
        base_image=base,
        spec=spec,
        out_path=planned_path,
        redraw=False,
    )
    assert returned_path == base
    assert not planned_path.exists()
    assert updated["placements"]["FIG 01"][2] - updated["placements"]["FIG 01"][0] == updated["placements"]["FIG 01"][3] - updated["placements"]["FIG 01"][1]
    assert "_replacement_clear_boxes" in updated


def test_hidden_normalize_keeps_replacement_inside_original_placeholder(tmp_path: Path):
    base = tmp_path / "base.png"
    Image.new("RGB", (400, 260), "white").save(base)
    spec = {
        "placeholders": [{"id": "FIG 02", "label": "Wide plot", "aspect": "2.5:1 wide"}],
        # A too-wide but geometry-QA-near placeholder. Hidden planning should
        # shrink inside this box, not grow taller or protrude outside it.
        "placements": {"FIG 02": [20, 80, 380, 180]},
    }
    returned_path, updated = normalize_placeholder_geometry(
        base_image=base,
        spec=spec,
        out_path=tmp_path / "planned.png",
        redraw=False,
    )
    assert returned_path == base
    box = updated["placements"]["FIG 02"]
    assert box[0] >= 20 and box[1] >= 80
    assert box[2] <= 380 and box[3] <= 180
    assert abs(((box[2] - box[0]) / (box[3] - box[1])) - 2.5) < 0.03
    assert updated["_replacement_clear_boxes"]["FIG 02"] == [20, 80, 380, 180]


def test_hidden_normalize_corrects_over_tall_color_consistent_placeholder(tmp_path: Path):
    base = tmp_path / "base.png"
    im = Image.new("RGB", (620, 420), "white")
    draw = ImageDraw.Draw(im)
    # Visible placeholder border is safely above the next section, but the LLM
    # detector included too much vertical whitespace below it.
    for x in range(40, 300, 28):
        draw.line([(x, 100), (min(x + 14, 300), 100)], fill=(120, 65, 150), width=3)
        draw.line([(x, 250), (min(x + 14, 300), 250)], fill=(120, 65, 150), width=3)
    for y in range(100, 250, 28):
        draw.line([(40, y), (40, min(y + 14, 250))], fill=(120, 65, 150), width=3)
        draw.line([(300, y), (300, min(y + 14, 250))], fill=(120, 65, 150), width=3)
    im.save(base)
    spec = {
        "placeholders": [{"id": "FIG 01", "label": "VBF diagram", "aspect": "1.2:1"}],
        "placements": {"FIG 01": [40, 105, 300, 330]},
    }
    _, updated = normalize_placeholder_geometry(
        base_image=base,
        spec=spec,
        out_path=tmp_path / "planned.png",
        redraw=False,
    )
    clear = updated["_replacement_clear_boxes"]["FIG 01"]
    box = updated["placements"]["FIG 01"]
    assert clear[1] <= 100
    assert clear[3] <= 255
    assert box[3] <= clear[3]
    assert abs(((box[2] - box[0]) / (box[3] - box[1])) - 1.2) < 0.03


def test_hidden_normalize_keeps_edge_detection_inside_canvas_gutter(tmp_path: Path):
    base = tmp_path / "base.png"
    Image.new("RGB", (1000, 900), "white").save(base)
    spec = {
        "placeholders": [{"id": "FIG 02", "label": "Wide distribution", "aspect": "2.5:1 wide"}],
        # Simulate an LLM detection that incorrectly starts at x=0 even though
        # scientific figure replacement must not be clipped by the canvas edge.
        "placements": {"FIG 02": [0, 300, 600, 540]},
    }
    _, updated = normalize_placeholder_geometry(
        base_image=base,
        spec=spec,
        out_path=tmp_path / "planned.png",
        redraw=False,
    )
    clear = updated["_replacement_clear_boxes"]["FIG 02"]
    box = updated["placements"]["FIG 02"]
    assert clear[0] >= 25
    assert box[0] >= clear[0]
    assert abs(((box[2] - box[0]) / (box[3] - box[1])) - 2.5) < 0.03


def test_hidden_normalize_recovers_vertical_extent_for_edge_wide_detection(tmp_path: Path):
    base = tmp_path / "base.png"
    Image.new("RGB", (1000, 900), "white").save(base)
    spec = {
        "placeholders": [{"id": "FIG 02", "label": "Wide distribution", "aspect": "2.5:1 wide"}],
        # Edge-touching detection whose recovered source box might otherwise
        # miss the lower dashed placeholder boundary.
        "placements": {"FIG 02": [0, 300, 600, 540]},
    }
    _, updated = normalize_placeholder_geometry(
        base_image=base,
        spec=spec,
        out_path=tmp_path / "planned.png",
        redraw=False,
    )
    clear = updated["_replacement_clear_boxes"]["FIG 02"]
    assert clear[3] >= 590


def test_hidden_normalize_expands_square_inner_detection_to_visible_panel(tmp_path: Path):
    base = tmp_path / "base.png"
    im = Image.new("RGB", (360, 360), "white")
    draw = ImageDraw.Draw(im)
    for x in range(50, 290, 24):
        draw.line([(x, 50), (min(x + 12, 290), 50)], fill=(170, 120, 25), width=3)
        draw.line([(x, 290), (min(x + 12, 290), 290)], fill=(170, 120, 25), width=3)
    for y in range(50, 290, 24):
        draw.line([(50, y), (50, min(y + 12, 290))], fill=(170, 120, 25), width=3)
        draw.line([(290, y), (290, min(y + 12, 290))], fill=(170, 120, 25), width=3)
    im.save(base)
    spec = {
        "placeholders": [{"id": "FIG 01", "label": "Square result", "aspect": "1:1 square"}],
        # Simulate the LLM detecting only the inner text/white region, not the
        # visible dashed placeholder boundary.
        "placements": {"FIG 01": [100, 100, 240, 240]},
    }
    _, updated = normalize_placeholder_geometry(
        base_image=base,
        spec=spec,
        out_path=tmp_path / "planned.png",
        redraw=False,
    )
    clear = updated["_replacement_clear_boxes"]["FIG 01"]
    box = updated["placements"]["FIG 01"]
    assert clear[0] <= 55 and clear[1] <= 55
    assert clear[2] >= 285 and clear[3] >= 285
    assert box[0] >= clear[0] and box[1] >= clear[1]
    assert box[2] <= clear[2] and box[3] <= clear[3]
    assert abs(((box[2] - box[0]) / (box[3] - box[1])) - 1.0) < 0.03


def test_replace_rejects_overlapping_figure_targets(tmp_path: Path):
    base = tmp_path / "base.png"
    Image.new("RGB", (220, 160), "white").save(base)
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    Image.new("RGB", (80, 80), "red").save(asset_dir / "fig-a.png")
    Image.new("RGB", (80, 80), "blue").save(asset_dir / "fig-b.png")
    spec = {
        "placeholders": [
            {"id": "FIG 01", "label": "A", "aspect": "1:1 square", "asset": "fig-a.png"},
            {"id": "FIG 02", "label": "B", "aspect": "1:1 square", "asset": "fig-b.png"},
        ],
        "placements": {
            "FIG 01": [40, 40, 120, 120],
            "FIG 02": [90, 40, 170, 120],
        },
        "_replacement_clear_boxes": {
            "FIG 01": [35, 35, 125, 125],
            "FIG 02": [85, 35, 175, 125],
        },
    }
    with pytest.raises(ValueError, match="overlaps"):
        replace_placeholders(base_image=base, spec=spec, asset_dir=asset_dir, out_path=tmp_path / "out.png")


def test_audit_generated_placeholder_geometry_rejects_wide_ribbon(tmp_path: Path):
    base = tmp_path / "base.png"
    im = Image.new("RGB", (1024, 1536), "white")
    draw = ImageDraw.Draw(im)
    draw.rectangle([40, 800, 984, 940], outline=(120, 65, 150), width=2)
    im.save(base)
    spec = {
        "placeholders": [{"id": "FIG 02", "label": "Wide plot", "aspect": "2.5:1 wide"}],
        "placements": {"FIG 02": [40, 800, 984, 940]},
    }
    issues = audit_generated_placeholder_geometry(base_image=base, spec=spec, ratio_tolerance=0.20)
    assert issues
    assert issues[0]["id"] == "FIG 02"
    assert issues[0]["actual_ratio"] > 5


def test_audit_tolerates_moderately_imperfect_wide_placeholder(tmp_path: Path):
    base = tmp_path / "base.png"
    im = Image.new("RGB", (900, 420), "white")
    draw = ImageDraw.Draw(im)
    # 600 / 300 = 2.0, outside the base 20% tolerance for an expected 2.5:1
    # placeholder but still usable for contained replacement planning.
    draw.rectangle([80, 60, 680, 360], outline=(40, 110, 190), width=2)
    im.save(base)
    spec = {
        "placeholders": [{"id": "FIG 02", "label": "Wide distribution", "aspect": "2.5:1 wide"}],
        "placements": {"FIG 02": [80, 60, 680, 360]},
    }
    assert audit_generated_placeholder_geometry(base_image=base, spec=spec, ratio_tolerance=0.20) == []


def test_audit_does_not_expand_square_placeholder_to_surrounding_card(tmp_path: Path):
    base = tmp_path / "base.png"
    im = Image.new("RGB", (4096, 6144), "white")
    draw = ImageDraw.Draw(im)
    for offset in range(8):
        draw.rounded_rectangle(
            [817 + offset, 3647 + offset, 4003 - offset, 5128 - offset],
            radius=40,
            outline=(40, 110, 190),
            width=1,
        )
    x0, y0, x1, y1 = 2104, 3872, 3274, 4874
    for x in range(x0, x1, 60):
        draw.line([(x, y0), (min(x + 12, x1), y0)], fill=(210, 140, 20), width=3)
        draw.line([(x, y1), (min(x + 12, x1), y1)], fill=(210, 140, 20), width=3)
    for y in range(y0, y1, 60):
        draw.line([(x0, y), (x0, min(y + 12, y1))], fill=(210, 140, 20), width=3)
        draw.line([(x1, y), (x1, min(y + 12, y1))], fill=(210, 140, 20), width=3)
    im.save(base)

    spec = {
        "placeholders": [{"id": "FIG 01", "label": "Result", "aspect": "1:1 square"}],
        "placements": {"FIG 01": [x0, y0, x1, y1]},
    }
    assert audit_generated_placeholder_geometry(base_image=base, spec=spec, ratio_tolerance=0.20) == []


def test_audit_tolerates_slightly_landscape_square_result_placeholder(tmp_path: Path):
    base = tmp_path / "base.png"
    im = Image.new("RGB", (800, 400), "white")
    draw = ImageDraw.Draw(im)
    draw.rectangle([80, 80, 336, 280], outline=(170, 120, 25), width=2)
    im.save(base)
    spec = {
        "placeholders": [{"id": "FIG 01", "label": "Result", "aspect": "1:1 square"}],
        # 256 / 200 = 1.28, intentionally outside base 20% tolerance but inside
        # the square-result tolerance used by strict replacement planning.
        "placements": {"FIG 01": [80, 80, 336, 280]},
    }
    assert audit_generated_placeholder_geometry(base_image=base, spec=spec, ratio_tolerance=0.20) == []


def test_audit_rejects_oversized_square_result_placeholder(tmp_path: Path):
    base = tmp_path / "base.png"
    im = Image.new("RGB", (1000, 1500), "white")
    draw = ImageDraw.Draw(im)
    draw.rectangle([100, 500, 500, 900], outline=(170, 120, 25), width=2)
    im.save(base)
    spec = {
        "placeholders": [{"id": "FIG 01", "label": "95% CL upper limit result", "aspect": "1:1 square"}],
        "placements": {"FIG 01": [100, 500, 500, 900]},
    }
    issues = audit_generated_placeholder_geometry(base_image=base, spec=spec, ratio_tolerance=0.20)
    assert issues
    assert issues[0]["id"] == "FIG 01"
    assert "too large" in issues[0]["message"]


def test_audit_rejects_too_low_square_result_placeholder(tmp_path: Path):
    base = tmp_path / "base.png"
    im = Image.new("RGB", (1000, 1500), "white")
    draw = ImageDraw.Draw(im)
    draw.rectangle([400, 1100, 700, 1400], outline=(170, 120, 25), width=2)
    im.save(base)
    spec = {
        "placeholders": [{"id": "FIG 01", "label": "95% CL upper limit result", "aspect": "1:1 square"}],
        "placements": {"FIG 01": [400, 1100, 700, 1400]},
    }
    issues = audit_generated_placeholder_geometry(base_image=base, spec=spec, ratio_tolerance=0.20)
    assert issues
    assert any("too low" in issue["message"] for issue in issues)


def test_audit_recovers_landscape_light_square_placeholder_from_partial_detection(tmp_path: Path):
    base = tmp_path / "base.png"
    im = Image.new("RGB", (1200, 1000), (10, 35, 70))
    draw = ImageDraw.Draw(im)
    # Light placeholder panel is actually landscape, but the simulated vision
    # detection only reports the centered label area.
    draw.rounded_rectangle([520, 500, 1050, 820], radius=20, fill=(245, 245, 245), outline=(210, 140, 30), width=3)
    draw.text((720, 620), "[FIG 01]", fill=(20, 20, 30))
    im.save(base)
    spec = {
        "placeholders": [{"id": "FIG 01", "label": "95% CL upper limit result", "aspect": "1:1 square"}],
        "placements": {"FIG 01": [650, 575, 900, 825]},
    }
    issues = audit_generated_placeholder_geometry(base_image=base, spec=spec, ratio_tolerance=0.20)
    assert issues
    assert any(issue["id"] == "FIG 01" and issue["actual_ratio"] > 1.4 for issue in issues)


def test_audit_rejects_dark_figure_card_surface(tmp_path: Path):
    base = tmp_path / "base.png"
    im = Image.new("RGB", (600, 600), (8, 18, 45))
    draw = ImageDraw.Draw(im)
    draw.rounded_rectangle([220, 240, 380, 400], radius=10, fill=(248, 248, 248), outline=(210, 150, 30), width=3)
    im.save(base)
    spec = {
        "placeholders": [{"id": "FIG 01", "label": "95% CL upper limit result", "aspect": "1:1 square"}],
        "placements": {"FIG 01": [220, 240, 380, 400]},
    }
    issues = audit_generated_placeholder_geometry(base_image=base, spec=spec, ratio_tolerance=0.20)
    assert any(issue.get("category") == "figure_surface" for issue in issues)
    assert any("too dark" in issue["message"] for issue in issues)


def test_audit_accepts_light_mat_around_figure_placeholder(tmp_path: Path):
    base = tmp_path / "base.png"
    im = Image.new("RGB", (600, 600), (8, 18, 45))
    draw = ImageDraw.Draw(im)
    draw.rounded_rectangle([170, 190, 430, 450], radius=22, fill=(246, 248, 252), outline=(205, 215, 230), width=2)
    draw.rounded_rectangle([220, 240, 380, 400], radius=10, fill=(252, 252, 252), outline=(210, 150, 30), width=3)
    im.save(base)
    spec = {
        "placeholders": [{"id": "FIG 01", "label": "95% CL upper limit result", "aspect": "1:1 square"}],
        "placements": {"FIG 01": [220, 240, 380, 400]},
    }
    issues = audit_generated_placeholder_geometry(base_image=base, spec=spec, ratio_tolerance=0.20)
    assert not [issue for issue in issues if issue.get("category") == "figure_surface"]


def test_hidden_normalize_refines_gray_wide_placeholder_inside_overlarge_section_detection(tmp_path: Path):
    base = tmp_path / "base.png"
    im = Image.new("RGB", (1024, 720), "white")
    draw = ImageDraw.Draw(im)
    # Simulate section text on the left and a neutral gray dashed placeholder
    # on the right.  The LLM seed includes the whole section, which previously
    # made the wide figure overlap the result placeholder below.
    for y in range(170, 310, 28):
        draw.line([(40, y), (350, y)], fill=(35, 35, 45), width=3)
    for x in range(400, 980, 28):
        draw.line([(x, 145), (min(x + 14, 980), 145)], fill=(120, 120, 130), width=3)
        draw.line([(x, 355), (min(x + 14, 980), 355)], fill=(120, 120, 130), width=3)
    for y in range(145, 355, 28):
        draw.line([(400, y), (400, min(y + 14, 355))], fill=(120, 120, 130), width=3)
        draw.line([(980, y), (980, min(y + 14, 355))], fill=(120, 120, 130), width=3)
    im.save(base)
    spec = {
        "placeholders": [{"id": "FIG 02", "label": "Wide distribution", "aspect": "2.5:1 wide"}],
        "placements": {"FIG 02": [120, 140, 1000, 492]},
    }
    _, updated = normalize_placeholder_geometry(
        base_image=base,
        spec=spec,
        out_path=tmp_path / "planned.png",
        redraw=False,
    )
    clear = updated["_replacement_clear_boxes"]["FIG 02"]
    box = updated["placements"]["FIG 02"]
    assert clear[0] >= 390
    assert clear[1] <= 150
    assert clear[3] <= 365
    assert box[0] >= clear[0]
    assert box[1] >= clear[1]
    assert box[2] <= clear[2]
    assert box[3] <= clear[3]
    assert abs(((box[2] - box[0]) / (box[3] - box[1])) - 2.5) < 0.03


def test_hidden_normalize_square_uses_dashed_bottom_not_lower_section_card(tmp_path: Path):
    base = tmp_path / "base.png"
    im = Image.new("RGB", (900, 700), "white")
    draw = ImageDraw.Draw(im)
    # Real placeholder: slightly landscape square-result dashed box.
    for x in range(330, 650, 26):
        draw.line([(x, 230), (min(x + 13, 650), 230)], fill=(130, 130, 140), width=3)
        draw.line([(x, 500), (min(x + 13, 650), 500)], fill=(130, 130, 140), width=3)
    for y in range(230, 500, 26):
        draw.line([(330, y), (330, min(y + 13, 500))], fill=(130, 130, 140), width=3)
        draw.line([(650, y), (650, min(y + 13, 500))], fill=(130, 130, 140), width=3)
    # A lower card/section border that should not be mistaken as the placeholder
    # bottom, even though it gives a more square-looking outer box.
    draw.line([(200, 610), (780, 610)], fill=(40, 100, 180), width=5)
    draw.line([(200, 230), (200, 610)], fill=(40, 100, 180), width=5)
    draw.line([(780, 230), (780, 610)], fill=(40, 100, 180), width=5)
    im.save(base)
    spec = {
        "placeholders": [{"id": "FIG 01", "label": "95% CL limit result", "aspect": "1:1 square"}],
        "placements": {"FIG 01": [360, 245, 620, 505]},
    }
    _, updated = normalize_placeholder_geometry(
        base_image=base,
        spec=spec,
        out_path=tmp_path / "planned.png",
        redraw=False,
    )
    clear = updated["_replacement_clear_boxes"]["FIG 01"]
    box = updated["placements"]["FIG 01"]
    assert clear[3] <= 515
    assert box[3] <= clear[3]
    assert abs(((box[2] - box[0]) / (box[3] - box[1])) - 1.0) < 0.03


def test_audit_allows_square_result_just_above_summary_boundary(tmp_path: Path):
    base = tmp_path / "base.png"
    im = Image.new("RGB", (1000, 1500), "white")
    draw = ImageDraw.Draw(im)
    # Bottom at 82.7% of canvas: visually above the lower summary band and should
    # not be rejected by an overly brittle boundary check.
    draw.rectangle([400, 940, 700, 1240], outline=(170, 120, 25), width=2)
    im.save(base)
    spec = {
        "placeholders": [{"id": "FIG 01", "label": "95% CL upper limit result", "aspect": "1:1 square"}],
        "placements": {"FIG 01": [400, 940, 700, 1240]},
    }
    issues = audit_generated_placeholder_geometry(base_image=base, spec=spec, ratio_tolerance=0.20)
    assert not [issue for issue in issues if "too low" in issue["message"]]
