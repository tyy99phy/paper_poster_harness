from pathlib import Path

from PIL import Image, ImageDraw

from poster_harness.replace import normalize_placeholder_geometry, replace_placeholders


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
        "placeholders": [{"id": "FIG 01", "label": "Result", "aspect": "1:1 square", "asset": "fig.png"}],
        "placements": {"FIG 01": [70, 35, 150, 115]},
        "_replacement_clear_boxes": {"FIG 01": [30, 30, 190, 125]},
    }
    out = tmp_path / "out.png"
    replace_placeholders(base_image=base, spec=spec, asset_dir=asset_dir, out_path=out)
    result = Image.open(out).convert("RGB")
    assert result.getpixel((40, 120)) == (255, 255, 255)


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
