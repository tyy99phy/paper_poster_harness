from poster_harness.assets import apply_detections_to_spec
from poster_harness.layout_contract import build_layout_contract, contract_boxes_for_image
from poster_harness.prompt import build_prompt


def _spec():
    return {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "sections": [
            {"id": 1, "title": "Motivation", "layout": "upper-left", "text": []},
            {"id": 2, "title": "Dataset", "layout": "upper-right", "text": []},
            {"id": 3, "title": "Strategy", "layout": "middle-left", "text": []},
            {"id": 4, "title": "Result", "layout": "middle-right", "text": []},
            {"id": 5, "title": "Summary", "layout": "bottom", "text": []},
        ],
        "placeholders": [
            {"id": "FIG 01", "section": 4, "label": "Observed limit", "aspect": "1:1 square", "role": "hero_result"},
            {"id": "FIG 02", "section": 3, "label": "Post-fit distribution", "aspect": "2.5:1 wide"},
            {"id": "FIG 03", "section": 1, "label": "Process A", "aspect": "1.2:1"},
            {"id": "FIG 04", "section": 1, "label": "Process B", "aspect": "1.2:1"},
        ],
        "conclusion": [],
    }


def test_build_layout_contract_preserves_placeholder_aspects():
    contract = build_layout_contract(_spec())
    rows = {row["id"]: row for row in contract["placeholders"]}
    assert set(rows) == {"FIG 01", "FIG 02", "FIG 03", "FIG 04"}
    assert rows["FIG 01"]["expected_aspect"] == 1.0
    assert rows["FIG 02"]["expected_aspect"] == 2.5
    assert abs(rows["FIG 03"]["expected_aspect"] - 1.2) < 0.01
    wide = rows["FIG 02"]["zone"]
    assert (wide[2] - wide[0]) / (wide[3] - wide[1]) > 2.3


def test_prompt_includes_layout_contract_as_soft_prior():
    spec = _spec()
    spec["layout_contract"] = build_layout_contract(spec)
    prompt = build_prompt(spec)
    assert "LAYOUT CONTRACT" in prompt
    assert "normalized poster fractions" in prompt
    assert "[FIG 01]: planned placeholder zone" in prompt
    assert "not pixel art instructions" in prompt


def test_apply_detections_attaches_contract_boxes_and_flags_far_detection():
    spec = _spec()
    spec["layout_contract"] = build_layout_contract(spec)
    boxes = contract_boxes_for_image(spec["layout_contract"], {"width": 1024, "height": 1536})
    near_fig01 = boxes["FIG 01"]
    detections = {
        "image_size": {"width": 1024, "height": 1536},
        "placeholders": [
            {"id": "FIG 01", "bbox": near_fig01, "confidence": 0.9},
            {"id": "FIG 02", "bbox": [800, 50, 980, 140], "confidence": 0.9},
        ],
    }
    updated = apply_detections_to_spec(spec, detections)
    assert "_layout_contract_boxes" in updated
    assert updated["_layout_contract_boxes"]["FIG 01"] == near_fig01
    issues = updated.get("_layout_contract_issues") or []
    assert [issue for issue in issues if issue["id"] == "FIG 02"]
