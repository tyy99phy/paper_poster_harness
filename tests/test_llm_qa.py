from pathlib import Path

from PIL import Image

from poster_harness.llm_stages import _deterministic_qa_checks, _qa_mode_instructions


def test_deterministic_qa_containment_uses_top_boundary_correctly():
    spec = {
        "project": {"title": "T"},
        "sections": [],
        "placeholders": [{"id": "FIG 01", "asset": "fig.png"}],
        "placements": {"FIG 01": [100, 120, 300, 320]},
        "_replacement_clear_boxes": {"FIG 01": [90, 110, 320, 340]},
    }
    result = _deterministic_qa_checks(spec, prompt=None, detected_placeholders=None)
    assert not [issue for issue in result["issues"] if issue.get("category") == "figure_containment"]


def test_deterministic_qa_flags_true_containment_violation():
    spec = {
        "project": {"title": "T"},
        "sections": [],
        "placeholders": [{"id": "FIG 01", "asset": "fig.png"}],
        "placements": {"FIG 01": [80, 120, 300, 320]},
        "_replacement_clear_boxes": {"FIG 01": [90, 110, 320, 340]},
    }
    result = _deterministic_qa_checks(spec, prompt=None, detected_placeholders=None)
    assert [issue for issue in result["issues"] if issue.get("category") == "figure_containment"]


def test_final_qa_instructions_trust_replacement_clear_boxes():
    text = "\n".join(_qa_mode_instructions("final"))
    assert "_replacement_clear_boxes" in text
    assert "approved final placeholder/cleanup boundary" in text
    assert "deterministic_prechecks as authoritative" in text


def test_panel_level_qa_warns_on_local_ratio_mismatch(tmp_path: Path):
    image = tmp_path / "poster.png"
    Image.new("RGB", (500, 500), "white").save(image)
    spec = {
        "project": {"title": "T"},
        "sections": [],
        "placeholders": [{"id": "FIG 01", "asset": "fig.png", "aspect": "1:1 square"}],
        "placements": {"FIG 01": [100, 100, 300, 180]},
        "_replacement_clear_boxes": {"FIG 01": [95, 95, 305, 185]},
    }
    result = _deterministic_qa_checks(spec, prompt=None, detected_placeholders=None, image_path=image)
    assert [issue for issue in result["issues"] if issue.get("category") == "panel_geometry"]


def test_panel_level_qa_warns_on_detection_misalignment(tmp_path: Path):
    image = tmp_path / "poster.png"
    Image.new("RGB", (500, 500), "white").save(image)
    spec = {
        "project": {"title": "T"},
        "sections": [],
        "placeholders": [{"id": "FIG 01", "asset": "fig.png", "aspect": "1:1 square"}],
        "placements": {"FIG 01": [100, 100, 250, 250]},
        "_replacement_clear_boxes": {"FIG 01": [95, 95, 255, 255]},
    }
    detected = {"placements": {"FIG 01": [300, 300, 450, 450]}}
    result = _deterministic_qa_checks(spec, prompt=None, detected_placeholders=detected, image_path=image)
    assert [issue for issue in result["issues"] if issue.get("category") == "panel_detection_alignment"]
