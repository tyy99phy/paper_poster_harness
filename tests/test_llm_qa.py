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


def test_final_qa_instructions_include_information_richness_check():
    text = "\n".join(_qa_mode_instructions("final"))
    assert "information-rich enough" in text
    assert "storyboard.information_plan.must_answer_questions" in text


def test_template_critique_normalizes_scores_and_repairs():
    from poster_harness.llm_stages import _normalize_template_critique

    result = _normalize_template_critique(
        {
            "passes": True,
            "summary": "Needs more content.",
            "scores": {"overall": 8, "artistry": 0.7, "information_density": 62, "placeholder_contract": 0.9},
            "issues": [
                {
                    "severity": "warning",
                    "category": "information_density",
                    "message": "Too sparse.",
                    "suggested_prompt_repair": "Add grounded fact badges.",
                }
            ],
            "prompt_repairs": [],
        }
    )
    assert result["passes"] is True
    assert result["scores"]["overall"] == 0.8
    assert result["scores"]["information_density"] == 0.62
    assert "Add grounded fact badges." in result["prompt_repairs"]


def test_template_critique_critical_issue_forces_fail():
    from poster_harness.llm_stages import _normalize_template_critique

    result = _normalize_template_critique(
        {
            "passes": True,
            "summary": "Fake plot present.",
            "scores": {"overall": 0.9},
            "issues": [{"severity": "critical", "category": "fake_science", "message": "A fake plot is visible."}],
            "prompt_repairs": [],
        }
    )
    assert result["passes"] is False


def test_template_critic_acceptance_thresholds():
    from poster_harness.cli import _template_critic_accepts

    config = {
        "autoposter": {
            "template_critic": {
                "require_pass": True,
                "min_overall_score": 0.7,
                "min_artistry_score": 0.6,
                "min_information_density_score": 0.6,
                "min_placeholder_contract_score": 0.7,
            }
        }
    }
    good = {"passes": True, "scores": {"overall": 0.8, "artistry": 0.7, "information_density": 0.7, "placeholder_contract": 0.8}}
    sparse = {"passes": True, "scores": {"overall": 0.8, "artistry": 0.7, "information_density": 0.4, "placeholder_contract": 0.8}}
    assert _template_critic_accepts(good, config)
    assert not _template_critic_accepts(sparse, config)


def test_regen_prompt_keeps_whole_poster_generation_and_no_overlay():
    from poster_harness.cli import _prompt_with_template_critic_repairs

    prompt = _prompt_with_template_critic_repairs(
        "BASE PROMPT",
        repairs=["Increase grounded badges.", "Improve HEP artistry."],
        round_index=1,
    )
    assert "BASE PROMPT" in prompt
    assert "REGENERATION CRITIQUE ROUND 1" in prompt
    assert "Increase grounded badges." in prompt
    assert "Regenerate a complete fresh poster" in prompt
    assert "overlay" not in prompt.lower()
