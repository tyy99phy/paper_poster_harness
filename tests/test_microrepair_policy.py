from poster_harness.cli import (
    _final_qa_is_micro_repairable,
    _template_critic_accepts,
    _template_critique_can_defer_to_downstream,
    _template_critique_is_micro_repairable,
)


def test_final_microrepair_allows_only_local_gamma_typo():
    qa = {
        "passes": False,
        "issues": [
            {
                "severity": "critical",
                "category": "text_accuracy",
                "message": "The title renders H→Zγ as H→Zy; this is a local Greek gamma glyph typo.",
                "suggested_fix": "Replace only the Latin y glyph with γ.",
            }
        ],
    }
    assert _final_qa_is_micro_repairable(qa)


def test_final_microrepair_rejects_figure_geometry_even_if_text_is_mentioned():
    qa = {
        "passes": False,
        "issues": [
            {
                "severity": "warning",
                "category": "figure_placement",
                "message": "FIG 02 overlaps nearby text and protrudes outside the placeholder boundary.",
                "suggested_fix": "Move or resize the figure and reflow text.",
            }
        ],
    }
    assert not _final_qa_is_micro_repairable(qa)


def test_final_microrepair_rejects_internal_text_cleanup_as_too_large():
    qa = {
        "passes": False,
        "issues": [
            {
                "severity": "critical",
                "category": "public_text_cleanliness",
                "message": "The poster includes internal workflow/placeholder notes in a conclusion bullet.",
                "suggested_fix": "Remove the internal workflow note.",
            }
        ],
    }
    assert not _final_qa_is_micro_repairable(qa)


def test_template_microrepair_rejects_placeholder_geometry():
    critique = {
        "passes": False,
        "issues": [
            {
                "severity": "critical",
                "category": "placeholder_aspect",
                "message": "FIG 01 is landscape but expected 1:1 square.",
                "suggested_prompt_repair": "Regenerate with a square placeholder.",
            }
        ],
    }
    assert not _template_critique_is_micro_repairable(critique)


def test_template_critic_geometry_can_defer_to_deterministic_qa_not_microrepair():
    critique = {
        "passes": False,
        "checks": {"no_internal_text": True, "no_fake_science": True, "placeholder_contract_clean": True},
        "issues": [
            {
                "severity": "warning",
                "category": "placeholder_aspect",
                "message": "FIG 02 may be too square for a 2.5:1 wide plot.",
                "suggested_prompt_repair": "Make FIG 02 wider.",
            }
        ],
    }
    assert not _template_critique_is_micro_repairable(critique)
    assert _template_critique_can_defer_to_downstream(critique)


def test_template_microrepair_rejects_score_failure_without_local_issue():
    critique = {
        "passes": False,
        "scores": {"overall": 0.4, "artistry": 0.3},
        "issues": [],
    }
    assert not _template_critique_is_micro_repairable(critique)


def test_template_critic_acceptance_honors_boolean_contract_checks():
    critique = {
        "passes": True,
        "scores": {
            "overall": 0.95,
            "artistry": 0.95,
            "information_density": 0.95,
            "placeholder_contract": 0.95,
        },
        "checks": {
            "information_plan_visible": True,
            "art_direction_strong": True,
            "placeholder_contract_clean": False,
            "no_internal_text": True,
            "no_fake_science": True,
        },
        "issues": [
            {
                "severity": "critical",
                "category": "placeholder_contract",
                "message": "FIG 01 aspect ratio is wrong and the placeholder geometry is too wide.",
            }
        ],
    }
    assert not _template_critic_accepts(critique, {"autoposter": {"template_critic": {"require_pass": True}}})


def test_template_critic_can_pass_geometry_warning_to_deterministic_qa():
    critique = {
        "passes": True,
        "scores": {
            "overall": 0.95,
            "artistry": 0.95,
            "information_density": 0.95,
            "placeholder_contract": 0.95,
        },
        "checks": {
            "information_plan_visible": True,
            "art_direction_strong": True,
            "placeholder_contract_clean": False,
            "no_internal_text": True,
            "no_fake_science": True,
        },
        "issues": [
            {
                "severity": "warning",
                "category": "placeholder_contract",
                "message": "FIG 02 may be slightly wide; deterministic geometry QA should verify.",
            }
        ],
    }
    assert _template_critic_accepts(critique, {"autoposter": {"template_critic": {"require_pass": True}}})
