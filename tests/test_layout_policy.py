from poster_harness.llm_stages import _clean_public_caption, _normalize_display_aspect


def test_wide_distribution_uses_source_aspect():
    asset = {
        "label": "HT distribution in signal and control regions",
        "caption": "post-fit distributions",
        "name": "Figure_002.png",
        "width": 5154,
        "height": 2061,
        "aspect": "2.5:1 wide",
    }
    assert _normalize_display_aspect("2.1:1 wide", asset=asset, label="HT distribution") == "2.5:1 wide"


def test_square_result_uses_source_aspect_over_poster_guess():
    asset = {"label": "Observed limit", "width": 2357, "height": 2357, "aspect": "1:1 square"}
    assert _normalize_display_aspect("2:1 wide", asset=asset, label="Observed limit") == "1:1 square"


def test_non_plot_aspect_is_preserved_when_only_raw_aspect_exists():
    asset = {"label": "Panoramic detector photograph"}
    assert _normalize_display_aspect("2.5:1 wide", asset=asset, label="Panoramic detector photograph") == "2.5:1 wide"


def test_instruction_caption_is_removed():
    assert _clean_public_caption("Use CMS blue for the primary observed limit") == ""
    assert _clean_public_caption("Observed and expected 95% CL limits versus mass.")


def test_figure_selection_ids_are_sequential_fig_numbers():
    from poster_harness.llm_stages import _normalize_figure_selection

    assets = [
        {"asset": "limit.png", "width": 200, "height": 200, "aspect": "1:1 square"},
        {"asset": "dist.png", "width": 500, "height": 200, "aspect": "2.5:1 wide"},
    ]
    spec = {
        "sections": [{"id": 1}, {"id": 2}],
        "placeholders": [
            {"id": "HERO_LIMIT", "section": 2},
            {"id": "STRATEGY_DIST", "section": 1},
        ],
    }
    result = {
        "selected_figures": [
            {"placeholder_id": "STRATEGY_DIST", "asset": "dist.png", "section": 1, "label": "Distribution", "aspect": "2.5:1 wide", "priority": 2, "rationale": "support"},
            {"placeholder_id": "HERO_LIMIT", "asset": "limit.png", "section": 2, "label": "Observed limit", "aspect": "1:1 square", "priority": 1, "rationale": "hero"},
        ],
        "selection_notes": [],
    }
    normalized = _normalize_figure_selection(result, spec, assets, limit=2)
    assert [row["placeholder_id"] for row in normalized["selected_figures"]] == ["FIG 01", "FIG 02"]
    assert [row["asset"] for row in normalized["selected_figures"]] == ["limit.png", "dist.png"]


def test_figure_selection_preserves_storyboard_fields():
    from poster_harness.llm_stages import _normalize_figure_selection

    assets = [{"asset": "mass.png", "width": 300, "height": 200, "aspect": "1.5:1 wide"}]
    spec = {
        "sections": [{"id": 1}, {"id": 5}],
        "placeholders": [{"id": "RESULT", "section": 1}],
    }
    result = {
        "selected_figures": [
            {
                "placeholder_id": "RESULT",
                "asset": "mass.png",
                "target_section": 5,
                "label": "Mass comparison",
                "aspect": "1.5:1 wide",
                "priority": 1,
                "role": "hero_result",
                "reason": "Best headline comparison plot.",
            }
        ],
        "selection_notes": [],
    }
    normalized = _normalize_figure_selection(result, spec, assets, limit=1)
    row = normalized["selected_figures"][0]
    assert row["section"] == 5
    assert row["target_section"] == 5
    assert row["role"] == "hero_result"
    assert row["reason"] == "Best headline comparison plot."
    assert row["rationale"] == "Best headline comparison plot."



def test_storyboard_information_plan_normalization():
    from poster_harness.llm_stages import _normalize_storyboard

    spec = {"project": {"title": "T"}, "sections": [{"id": 1, "title": "Result"}]}
    result = {
        "meta": {},
        "core_message": "Headline result",
        "sections": [{"id": 1, "title": "Result", "role": "result", "synopsis": "", "key_claims": ["Public claim"], "text_budget": "2 bullets", "preferred_visual": "plot"}],
        "visual_assets": [],
        "layout_tree": {"reading_order": [1], "hero_section": 1},
        "information_plan": {"data_badges": [{"label": "Dataset", "value": "13 TeV"}], "must_answer_questions": ["What is measured?"]},
        "qa_questions": ["What is measured?"],
    }
    normalized = _normalize_storyboard(result, spec, [])
    plan = normalized["information_plan"]
    assert "Dataset: 13 TeV" in plan["data_badges"]
    assert "Public claim" in plan["display_facts"]
    assert "What is measured?" in plan["must_answer_questions"]


def test_physics_quiz_and_copy_deck_normalization():
    from poster_harness.llm_stages import _normalize_copy_deck, _normalize_physics_quiz

    spec = {
        "project": {"title": "T"},
        "sections": [{"id": 1, "title": "Result"}, {"id": 2, "title": "Method"}],
        "placeholders": [{"id": "FIG 01", "section": 1, "label": "Limit"}],
    }
    assets = [{"asset": "limit.png", "label": "Limit"}]
    quiz = _normalize_physics_quiz(
        {
            "quiz_items": [
                {
                    "id": "custom",
                    "aspect": "result",
                    "question": "What is the headline result?",
                    "answer": "No significant excess.",
                    "poster_priority": "must",
                    "target_section": 1,
                    "linked_assets": ["limit.png", "missing.png"],
                }
            ],
            "coverage_notes": ["cover result"],
        },
        spec,
        assets,
        limit=4,
    )
    assert quiz["quiz_items"][0]["id"] == "Q01"
    assert quiz["quiz_items"][0]["aspect"] == "headline_result"
    assert quiz["quiz_items"][0]["linked_assets"] == ["limit.png"]

    deck = _normalize_copy_deck(
        {
            "copy_units": [
                {
                    "target_section": 1,
                    "type": "headline",
                    "text": "No significant excess",
                    "max_chars": 42,
                    "priority": "must",
                    "quiz_ids": ["Q01", "Q99"],
                    "placeholder_id": "FIG 01",
                }
            ]
        },
        spec,
        quiz,
        limit=8,
    )
    unit = deck["copy_units"][0]
    assert unit["id"] == "C01"
    assert unit["type"] == "hero_headline"
    assert unit["quiz_ids"] == ["Q01"]
    assert unit["placeholder_id"] == "FIG 01"


def test_generic_region_matrix_is_removed_but_concrete_regions_survive():
    from poster_harness.llm_stages import _normalize_copy_deck

    spec = {
        "project": {"title": "T"},
        "sections": [{"id": 2, "title": "Analysis"}],
        "placeholders": [],
    }
    deck = _normalize_copy_deck(
        {
            "copy_units": [
                {
                    "target_section": 2,
                    "type": "region_matrix",
                    "text": "pp → SS μμ → VBF topology → SR/CR → fit",
                    "priority": "must",
                    "quiz_ids": [],
                },
                {
                    "target_section": 2,
                    "type": "region_matrix",
                    "text": "SR: 2 SS μ; WZ CR: 3μ with OS pair near mZ",
                    "priority": "must",
                    "quiz_ids": [],
                },
            ]
        },
        spec,
        {},
        limit=8,
    )
    assert [unit["text"] for unit in deck["copy_units"]] == ["SR: 2 SS μ; WZ CR: 3μ with OS pair near mZ"]


def test_layout_contract_preserves_pixel_aspect_on_portrait_canvas():
    from poster_harness.layout_contract import build_layout_contract, contract_boxes_for_image

    spec = {
        "sections": [
            {"id": 1, "title": "Motivation"},
            {"id": 2, "title": "Strategy"},
            {"id": 3, "title": "Result"},
        ],
        "placeholders": [
            {"id": "FIG 01", "section": 1, "aspect": "1:1 square", "label": "Square result", "role": "hero_result"},
            {"id": "FIG 02", "section": 2, "aspect": "2.5:1 wide", "label": "Wide fit plot"},
            {"id": "FIG 03", "section": 3, "aspect": "1.2:1", "label": "Moderate process diagram"},
        ],
    }
    contract = build_layout_contract(spec, canvas_width=1024, canvas_height=1536)
    boxes = contract_boxes_for_image(contract, (1024, 1536))
    expected = {"FIG 01": 1.0, "FIG 02": 2.5, "FIG 03": 1.2}
    for fig_id, ratio in expected.items():
        x0, y0, x1, y1 = boxes[fig_id]
        actual = (x1 - x0) / (y1 - y0)
        assert abs(actual / ratio - 1.0) < 0.04


def test_layout_contract_keeps_section5_hero_above_conclusion_zone():
    from poster_harness.layout_contract import build_layout_contract, contract_boxes_for_image

    spec = {
        "storyboard": {"layout_tree": {"hero_section": 5}},
        "sections": [
            {"id": 1, "title": "Motivation"},
            {"id": 2, "title": "Dataset"},
            {"id": 3, "title": "Fit"},
            {"id": 4, "title": "Background"},
            {"id": 5, "title": "Results and interpretation"},
        ],
        "placeholders": [
            {
                "id": "FIG 01",
                "section": 5,
                "aspect": "1:1 square",
                "label": "95% CL limits",
                "role": "hero_result",
            }
        ],
    }
    contract = build_layout_contract(spec, canvas_width=1024, canvas_height=1536)
    box = contract_boxes_for_image(contract, (1024, 1536))["FIG 01"]
    actual = (box[2] - box[0]) / (box[3] - box[1])
    assert abs(actual - 1.0) < 0.04
    assert box[3] / 1536 <= 0.835
