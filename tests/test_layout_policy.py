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
