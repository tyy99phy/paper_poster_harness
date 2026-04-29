from poster_harness.prompt import build_prompt, sanitize_public_text


def test_public_filter_removes_internal_conclusion():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "sections": [],
        "placeholders": [],
        "forbidden_phrases": ["placeholder explanation"],
        "conclusion": ["A public result.", "placeholder explanation should not leak"],
    }
    prompt = build_prompt(spec)
    assert "A public result." in prompt
    assert "placeholder explanation should not leak" not in prompt


def test_sanitize_public_text_removes_workflow_lines():
    cleaned = sanitize_public_text("keep this\ninternal workflow note\nkeep that", ["internal workflow"])
    assert "keep this" in cleaned
    assert "keep that" in cleaned
    assert "internal workflow" not in cleaned


def test_prompt_has_positive_art_direction():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {"art_direction": "cinematic abstract detector art"},
        "sections": [],
        "placeholders": [],
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    assert "POSITIVE ART DIRECTION" in prompt
    assert "cinematic abstract detector art" in prompt


def test_prompt_marks_flowchart_items_as_node_labels_not_instruction_text():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "sections": [{"id": 1, "title": "Method", "flowchart": ["A", "B"], "text": []}],
        "placeholders": [],
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    assert "Node label" in prompt
    assert "do NOT render instruction sentences verbatim" in prompt


def test_prompt_includes_hep_poster_grammar_and_density_rules():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "sections": [],
        "placeholders": [],
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    assert "HEP POSTER DESIGN GRAMMAR" in prompt
    assert "CERN/LHCC-style scientific story spine" in prompt
    assert "TEXT DENSITY AND READABILITY" in prompt
    assert "FIGURE-LED COMPOSITION" in prompt
    assert "one dominant hero region" in prompt


def test_result_placeholder_gets_hero_layout_hint():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "sections": [{"id": 4, "title": "Results", "layout": "card", "text": []}],
        "placeholders": [
            {"id": "FIG 01", "section": 4, "label": "Observed 95% CL exclusion limit", "aspect": "1:1 square"}
        ],
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    assert "dominant key-result figure" in prompt
    assert "largest readable near-square slot" in prompt


def test_prompt_has_exact_source_aspect_instruction():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "sections": [{"id": 1, "title": "Result", "layout": "card", "text": []}],
        "placeholders": [
            {"id": "FIG 01", "section": 1, "label": "Observed limit", "aspect": "1:1 square"}
        ],
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    assert "EXACT source-image aspect ratio 1:1 square" in prompt
    assert "near-square; do not stretch it into a landscape strip" in prompt
    assert "target visible box" in prompt
    assert "If only 230 px height is available" in prompt


def test_prompt_allows_non_square_section_blocks():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "sections": [],
        "placeholders": [],
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    assert "Cards/blocks do not need to be square" in prompt
    assert "circular callouts" in prompt


def test_prompt_omits_captions_next_to_placeholders():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "sections": [{"id": 1, "title": "Result", "layout": "card", "text": [], "caption": "Do not crowd the figure."}],
        "placeholders": [{"id": "FIG 01", "section": 1, "label": "Result plot", "aspect": "1:1 square"}],
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    assert "Do not crowd the figure" not in prompt
    assert "Do not render separate captions directly above or below" in prompt


def test_prompt_gives_wide_placeholder_section_layout_guard():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "sections": [{"id": 3, "title": "Strategy", "layout": "card", "text": []}],
        "placeholders": [{"id": "FIG 02", "section": 3, "label": "Post-fit distribution", "aspect": "2.5:1 wide"}],
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    assert "Wide-figure section layout" in prompt
    assert "not a full-width ribbon" in prompt
    assert "600–720 px wide and 240–300 px tall" in prompt


def test_prompt_prefers_geometry_over_conflicting_two_column_layout():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "sections": [
            {"id": 3, "title": "Strategy", "layout": "middle-left", "text": []},
            {"id": 4, "title": "Result", "layout": "middle-right", "text": []},
        ],
        "placeholders": [
            {"id": "FIG 01", "section": 4, "label": "Observed result limit", "aspect": "1:1 square"},
            {"id": "FIG 02", "section": 3, "label": "Post-fit distribution", "aspect": "2.5:1 wide"},
        ],
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    assert "GEOMETRY-FIRST LAYOUT OVERRIDE" in prompt
    assert "ignore that phrase and preserve the placeholder geometry" in prompt
    assert "do not place those two sections side-by-side" in prompt


def test_prompt_limits_text_when_placeholder_needs_geometry():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "sections": [
            {
                "id": 3,
                "title": "Strategy",
                "layout": "card",
                "text": [{"title": "Input", "body": ["Body"], "bullets": ["A", "B", "C"]}],
                "flowchart": ["one", "two"],
            }
        ],
        "placeholders": [{"id": "FIG 02", "section": 3, "label": "Post-fit distribution", "aspect": "2.5:1 wide"}],
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    assert "render at most 2 short bullets" in prompt
    assert "Bullet: A" in prompt
    assert "Bullet: B" in prompt
    assert "Bullet: C" not in prompt
    assert "Optional compact public text-only analysis flowchart" in prompt
