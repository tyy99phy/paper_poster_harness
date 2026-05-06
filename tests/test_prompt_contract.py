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
    assert "PUBLICATION FILTER" not in prompt


def test_sanitize_public_text_removes_workflow_lines():
    cleaned = sanitize_public_text("keep this\ninternal workflow note\nkeep that", ["internal workflow"])
    assert "keep this" in cleaned
    assert "keep that" in cleaned
    assert "internal workflow" not in cleaned


def test_prompt_has_positive_art_direction_before_placeholder_contract():
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
    assert prompt.index("POSITIVE ART DIRECTION") < prompt.index("ABSOLUTE PLACEHOLDER CONTRACT")


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
    assert "FIGURE CARD SURFACE POLICY" in prompt
    assert "TYPOGRAPHY SYSTEM" in prompt
    assert "COLOR AND MATERIAL SYSTEM" in prompt
    assert "one dominant hero region" in prompt
    assert "Do not put physics symbols" in prompt
    assert "particle-labeled icons" in prompt


def test_placeholder_reference_list_keeps_source_aspect_without_pixel_blueprint():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "sections": [{"id": 4, "title": "Results", "layout": "card", "text": []}],
        "placeholders": [
            {"id": "FIG 01", "section": 4, "label": "Observed 95% CL exclusion limit", "aspect": "1:1 square", "asset": "limit.png"}
        ],
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    assert "PLACEHOLDER REFERENCE LIST" in prompt
    assert "label \"Observed 95% CL exclusion limit\"" in prompt
    assert "aspect ratio \"1:1 square\"" in prompt
    assert "visual shape: true square" in prompt
    assert "width:height = X:Y" in prompt
    assert "not three, four, or seven heights" in prompt
    assert "limit.png" not in prompt
    assert "target visible box" not in prompt
    assert "PLACEHOLDER GEOMETRY BLUEPRINT" not in prompt


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
    assert "[FIG 01]: blank rectangular placeholder only" in prompt
    assert "aspect ratio \"1:1 square\"" in prompt
    assert "28-33% of the canvas width" in prompt
    assert "above the bottom summary/conclusion modules" in prompt
    assert "square light/white fill" in prompt
    assert "surrounding figure card/mat must also be light" in prompt
    assert "Inside" not in prompt  # use concise section-level instructions, not verbose geometry micro-management
    assert "Hard rejection rule" not in prompt


def test_prompt_forbids_dark_figure_blocks():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {
            "typography": "custom condensed editorial sans",
            "color_palette": "navy atmosphere, cobalt accent, pearl cards",
            "figure_surface": "strict pearl cards for all plots",
        },
        "sections": [{"id": 4, "title": "Result", "layout": "hero", "text": []}],
        "placeholders": [
            {"id": "FIG 01", "section": 4, "label": "Observed limit", "aspect": "1:1 square"}
        ],
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    assert "custom condensed editorial sans" in prompt
    assert "navy atmosphere, cobalt accent, pearl cards" in prompt
    assert "strict pearl cards for all plots" in prompt
    assert "never inside a dark navy/purple/black content block" in prompt
    assert "Do not put a chart placeholder inside a dark navy, purple, black, or saturated card" in prompt
    assert "Never use dark-filled blocks for chart/plot/diagram sections" in prompt


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


def test_prompt_gives_wide_placeholder_design_guard_without_pixels():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "sections": [{"id": 3, "title": "Strategy", "layout": "card", "text": []}],
        "placeholders": [{"id": "FIG 02", "section": 3, "label": "Post-fit distribution", "aspect": "2.5:1 wide"}],
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    assert "Wide-figure section design" in prompt
    assert "thin ribbon" in prompt
    assert "visual shape: substantial wide panel" in prompt
    assert "width:height about 2.5:1" in prompt
    assert "600–720 px" not in prompt
    assert "1024×1536" not in prompt


def test_prompt_gives_moderate_landscape_guard():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "sections": [{"id": 5, "title": "Global result", "layout": "card", "text": []}],
        "placeholders": [{"id": "FIG 01", "section": 5, "label": "Mass comparison", "aspect": "1.49:1 wide"}],
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    assert "visual shape: moderate landscape panel" in prompt
    assert "Moderate-landscape placeholder section design" in prompt
    assert "about one and a half times wider than tall" in prompt
    assert "not a panoramic banner" in prompt
    assert "3:1 banner" in prompt


def test_prompt_gives_near_square_placeholder_guard():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "sections": [{"id": 1, "title": "Process diagrams", "layout": "card", "text": []}],
        "placeholders": [
            {"id": "FIG 03", "section": 1, "label": "Process A", "aspect": "1.2:1"},
            {"id": "FIG 04", "section": 1, "label": "Process B", "aspect": "1.2:1"},
        ],
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    assert "A box labeled 1.2:1 must be near-square" in prompt
    assert "Near-square placeholder section design" in prompt
    assert "not a 1.5:1 landscape card" in prompt
    assert "Do not use 1.5:1" in prompt


def test_prompt_gives_mixed_portrait_square_placeholder_guard_and_allowed_ids():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "sections": [{"id": 3, "title": "Validation", "layout": "card", "text": []}],
        "placeholders": [
            {"id": "FIG 03", "section": 3, "label": "Portrait validation", "aspect": "1:1.16"},
            {"id": "FIG 04", "section": 3, "label": "Square calibration", "aspect": "1:1 square"},
        ],
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    assert "Exact allowed placeholder IDs: [FIG 03], [FIG 04]" in prompt
    assert "Do not render any other [FIG NN] box" in prompt
    assert "Mixed portrait/square placeholder section design" in prompt
    assert "only slightly taller than wide" in prompt
    assert "not make it a narrow tall column" in prompt


def test_prompt_removes_geometry_first_layout_override():
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
    assert "GEOMETRY-FIRST LAYOUT OVERRIDE" not in prompt
    assert "ignore that phrase and preserve the placeholder geometry" not in prompt
    assert "do not place those two sections side-by-side" not in prompt
    assert "INTER-PLACEHOLDER GUTTER RULE" in prompt
    assert "must be in separate non-overlapping modules" in prompt


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


def test_decorative_constraints_are_rendered_as_positive_guidance():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "sections": [],
        "placeholders": [],
        "decorative_art_constraints": [
            "Do not draw Feynman diagrams in decorative areas.",
            "Do not put particle labels such as mu, nu, j, q, W, N in decorative header artwork.",
        ],
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    assert "DECORATIVE ART GUIDANCE" in prompt
    assert "reserve Feynman diagrams for [FIG NN] placeholders only" in prompt
    assert "abstract geometric motifs" in prompt


def test_prompt_includes_storyboard_as_internal_design_brief():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "storyboard": {
            "core_message": "A precision mass measurement tests electroweak consistency.",
            "sections": [
                {
                    "id": 4,
                    "role": "hero_result",
                    "synopsis": "The comparison plot carries the headline.",
                    "key_claims": ["Mass agrees with global context."],
                    "text_budget": "one sentence + 2 bullets",
                    "preferred_visual": "global comparison plot",
                }
            ],
            "layout_tree": {"reading_order": [1, 4], "hero_section": 4, "hero_visual_role": "headline comparison"},
        },
        "sections": [{"id": 4, "title": "Result", "layout": "hero", "text": []}],
        "placeholders": [{"id": "FIG 01", "section": 4, "label": "Mass comparison", "aspect": "1.49:1 wide", "role": "hero_result"}],
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    assert "NARRATIVE STORYBOARD" in prompt
    assert "Core message to make visually obvious" in prompt
    assert "Story role: hero_result" in prompt
    assert "Hero visual priority" in prompt
    assert "do not render this heading" in prompt
