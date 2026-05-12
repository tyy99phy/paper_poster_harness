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
    assert "30-34% of the canvas width" in prompt
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


def test_prompt_includes_information_density_plan_from_storyboard():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {"information_density": "rich but readable"},
        "storyboard": {
            "information_plan": {
                "density_target": "Paper2Poster-rich",
                "data_badges": ["13 TeV dataset", {"label": "Channel", "value": "same-sign dimuon"}],
                "display_facts": ["The analysis targets vector-boson fusion topology."],
                "must_answer_questions": ["What is the headline result?"],
                "visual_story_units": ["hero result card plus supporting strategy card"],
            }
        },
        "sections": [],
        "placeholders": [],
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    assert "INFORMATION DENSITY TARGET" in prompt
    assert "rich but readable" in prompt
    assert "Do not make the poster a sparse cover illustration" in prompt
    assert "13 TeV dataset" in prompt
    assert "same-sign dimuon" in prompt
    assert "What is the headline result?" in prompt


def test_prompt_uses_copy_deck_as_authoritative_public_text():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "sections": [
            {
                "id": 1,
                "title": "Result",
                "layout": "hero",
                "text": [{"title": "Old", "body": ["Old paragraph should not render"], "bullets": ["Old bullet"]}],
            }
        ],
        "placeholders": [{"id": "FIG 01", "section": 1, "label": "Observed limit", "aspect": "1:1 square"}],
        "physics_quiz": {
            "quiz_items": [
                {
                    "id": "Q01",
                    "aspect": "headline_result",
                    "question": "What is the headline result?",
                    "answer": "No significant excess is observed.",
                    "poster_priority": "must",
                }
            ]
        },
        "copy_deck": {
            "copy_units": [
                {
                    "id": "C01",
                    "target_section": 1,
                    "type": "hero_headline",
                    "text": "No significant excess is observed",
                    "max_chars": 42,
                    "priority": "must",
                    "quiz_ids": ["Q01"],
                },
                {
                    "id": "C02",
                    "target_section": 1,
                    "type": "figure_headline",
                    "text": "Observed and expected limits",
                    "max_chars": 36,
                    "priority": "should",
                    "placeholder_id": "FIG 01",
                    "quiz_ids": ["Q01"],
                },
            ],
            "coverage_notes": ["Must cover the headline result."],
        },
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    assert "PHYSICS QUIZ COVERAGE TARGET" in prompt
    assert "PUBLIC COPY DECK" in prompt
    assert "No significant excess is observed" in prompt
    assert "Observed and expected limits" in prompt
    assert "Authoritative copy deck text for this section" in prompt
    assert "do not render C/Q IDs or evidence" in prompt
    assert "Old paragraph should not render" not in prompt
    assert "Old bullet" not in prompt


def test_copy_deck_section_title_replaces_section_heading_not_body_text():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "sections": [{"id": 2, "title": "Analysis strategy", "layout": "card", "text": []}],
        "placeholders": [],
        "copy_deck": {
            "copy_units": [
                {
                    "id": "C01",
                    "target_section": 2,
                    "type": "section_title",
                    "text": "02 Dataset and event signature",
                    "max_chars": 32,
                    "priority": "must",
                    "evidence": "section plan",
                    "quiz_ids": [],
                },
                {
                    "id": "C02",
                    "target_section": 2,
                    "type": "selection_cut",
                    "text": "VBF jets: |Δηjj| > 2.5, mjj > 750 GeV",
                    "max_chars": 54,
                    "priority": "must",
                    "evidence": "selection",
                    "quiz_ids": [],
                },
            ],
            "coverage_notes": [],
        },
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    assert 'Section 2, layout: card, title: "02 Dataset and event signature"' in prompt
    assert "Visible section heading is taken from the copy deck" in prompt
    assert 'section_title, priority=must' not in prompt
    assert 'selection_cut, priority=must' in prompt
    assert "VBF jets" in prompt


def test_copy_deck_specialist_units_suppress_generic_flowchart():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "sections": [
            {
                "id": 2,
                "title": "Analysis strategy",
                "layout": "card",
                "text": [],
                "flowchart": ["pp collisions", "candidate events", "SR/CR", "fit"],
            }
        ],
        "placeholders": [],
        "copy_deck": {
            "copy_units": [
                {
                    "id": "C01",
                    "target_section": 2,
                    "type": "region_matrix",
                    "text": "SR: 2 SS μ; WZ CR: 3μ with OS pair near mZ",
                    "max_chars": 56,
                    "priority": "must",
                    "evidence": "regions",
                    "quiz_ids": [],
                }
            ],
            "coverage_notes": [],
        },
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    assert "region_matrix, priority=must" in prompt
    assert "SR: 2 SS μ; WZ CR" in prompt
    assert "Node label" not in prompt
    assert "pp collisions" not in prompt


def test_copy_deck_specialist_units_allow_concrete_rewritten_flowchart():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "sections": [
            {
                "id": 2,
                "title": "Analysis strategy",
                "layout": "card",
                "text": [],
                "flowchart": [
                    "Run 2 138 fb-1 pp at 13 TeV",
                    ">=2 same-sign muons: pT>30 GeV, |eta|<2.4",
                    "WZ CR | top CR | nonprompt estimate",
                    "Profile-likelihood CLs fit",
                ],
            }
        ],
        "placeholders": [],
        "copy_deck": {
            "copy_units": [
                {
                    "id": "C01",
                    "target_section": 2,
                    "type": "region_matrix",
                    "text": "SR: 2 SS μ; WZ CR: 3μ with OS pair near mZ",
                    "max_chars": 56,
                    "priority": "must",
                    "evidence": "regions",
                    "quiz_ids": [],
                }
            ],
            "coverage_notes": [],
        },
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    assert "polished public text-only HEP analysis schematic" in prompt
    assert "Node label" in prompt
    assert "pT>30 GeV" in prompt
    assert "Profile-likelihood CLs fit" in prompt


def test_square_hero_layout_suppresses_summary_strip_and_optional_fit_chips():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "sections": [
            {
                "id": 5,
                "title": "Results",
                "layout": "large lower hero result card plus summary strip",
                "text": [],
            }
        ],
        "placeholders": [
            {
                "id": "FIG 01",
                "section": 5,
                "label": "Observed result limit",
                "aspect": "1:1 square",
                "role": "hero_result",
            }
        ],
        "copy_deck": {
            "copy_units": [
                {
                    "id": "C01",
                    "target_section": 5,
                    "type": "hero_headline",
                    "text": "No significant excess",
                    "max_chars": 42,
                    "priority": "must",
                    "evidence": "result",
                    "quiz_ids": [],
                },
                {
                    "id": "C02",
                    "target_section": 5,
                    "type": "fit_strategy",
                    "text": "Profile likelihood with nuisance parameters",
                    "max_chars": 60,
                    "priority": "should",
                    "evidence": "method",
                    "quiz_ids": [],
                },
            ],
            "coverage_notes": [],
        },
        "conclusion": [],
    }
    prompt = build_prompt(spec)
    section_line = next(line for line in prompt.splitlines() if line.startswith("Section 5,"))
    assert "square-first hero layout" in section_line
    assert "plus summary strip" not in section_line
    assert "No significant excess" in prompt
    assert "Profile likelihood with nuisance parameters" not in prompt


def test_single_copy_deck_conclusion_is_augmented_with_public_conclusions():
    spec = {
        "project": {"title": "T", "topic": "T"},
        "style": {},
        "sections": [{"id": 5, "title": "Results", "layout": "card", "text": []}],
        "placeholders": [],
        "copy_deck": {
            "copy_units": [
                {
                    "id": "C01",
                    "target_section": 5,
                    "type": "conclusion",
                    "text": "Leading high-mass LHC constraints",
                    "max_chars": 60,
                    "priority": "must",
                    "evidence": "result",
                    "quiz_ids": [],
                }
            ],
            "coverage_notes": [],
        },
        "conclusion": [
            "No significant excess is observed.",
            "First collider limit on the effective dimuon Majorana mass.",
            "Do not use this placeholder explanation.",
        ],
        "forbidden_phrases": ["placeholder explanation"],
    }
    prompt = build_prompt(spec)
    section_block = prompt[prompt.index("Section 5,") : prompt.index("Conclusion box")]
    assert "Leading high-mass LHC constraints" not in section_block
    assert "Leading high-mass LHC constraints" in prompt
    assert "No significant excess is observed." in prompt
    assert "First collider limit on the effective dimuon Majorana mass." in prompt
    assert "placeholder explanation" not in prompt
