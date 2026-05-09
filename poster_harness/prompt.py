from __future__ import annotations

import re
from typing import Any

from .schemas import DEFAULT_FORBIDDEN_PHRASES

INTERNAL_DEFAULTS = DEFAULT_FORBIDDEN_PHRASES


HEP_POSTER_GRAMMAR = [
    "Use a CERN/LHCC-style scientific story spine: Motivation → dataset/selection → strategy/background → key result → interpretation/summary.",
    "Use a strong top banner occupying roughly 10-16% of the canvas with title, author/institute line, and one abstract detector/beam artwork zone.",
    "Use colored section ribbons or numbered tabs so the viewer can scan the poster from 2 meters away.",
    "Create one dominant hero region for the headline result; do not divide the body into many equal tiny boxes.",
    "Cards/blocks do not need to be square: use wide bands, tall sidebars, pill headers, circular callouts, curved connectors, and staggered panels when it improves hierarchy.",
    "Keep the body modular: 4-6 large cards or shaped blocks, 2-3 columns, generous gutters, and a clear left-to-right/top-to-bottom reading path.",
    "Let 40-55% of the body be reserved for figure placeholders; HEP posters are figure-led, not paragraph-led.",
    "Use small badges for dataset, collision energy, luminosity, channel, or experiment only when present in the supplied text; never invent numbers.",
]

TEXT_DENSITY_RULES = [
    "Compress prose into poster bullets: short noun phrases, ideally under 11 words each.",
    "Maintain Paper2Poster-style information richness: enough compact facts to answer the central reader questions, not just decorative headings.",
    "Avoid microscopic paragraphs, footnote blocks, dense equations, and reference lists in the rendered poster.",
    "Prefer 2-4 bullets per card and at most one short sentence per text block; preserve meaning without adding new science.",
    "Use public fact chips and numeric badges for explicitly grounded dataset/result facts; do not invent numbers.",
    "If text competes with a result figure, shrink or omit lower-priority text first and enlarge/preserve the figure placeholder.",
]

FIGURE_COMPOSITION_RULES = [
    "Allocate the largest placeholder to the main result/limit/cross-section/significance plot.",
    "Every placeholder rectangle must match its selected source image aspect ratio; enlarge or reshape the surrounding block instead of stretching the placeholder.",
    "Method, detector, topology, or control-region placeholders should support the story, not dominate it unless the paper is instrumentation-focused.",
    "Dense multi-panel HEP plots need large absolute area; preserve their native wide/tall ratio rather than forcing a square slot.",
    "Placeholders should align to the card grid and be easy to replace: rectangular, unobstructed, with visible margins.",
    "All figure-containing cards must use a light paper/lab-white surface around the placeholder; dark or saturated fills may be outer accents only, never the block background directly behind a future white plot.",
    "If a section mixes portrait, near-square, and square plots, give each placeholder its own correctly shaped light mat; do not force them into equal tall columns or equal square tiles.",
]

FIGURE_SURFACE_RULES = [
    "Mandatory: every [FIG NN] placeholder sits inside a light neutral figure card or mat (warm white, pearl, very pale blue, or very pale gray), never inside a dark navy/purple/black content block.",
    "The placeholder rectangle and its immediate surrounding card surface should read as one clean paper-like area, so a future white-background plot blends naturally instead of looking pasted onto a dark sticker.",
    "Dark cinematic backgrounds, detector rings, beamline glow, and saturated accent colors may frame figure cards from the outside, appear in margins, headers, side rails, numbered tabs, or corner ornaments, but must not fill the main figure card.",
    "For a hero result, create drama through scale, gold/blue outlines, shadows, halos, ribbons, and nearby abstract art; keep the actual result card interior light.",
    "For multi-figure sections, use a shared light cluster panel with thin pale separators and subtle shadows rather than separate dark tiles.",
    "Keep decorative texture out of figure mats; use only a quiet solid or near-solid light fill inside any chart/plot/diagram block.",
]

TYPOGRAPHY_RULES = [
    "Use a modern Swiss/editorial sans-serif feeling: Helvetica/Inter/Source-Sans-like, condensed bold title, crisp section headers, and regular-weight readable body text.",
    "Use a strict type scale: very large title, large numbered section titles, medium subheads, compact bullets; avoid many competing font styles.",
    "Create emphasis with weight, color, pill labels, and spacing rather than dense prose or decorative typefaces.",
    "Keep scientific symbols legible and sparse; do not use decorative equation-like typography outside public text.",
]

COLOR_MATERIAL_RULES = [
    "Palette should feel premium rather than default-PPT: deep navy/indigo atmospheric field, CMS/cobalt blue, violet/magenta contrast, restrained amber/gold highlights, and warm-white figure cards.",
    "Maintain high contrast for text: dark text on light content cards, white text only in the top banner or non-figure dark accents.",
    "Use gradients, glows, and metallic accents in the background and frames, while keeping text cards and figure cards calm.",
    "Avoid large flat dark rectangles behind plots; avoid random rainbow palettes, neon clutter, and low-contrast gray text.",
]


def sanitize_public_text(text: str, forbidden: list[str] | None = None) -> str:
    forbidden = (forbidden or []) + INTERNAL_DEFAULTS
    lines = []
    for line in text.splitlines():
        low = line.lower()
        if any(p.lower() in low for p in forbidden):
            continue
        lines.append(line)
    return "\n".join(lines)


def _q(text: str) -> str:
    return text.replace('"', "'")


def _sentence(text: Any) -> str:
    value = str(text or "").strip()
    return value if value.endswith((".", "!", "?")) else value + "."


def build_prompt(spec: dict[str, Any]) -> str:
    project = spec.get("project", {})
    style = spec.get("style", {})
    sections = spec.get("sections", [])
    placeholders = spec.get("placeholders", [])
    conclusion = [c for c in spec.get("conclusion", []) if not _is_internal(c, spec)]

    lines: list[str] = []
    lines += [
        "Create a complete vertical academic conference poster as one finished design.",
        "",
        f"Topic: {project.get('topic', project.get('title', 'scientific research poster'))}.",
        f"Audience: {project.get('audience', 'academic conference audience')}.",
        f"Overall style: {style.get('summary', 'premium scientific conference poster, modern, artistic, clean, not a collage')}.",
        "",
        "POSITIVE ART DIRECTION:",
        f"- Art direction: {_sentence(style.get('art_direction', 'premium editorial science design with layered abstract geometry, luminous gradients, subtle depth, and clear hierarchy'))}",
        f"- Layout rhythm: {_sentence(style.get('layout_rhythm', 'asymmetric but balanced; avoid uniformly tiled white boxes'))}",
        f"- Background texture: {_sentence(style.get('background_texture', 'soft abstract scientific texture behind content, never fake data'))}",
        f"- Typography: {_sentence(style.get('typography', 'modern editorial sans-serif typography with a clear type scale, bold condensed title, crisp section headers, and readable compact bullets'))}",
        f"- Color/material palette: {_sentence(style.get('color_palette', 'deep navy/indigo atmosphere with cobalt blue, violet, restrained gold accents, and warm-white content/figure cards'))}",
        f"- Figure-card surface: {_sentence(style.get('figure_surface', 'all plot/diagram placeholders must sit on light paper-like cards or mats, never on dark content blocks'))}",
        "- Start from a beautiful editorial composition, not from a slide deck or wireframe.",
        "- Use abstract, non-data artwork outside placeholders: gradients, light trails, detector-like geometry, atmospheric glow, soft depth, and subtle material texture.",
        "- Decorative icons/badges outside placeholders must be generic only (stars, checks, arrows, abstract circles). Do not put physics symbols, particle labels, equations, ΔL=2, μ/ν/q/W/N, axes, vertices, or event-display schematics in decorative badges.",
        "- Flowcharts, summary pills, and callout badges must not turn scientific shorthand into icons. Use plain words in normal text (for example 'dimuon' or 'muon-sector limit'), never standalone μμ/ν/W/N/ΔL/equation badges or pictograms.",
        "- Give the poster varied visual mass: one dominant hero region, secondary supporting modules, small badges, shaped callouts, and breathing room.",
        "- Section blocks may be rounded rectangles, capsules, circular badges, vertical sidebars, L-shaped wraps, staggered panels, or translucent overlays when this improves hierarchy.",
        "- Keep every scientific figure placeholder itself a clean rectangle on a light paper-like figure card, but make the surrounding poster expressive and premium.",
        "- Use official-looking identity as plain text badges only; do not hallucinate complex collaboration or institute logos unless explicitly supplied as assets.",
        "",
    ]

    storyboard = spec.get("storyboard") if isinstance(spec.get("storyboard"), dict) else {}
    storyboard_sections = _storyboard_section_map(storyboard)
    if storyboard:
        lines += [
            "NARRATIVE STORYBOARD (internal design brief; do not render this heading or the word storyboard):",
        ]
        core_message = str(storyboard.get("core_message") or storyboard.get("meta", {}).get("one_sentence_takeaway") or "").strip()
        if core_message:
            lines.append(f"- Core message to make visually obvious: {_q(_design_brief_safe_text(core_message))}")
        layout_tree = storyboard.get("layout_tree") if isinstance(storyboard.get("layout_tree"), dict) else {}
        reading_order = layout_tree.get("reading_order") or []
        if reading_order:
            lines.append(f"- Reading order should follow section ids: {', '.join(str(item) for item in reading_order)}.")
        if layout_tree.get("hero_section"):
            lines.append(f"- Hero section id: {layout_tree.get('hero_section')}; hero visual role: {_q(str(layout_tree.get('hero_visual_role') or 'headline result figure'))}.")
        if layout_tree.get("layout_intent"):
            lines.append(f"- Layout intent: {_q(_design_brief_safe_text(str(layout_tree.get('layout_intent'))))}")
        for sec in storyboard.get("sections") or []:
            if not isinstance(sec, dict):
                continue
            claims = [_design_brief_safe_text(str(item).strip()) for item in sec.get("key_claims") or [] if str(item).strip()]
            claim_text = "; ".join(claims[:3])
            preferred_visual = _design_brief_safe_text(str(sec.get("preferred_visual") or ""))
            lines.append(
                f"- Section {sec.get('id')}: role={_q(str(sec.get('role') or 'section'))}; "
                f"text budget={_q(str(sec.get('text_budget') or 'compact bullets'))}; "
                f"preferred visual={_q(preferred_visual)}"
                + (f"; key claims={_q(claim_text)}" if claim_text else "")
            )
        lines += [
            "- Use this storyboard only to guide hierarchy, reading path, and text compression; render only the public section text specified below.",
            "- Storyboard science terms are semantic guidance only: never render them as standalone icons, circular badges, particle-symbol marks, fake diagrams, or flowchart pictograms outside [FIG NN] placeholders.",
            "",
        ]

    information_plan = storyboard.get("information_plan") if isinstance(storyboard.get("information_plan"), dict) else {}
    lines += _information_density_prompt_lines(style, information_plan)

    grammar_rules = _style_rule_list(style, "hep_poster_grammar", HEP_POSTER_GRAMMAR)
    density_rules = _style_rule_list(style, "text_density", TEXT_DENSITY_RULES)
    figure_rules = _style_rule_list(style, "figure_composition", FIGURE_COMPOSITION_RULES)

    lines += ["HEP POSTER DESIGN GRAMMAR (calibrated from public CERN/LHCC posters):"]
    for rule in grammar_rules:
        lines.append(f"- {rule}")
    lines += ["", "TEXT DENSITY AND READABILITY:"]
    for rule in density_rules:
        lines.append(f"- {rule}")
    lines += ["", "FIGURE-LED COMPOSITION:"]
    for rule in figure_rules:
        lines.append(f"- {rule}")
    lines += ["", "FIGURE CARD SURFACE POLICY (mandatory for visual integration):"]
    for rule in FIGURE_SURFACE_RULES:
        lines.append(f"- {rule}")
    lines += ["", "TYPOGRAPHY SYSTEM:"]
    for rule in _style_rule_list(style, "typography_rules", TYPOGRAPHY_RULES):
        lines.append(f"- {rule}")
    lines += ["", "COLOR AND MATERIAL SYSTEM:"]
    for rule in _style_rule_list(style, "color_material_rules", COLOR_MATERIAL_RULES):
        lines.append(f"- {rule}")

    lines += [
        "",
        "Canvas/layout:",
        f"- Portrait poster, approximately {_sentence(style.get('aspect', 'A0 vertical / 2:3 ratio'))}",
        f"- Top band: {_sentence(style.get('top_band', 'dark navy title band with identity area on left and subtle scientific art on right'))}",
        f"- Main body: {_sentence(style.get('body_layout', 'major modules on a light background with translucent rounded white cards'))}",
        f"- Color grammar: {_sentence(style.get('color_grammar', 'primary signal = blue; secondary signal = warm red; use consistently'))}",
        "- Avoid old poster rectangular-crop feeling. Make it feel designed from scratch.",
        "",
    ]
    layout_contract = spec.get("layout_contract") if isinstance(spec.get("layout_contract"), dict) else {}
    if layout_contract:
        lines += _layout_contract_prompt_lines(layout_contract)
        lines.append("")

    lines += [
        "Top title band text:",
        f"Main title: \"{_q(project.get('title', 'Untitled Scientific Poster'))}\"",
        f"Author line: \"{_q(project.get('authors', ''))}\"",
    ]
    if project.get("subtitle"):
        lines.append(f"Small subtitle: \"{_q(project['subtitle'])}\"")
    if project.get("identity"):
        lines.append(f"Left identity block: \"{_q(project['identity'])}\".")
    lines.append("")

    lines += [
        "ABSOLUTE PLACEHOLDER CONTRACT (mandatory; failure means the design is rejected):",
        "- Do NOT draw, approximate, summarize, miniaturize, stylize, or recreate ANY scientific figure content.",
        "- This prohibition includes plots, axes, bars, bins, curves, legends, heatmaps, tables, Feynman diagrams, detector diagrams, process diagrams, equations-as-figures, screenshots, or tiny preview thumbnails.",
        "- Even if a placeholder label says 'diagram', 'plot', 'distribution', 'limits', or 'table', you must NOT draw that item. Draw a blank placeholder only.",
        "- Every location where a plot/diagram/table/image should go must be a clean rectangular placeholder panel.",
        "- Use ONLY the placeholder IDs explicitly listed in this prompt. Do not invent, add, skip ahead to, or render any extra placeholder such as [FIG 06] unless it is listed in the PLACEHOLDER REFERENCE LIST.",
        "- Each placeholder panel must contain ONLY three text elements: the exact ID such as [FIG 01], the intended content label, and the requested aspect ratio.",
        "- Placeholder IDs [FIG 01], [FIG 02], etc. must be large, legible, and visually centered or top-left inside the box.",
        "- Placeholder boxes must be light neutral panels with dashed borders or subtle tint; inside must remain empty except the required ID/label/aspect text.",
        "- The whole figure-containing block/card behind each placeholder must also be light neutral or paper-like. Do not put a chart placeholder inside a dark navy, purple, black, or saturated card.",
        "- Placeholder geometry is audited after generation. A box labeled 1:1 must be visibly square; a box labeled 2.5:1 must be about 2.5 times wider than tall.",
        "- A box labeled 1.2:1 must be near-square: only about 20% wider than tall. Do not draw it as a generic 1.5:1 landscape card.",
        "- Do not draw a generic 1.6:1 landscape rectangle for every placeholder. Each placeholder must use its own declared ratio.",
        "- Rejected geometry examples: a 2.5:1 plot as a 950×90 ribbon, a 1:1 result as a 300×190 landscape card, or any placeholder whose text cluster is detected instead of the whole figure panel.",
        "- No fake data and no real-looking scientific content may appear inside placeholders. Faint abstract grid texture is allowed only if it cannot be mistaken for data.",
        "- Outside placeholders, do not draw particle-labeled icons, equation badges, process badges, or symbolic physics diagrams. Use public text only for scientific claims and generic abstract icons for decoration.",
        "- Later, these placeholders will be replaced with real figures. Therefore they must be rectangular, unobstructed, and have enough padding.",
        "- Never add captions under placeholders that describe drawn content as if the figure were already present; captions may describe the future intended content only.",
        "",
    ]

    decorative_constraints = spec.get("decorative_art_constraints") or []
    if decorative_constraints:
        lines += ["DECORATIVE ART GUIDANCE:"]
        for item in decorative_constraints:
            lines.append(f"- {_q(_positive_decorative_guidance(str(item)))}")
        lines.append("")

    if placeholders:
        allowed_placeholder_ids = ", ".join(f"[{str(fig.get('id', 'FIG ??'))}]" for fig in placeholders)
        lines += [
            "PLACEHOLDER REFERENCE LIST (aspect ratios are mandatory; no pixel-level target is required):",
        ]
        for fig in placeholders:
            aspect_text = str(fig.get("aspect") or "1:1 square")
            shape_hint = _aspect_shape_hint(aspect_text)
            hint_suffix = f"; visual shape: {shape_hint}" if shape_hint else ""
            lines.append(
                f"- [{fig.get('id', 'FIG ??')}]: label \"{_q(str(fig.get('label') or 'figure'))}\"; "
                f"aspect ratio \"{_q(aspect_text)}\"{hint_suffix}."
            )
        lines += [
            "- Aspect ratios describe the visible dashed placeholder rectangle, not the surrounding decorative card. The dashed rectangle itself is what will be audited.",
            f"- Exact allowed placeholder IDs: {allowed_placeholder_ids}. Do not render any other [FIG NN] box.",
            "- Interpret X:Y literally as width:height = X:Y. For example, 2.5:1 means the width is about two and a half heights, not three, four, or seven heights.",
            "- A square placeholder should look square; a wide placeholder should have real vertical presence and should not become a thin banner.",
            "- A moderate landscape placeholder such as 1.49:1 should look only about one and a half times wider than tall; never stretch it into a panoramic 2.5:1 or 3:1 banner.",
            "- A portrait or near-portrait placeholder such as 1:1.16 should look only slightly taller than wide; do not draw it as a narrow vertical strip or tall column.",
            "- A square headline-result placeholder should be prominent but not oversized: keep its side around 28-33% of the canvas width, never a giant sticker covering most of the result card.",
            "- For square placeholders, the visible light/white placeholder fill itself must also be square; do not put the [FIG NN] label inside a wide white rounded rectangle and only draw a square-ish dashed fragment inside it.",
            "- Keep the square headline-result placeholder clearly above the bottom summary/conclusion modules; its bottom edge should sit before the lower fifth of the poster begins, with an obvious gutter below it.",
            "- If a wide placeholder would become too thin at full poster width, make it narrower or make its section taller rather than stretching it into a ribbon.",
            "- If prose, badges, or flowcharts compete with a placeholder, simplify those elements before distorting the placeholder shape.",
            "- Every dashed placeholder must be an interior figure slot, not the outer border of a whole section card.",
            "- Keep a clearly visible gutter between every pair of dashed placeholders; placeholder rectangles must never touch, overlap, or share a border.",
            "- Each placeholder's surrounding figure mat/card must be light and quiet; if a section has a dramatic dark background, place the figure in a large warm-white inset card with padding.",
            "",
        ]
        wide_ids = [
            str(fig.get("id"))
            for fig in placeholders
            if (_parse_aspect_ratio_text(str(fig.get("aspect") or "")) or 1.0) >= 2.0
        ]
        square_ids = [
            str(fig.get("id"))
            for fig in placeholders
            if 0.90 <= (_parse_aspect_ratio_text(str(fig.get("aspect") or "")) or 1.0) <= 1.10
        ]
        if wide_ids and square_ids:
            lines += [
                "INTER-PLACEHOLDER GUTTER RULE:",
                f"- The wide placeholder(s) {', '.join(f'[{item}]' for item in wide_ids)} and square result placeholder(s) {', '.join(f'[{item}]' for item in square_ids)} must be in separate non-overlapping modules with an obvious blank/card gutter between them.",
                "- Do not stack a wide distribution placeholder so close to a square result placeholder that a replacement figure could intrude into the next module.",
                "",
            ]

    # Section details.
    for sec in sections:
        sid = sec.get("id")
        title = sec.get("title", f"Section {sid}")
        layout = sec.get("layout", "card")
        sec_figs = [p for p in placeholders if p.get("section") == sid]
        bullet_budget = _section_bullet_budget(sec_figs)
        lines.append(f"Section {sid}, layout: {layout}, title: \"{sid}  {_q(title)}\"")
        storyboard_sec = storyboard_sections.get(int(sid)) if sid is not None else None
        if storyboard_sec:
            if storyboard_sec.get("role"):
                lines.append(f"Story role: {_q(str(storyboard_sec.get('role')))}")
            if storyboard_sec.get("synopsis"):
                lines.append(f"Story synopsis for layout emphasis (do not render verbatim): {_q(str(storyboard_sec.get('synopsis')))}")
            if storyboard_sec.get("text_budget"):
                lines.append(f"Story text budget: {_q(str(storyboard_sec.get('text_budget')))}")
        if sec.get("text"):
            if bullet_budget is not None:
                lines.append(
                    f"Text budget for this figure-containing section: render at most {bullet_budget} short bullets; omit lower-priority bullets before shrinking placeholders."
                )
            lines.append("Text content:")
            for t in sec["text"]:
                if isinstance(t, str):
                    clean = sanitize_public_text(t, spec.get("forbidden_phrases"))
                    if clean:
                        lines.append(f"- {_q(clean)}")
                elif isinstance(t, dict):
                    if t.get("title"):
                        lines.append(f"- Block title: \"{_q(t['title'])}\"")
                    for b in t.get("body", []):
                        clean = sanitize_public_text(str(b), spec.get("forbidden_phrases"))
                        if clean:
                            lines.append(f"  Body: {_q(clean)}")
                    rendered_bullets = 0
                    for b in t.get("bullets", []):
                        if bullet_budget is not None and rendered_bullets >= bullet_budget:
                            continue
                        clean = sanitize_public_text(str(b), spec.get("forbidden_phrases"))
                        if clean:
                            lines.append(f"  Bullet: {_q(clean)}")
                            rendered_bullets += 1
        if sec_figs:
            hero_figs = [fig for fig in sec_figs if "hero" in str(fig.get("role") or fig.get("group") or "").lower()]
            if hero_figs:
                lines.append(
                    "Hero visual priority: make this section's hero placeholder visibly dominant within its card while preserving the exact placeholder aspect ratio and light figure surface."
                )
            ratios = [_parse_aspect_ratio_text(str(fig.get("aspect") or "")) or 1.0 for fig in sec_figs]
            if any(ratio < 0.92 for ratio in ratios):
                shape_notes = ", ".join(
                    f"[{fig.get('id')}]=~{(_parse_aspect_ratio_text(str(fig.get('aspect') or '')) or 1.0):.2f}:1"
                    for fig in sec_figs
                )
                lines.append(
                    "Mixed portrait/square placeholder section design: "
                    f"{shape_notes}. Preserve each placeholder's own ratio. "
                    "A near-portrait 0.86:1 or 1:1.16 placeholder is only slightly taller than wide; do not make it a narrow tall column. "
                    "A square placeholder must stay square. If space is tight, enlarge the whole section or reduce prose rather than distorting either placeholder."
                )
            elif any(ratio >= 2.0 for ratio in ratios):
                wide_fig = next(fig for fig in sec_figs if (_parse_aspect_ratio_text(str(fig.get("aspect") or "")) or 1.0) >= 2.0)
                wide_ratio = _parse_aspect_ratio_text(str(wide_fig.get("aspect") or "")) or 2.5
                lines.append(
                    f"Wide-figure section design: draw the dashed placeholder as width:height about {wide_ratio:.1f}:1; "
                    "give it a substantial, readable light figure zone; reduce nearby prose before making it look like a thin ribbon."
                )
            elif any(1.35 <= ratio < 2.0 for ratio in ratios):
                moderate_shapes = ", ".join(
                    f"[{fig.get('id')}]=~{(_parse_aspect_ratio_text(str(fig.get('aspect') or '')) or 1.0):.2f}:1"
                    for fig in sec_figs
                )
                lines.append(
                    "Moderate-landscape placeholder section design: "
                    f"{moderate_shapes}. A 1.49:1 placeholder is only moderately landscape — about one and a half times wider than tall. "
                    "Do not draw it as a panoramic hero ribbon, full-width strip, 2.5:1 panel, or 3:1 banner. "
                    "Give the plot enough vertical height and, if necessary, place text beside it rather than stretching the placeholder across the card."
                )
            elif any(0.92 <= ratio <= 1.08 for ratio in ratios):
                lines.append(
                    "Square-placeholder section design: reserve a true square dashed rectangle with square light/white fill; the surrounding figure card/mat must also be light, although it may have dramatic outer outlines, halos, ribbons, or side accents. The placeholder itself cannot be a landscape box or wide white rounded rectangle. For a headline result, make the square substantial yet moderate, with visible surrounding light-card breathing room; do not let it descend into the bottom summary/conclusion zone."
                )
            elif all(1.08 < ratio < 1.35 for ratio in ratios):
                near_square_shapes = ", ".join(
                    f"[{fig.get('id')}]=~{(_parse_aspect_ratio_text(str(fig.get('aspect') or '')) or 1.0):.1f}:1"
                    for fig in sec_figs
                )
                lines.append(
                    "Near-square placeholder section design: "
                    f"{near_square_shapes}; each dashed rectangle should look almost square, only slightly wider than tall. "
                    "For paired near-square placeholders, each visible dashed box must be wider than tall (for 1.2:1, height is about 83% of width), not portrait or square-tall. "
                    "Do not use 1.5:1, 16:10, portrait, or generic landscape tiles for these placeholders."
                )
            elif len(sec_figs) >= 2:
                lines.append(
                    "Multi-placeholder section design: draw separate figure tiles with visible gutters; the surrounding card may be expressive, but the placeholder rectangles must stay clean and keep their stated ratios."
                )
            lines.append("Use these blank placeholders in this section (do not draw their content):")
            for fig in sec_figs:
                aspect_text = str(fig.get("aspect", "1:1 square"))
                lines.append(
                    f"- [{fig['id']}]: blank rectangular placeholder only; label \"{_q(fig.get('label','figure'))}\"; "
                    f"aspect ratio \"{_q(aspect_text)}\"; inside the box write only the exact ID, label, and aspect ratio."
                )
            lines.append("Do not render separate captions directly above or below these placeholder boxes; keep their replacement area visually clear.")
            lines.append("Keep bullets/text, flowcharts, badges, and decorative artwork outside every placeholder rectangle with a clear visual gutter.")
            lines.append("Use a light neutral figure-card surface for this whole figure area; dark or saturated colors are allowed only as thin framing accents outside the light card.")
        if sec.get("flowchart"):
            wide_sec = any((_parse_aspect_ratio_text(str(fig.get("aspect") or "")) or 1.0) >= 2.0 for fig in sec_figs)
            if wide_sec:
                lines.append(
                    "Optional compact public text-only analysis flowchart: draw it only if the wide placeholder remains visually substantial; otherwise omit the flowchart before shrinking the placeholder."
                )
            else:
                lines.append("Draw this as a simple public text-only analysis flowchart, not a source-figure placeholder:")
            lines.append("- Render only concise node labels and arrows derived from the following items; do NOT render instruction sentences verbatim.")
            lines.append("- Flowchart style is text-only: rounded text boxes plus connecting arrows only. No circular node icons, no pictograms, no mini charts, no particle/equation symbols, and no symbol badges.")
            for item in sec["flowchart"]:
                lines.append(f"- Node label: \"{_q(_flowchart_label_text(str(item)))}\"")
        if sec.get("caption") and not sec_figs:
            clean_caption = sanitize_public_text(str(sec["caption"]), spec.get("forbidden_phrases")).strip()
            if clean_caption:
                lines.append(f"Caption: \"{_q(clean_caption)}\"")
        lines.append("")

    if conclusion:
        lines.append("Conclusion box titled \"Conclusion and prospects\" with public bullets:")
        for bullet in conclusion:
            lines.append(f"- \"{_q(str(bullet))}\"")
        if spec.get("closing"):
            lines.append(f"Closing line: \"{_q(spec['closing'])}\"")
        lines.append("Conclusion hard constraint: the conclusion box must contain ONLY the public bullets and closing line listed above. Do not add any workflow, placeholder, replacement, validation, TODO, or production-process bullet.")
        lines.append("")

    lines += [
        "Layout quality constraints:",
        "- Copy all section titles, placeholder labels, and public conclusion bullets exactly as supplied; do not misspell or paraphrase headings.",
        "- Do not invent extra placeholder IDs. Render exactly the listed [FIG NN] placeholders and no additional figure boxes.",
        "- Make placeholder boxes geometrically clean and easy to detect for replacement: blank interior, dashed/subtle border, exact [FIG NN] text, label, and aspect-ratio text.",
        "- Match each placeholder's declared aspect ratio closely; width/height should visibly agree with the source ratio, not just the label text.",
        "- Keep all section cards aligned to a coherent grid, but avoid a monotonous tiled template; use editorial hierarchy, varied card scale, non-square cards, and shaped callouts.",
        "- The poster must not look like a simple wireframe; it should feel like a finished premium visual design with placeholders ready for production replacement.",
        "- Use modern rounded cards, subtle shadows, scientific background art, glow/depth effects, and coherent accent colors.",
        "- Use sophisticated typography and palette: editorial sans-serif type, disciplined type scale, deep atmospheric background, cobalt/violet/gold accents, and warm-white content cards.",
        "- Never use dark-filled blocks for chart/plot/diagram sections; if the surrounding poster is dark, put every figure on a large light paper card or mat.",
        "- If you use summary pills or badges, keep their icons generic and their labels word-based; do not use standalone physics-symbol badges such as μμ, ν, W, N, ΔL, or equation fragments.",
        "- Preserve enough whitespace for later replacement; no decorative artwork may overlap a placeholder rectangle.",
        "- Keep placeholder-to-placeholder gutters obvious; final real figures must fit inside their own dashed boxes without protruding into neighboring modules.",
        "- Visually group many placeholders into one hero result, supporting analysis figures, and smaller diagnostics rather than equal tiles.",
    ]
    return "\n".join(lines).strip() + "\n"


def _positive_decorative_guidance(text: str) -> str:
    stripped = text.strip()
    lowered = stripped.lower().rstrip(".")
    rewrites = {
        "do not draw feynman diagrams in decorative areas": "Keep decorative art abstract; reserve Feynman diagrams for [FIG NN] placeholders only.",
        "do not put particle labels such as mu, nu, j, q, w, n in decorative header artwork": "Use abstract geometric motifs rather than specific particle notation in header artwork.",
        "do not add feynman diagrams, interaction vertices, or physics-process diagrams outside labeled placeholders": "Reserve process diagrams, interaction vertices, and physics schematics for labeled [FIG NN] placeholders; use abstract motion and detector-inspired forms elsewhere.",
        "do not add physics arrows, interaction vertices, or process diagrams outside labeled placeholders": "Reserve physics arrows, interaction vertices, and process diagrams for labeled [FIG NN] placeholders; use abstract motion and detector-inspired forms elsewhere.",
        "do not draw fake plots, fake tables, or unlabeled scientific diagrams outside placeholders": "Focus decorative art on abstract atmosphere, geometry, light, and texture; all plots, tables, and scientific diagrams belong only in labeled placeholders.",
    }
    if lowered in rewrites:
        return rewrites[lowered]
    if lowered.startswith("do not "):
        return "Use abstract decorative design instead; " + stripped[7:]
    return stripped


def _layout_contract_prompt_lines(contract: dict[str, Any]) -> list[str]:
    rows = [
        "LAYOUT CONTRACT (soft visual prior; replacement QA will check it):",
        "- Coordinates below are normalized poster fractions [x0, y0, x1, y1], not pixel art instructions.",
        "- Keep each visible dashed [FIG NN] rectangle roughly inside its planned zone/search zone and in the named section, while preserving artistic freedom for card shape, glow, ribbons, and surrounding decoration.",
        "- Do not use these coordinates as a reason to draw fake figure content; they only reserve blank placeholder rectangles for later real source figures.",
    ]
    for item in contract.get("placeholders") or []:
        if not isinstance(item, dict):
            continue
        fig_id = str(item.get("id") or "").strip()
        if not fig_id:
            continue
        rows.append(
            f"- [{fig_id}]: planned placeholder zone {_format_norm_box(item.get('zone'))}; "
            f"search zone {_format_norm_box(item.get('search_zone'))}; "
            f"aspect {item.get('aspect') or item.get('expected_aspect')}; "
            f"label \"{_q(str(item.get('label') or 'figure'))}\"."
        )
    return rows


def _format_norm_box(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return "[unspecified]"
    return "[" + ", ".join(f"{float(v):.2f}" for v in value[:4]) + "]"


def _flowchart_label_text(text: str) -> str:
    """Make flowchart labels less likely to become physics-symbol icons.

    Public scientific shorthand can still appear in ordinary prose, but flowchart
    nodes are graphically close to icons.  Expanding compact symbols here keeps
    the node text semantic and discourages standalone μμ/particle badges.
    """
    value = str(text or "").strip()
    replacements = [
        (r"same-sign\s*μμ", "same-sign dimuon"),
        (r"\bμμ\b", "dimuon"),
        (r"\bμ\s*μ\b", "dimuon"),
        (r"\bmu\s*mu\b", "dimuon"),
        (r"\bν\b", "neutrino"),
        (r"\bnu\b", "neutrino"),
    ]
    for pattern, repl in replacements:
        value = re.sub(pattern, repl, value, flags=re.IGNORECASE)
    return value


def _design_brief_safe_text(text: str) -> str:
    """Sanitize internal storyboard prose before it enters the image prompt.

    The storyboard is not rendered verbatim, but visual models may still convert
    compact physics notation into decorative icons.  Use word forms for the
    highest-risk symbols while preserving the meaning for layout planning.
    """
    value = str(text or "")
    replacements = {
        "μμ": "dimuon",
        "μ μ": "dimuon",
        "μ/ν/q/W/N": "muon/neutrino/quark/W-boson/heavy-neutrino symbols",
        "ν": "neutrino",
        "ΔL": "lepton-number change",
        "|VμN|²": "muon-heavy-neutrino mixing squared",
        "|V_{μN}|²": "muon-heavy-neutrino mixing squared",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def _parse_aspect_ratio_text(aspect: str) -> float | None:
    text = str(aspect or "").strip().lower()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[:/]\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return 1.0 if "square" in text else None
    den = float(match.group(2))
    return float(match.group(1)) / den if den else None


def _aspect_shape_hint(aspect: str) -> str:
    ratio = _parse_aspect_ratio_text(aspect)
    if ratio is None:
        return "clean rectangle matching the stated source aspect"
    if 0.92 <= ratio <= 1.08:
        return "true square; width and height should look equal"
    if ratio >= 2.0:
        return f"substantial wide panel with width:height about {ratio:.1f}:1; not 3.5:1, 4:1, or a thin ribbon"
    if ratio > 1.0:
        if ratio < 1.35:
            return f"near-square panel with width:height about {ratio:.1f}:1; only slightly wider than tall, not a 1.5:1 landscape card"
        return f"moderate landscape panel with width:height about {ratio:.1f}:1; not a panoramic banner"
    return f"portrait panel, about {(1.0 / ratio):.1f} times taller than wide"


def _section_bullet_budget(sec_figs: list[dict[str, Any]]) -> int | None:
    if not sec_figs:
        return None
    ratios = [_parse_aspect_ratio_text(str(fig.get("aspect") or "")) or 1.0 for fig in sec_figs]
    if any(ratio >= 2.0 for ratio in ratios):
        return 2
    if any(0.92 <= ratio <= 1.08 for ratio in ratios):
        return 2
    return 3


def _storyboard_section_map(storyboard: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for item in storyboard.get("sections") or []:
        if not isinstance(item, dict):
            continue
        try:
            section_id = int(item.get("id"))
        except Exception:
            continue
        out[section_id] = item
    return out


def _information_density_prompt_lines(style: dict[str, Any], information_plan: dict[str, Any]) -> list[str]:
    density_target = str(
        style.get("information_density")
        or information_plan.get("density_target")
        or "Paper2Poster-rich but readable: 14-24 concise public information units, no paragraphs"
    ).strip()
    data_badges = [str(item).strip() for item in information_plan.get("data_badges") or [] if str(item).strip()][:8]
    display_facts = [str(item).strip() for item in information_plan.get("display_facts") or [] if str(item).strip()][:18]
    must_answer = [str(item).strip() for item in information_plan.get("must_answer_questions") or [] if str(item).strip()][:8]
    visual_story_units = [str(item).strip() for item in information_plan.get("visual_story_units") or [] if str(item).strip()][:8]

    lines = [
        "INFORMATION DENSITY TARGET:",
        f"- Target: {_q(density_target)}.",
        "- Do not make the poster a sparse cover illustration. It should communicate enough public content for a conference viewer to understand the paper's motivation, method, key figures, and conclusion.",
        "- Prefer compact information architecture: 4-6 section modules, 12-20 total short bullets/fact chips, 3-6 small badges, and a concise conclusion strip.",
        "- Keep public facts legible and truthful. If the image model cannot fit a fact clearly, omit the lowest-priority fact rather than shrinking text to unreadable size or inventing abbreviations.",
        "- Numeric badges are allowed only for numbers explicitly present in the supplied public text/assets; never invent luminosities, energies, masses, limits, years, or confidence levels.",
        "- Use figure placeholders as information anchors: each figure card should have a short nearby public headline explaining why the future real figure matters, without describing fake drawn contents.",
    ]
    if data_badges:
        lines.append("- Candidate public data/fact badges to render when space allows:")
        for item in data_badges:
            lines.append(f"  • {_q(_design_brief_safe_text(item))}")
    if display_facts:
        lines.append("- Candidate high-value public facts/claims to distribute across modules:")
        for item in display_facts[:12]:
            lines.append(f"  • {_q(_design_brief_safe_text(item))}")
    if visual_story_units:
        lines.append("- Visual story units to make obvious through hierarchy, not fake scientific drawings:")
        for item in visual_story_units:
            lines.append(f"  • {_q(_design_brief_safe_text(item))}")
    if must_answer:
        lines.append("- A viewer should be able to answer these from the final poster; use them to prioritize content, not as rendered questions:")
        for item in must_answer[:6]:
            lines.append(f"  • {_q(_design_brief_safe_text(item))}")
    lines.append("")
    return lines


def _style_rule_list(style: dict[str, Any], key: str, defaults: list[str]) -> list[str]:
    raw = style.get(key)
    if isinstance(raw, list):
        rules = [str(item).strip() for item in raw if str(item).strip()]
        return rules or defaults
    if isinstance(raw, str) and raw.strip():
        parts = [part.strip(" -;\n\t") for part in re.split(r"[\n;]+", raw) if part.strip(" -;\n\t")]
        return parts or [raw.strip()]
    return defaults


def _is_internal(text: str, spec: dict[str, Any]) -> bool:
    low = str(text).lower()
    forbidden = spec.get("forbidden_phrases", []) + INTERNAL_DEFAULTS
    return any(str(p).lower() in low for p in forbidden)
