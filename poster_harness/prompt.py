from __future__ import annotations

import re
from pathlib import Path
from typing import Any

INTERNAL_DEFAULTS = [
    "replace placeholder",
    "validated cms/pptx plots",
    "preserving the generated design language",
    "next step:",
    "internal",
    "draft",
    "TODO",
    "to be replaced",
    "placeholder figures",
    "use a clean",
    "use cms blue",
    "bottom strip should",
    "should read as",
    "caption should",
    "source text",
    "source document",
]


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
    "Avoid microscopic paragraphs, footnote blocks, dense equations, and reference lists in the rendered poster.",
    "Prefer 2-4 bullets per card and at most one short sentence per text block; preserve meaning without adding new science.",
    "If text competes with a result figure, shrink text first and enlarge the figure placeholder.",
]

FIGURE_COMPOSITION_RULES = [
    "Allocate the largest placeholder to the main result/limit/cross-section/significance plot.",
    "Every placeholder rectangle must match its selected source image aspect ratio; enlarge or reshape the surrounding block instead of stretching the placeholder.",
    "Method, detector, topology, or control-region placeholders should support the story, not dominate it unless the paper is instrumentation-focused.",
    "Dense multi-panel HEP plots need large absolute area; preserve their native wide/tall ratio rather than forcing a square slot.",
    "Placeholders should align to the card grid and be easy to replace: rectangular, unobstructed, with visible margins.",
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
        "ABSOLUTE PLACEHOLDER CONTRACT (mandatory; failure means the design is rejected):",
        "- Do NOT draw, approximate, summarize, miniaturize, stylize, or recreate ANY scientific figure content.",
        "- This prohibition includes plots, axes, bars, bins, curves, legends, heatmaps, tables, Feynman diagrams, detector diagrams, process diagrams, equations-as-figures, screenshots, or tiny preview thumbnails.",
        "- Even if a placeholder label says 'diagram', 'plot', 'distribution', 'limits', or 'table', you must NOT draw that item. Draw a blank placeholder only.",
        "- Every location where a plot/diagram/table/image should go must be a clean rectangular placeholder panel.",
        "- Each placeholder panel must contain ONLY three text elements: the exact ID such as [FIG 01], the intended content label, and the requested aspect ratio.",
        "- Placeholder IDs [FIG 01], [FIG 02], etc. must be large, legible, and visually centered or top-left inside the box.",
        "- Placeholder boxes must be light neutral panels with dashed borders or subtle tint; inside must remain empty except the required ID/label/aspect text.",
        "- Placeholder geometry is audited after generation. A box labeled 1:1 must be visibly square; a box labeled 2.5:1 must be about 2.5 times wider than tall.",
        "- Do not draw a generic 1.6:1 landscape rectangle for every placeholder. Each placeholder must use its own declared ratio.",
        "- Rejected geometry examples: a 2.5:1 plot as a 950×90 ribbon, a 1:1 result as a 300×190 landscape card, or any placeholder whose text cluster is detected instead of the whole figure panel.",
        "- No fake data and no real-looking scientific content may appear inside placeholders. Faint abstract grid texture is allowed only if it cannot be mistaken for data.",
        "- Later, these placeholders will be replaced with real figures. Therefore they must be rectangular, unobstructed, and have enough padding.",
        "- Never add captions under placeholders that describe drawn content as if the figure were already present; captions may describe the future intended content only.",
        "",
        "PUBLICATION FILTER:",
        "- Do not include workflow notes, TODOs, internal comments, replacement instructions, or any text about placeholders being replaced later.",
        "- The final poster should read like a public conference poster, not a production memo.",
        "- The word 'placeholder' itself should not appear as public-facing poster text except inside the neutral figure boxes if needed for detection.",
        "",
    ]
    decorative_constraints = spec.get("decorative_art_constraints") or []
    if decorative_constraints:
        lines += ["DECORATIVE ART CONSTRAINTS:"]
        for item in decorative_constraints:
            lines.append(f"- {_q(str(item))}")
        lines.append("")

    grammar_rules = _style_rule_list(style, "hep_poster_grammar", HEP_POSTER_GRAMMAR)
    density_rules = _style_rule_list(style, "text_density", TEXT_DENSITY_RULES)
    figure_rules = _style_rule_list(style, "figure_composition", FIGURE_COMPOSITION_RULES)

    lines += [
        "HEP POSTER DESIGN GRAMMAR (calibrated from public CERN/LHCC posters):",
    ]
    for rule in grammar_rules:
        lines.append(f"- {rule}")
    lines += [
        "",
        "TEXT DENSITY AND READABILITY:",
    ]
    for rule in density_rules:
        lines.append(f"- {rule}")
    lines += [
        "",
        "FIGURE-LED COMPOSITION:",
    ]
    for rule in figure_rules:
        lines.append(f"- {rule}")

    lines += [
        "",
        "POSITIVE ART DIRECTION:",
        f"- Art direction: {style.get('art_direction', 'premium editorial science design with layered abstract geometry, luminous gradients, subtle depth, and clear hierarchy')}.",
        f"- Layout rhythm: {style.get('layout_rhythm', 'asymmetric but balanced; avoid uniformly tiled white boxes')}.",
        f"- Background texture: {style.get('background_texture', 'soft abstract scientific texture behind content, never fake data')}.",
        "- You are encouraged to add abstract, non-data artwork outside placeholders: gradients, light trails, detector-like geometry, depth, glow, and subtle material texture.",
        "- Section blocks may be varied shapes: rounded rectangles, capsules, circular badges, vertical sidebars, overlapping translucent panels, and curved connector lines are allowed when they do not touch figure placeholders.",
        "- Keep every scientific figure placeholder itself a clean rectangle for replacement, but the surrounding block/card can be non-square or more expressive.",
        "- Keep the placeholder contract strict, but do not let the poster become a wireframe; the surrounding poster should feel like a finished premium visual design.",
        "- Use official-looking identity as plain text badges only; do not hallucinate complex collaboration or institute logos unless explicitly supplied as assets.",
        "",
        "Canvas/layout:",
        f"- Portrait poster, approximately {style.get('aspect', 'A0 vertical / 2:3 ratio')}.",
        f"- Top band: {style.get('top_band', 'dark navy title band with identity area on left and subtle scientific art on right')}.",
        f"- Main body: {style.get('body_layout', 'major modules on a light background with translucent rounded white cards')}.",
        f"- Color grammar: {style.get('color_grammar', 'primary signal = blue; secondary signal = warm red; use consistently')}.",
        "- Avoid old poster rectangular-crop feeling. Make it feel designed from scratch.",
        "",
        "Top title band text:",
        f"Main title: \"{_q(project.get('title', 'Untitled Scientific Poster'))}\"",
        f"Author line: \"{_q(project.get('authors', ''))}\"",
    ]
    if project.get("subtitle"):
        lines.append(f"Small subtitle: \"{_q(project['subtitle'])}\"")
    if project.get("identity"):
        lines.append(f"Left identity block: \"{_q(project['identity'])}\".")
    lines.append("")

    if placeholders:
        lines += [
            "PLACEHOLDER GEOMETRY BLUEPRINT (must be followed visually, not only written as text):",
        ]
        for fig in placeholders:
            lines.append(f"- [{fig.get('id', 'FIG ??')}]: {_placeholder_geometry_blueprint(fig)}")
        lines += [
            "- The listed dimensions are readability targets on a 1024×1536 canvas; the aspect ratio itself is mandatory.",
            "- For wide placeholders, do not make a full-width ribbon unless the height also scales with the ratio. Example: 950 px wide at 2.5:1 requires about 380 px height; if that is too tall, use a shorter but still large box such as 650×260.",
            "- For square placeholders, reserve a genuinely square panel; do not place them inside landscape cards that will later crop or letterbox the real figure.",
            "- If text competes with placeholder geometry, shrink/remove text before breaking the declared aspect ratio.",
            "- If a surrounding section/card is too narrow, reshape the surrounding card; never distort the placeholder ratio.",
            "- Curved/circular/pill design elements are allowed around cards, but figure placeholder boxes remain clean rectangles with the exact geometry above.",
            "",
        ]

    # Section details.
    for sec in sections:
        sid = sec.get("id")
        title = sec.get("title", f"Section {sid}")
        layout = sec.get("layout", "card")
        lines.append(f"Section {sid}, layout: {layout}, title: \"{sid}  {_q(title)}\"")
        if sec.get("text"):
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
                    for b in t.get("bullets", []):
                        clean = sanitize_public_text(str(b), spec.get("forbidden_phrases"))
                        if clean:
                            lines.append(f"  Bullet: {_q(clean)}")
        sec_figs = [p for p in placeholders if p.get("section") == sid]
        if sec_figs:
            lines.append("Use these blank placeholders in this section (do not draw their content):")
            for fig in sec_figs:
                aspect_text = str(fig.get("aspect", "1:1 square"))
                lines.append(
                    f"- Draw a BLANK dashed rectangle for [{fig['id']}] with EXACT source-image aspect ratio {aspect_text}. "
                    f"Inside it write only: \"[{fig['id']}]\", \"{_q(fig.get('label','figure'))}\", and \"aspect {aspect_text}\". Do not draw the figure."
                )
                ratio_hint = _aspect_ratio_hint(aspect_text)
                if ratio_hint:
                    lines.append(f"  Aspect-ratio hard requirement for [{fig['id']}]: {ratio_hint}")
                hint = _placeholder_layout_hint(fig)
                if hint:
                    lines.append(f"  Placement/size hint for [{fig['id']}]: {hint}")
            lines.append("Do not render separate captions directly above or below these placeholder boxes; keep padding clear for later figure replacement.")
        if sec.get("flowchart"):
            lines.append("Draw this as a simple public text-only analysis flowchart, not a source-figure placeholder:")
            lines.append("- Render only concise node labels and arrows derived from the following items; do NOT render instruction sentences verbatim.")
            for item in sec["flowchart"]:
                lines.append(f"- Node label: \"{_q(str(item))}\"")
        if sec.get("caption") and not sec_figs:
            clean_caption = sanitize_public_text(str(sec['caption']), spec.get("forbidden_phrases")).strip()
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
        "- Make placeholder boxes geometrically clean and easy to detect for replacement; blank interior, dashed border, exact [FIG NN] text.",
        "- Match each placeholder's declared aspect ratio closely; width/height should visibly agree with the source ratio, not just the label text.",
        "- Leave at least 12 px visual padding around every placeholder.",
        "- Keep all section cards aligned to a consistent grid, but avoid a monotonous tiled template; use editorial hierarchy, varied card scale, non-square cards, and shaped callouts.",
        "- The poster must not look like a simple wireframe; it should still be an artistic final poster design with placeholders ready for production replacement.",
        "- Use modern rounded cards, subtle shadows, scientific background art, glow/depth effects, and coherent accent colors.",
        "- Give wide scientific-plot placeholders enough absolute size for readability while preserving their declared source aspect ratio.",
        "- Never solve a wide plot by making it an ultra-thin strip; reduce nearby text or give the plot a taller module instead.",
        "- If there are many placeholders, visually group them into one hero result, two supporting analysis figures, and smaller diagnostics rather than equal tiles.",
        "- Preserve enough whitespace for later replacement; no decorative artwork may overlap a placeholder rectangle.",
    ]
    return "\n".join(lines).strip() + "\n"


def _aspect_ratio_hint(aspect: str) -> str:
    ratio = _parse_aspect_ratio_text(aspect)
    if ratio is None:
        return ""
    if 0.92 <= ratio <= 1.08:
        return "draw this placeholder near-square; do not stretch it into a landscape strip."
    if ratio > 1:
        return f"draw width:height approximately {ratio:.2f}:1; preserve the wide source ratio by enlarging the surrounding card if needed."
    return f"draw width:height approximately 1:{(1.0 / ratio):.2f}; preserve the tall source ratio by enlarging the surrounding card if needed."


def _placeholder_geometry_blueprint(fig: dict[str, Any]) -> str:
    aspect = str(fig.get("aspect") or "1:1 square")
    ratio = _parse_aspect_ratio_text(aspect)
    label = str(fig.get("label") or "figure")
    if ratio is None:
        return f"source aspect {aspect}; draw a clean rectangular placeholder for {label!r}."
    width, height = _suggested_placeholder_size(ratio, label=label)
    if 0.92 <= ratio <= 1.08:
        ratio_text = "near-square 1:1"
    elif ratio > 1:
        ratio_text = f"width:height ≈ {ratio:.2f}:1"
    else:
        ratio_text = f"width:height ≈ 1:{(1.0 / ratio):.2f}"
    return (
        f"{ratio_text}; target visible box about {width}×{height} px on a 1024×1536 canvas; "
        f"label '{label}'; do not substitute a generic landscape slot."
    )


def _suggested_placeholder_size(ratio: float, *, label: str) -> tuple[int, int]:
    low = label.lower()
    hero = any(key in low for key in ("limit", "result", "constraint", "cross section", "significance", "exclusion", "observed"))
    if 0.92 <= ratio <= 1.08:
        side = 330 if hero else 220
        return side, side
    if ratio > 1:
        width = 680 if ratio >= 2.2 else (300 if ratio <= 1.35 else 460)
        height = max(120, int(round(width / ratio)))
        return width, height
    height = 360 if hero else 260
    width = max(120, int(round(height * ratio)))
    return width, height


def _parse_aspect_ratio_text(aspect: str) -> float | None:
    text = str(aspect or "").strip().lower()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[:/]\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return 1.0 if "square" in text else None
    den = float(match.group(2))
    return float(match.group(1)) / den if den else None


def _style_rule_list(style: dict[str, Any], key: str, defaults: list[str]) -> list[str]:
    raw = style.get(key)
    if isinstance(raw, list):
        rules = [str(item).strip() for item in raw if str(item).strip()]
        return rules or defaults
    if isinstance(raw, str) and raw.strip():
        parts = [part.strip(" -;\n\t") for part in re.split(r"[\n;]+", raw) if part.strip(" -;\n\t")]
        return parts or [raw.strip()]
    return defaults


def _placeholder_layout_hint(fig: dict[str, Any]) -> str:
    label = str(fig.get("label") or "").lower()
    aspect = str(fig.get("aspect") or "").lower()
    if any(key in label for key in ("limit", "95%", "result", "constraint", "upper", "cross section", "significance", "exclusion", "measurement", "observation", "interpretation")):
        return "make this a dominant key-result figure, not a thumbnail; allocate the largest readable near-square slot in its section and reduce neighboring text if needed."
    if any(key in label for key in ("distribution", "region", "background", "fit", "control", "post-fit", "pre-fit", "multi-panel", "unfolded")) or "wide" in aspect:
        return "make this wide or multi-panel plot a large landscape panel with real vertical height; use full width only if the height scales with the declared ratio, never as a ribbon-thin strip."
    if any(key in label for key in ("detector", "event display", "topology", "selection", "strategy", "workflow")):
        return "use this as a clear supporting visual near the method text; keep it medium-sized and unobstructed."
    return ""


def _is_internal(text: str, spec: dict[str, Any]) -> bool:
    low = str(text).lower()
    forbidden = spec.get("forbidden_phrases", []) + INTERNAL_DEFAULTS
    return any(str(p).lower() in low for p in forbidden)
