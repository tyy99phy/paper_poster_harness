from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from .latex_utils import clean_latex_inline, extract_latex_braced
from .llm import ChatGPTAccountResponsesProvider
from .prompt import INTERNAL_DEFAULTS, sanitize_public_text
from .schemas import (
    DEFAULT_FORBIDDEN_PHRASES,
    copy_deck_schema,
    default_poster_spec,
    figure_selection_schema,
    normalize_assets_manifest,
    normalize_placeholder_id,
    placeholder_detection_schema,
    physics_quiz_schema,
    poster_qa_schema,
    poster_template_critic_schema,
    poster_spec_schema,
    storyboard_schema,
)


SYSTEM_PROMPT = (
    "You are building internal poster-harness stage outputs for a scientific poster pipeline. "
    "Return JSON only. Keep text public-facing. Never invent scientific results that are not grounded in the provided text or assets."
)


FLOWCHART_NODE_RULES: list[str] = [
    "When a section gets a 'flowchart' field, treat each node as a concrete information capsule, not a generic stage label.",
    "Each flowchart node should contain at least one paper-specific number, variable, threshold, region label, fitted observable, uncertainty label, or statistical-method detail.",
    "Bad node: 'preselection'. Good node: '>=2 same-sign muons: pT>30 GeV, |eta|<2.4; >=2 jets pT>30 GeV'.",
    "Bad node: 'signal region'. Good node: 'HMN SR binned in Delta phi(ll) | Weinberg SR binned in pTmiss, split at 50 GeV'.",
    "Bad node: 'fit'. Good node: 'Simultaneous SR/CR fit; WZ/top/fake norms constrained; profile-likelihood CLs'.",
    "Use a '|' character inside a node label only when the analysis genuinely branches into parallel SR/CR/category paths.",
    "Order nodes in true data-flow order: dataset -> object/event selection -> categorization -> SR/CR definition -> fit/model -> output result.",
    "Keep each node under 22 words; use the paper's own symbols and unit suffixes when explicitly present.",
    "Produce 4-5 nodes total; fewer is fine. Do not pad with generic stages.",
    "Skip the flowchart entirely or leave it empty if the source text does not provide concrete data-flow details.",
]


def draft_spec_from_text(
    text: str,
    assets_manifest: Any = None,
    *,
    provider: ChatGPTAccountResponsesProvider | None = None,
    project_overrides: Mapping[str, Any] | None = None,
    style_overrides: Mapping[str, Any] | None = None,
    extra_instructions: str | None = None,
) -> dict[str, Any]:
    provider = provider or ChatGPTAccountResponsesProvider()
    assets = normalize_assets_manifest(assets_manifest)
    title = str((project_overrides or {}).get("title") or _guess_title(text))
    base = default_poster_spec(title=title)
    abstract = _guess_abstract(text)
    if abstract:
        base["sections"][0]["text"][0]["body"] = [abstract]
    if project_overrides:
        base["project"].update(dict(project_overrides))
    if style_overrides:
        base["style"].update(dict(style_overrides))
    if assets:
        for placeholder, asset in zip(base["placeholders"], assets):
            placeholder["asset"] = asset["asset"]
            placeholder["label"] = asset.get("label") or placeholder["label"]

    prompt = _compose_prompt(
        header="Draft a poster_spec-compatible JSON object.",
        instructions=[
            "Use the supplied paper/talk text to create a public-facing poster specification.",
            "Preserve the existing harness structure: project, style, sections, placeholders, placements, conclusion, closing.",
            "Keep 4-6 sections unless the source clearly needs a different count.",
            "Follow HEP poster rhetoric: motivation/context, dataset/object selection, analysis/background strategy, key results, interpretation/summary.",
            "For a professional HEP audience, make dataset/selection and analysis-strategy sections analysis-specific, not generic. Extract concrete object thresholds, SR/CR definitions, binning variables, fit observable, likelihood/CLs method, normalization factors, and dominant uncertainties when present.",
            "Do not use a generic workflow like 'pp collisions → candidates → topology → SR/CR → fit' as the main analysis graphic. If a flowchart is useful, make it a compact HEP region/fit schematic with concrete SR bins, CR labels, fitted observables, and nuisance/systematic blocks.",
            *FLOWCHART_NODE_RULES,
            "Keep rendered text compact but information-rich: prefer short public bullets, data badges, and one-line claims over paragraphs.",
            "Do not make a sparse cover image. Aim for 14-24 public information units across the poster: section claims, badges, concise bullets, and conclusion takeaways.",
            "Each section should have at most one short body sentence plus 2-4 high-value bullets unless the source requires otherwise; use more sections or badges rather than long paragraphs.",
            "Each placeholder must have id, section, label, aspect, and asset filename.",
            "Plan one hero placeholder for the headline result/limit/cross-section plot and 2-3 supporting placeholders for strategy/background/context.",
            "Captions must be public-facing scientific descriptions, not design instructions; avoid wording like 'Use...', 'Draw...', 'Place...', or 'Bottom strip should...'.",
            "Do not leak workflow notes, TODOs, or internal replacement language into public text.",
            "When asset usage is uncertain, keep the placeholder label descriptive and use filenames from the provided asset manifest only when plausible.",
            extra_instructions or "",
        ],
        context={
            "project_overrides": dict(project_overrides or {}),
            "style_overrides": dict(style_overrides or {}),
            "assets_manifest": assets,
            "source_text_excerpt": _truncate(text, 30000),
            "starter_spec": base,
        },
    )
    envelope = provider.generate_json(
        stage_name="draft_spec_from_text",
        prompt=prompt,
        schema=poster_spec_schema(),
        system_prompt=SYSTEM_PROMPT,
    )
    merged = _deep_merge(base, envelope["result"])
    merged = _normalize_spec(merged, assets)
    envelope["result"] = merged
    return envelope


def storyboard_from_text(
    text: str,
    assets_manifest: Any = None,
    *,
    spec: Mapping[str, Any] | None = None,
    provider: ChatGPTAccountResponsesProvider | None = None,
    extra_instructions: str | None = None,
) -> dict[str, Any]:
    """Create a compact Paper2Poster-style storyboard before figure selection.

    This is an internal planning artifact, not public poster text.  It gives the
    downstream selector and image-generation prompt a stable semantic spine:
    section roles, visual priorities, reading order, and reader-understanding
    questions.
    """
    provider = provider or ChatGPTAccountResponsesProvider()
    assets = normalize_assets_manifest(assets_manifest)
    target_spec = _normalize_spec(copy.deepcopy(dict(spec or default_poster_spec(_guess_title(text)))), assets)
    prompt = _compose_prompt(
        header="Draft a storyboard JSON for a scientific poster generation pipeline.",
        instructions=[
            "Create an internal storyboard that compresses the paper into a poster narrative.",
            "Do not write design-process or workflow text that should appear on the poster; this is an internal planning artifact.",
            "Use the existing poster_spec sections as the section scaffold. Preserve section ids and titles where possible.",
            "For each section, assign a semantic role such as motivation, dataset, method, validation, result, interpretation, or outlook.",
            "Write concise public-facing synopses and key claims grounded only in the provided paper text.",
            "For HEP analyses, preserve specialist analysis details when present: object selections, SR binning variables, CR definitions, fitted discriminant, profile-likelihood/CLs strategy, floating background normalizations, nuisance parameters, and leading statistical/systematic uncertainties.",
            "Professional HEP posters should not spend scarce space on generic 'data → selection → fit' pipelines. Prefer region matrices, fit-model schematics, and short analysis-specific callouts.",
            *FLOWCHART_NODE_RULES,
            "Assign text budgets in practical poster terms, e.g. 'title + 2 bullets' or 'one sentence + 3 short bullets'.",
            "Describe the preferred visual role for each section and map useful assets to target sections when supported by captions/labels.",
            "Mark one hero section and one hero visual role for the headline result or central method.",
            "Add an information_plan object with: density_target, data_badges, must_answer_questions, display_facts, and visual_story_units.",
            "The information_plan should preserve Paper2Poster-style information richness: enough concise facts for a reader to understand motivation, method, dataset, headline result, and interpretation without reading the paper.",
            "Prefer public numeric badges only when the numbers are explicitly present in source text/assets; otherwise use qualitative fact badges.",
            "Write 4-8 reader-understanding questions that a good poster should enable a viewer to answer.",
            "Never invent numeric results, luminosities, limits, channels, or claims not present in the source text/assets.",
            extra_instructions or "",
        ],
        context={
            "poster_spec_sections": target_spec.get("sections") or [],
            "poster_project": target_spec.get("project") or {},
            "assets_manifest": assets,
            "source_text_excerpt": _truncate(text, 30000),
        },
    )
    envelope = provider.generate_json(
        stage_name="storyboard_from_text",
        prompt=prompt,
        schema=storyboard_schema(),
        system_prompt=SYSTEM_PROMPT,
    )
    envelope["result"] = _normalize_storyboard(envelope["result"], target_spec, assets)
    return envelope


def physics_quiz_from_text(
    text: str,
    assets_manifest: Any = None,
    *,
    spec: Mapping[str, Any] | None = None,
    storyboard: Mapping[str, Any] | None = None,
    provider: ChatGPTAccountResponsesProvider | None = None,
    max_questions: int | None = None,
    extra_instructions: str | None = None,
) -> dict[str, Any]:
    """Generate a PaperQuiz-lite HEP comprehension target.

    The quiz is an internal planning/evaluation artifact.  It should describe
    what a conference viewer ought to learn from the poster; the questions are
    not rendered verbatim on the poster.
    """

    provider = provider or ChatGPTAccountResponsesProvider()
    assets = normalize_assets_manifest(assets_manifest)
    target_spec = _normalize_spec(copy.deepcopy(dict(spec or default_poster_spec(_guess_title(text)))), assets)
    limit = max(8, min(24, int(max_questions or 16)))
    prompt = _compose_prompt(
        header="Draft a HEP PaperQuiz-lite JSON object for poster information planning.",
        instructions=[
            f"Create {limit} concise quiz items that a good scientific poster for this paper should enable a viewer to answer.",
            "This is internal planning/evaluation data, not public poster text. Do not ask the image model to render these questions.",
            "Use high-energy-physics poster aspects: physics target, dataset/channel, object/event selection, analysis strategy, background/control regions, statistical method, systematic uncertainty, headline result, interpretation, and figure evidence.",
            "Include specialist HEP questions about SR/CR definitions, discriminating variables, simultaneous-fit structure, nuisance/systematic treatment, floating normalization factors, and dominant uncertainty sources whenever the paper provides them.",
            "For result papers with multiple interpretations, include quiz items for each headline numerical result, including observed and expected limits for secondary interpretations when present.",
            "Do not settle for generic 'SR/CR fits' questions if the source gives named control regions or fit ingredients. Ask concrete questions about named CRs, binning variables, free normalization factors, profile likelihood/CLs, and dominant statistical/template uncertainties when present.",
            "Every quiz item must be answerable from explicit source text, figure captions, or assets. Never invent luminosities, energies, masses, channels, significances, or limits.",
            "For each item, provide a short answer, 0-4 multiple-choice options if useful, source_evidence, poster_priority, target_section, recommended_copy, and linked_assets when relevant.",
            "Mark only the most central 6-10 items as poster_priority='must'; secondary items should be 'should' or 'could'.",
            "Recommended copy should be a short public phrase suitable for a badge, callout, headline, or bullet; keep it under about 70 characters when possible.",
            extra_instructions or "",
        ],
        context={
            "max_questions": limit,
            "poster_spec": target_spec,
            "storyboard": dict(storyboard or {}),
            "assets_manifest": assets,
            "source_text_excerpt": _truncate(text, 30000),
        },
    )
    envelope = provider.generate_json(
        stage_name="physics_quiz_from_text",
        prompt=prompt,
        schema=physics_quiz_schema(),
        system_prompt=SYSTEM_PROMPT,
    )
    envelope["result"] = _normalize_physics_quiz(envelope["result"], target_spec, assets, limit=limit)
    return envelope


def copy_deck_from_text(
    text: str,
    assets_manifest: Any = None,
    *,
    spec: Mapping[str, Any] | None = None,
    storyboard: Mapping[str, Any] | None = None,
    physics_quiz: Mapping[str, Any] | None = None,
    figure_selection: Mapping[str, Any] | None = None,
    provider: ChatGPTAccountResponsesProvider | None = None,
    max_units: int | None = None,
    extra_instructions: str | None = None,
) -> dict[str, Any]:
    """Create a grounded public copy deck for the image-generation prompt."""

    provider = provider or ChatGPTAccountResponsesProvider()
    assets = normalize_assets_manifest(assets_manifest)
    target_spec = _normalize_spec(copy.deepcopy(dict(spec or default_poster_spec(_guess_title(text)))), assets)
    limit = max(10, min(36, int(max_units or 28)))
    prompt = _compose_prompt(
        header="Draft a public copy_deck JSON object for a placeholder-first scientific poster.",
        instructions=[
            "Create concise public text units that the image-generation model should render on the poster.",
            "Use the physics_quiz as the coverage target: must-priority quiz items should be answerable from rendered copy plus future real figure placeholders.",
            "Use the storyboard as the narrative spine and the poster_spec sections as the section scaffold. Preserve section ids.",
            "Do not create copy units for the exact main title, author line, or identity line; those are controlled by poster_spec.project and rendered separately.",
            "Write compact, camera-ready poster copy: short headlines, badges, figure-near headlines, bullets, callouts, and conclusion takeaways.",
            "For target_section roles like dataset, selection, method, strategy, fit, or background, prioritize analysis-specific copy over general HEP knowledge. Use types such as selection_cut, region_matrix, fit_strategy, and uncertainty when applicable.",
            "Section 2 should favor a concrete object/region-definition matrix: lepton/jet thresholds, VBF cuts, b/tau/extra-lepton vetoes, SR binning variables, and named CRs if present.",
            "Section 3 should favor a concrete fit-strategy summary: fitted discriminant, SR/CR simultaneous fit, background-only post-fit comparison, floating background normalizations, profile likelihood/CLs, nuisance parameters, and leading statistical/systematic uncertainties if present.",
            "Result/conclusion units must cover every headline interpretation, not only the primary limit plot. If the paper gives both heavy-neutrino and Weinberg-operator limits, include both; preserve observed and expected numerical limits such as |mμμ| observed/expected values when explicitly present.",
            "Never output a region_matrix unit that is merely a generic pipeline such as 'pp → candidates → SR/CR → fit'. A region_matrix must name concrete SR bins, CRs, or fitted regions; otherwise omit it.",
            "If the paper names WZ, b-tagged, WZb, top, fake/nonprompt, validation, or control regions, include those names in section 2/3 copy units rather than a generic flowchart.",
            "If the paper says statistical/template uncertainty dominates or mentions Barlow-Beeston-lite, CLs, profile likelihood, nuisance parameters, or floating normalizations, include at least one fit_strategy or uncertainty unit for it.",
            "Do not write paragraphs. Most units should be under 70 characters; badges should be under 36 characters; hero headlines can be under 90 characters.",
            "Ground every text unit in explicit source evidence, a quiz answer, a figure caption, or the supplied poster_spec. Never invent scientific numbers or claims.",
            "For each placeholder/selected figure, add a nearby figure_headline if the evidence supports it; do not describe fake drawn contents and do not ask the model to draw scientific data.",
            "Use priority='must' for the minimum public copy that must be rendered; use 'should' and 'could' for optional density.",
            "Treat the copy deck as an exhaustive public body-text plan: do not add generic future-prospect, impact, or methodology slogans unless explicitly grounded.",
            "Do not include internal workflow language, prompt instructions, placeholder explanations, TODOs, or replacement-process text in public copy.",
            "Return at most max_units copy_units. If space is tight, omit could-priority units before shortening figures/placeholders.",
            extra_instructions or "",
        ],
        context={
            "max_units": limit,
            "poster_spec": target_spec,
            "storyboard": dict(storyboard or {}),
            "physics_quiz": dict(physics_quiz or {}),
            "figure_selection": dict(figure_selection or {}),
            "assets_manifest": assets,
            "source_text_excerpt": _truncate(text, 30000),
        },
    )
    envelope = provider.generate_json(
        stage_name="copy_deck_from_text",
        prompt=prompt,
        schema=copy_deck_schema(),
        system_prompt=SYSTEM_PROMPT,
    )
    envelope["result"] = _normalize_copy_deck(envelope["result"], target_spec, physics_quiz or {}, limit=limit)
    return envelope


def select_figures(
    text: str,
    assets_manifest: Any,
    *,
    spec: Mapping[str, Any] | None = None,
    storyboard: Mapping[str, Any] | None = None,
    provider: ChatGPTAccountResponsesProvider | None = None,
    max_figures: int | None = None,
    extra_instructions: str | None = None,
) -> dict[str, Any]:
    provider = provider or ChatGPTAccountResponsesProvider()
    assets = normalize_assets_manifest(assets_manifest)
    target_spec = _normalize_spec(copy.deepcopy(dict(spec or default_poster_spec(_guess_title(text)))), assets)
    limit = max_figures or len(target_spec.get("placeholders", [])) or 8
    prompt = _compose_prompt(
        header="Select the most valuable real figures/assets for the poster.",
        instructions=[
            "Pick at most the available placeholder count or max_figures, whichever is smaller.",
            "Prefer figures that communicate the main physics motivation, method, key backgrounds, and headline results.",
            "Assign priority so the headline result/limit/cross-section/significance plot becomes the hero placeholder.",
            "Deprioritize dense diagnostic variants unless they are essential to the scientific claim.",
            "Map each selected asset onto a sequential placeholder_id exactly like FIG 01, FIG 02, ...; do not invent semantic IDs.",
            "Use storyboard target sections/roles when supplied, but keep the final section id compatible with the poster_spec.",
            "For every selected figure, include role, target_section, and reason when possible. Use role values like hero_result, supporting_validation, method_flow, context, diagnostic, or table.",
            "The placeholder aspect must match the selected source asset aspect ratio; do not invent a more convenient poster ratio.",
            "If a source plot is very wide, allocate a wider card or full-width band rather than squeezing or changing its aspect.",
            "Give plots enough absolute size for axes, legends, and labels while preserving the source aspect ratio.",
            "Defer logos, contact sheets, decorative files, or clearly redundant variants unless they are uniquely necessary.",
            extra_instructions or "",
        ],
        context={
            "max_figures": limit,
            "poster_spec": target_spec,
            "storyboard": dict(storyboard or {}),
            "assets_manifest": assets,
            "source_text_excerpt": _truncate(text, 5000),
        },
    )
    envelope = provider.generate_json(
        stage_name="select_figures",
        prompt=prompt,
        schema=figure_selection_schema(),
        system_prompt=SYSTEM_PROMPT,
    )
    envelope["result"] = _normalize_figure_selection(envelope["result"], target_spec, assets, limit=limit)
    return envelope


def detect_placeholders_from_image(
    image_path: str | Path,
    expected_placeholders: Sequence[Any] | None = None,
    *,
    provider: ChatGPTAccountResponsesProvider | None = None,
    image_detail: str = "high",
    extra_instructions: str | None = None,
) -> dict[str, Any]:
    provider = provider or ChatGPTAccountResponsesProvider()
    path = Path(image_path)
    width, height = Image.open(path).size
    expected = _normalize_expected_placeholders(expected_placeholders)
    prompt = _compose_prompt(
        header="Locate poster placeholder boxes from an image.",
        instructions=[
            "Return bounding boxes in image pixel coordinates [x0, y0, x1, y1].",
            "Prefer the visible clean placeholder panel edges, not the outer card boundary.",
            "If a placeholder id is missing or ambiguous, still report the best-matching visible placeholder and note the ambiguity.",
            extra_instructions or "",
        ],
        context={
            "image_path": str(path),
            "image_size": {"width": width, "height": height},
            "expected_placeholders": expected,
        },
    )
    envelope = provider.generate_json(
        stage_name="detect_placeholders_from_image",
        prompt=prompt,
        schema=placeholder_detection_schema(),
        system_prompt=SYSTEM_PROMPT,
        image_paths=[path],
        image_detail=image_detail,
    )
    result = _normalize_placeholder_detection(envelope["result"], width=width, height=height, expected=expected)
    _require_complete_placeholder_detection(result, expected)
    envelope["result"] = result
    return envelope


def qa_poster(
    spec: Mapping[str, Any],
    *,
    prompt: str | None = None,
    image_path: str | Path | None = None,
    detected_placeholders: Mapping[str, Any] | None = None,
    provider: ChatGPTAccountResponsesProvider | None = None,
    qa_mode: str = "placeholder",
    extra_instructions: str | None = None,
) -> dict[str, Any]:
    provider = provider or ChatGPTAccountResponsesProvider()
    if qa_mode not in {"placeholder", "final"}:
        raise ValueError("qa_mode must be 'placeholder' or 'final'")
    normalized_spec = _normalize_spec(copy.deepcopy(dict(spec)), normalize_assets_manifest(None))
    prechecks = _deterministic_qa_checks(
        normalized_spec,
        prompt=prompt,
        detected_placeholders=detected_placeholders,
        image_path=image_path,
        qa_mode=qa_mode,
    )
    image_paths = [image_path] if image_path else None
    mode_instructions = _qa_mode_instructions(qa_mode)
    detected_context = _qa_detected_placeholders_context(detected_placeholders or {}, qa_mode)
    prompt_text = _compose_prompt(
        header=f"Quality-assure the poster in {qa_mode!r} mode.",
        instructions=[
            *mode_instructions,
            extra_instructions or "",
        ],
        context={
            "qa_mode": qa_mode,
            "poster_spec": normalized_spec,
            "render_prompt": prompt or "",
            "detected_placeholders": detected_context,
            "deterministic_prechecks": prechecks,
        },
    )
    envelope = provider.generate_json(
        stage_name=f"qa_poster_{qa_mode}",
        prompt=prompt_text,
        schema=poster_qa_schema(),
        system_prompt=SYSTEM_PROMPT,
        image_paths=image_paths,
    )
    envelope["result"] = _merge_qa_results(prechecks, envelope["result"])
    return envelope


def _qa_detected_placeholders_context(detected_placeholders: Mapping[str, Any], qa_mode: str) -> dict[str, Any]:
    detected = dict(detected_placeholders or {})
    if qa_mode != "final":
        return detected
    # In final mode, labels like "[FIG 02]" are expected in the pre-replacement
    # detection JSON but should not appear in the final rendered poster.  Passing
    # those labels/notes to the VLM makes it prone to claiming that placeholder
    # text is still visible.  Keep only geometry.
    return {
        "image_size": detected.get("image_size") or {},
        "placements": detected.get("placements") or {},
    }


def critique_poster_template(
    spec: Mapping[str, Any],
    *,
    prompt: str,
    image_path: str | Path,
    provider: ChatGPTAccountResponsesProvider | None = None,
    extra_instructions: str | None = None,
) -> dict[str, Any]:
    """Critique a generated placeholder poster before real-figure replacement.

    This stage is deliberately a *regeneration* critic, not a repair/overlay
    planner.  If information density, artistry, text quality, or placeholder
    contract quality is weak, it returns concise prompt repairs that should be
    fed into another full image-generation attempt.
    """
    provider = provider or ChatGPTAccountResponsesProvider()
    path = Path(image_path)
    normalized_spec = _normalize_spec(copy.deepcopy(dict(spec)), normalize_assets_manifest(None))
    prompt_text = _compose_prompt(
        header="Critique this generated scientific poster template before figure replacement.",
        instructions=[
            "This is the pre-replacement template. Scientific figures must still be blank [FIG NN] placeholders; do not penalize placeholders merely for being blank.",
            "Judge whether the complete generated poster is already strong enough to proceed to deterministic real-figure replacement.",
            "Evaluate five dimensions: artistic/editorial quality, information density, placeholder contract cleanliness, text legibility/typos, and figure-card integration.",
            "Use storyboard.information_plan and storyboard.qa_questions as the information target: the poster should visibly contain enough compact public facts/badges/bullets to answer the central reader questions.",
            "Do not require deterministic text overlays or manual post-editing. If text/information is weak, propose prompt repairs for regenerating the whole poster with image_generation.",
            "Fail if the poster is merely decorative, too sparse, PPT-like, has serious text corruption, leaks internal workflow text, has dark figure blocks, or contains fake scientific plots/diagrams outside placeholders.",
            "Fail if any placeholder appears to contain real/fake scientific content, is missing/duplicated, unreadable, or clearly violates its declared aspect ratio.",
            "Do not penalize placeholder labels or aspect-ratio text when they are inside the dashed placeholder; the contract requires each placeholder to contain exactly the ID, intended label, and aspect ratio.",
            "Prompt repairs must never contradict the placeholder contract: do not ask for only the [FIG NN] token, do not remove aspect-ratio text, and do not move labels outside the placeholder.",
            "Keep prompt_repairs concrete, short, and directly usable as additional image-generation instructions.",
            extra_instructions or "",
        ],
        context={
            "poster_spec": normalized_spec,
            "render_prompt_excerpt": _truncate(prompt, 9000),
            "acceptance_rule": "Pass only if this template should proceed to placeholder detection and replacement without manual intervention.",
        },
    )
    envelope = provider.generate_json(
        stage_name="critique_poster_template",
        prompt=prompt_text,
        schema=poster_template_critic_schema(),
        system_prompt=SYSTEM_PROMPT,
        image_paths=[path],
        image_detail="high",
    )
    envelope["result"] = _normalize_template_critique(envelope["result"])
    return envelope


def _qa_mode_instructions(qa_mode: str) -> list[str]:
    if qa_mode == "placeholder":
        return [
            "This is the pre-replacement placeholder poster. Enforce the placeholder contract strictly.",
            "Every scientific figure/table/diagram area must be a blank neutral placeholder box with a visible exact label [FIG NN].",
            "Inside each placeholder, only the ID, intended content label, and aspect-ratio text are allowed.",
            "For deterministic replacement, the [FIG NN] ID, clean rectangular boundary, blank scientific content, and geometry are critical. Minor label paraphrasing, line breaks, or typography differences in the intended label/aspect text are warnings, not critical failures.",
            "Simple public text-only analysis flowcharts outside placeholders are allowed when they are explicitly specified by the poster spec; do not confuse them with source scientific figures.",
            "Fail with a critical issue if any placeholder contains real or fake scientific content: axes, bins, curves, legends, Feynman lines, process diagrams, heatmaps, tables, or thumbnails.",
            "Fail with a critical issue if any expected [FIG NN] label is missing, unreadable, or not associated with a clean rectangular placeholder.",
            "Also check public text cleanliness, duplicate placeholder ids, and detected placement plausibility.",
        ]
    return [
        "This is the post-replacement final poster. Real scientific figures are now expected inside the previously detected boxes.",
        "Do not require blank placeholders or visible [FIG NN] labels in final mode.",
        "Check that public text is clean, final figures appear plausible and non-fabricated, and no internal workflow text leaked into the poster.",
        "Flag missing, badly cropped, stretched, or unreadable replaced figures.",
        "The poster_spec may include placements and _replacement_clear_boxes. Treat _replacement_clear_boxes as the approved final placeholder/cleanup boundary; the old dashed border is normally removed in final mode.",
        "A white publication-style frame around a real figure is allowed if it stays inside the approved cleanup boundary. Do not fail only because the white frame covers the old placeholder label or dashed border.",
        "Use deterministic_prechecks as authoritative for coordinate containment/overlap unless the image shows an obvious contradiction. If deterministic_prechecks contain no figure_containment or figure_overlap issue, do not invent speculative critical containment failures from visual uncertainty alone.",
        "CRITICAL: Fail only when a real scientific figure or its white replacement frame visibly extends outside its approved placeholder/cleanup boundary, is badly cropped, or clearly overlaps another real figure.",
        "CRITICAL: Check that figures do not overlap with each other. Each figure should have a clear visual gutter separating it from neighbors.",
        "Check that the poster structure remains plausible for a public scientific conference poster.",
        "If the poster_spec includes storyboard.qa_questions or storyboard.information_plan.must_answer_questions, verify that the visible poster appears information-rich enough to answer the central public questions; warn if it is merely decorative or too sparse.",
    ]


def _compose_prompt(*, header: str, instructions: Sequence[str], context: Mapping[str, Any]) -> str:
    lines = [header, ""]
    lines.append("Instructions:")
    for item in instructions:
        item = str(item).strip()
        if item:
            lines.append(f"- {item}")
    lines += ["", "Context JSON:", json.dumps(context, ensure_ascii=False, indent=2)]
    return "\n".join(lines)


def _guess_title(text: str) -> str:
    latex_title = extract_latex_braced(text, "title")
    if latex_title:
        cleaned = clean_latex_inline(latex_title)
        if 12 <= len(cleaned) <= 240:
            return cleaned
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("---") or line.startswith("\\"):
            continue
        if line.lower() in {"cms paper", "abstract"}:
            continue
        if line.startswith("#"):
            line = line.lstrip("#").strip()
        line = clean_latex_inline(line)
        if 12 <= len(line) <= 180:
            return line
    return "Untitled Scientific Poster"


def _guess_abstract(text: str) -> str:
    latex_abstract = extract_latex_braced(text, "abstract")
    if latex_abstract:
        return clean_latex_inline(latex_abstract)[:700]
    lowered = text.lower()
    idx = lowered.find("abstract")
    snippet = text[idx : idx + 1800] if idx >= 0 else text[:1800]
    snippet = re.sub(r"\s+", " ", snippet).strip()
    sentences = re.split(r"(?<=[.!?])\s+", snippet)
    return clean_latex_inline(" ".join(sentences[:2]))[:700]


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, Mapping):
        merged = copy.deepcopy(base)
        for key, value in override.items():
            merged[key] = _deep_merge(base.get(key), value)
        return merged
    if isinstance(base, list) and isinstance(override, list):
        if not base:
            return copy.deepcopy(override)
        if override and all(isinstance(item, Mapping) for item in override):
            return [copy.deepcopy(item) for item in override]
        return copy.deepcopy(override)
    return copy.deepcopy(override)


def _normalize_spec(spec: dict[str, Any], assets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base = default_poster_spec(str(spec.get("project", {}).get("title") or "Untitled Scientific Poster"))
    merged = _deep_merge(base, spec)
    merged.setdefault("placements", {})
    merged.setdefault("closing", base["closing"])
    merged["forbidden_phrases"] = _dedupe_strings(list(DEFAULT_FORBIDDEN_PHRASES) + list(merged.get("forbidden_phrases") or []))

    sections: list[dict[str, Any]] = []
    for idx, section in enumerate(merged.get("sections") or [], start=1):
        row = dict(section)
        row["id"] = int(row.get("id") or idx)
        row["title"] = str(row.get("title") or f"Section {row['id']}")
        row["layout"] = str(row.get("layout") or "card")
        row["text"] = [_normalize_text_block(item) for item in row.get("text") or []] or [{"title": row["title"], "body": [], "bullets": []}]
        if row.get("flowchart"):
            row["flowchart"] = [_clean_flowchart_item(str(item)) for item in row["flowchart"] if str(item).strip()]
            row["flowchart"] = [item for item in row["flowchart"] if item]
        if row.get("caption"):
            row["caption"] = _clean_public_caption(str(row.get("caption") or ""))
        sections.append(row)
    merged["sections"] = sections or base["sections"]

    asset_names = [asset["asset"] for asset in assets]
    placeholders: list[dict[str, Any]] = []
    source_placeholders = merged.get("placeholders") or []
    if not source_placeholders:
        source_placeholders = copy.deepcopy(base["placeholders"])
    for idx, placeholder in enumerate(source_placeholders, start=1):
        row = dict(placeholder)
        row["id"] = str(row.get("id") or normalize_placeholder_id(idx))
        row["section"] = _coerce_section(row.get("section"), sections)
        row["label"] = str(row.get("label") or f"Figure {idx}")
        row["aspect"] = str(row.get("aspect") or "1:1 square")
        if not row.get("asset"):
            row["asset"] = asset_names[idx - 1] if idx - 1 < len(asset_names) else f"fig{idx:02d}.png"
        placeholders.append(row)
    if assets:
        for placeholder, asset in zip(placeholders, assets):
            placeholder.setdefault("asset", asset["asset"])
    merged["placeholders"] = placeholders
    merged["conclusion"] = [sanitize_public_text(str(item), merged["forbidden_phrases"]).strip() for item in merged.get("conclusion") or []]
    merged["conclusion"] = [item for item in merged["conclusion"] if item] or base["conclusion"]
    merged["project"]["title"] = sanitize_public_text(str(merged["project"].get("title") or base["project"]["title"]), merged["forbidden_phrases"]).strip() or base["project"]["title"]
    merged["project"]["topic"] = str(merged["project"].get("topic") or merged["project"]["title"])
    return merged


def _clean_public_caption(text: str) -> str:
    text = sanitize_public_text(str(text)).strip()
    lowered = text.lower()
    instruction_starts = (
        "use ",
        "draw ",
        "place ",
        "put ",
        "make ",
        "bottom strip should",
        "caption should",
    )
    instruction_phrases = (
        " should read",
        " should be",
        "use cms",
        "use the ",
        "as the executive summary",
    )
    if lowered.startswith(instruction_starts) or any(phrase in lowered for phrase in instruction_phrases):
        return ""
    return text


def _clean_flowchart_item(text: str) -> str:
    text = sanitize_public_text(str(text)).strip()
    lowered = text.lower()
    if lowered.startswith(("use ", "draw ", "place ", "make ")):
        return ""
    return text


def _normalize_storyboard(
    result: Mapping[str, Any],
    spec: Mapping[str, Any],
    assets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sections_by_id = {int(section.get("id") or idx): dict(section) for idx, section in enumerate(spec.get("sections") or [], start=1)}
    valid_section_ids = sorted(sections_by_id) or [1]
    asset_index = {str(asset.get("asset")): dict(asset) for asset in assets}

    meta = dict(result.get("meta") or {})
    project = dict(spec.get("project") or {})
    normalized_sections: list[dict[str, Any]] = []
    raw_sections = list(result.get("sections") or [])
    if not raw_sections:
        raw_sections = [
            {
                "id": section_id,
                "title": row.get("title") or f"Section {section_id}",
                "role": "poster_section",
                "synopsis": "",
                "key_claims": [],
                "text_budget": "one sentence + 2 short bullets",
                "preferred_visual": "",
            }
            for section_id, row in sections_by_id.items()
        ]
    for idx, item in enumerate(raw_sections, start=1):
        row = dict(item)
        section_id = _coerce_section(row.get("id"), spec.get("sections") or [])
        seed = sections_by_id.get(section_id, {})
        normalized_sections.append(
            {
                "id": section_id,
                "title": sanitize_public_text(str(row.get("title") or seed.get("title") or f"Section {section_id}")).strip() or f"Section {section_id}",
                "role": _compact_token(str(row.get("role") or "poster_section"), default="poster_section"),
                "synopsis": sanitize_public_text(str(row.get("synopsis") or "")).strip(),
                "key_claims": _dedupe_strings([sanitize_public_text(str(claim)).strip() for claim in row.get("key_claims") or [] if str(claim).strip()])[:5],
                "text_budget": str(row.get("text_budget") or "one sentence + 2 short bullets"),
                "preferred_visual": str(row.get("preferred_visual") or ""),
                "visual_keywords": _dedupe_strings([str(item).strip() for item in row.get("visual_keywords") or [] if str(item).strip()])[:8],
            }
        )
    # Keep the order stable and avoid duplicate section entries.
    seen_sections: set[int] = set()
    deduped_sections: list[dict[str, Any]] = []
    for row in normalized_sections:
        if int(row["id"]) in seen_sections:
            continue
        seen_sections.add(int(row["id"]))
        deduped_sections.append(row)

    visual_assets: list[dict[str, Any]] = []
    for item in result.get("visual_assets") or []:
        row = dict(item)
        asset_name = str(row.get("asset") or "")
        if asset_name and asset_name not in asset_index:
            continue
        asset = asset_index.get(asset_name, {})
        visual_assets.append(
            {
                "asset": asset_name,
                "caption": sanitize_public_text(str(row.get("caption") or asset.get("caption") or asset.get("label") or asset_name)).strip(),
                "figure_type": _compact_token(str(row.get("figure_type") or asset.get("kind") or "figure"), default="figure"),
                "relevance": sanitize_public_text(str(row.get("relevance") or "")).strip(),
                "target_section": _coerce_section(row.get("target_section"), spec.get("sections") or []),
                "role": _compact_token(str(row.get("role") or "supporting_visual"), default="supporting_visual"),
            }
        )

    information_plan = _normalize_information_plan(result.get("information_plan") or {}, result, deduped_sections)

    layout_tree = dict(result.get("layout_tree") or {})
    reading_order = []
    for value in layout_tree.get("reading_order") or valid_section_ids:
        section_id = _coerce_section(value, spec.get("sections") or [])
        if section_id not in reading_order:
            reading_order.append(section_id)
    hero_section = _coerce_section(layout_tree.get("hero_section") or (reading_order[-1] if reading_order else valid_section_ids[-1]), spec.get("sections") or [])

    qa_questions = [
        sanitize_public_text(str(item)).strip()
        for item in result.get("qa_questions") or []
        if str(item).strip()
    ][:8]
    if not qa_questions:
        qa_questions = ["What is the central scientific question?", "What is the headline result?", "Which figure supports the main conclusion?"]

    return {
        "meta": {
            "title": str(meta.get("title") or project.get("title") or ""),
            "authors": str(meta.get("authors") or project.get("authors") or ""),
            "audience": str(meta.get("audience") or project.get("audience") or "academic conference audience"),
            "one_sentence_takeaway": sanitize_public_text(str(meta.get("one_sentence_takeaway") or "")).strip(),
        },
        "core_message": sanitize_public_text(str(result.get("core_message") or meta.get("one_sentence_takeaway") or project.get("topic") or "")).strip(),
        "sections": deduped_sections,
        "visual_assets": visual_assets,
        "layout_tree": {
            "reading_order": reading_order,
            "hero_section": hero_section,
            "hero_visual_role": str(layout_tree.get("hero_visual_role") or "headline result figure"),
            "layout_intent": sanitize_public_text(str(layout_tree.get("layout_intent") or "clear top-to-bottom scientific reading path")).strip(),
            "section_grouping": [str(item).strip() for item in layout_tree.get("section_grouping") or [] if str(item).strip()][:6],
        },
        "information_plan": information_plan,
        "qa_questions": qa_questions,
    }


def _normalize_information_plan(raw: Any, result: Mapping[str, Any], sections: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    plan = dict(raw or {}) if isinstance(raw, Mapping) else {}

    def clean_items(values: Any, *, limit: int) -> list[str]:
        cleaned: list[str] = []
        for item in values or []:
            if isinstance(item, Mapping):
                label = sanitize_public_text(str(item.get("label") or item.get("name") or "")).strip()
                value = sanitize_public_text(str(item.get("value") or item.get("text") or item.get("fact") or "")).strip()
                text = f"{label}: {value}" if label and value else (label or value)
            else:
                text = sanitize_public_text(str(item)).strip()
            if text and text not in cleaned:
                cleaned.append(text)
            if len(cleaned) >= limit:
                break
        return cleaned

    display_facts = clean_items(plan.get("display_facts") or plan.get("key_facts") or [], limit=18)
    if not display_facts:
        for section in sections:
            for claim in section.get("key_claims") or []:
                clean = sanitize_public_text(str(claim)).strip()
                if clean and clean not in display_facts:
                    display_facts.append(clean)
                if len(display_facts) >= 18:
                    break
            if len(display_facts) >= 18:
                break

    must_answer = clean_items(plan.get("must_answer_questions") or result.get("qa_questions") or [], limit=8)
    return {
        "density_target": sanitize_public_text(str(plan.get("density_target") or "Paper2Poster-rich but readable: 14-24 concise public information units, no paragraphs")).strip(),
        "data_badges": clean_items(plan.get("data_badges") or plan.get("badges") or [], limit=8),
        "display_facts": display_facts,
        "must_answer_questions": must_answer,
        "visual_story_units": clean_items(plan.get("visual_story_units") or [], limit=8),
    }


def _normalize_physics_quiz(
    result: Mapping[str, Any],
    spec: Mapping[str, Any],
    assets: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> dict[str, Any]:
    asset_names = {str(asset.get("asset")) for asset in assets}
    items: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for idx, item in enumerate(result.get("quiz_items") or [], start=1):
        if not isinstance(item, Mapping):
            continue
        row = dict(item)
        question = sanitize_public_text(str(row.get("question") or "")).strip()
        answer = sanitize_public_text(str(row.get("answer") or "")).strip()
        if not question or not answer:
            continue
        qkey = re.sub(r"\s+", " ", question).strip().lower()
        if qkey in seen_questions:
            continue
        seen_questions.add(qkey)
        priority = str(row.get("poster_priority") or row.get("priority") or "should").strip().lower()
        if priority not in {"must", "should", "could"}:
            priority = "should"
        linked_assets = [
            str(asset).strip()
            for asset in row.get("linked_assets") or []
            if str(asset).strip() and (not asset_names or str(asset).strip() in asset_names)
        ][:4]
        options = [
            sanitize_public_text(str(option)).strip()
            for option in row.get("options") or []
            if str(option).strip()
        ][:4]
        items.append(
            {
                "id": str(row.get("id") or f"Q{len(items) + 1:02d}"),
                "aspect": _quiz_aspect(str(row.get("aspect") or "")),
                "question": question,
                "options": options,
                "answer": answer,
                "poster_priority": priority,
                "source_evidence": _truncate(sanitize_public_text(str(row.get("source_evidence") or row.get("evidence") or "")).strip(), 360),
                "target_section": _coerce_section(row.get("target_section"), spec.get("sections") or []),
                "recommended_copy": _truncate(sanitize_public_text(str(row.get("recommended_copy") or answer)).strip(), 120),
                "linked_assets": _dedupe_strings(linked_assets),
            }
        )
        if len(items) >= limit:
            break
    if not items:
        raise RuntimeError("physics_quiz_from_text: LLM returned no usable quiz_items in strict mode")
    for idx, row in enumerate(items, start=1):
        row["id"] = f"Q{idx:02d}"
    coverage_notes = [
        sanitize_public_text(str(note)).strip()
        for note in result.get("coverage_notes") or []
        if str(note).strip()
    ][:8]
    return {"quiz_items": items, "coverage_notes": coverage_notes}


def _normalize_copy_deck(
    result: Mapping[str, Any],
    spec: Mapping[str, Any],
    physics_quiz: Mapping[str, Any],
    *,
    limit: int,
) -> dict[str, Any]:
    valid_quiz_ids = {str(item.get("id")) for item in physics_quiz.get("quiz_items") or [] if isinstance(item, Mapping)}
    valid_placeholder_ids = {str(item.get("id")) for item in spec.get("placeholders") or [] if isinstance(item, Mapping)}
    units: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for item in result.get("copy_units") or []:
        if not isinstance(item, Mapping):
            continue
        row = dict(item)
        text = sanitize_public_text(str(row.get("text") or ""), spec.get("forbidden_phrases")).strip()
        if not text:
            continue
        text = re.sub(r"\s+", " ", text)
        key = text.lower()
        if key in seen_text:
            continue
        if _looks_like_project_title_copy(row, text, spec):
            continue
        ctype = _copy_unit_type(str(row.get("type") or "bullet"))
        if ctype == "region_matrix" and _is_generic_region_matrix_text(text):
            continue
        seen_text.add(key)
        priority = str(row.get("priority") or "should").strip().lower()
        if priority not in {"must", "should", "could"}:
            priority = "should"
        try:
            max_chars = int(row.get("max_chars") or len(text) + 8)
        except Exception:
            max_chars = len(text) + 8
        max_chars = max(16, min(120, max_chars))
        placeholder_id = str(row.get("placeholder_id") or "").strip()
        if placeholder_id and valid_placeholder_ids and placeholder_id not in valid_placeholder_ids:
            placeholder_id = ""
        quiz_ids = [
            str(qid).strip()
            for qid in row.get("quiz_ids") or []
            if str(qid).strip() and (not valid_quiz_ids or str(qid).strip() in valid_quiz_ids)
        ][:6]
        units.append(
            {
                "id": str(row.get("id") or f"C{len(units) + 1:02d}"),
                "target_section": _coerce_section(row.get("target_section"), spec.get("sections") or []),
                "type": ctype,
                "text": _truncate(text, max_chars + 24),
                "max_chars": max_chars,
                "priority": priority,
                "evidence": _truncate(sanitize_public_text(str(row.get("evidence") or row.get("source_evidence") or "")).strip(), 360),
                "quiz_ids": _dedupe_strings(quiz_ids),
                "placeholder_id": placeholder_id,
                "placement_hint": sanitize_public_text(str(row.get("placement_hint") or "")).strip(),
                "render_style": sanitize_public_text(str(row.get("render_style") or "")).strip(),
            }
        )
        if len(units) >= limit:
            break
    if not units:
        raise RuntimeError("copy_deck_from_text: LLM returned no usable copy_units in strict mode")
    for idx, row in enumerate(units, start=1):
        row["id"] = f"C{idx:02d}"
    section_copy = []
    for item in result.get("section_copy") or []:
        if not isinstance(item, Mapping):
            continue
        row = dict(item)
        section_copy.append(
            {
                "section": _coerce_section(row.get("section"), spec.get("sections") or []),
                "role": _compact_token(str(row.get("role") or "section"), default="section"),
                "must_units": [
                    str(unit).strip()
                    for unit in row.get("must_units") or []
                    if str(unit).strip()
                ][:8],
                "target_density": sanitize_public_text(str(row.get("target_density") or "")).strip(),
            }
        )
    coverage_notes = [
        sanitize_public_text(str(note)).strip()
        for note in result.get("coverage_notes") or []
        if str(note).strip()
    ][:8]
    return {"copy_units": units, "section_copy": section_copy, "coverage_notes": coverage_notes}


def _quiz_aspect(value: str) -> str:
    text = _compact_token(value, default="paper_understanding")
    aliases = {
        "physics": "physics_target",
        "target": "physics_target",
        "data": "dataset_channel",
        "dataset": "dataset_channel",
        "channel": "dataset_channel",
        "selection": "event_selection",
        "method": "analysis_strategy",
        "strategy": "analysis_strategy",
        "background": "background_control",
        "fit": "statistical_method",
        "statistics": "statistical_method",
        "uncertainty": "systematics",
        "result": "headline_result",
        "interpretation": "interpretation",
        "figure": "figure_evidence",
    }
    return aliases.get(text, text)


def _copy_unit_type(value: str) -> str:
    text = _compact_token(value, default="bullet")
    allowed = {
        "hero_headline",
        "section_title",
        "subhead",
        "bullet",
        "badge",
        "figure_headline",
        "selection_cut",
        "region_matrix",
        "fit_strategy",
        "uncertainty",
        "conclusion",
        "callout",
    }
    aliases = {
        "headline": "hero_headline",
        "hero": "hero_headline",
        "title": "section_title",
        "chip": "badge",
        "fact_badge": "badge",
        "figure_caption": "figure_headline",
        "figure_label": "figure_headline",
        "cut": "selection_cut",
        "cutflow": "selection_cut",
        "selection": "selection_cut",
        "sr": "region_matrix",
        "cr": "region_matrix",
        "sr_cr": "region_matrix",
        "region": "region_matrix",
        "regions": "region_matrix",
        "fit": "fit_strategy",
        "likelihood": "fit_strategy",
        "systematic": "uncertainty",
        "systematics": "uncertainty",
        "nuisance": "uncertainty",
        "summary": "conclusion",
        "takeaway": "conclusion",
    }
    text = aliases.get(text, text)
    return text if text in allowed else "bullet"


def _is_generic_region_matrix_text(text: str) -> bool:
    low = str(text or "").lower()
    if "sr/cr" not in low and "signal/control" not in low and "control region" not in low:
        return False
    concrete_markers = [
        "wz",
        "b-tag",
        "b tagged",
        "wzb",
        "nonprompt",
        "fake",
        "Δ",
        "dphi",
        "pTmiss".lower(),
        "mjj",
        "750",
        "2.5",
        "0.75",
        "30 gev",
        "profile",
        "cls",
        "nuisance",
        "barlow",
        "normalization",
    ]
    if any(marker.lower() in low for marker in concrete_markers):
        return False
    generic_markers = ["pp", "candidate", "topology", "→", "->", "fit"]
    return sum(1 for marker in generic_markers if marker in low) >= 3


def _looks_like_project_title_copy(row: Mapping[str, Any], text: str, spec: Mapping[str, Any]) -> bool:
    project = spec.get("project") if isinstance(spec.get("project"), Mapping) else {}
    title = str(project.get("title") or "")
    if not title:
        return False
    placement = str(row.get("placement_hint") or "").lower()
    evidence = str(row.get("evidence") or row.get("source_evidence") or "").lower()
    ctype = _copy_unit_type(str(row.get("type") or ""))
    if "title" in placement or "title band" in placement or "project title" in evidence:
        return True
    text_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    title_tokens = set(re.findall(r"[a-z0-9]+", title.lower()))
    if ctype in {"hero_headline", "section_title"} and title_tokens:
        overlap = len(text_tokens & title_tokens) / max(1, len(text_tokens))
        return overlap >= 0.72 and len(text_tokens) >= 5
    return False


def _normalize_template_critique(result: Mapping[str, Any]) -> dict[str, Any]:
    raw_scores = dict(result.get("scores") or {})
    score_keys = [
        "overall",
        "artistry",
        "information_density",
        "placeholder_contract",
        "text_quality",
        "figure_integration",
    ]
    scores: dict[str, float] = {}
    for key in score_keys:
        scores[key] = _clamp_score(raw_scores.get(key), default=0.0)
    if not scores["overall"]:
        nonzero = [scores[key] for key in score_keys[1:] if scores[key] > 0]
        scores["overall"] = sum(nonzero) / len(nonzero) if nonzero else 0.0

    issues: list[dict[str, str]] = []
    for item in result.get("issues") or []:
        row = dict(item or {})
        severity = str(row.get("severity") or "warning").lower()
        if severity not in {"critical", "warning", "info"}:
            severity = "warning"
        message = sanitize_public_text(str(row.get("message") or "")).strip()
        if not message:
            continue
        issues.append(
            {
                "severity": severity,
                "category": str(row.get("category") or "template_quality"),
                "message": message,
                "location": str(row.get("location") or ""),
                "suggested_prompt_repair": sanitize_public_text(str(row.get("suggested_prompt_repair") or "")).strip(),
            }
        )
    prompt_repairs = [
        sanitize_public_text(str(item)).strip()
        for item in result.get("prompt_repairs") or []
        if str(item).strip()
    ][:8]
    for issue in issues:
        repair = issue.get("suggested_prompt_repair", "")
        if repair and repair not in prompt_repairs:
            prompt_repairs.append(repair)
        if len(prompt_repairs) >= 8:
            break
    checks = dict(result.get("checks") or {})
    passes = bool(result.get("passes")) and not any(issue["severity"] == "critical" for issue in issues)
    return {
        "passes": passes,
        "summary": sanitize_public_text(str(result.get("summary") or "")).strip(),
        "scores": scores,
        "issues": issues,
        "checks": {str(key): bool(value) for key, value in checks.items()},
        "prompt_repairs": prompt_repairs,
    }


def _clamp_score(value: Any, *, default: float) -> float:
    try:
        score = float(value)
    except Exception:
        return default
    if score > 1.0 and score <= 10.0:
        score = score / 10.0
    if score > 10.0 and score <= 100.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def _compact_token(value: str, *, default: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_ -]+", "", value).strip().lower().replace(" ", "_").replace("-", "_")
    text = re.sub(r"_+", "_", text).strip("_")
    return text or default


def _normalize_text_block(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        clean = sanitize_public_text(value).strip()
        return {"title": "", "body": [clean] if clean else [], "bullets": []}
    row = dict(value or {})
    row["title"] = str(row.get("title") or "")
    row["body"] = [sanitize_public_text(str(item)).strip() for item in row.get("body") or []]
    row["body"] = [item for item in row["body"] if item]
    row["bullets"] = [sanitize_public_text(str(item)).strip() for item in row.get("bullets") or []]
    row["bullets"] = [item for item in row["bullets"] if item]
    return row


def _coerce_section(value: Any, sections: Sequence[Mapping[str, Any]]) -> int:
    valid = {int(section["id"]) for section in sections if section.get("id") is not None}
    try:
        section_id = int(value)
    except Exception:
        section_id = min(valid) if valid else 1
    return section_id if section_id in valid else (min(valid) if valid else 1)


def _guess_aspect(asset: Mapping[str, Any]) -> str:
    text = " ".join(str(asset.get(key) or "") for key in ("label", "caption", "name")).lower()
    if "table" in text:
        return "2:1 wide"
    if "wide" in text or "roc" in text:
        return "2.4:1 wide"
    return "1:1 square"


def _normalize_figure_selection(
    result: Mapping[str, Any],
    spec: Mapping[str, Any],
    assets: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> dict[str, Any]:
    asset_index = {str(asset["asset"]): asset for asset in assets}
    placeholders = list(spec.get("placeholders") or [])
    normalized: list[dict[str, Any]] = []
    raw_selected = list(result.get("selected_figures") or [])
    if not raw_selected:
        raise RuntimeError("select_figures: LLM returned no selected_figures in strict mode")
    selected_for_ids = sorted(raw_selected[:limit], key=_figure_selection_priority)
    for idx, item in enumerate(selected_for_ids, start=1):
        row = dict(item)
        placeholder = placeholders[idx - 1] if idx - 1 < len(placeholders) else {}
        asset_name = str(row.get("asset") or placeholder.get("asset") or "")
        if asset_name not in asset_index:
            raise RuntimeError(f"select_figures: LLM selected unknown asset {asset_name!r}")
        asset = asset_index.get(asset_name, {})
        label = str(row.get("label") or asset.get("label") or placeholder.get("label") or f"Figure {idx}")
        raw_aspect = str(row.get("aspect") or placeholder.get("aspect") or _guess_aspect(asset))
        aspect = _normalize_display_aspect(raw_aspect, asset=asset, label=label)
        raw_section = row.get("section")
        if raw_section is None:
            raw_section = row.get("target_section")
        if raw_section is None:
            raw_section = placeholder.get("section")
        section = _coerce_section(raw_section, spec.get("sections") or [])
        role = _compact_token(str(row.get("role") or placeholder.get("role") or ("hero_result" if idx == 1 else "supporting_visual")), default=("hero_result" if idx == 1 else "supporting_visual"))
        reason = str(row.get("reason") or row.get("rationale") or "LLM-selected figure for poster coverage.")
        normalized.append(
            {
                "placeholder_id": normalize_placeholder_id(idx),
                "asset": asset_name or str(asset.get("asset") or f"fig{idx:02d}.png"),
                "section": section,
                "target_section": section,
                "role": role,
                "label": label,
                "aspect": aspect,
                "priority": int(row.get("priority") or idx),
                "rationale": reason,
                "reason": reason,
                "source_path": str(row.get("source_path") or asset.get("path") or asset_name),
                "notes": [str(note) for note in row.get("notes") or asset.get("notes") or []],
            }
        )
    selected_assets = {item["asset"] for item in normalized}
    deferred = [dict(item) for item in result.get("deferred_assets") or []]
    seen_deferred = {str(item.get("asset")) for item in deferred}
    for asset in assets:
        if asset["asset"] not in selected_assets and asset["asset"] not in seen_deferred:
            deferred.append({"asset": asset["asset"], "reason": "Not selected for the current poster pass."})
    return {
        "selected_figures": normalized,
        "deferred_assets": deferred,
        "selection_notes": [str(note) for note in result.get("selection_notes") or []],
    }


def _figure_selection_priority(item: Any) -> int:
    try:
        return int(dict(item).get("priority") or 999)
    except Exception:
        return 999


def _normalize_display_aspect(aspect: str, *, asset: Mapping[str, Any], label: str) -> str:
    """Return the source-figure aspect ratio for placeholder planning.

    The image-generation step creates blank boxes that will be deterministically
    replaced by real paper figures. To avoid post-replacement letterboxing, every
    placeholder should match the selected source asset's native aspect ratio.
    Readability is handled by allocating a larger slot, not by warping or
    moderating the ratio.
    """
    source_aspect = _source_asset_aspect(asset)
    if source_aspect:
        return source_aspect
    return str(aspect or "1:1 square")


def _source_asset_aspect(asset: Mapping[str, Any]) -> str:
    raw = str(asset.get("aspect") or "").strip()
    ratio = _parse_aspect_ratio(raw)
    if ratio is None:
        try:
            width = float(asset.get("width") or 0)
            height = float(asset.get("height") or 0)
        except Exception:
            width = height = 0.0
        if width > 0 and height > 0:
            ratio = width / height
    if ratio is None or ratio <= 0:
        return ""
    if abs(ratio - 1.0) < 0.08:
        return "1:1 square"
    if ratio > 1:
        num = _format_ratio_number(ratio)
        text = f"{num}:1"
        return f"{text} wide" if ratio >= 1.35 else text
    inv = 1.0 / ratio
    num = _format_ratio_number(inv)
    text = f"1:{num}"
    return f"{text} tall" if inv >= 1.35 else text


def _format_ratio_number(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _parse_aspect_ratio(aspect: str) -> float | None:
    text = str(aspect or "").strip().lower()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[:/]\s*([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        den = float(match.group(2))
        return float(match.group(1)) / den if den else None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*x\s*([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        den = float(match.group(2))
        return float(match.group(1)) / den if den else None
    return None


def _normalize_expected_placeholders(expected_placeholders: Sequence[Any] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(expected_placeholders or [], start=1):
        if isinstance(item, Mapping):
            row = dict(item)
            normalized.append({"id": str(row.get("id") or normalize_placeholder_id(idx)), "label": str(row.get("label") or row.get("asset") or row.get("id") or f"Figure {idx}")})
        else:
            normalized.append({"id": str(item), "label": str(item)})
    return normalized


def _normalize_placeholder_detection(
    result: Mapping[str, Any],
    *,
    width: int,
    height: int,
    expected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_index = {str(item["id"]): dict(item) for item in expected}
    normalized_placeholders: list[dict[str, Any]] = []
    raw = list(result.get("placeholders") or [])
    if not raw and expected:
        raw = [{"id": item["id"], "label": item.get("label", item["id"]), "bbox": [0, 0, 0, 0], "confidence": 0.0, "notes": ["not detected"]} for item in expected]
    for idx, item in enumerate(raw, start=1):
        row = dict(item)
        placeholder_id = str(row.get("id") or normalize_placeholder_id(idx))
        bbox = _clamp_bbox(row.get("bbox"), width=width, height=height)
        seed = expected_index.get(placeholder_id, {})
        normalized_placeholders.append(
            {
                "id": placeholder_id,
                "label": str(row.get("label") or seed.get("label") or placeholder_id),
                "bbox": bbox,
                "confidence": float(row.get("confidence") or 0.0),
                "notes": [str(note) for note in row.get("notes") or []],
            }
        )
    placements = {row["id"]: row["bbox"] for row in normalized_placeholders if any(row["bbox"])}
    return {
        "image_size": {"width": width, "height": height},
        "placeholders": normalized_placeholders,
        "placements": placements,
    }


def _require_complete_placeholder_detection(result: Mapping[str, Any], expected: Sequence[Mapping[str, Any]]) -> None:
    placements = result.get("placements") if isinstance(result, Mapping) else {}
    if not isinstance(placements, Mapping) or not placements:
        raise RuntimeError("detect_placeholders_from_image: LLM returned no usable placeholder placements")
    if expected:
        missing = [str(item["id"]) for item in expected if str(item["id"]) not in placements]
        if missing:
            raise RuntimeError(
                "detect_placeholders_from_image: LLM detection incomplete for expected placeholders: "
                + ", ".join(missing)
            )


def _clamp_bbox(value: Any, *, width: int, height: int) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) < 4:
        return [0, 0, 0, 0]
    x0, y0, x1, y1 = [int(round(float(item))) for item in value[:4]]
    x0 = max(0, min(width, x0))
    y0 = max(0, min(height, y0))
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1]


def _deterministic_qa_checks(
    spec: Mapping[str, Any],
    *,
    prompt: str | None,
    detected_placeholders: Mapping[str, Any] | None,
    image_path: str | Path | None = None,
    qa_mode: str | None = None,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    forbidden = _dedupe_strings(list(DEFAULT_FORBIDDEN_PHRASES) + list(spec.get("forbidden_phrases") or []) + list(INTERNAL_DEFAULTS))
    offending_lines = _find_forbidden_lines(spec, forbidden)
    for line in offending_lines:
        issues.append(
            {
                "severity": "warning",
                "category": "public_text",
                "message": f"Contains forbidden/internal phrase: {line['phrase']}",
                "location": line["location"],
                "suggested_fix": "Rewrite or remove workflow/internal wording before poster generation.",
            }
        )

    placeholder_ids = [str(item.get("id")) for item in spec.get("placeholders") or []]
    duplicates = sorted({item for item in placeholder_ids if placeholder_ids.count(item) > 1 and item})
    for dup in duplicates:
        issues.append(
            {
                "severity": "critical",
                "category": "placeholders",
                "message": f"Duplicate placeholder id: {dup}",
                "location": "spec.placeholders",
                "suggested_fix": "Ensure every placeholder id is unique (FIG 01, FIG 02, ...).",
            }
        )

    missing_assets = [item for item in spec.get("placeholders") or [] if not str(item.get("asset") or "").strip()]
    for item in missing_assets:
        issues.append(
            {
                "severity": "warning",
                "category": "placeholders",
                "message": f"Placeholder {item.get('id')} is missing an asset filename.",
                "location": "spec.placeholders",
                "suggested_fix": "Assign an asset filename or intentionally defer the placeholder.",
            }
        )

    detected = dict(detected_placeholders or {})
    placements = detected.get("placements") if isinstance(detected, Mapping) else {}
    if placements:
        missing_detected = [pid for pid in placeholder_ids if pid not in placements]
        for pid in missing_detected:
            issues.append(
                {
                    "severity": "warning",
                    "category": "detection",
                    "message": f"No detected placement for {pid}.",
                    "location": "detected_placeholders.placements",
                    "suggested_fix": "Manually review the rendered poster or rerun visual detection.",
                }
            )

    # Check figure containment: placements must be inside clear boxes
    placements_map = spec.get("placements") or {}
    clear_map = spec.get("_replacement_clear_boxes") or {}
    for fig_id, raw_box in placements_map.items():
        raw_clear = clear_map.get(fig_id)
        if not raw_clear:
            continue
        try:
            bx0, by0, bx1, by1 = [int(round(float(v))) for v in raw_box]
            cx0, cy0, cx1, cy1 = [int(round(float(v))) for v in raw_clear]
        except Exception:
            continue
        margin = 2
        if bx0 < cx0 - margin or by0 < cy0 - margin or bx1 > cx1 + margin or by1 > cy1 + margin:
            issues.append(
                {
                    "severity": "warning",
                    "category": "figure_containment",
                    "message": f"{fig_id} placement {list(raw_box)} may extend beyond placeholder boundary {list(raw_clear)}",
                    "location": f"spec.placements[{fig_id}]",
                    "suggested_fix": "Ensure figure placement is strictly inside the detected placeholder boundary.",
                }
            )

    # Check for figure-figure overlap
    fig_ids = list(placements_map.keys())
    for i in range(len(fig_ids)):
        for j in range(i + 1, len(fig_ids)):
            id_i, id_j = fig_ids[i], fig_ids[j]
            try:
                ax0, ay0, ax1, ay1 = [int(round(float(v))) for v in placements_map[id_i]]
                bxx0, byy0, bxx1, byy1 = [int(round(float(v))) for v in placements_map[id_j]]
            except Exception:
                continue
            ox0 = max(ax0, bxx0)
            oy0 = max(ay0, byy0)
            ox1 = min(ax1, bxx1)
            oy1 = min(ay1, byy1)
            if ox1 > ox0 and oy1 > oy0:
                overlap = (ox1 - ox0) * (oy1 - oy0)
                area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
                area_b = max(1, (bxx1 - bxx0) * (byy1 - byy0))
                ratio = overlap / min(area_a, area_b)
                if ratio > 0.05:
                    issues.append(
                        {
                            "severity": "warning",
                            "category": "figure_overlap",
                            "message": f"{id_i} and {id_j} overlap by {overlap} pixels ({ratio:.1%})",
                            "location": "spec.placements",
                            "suggested_fix": "Add a clear gutter between figure placeholders or reduce figure sizes.",
                        }
                    )

    panel_checks = _panel_level_qa_checks(
        spec,
        detected_placeholders=detected_placeholders,
        image_path=image_path,
        qa_mode=qa_mode or "",
    )
    issues.extend(panel_checks)

    passes = not any(issue["severity"] == "critical" for issue in issues)
    checks = {
        "public_text_clean": not offending_lines,
        "placeholders_accounted_for": not bool(placeholder_ids) or (not placements or all(pid in placements for pid in placeholder_ids)),
        "section_count": len(spec.get("sections") or []),
        "placeholder_count": len(spec.get("placeholders") or []),
        "panel_level_check_count": len(panel_checks),
    }
    summary = "No critical QA issues found." if passes else "Critical QA issues found; review placeholder metadata."
    score = max(0.0, 1.0 - 0.15 * len([issue for issue in issues if issue["severity"] == "critical"]) - 0.05 * len([issue for issue in issues if issue["severity"] == "warning"]))
    repairs = [issue["suggested_fix"] for issue in issues if issue.get("suggested_fix")]
    return {
        "passes": passes,
        "summary": summary,
        "score": round(score, 3),
        "issues": issues,
        "checks": checks,
        "recommended_repairs": _dedupe_strings(repairs),
    }


def _panel_level_qa_checks(
    spec: Mapping[str, Any],
    *,
    detected_placeholders: Mapping[str, Any] | None,
    image_path: str | Path | None,
    qa_mode: str,
) -> list[dict[str, Any]]:
    """Lightweight local checks around each placement/cleanup panel.

    This is inspired by Paper2Poster's panel-level visual loop, but deliberately
    deterministic and low-intrusion: it checks local geometry, image bounds, and
    detector/spec alignment before the LLM QA judges the full poster.
    """
    placements = spec.get("placements") or {}
    if not isinstance(placements, Mapping) or not placements:
        return []
    clear_map = spec.get("_replacement_clear_boxes") or {}
    placeholders = {str(item.get("id")): dict(item) for item in spec.get("placeholders") or []}
    detected_map = {}
    detected = dict(detected_placeholders or {})
    if isinstance(detected.get("placements"), Mapping):
        detected_map = dict(detected.get("placements") or {})
    canvas_size: tuple[int, int] | None = None
    if image_path:
        try:
            with Image.open(image_path) as im:
                canvas_size = im.size
        except Exception:
            canvas_size = None

    issues: list[dict[str, Any]] = []
    for fig_id, raw_box in placements.items():
        fig_id = str(fig_id)
        box = _read_numeric_box(raw_box)
        if not box:
            continue
        scope_box = _read_numeric_box(clear_map.get(fig_id)) or box
        if canvas_size:
            if not _box_inside_canvas(scope_box, canvas_size) or not _box_inside_canvas(box, canvas_size):
                issues.append(
                    {
                        "severity": "warning",
                        "category": "panel_geometry",
                        "message": f"{fig_id} local panel box is outside the rendered poster bounds.",
                        "location": f"spec.placements[{fig_id}]",
                        "suggested_fix": "Rerun placeholder detection/normalization so every local figure panel stays inside the image canvas.",
                    }
                )
            min_side = min(box[2] - box[0], box[3] - box[1])
            if min_side < max(24, min(canvas_size) * 0.025):
                issues.append(
                    {
                        "severity": "warning",
                        "category": "panel_geometry",
                        "message": f"{fig_id} local figure crop is very small relative to the poster.",
                        "location": f"spec.placements[{fig_id}]",
                        "suggested_fix": "Allocate a larger local panel or choose a less dense figure for this placeholder.",
                    }
                )

        ph = placeholders.get(fig_id, {})
        expected_ratio = _parse_aspect_ratio(str(ph.get("aspect") or ""))
        if expected_ratio:
            actual_ratio = (box[2] - box[0]) / max(1, box[3] - box[1])
            rel_error = abs(actual_ratio / max(expected_ratio, 0.01) - 1.0)
            if rel_error > 0.18:
                issues.append(
                    {
                        "severity": "warning",
                        "category": "panel_geometry",
                        "message": f"{fig_id} placement ratio {actual_ratio:.2f}:1 differs from declared aspect {expected_ratio:.2f}:1.",
                        "location": f"spec.placements[{fig_id}]",
                        "suggested_fix": "Use the source asset aspect ratio when planning the local replacement crop.",
                    }
                )

        if fig_id in detected_map:
            detected_box = _read_numeric_box(detected_map.get(fig_id))
            if detected_box:
                iou = _box_iou(scope_box, detected_box)
                # In final mode normalize_placeholder_geometry can intentionally
                # shrink/shift the replacement target inside the original
                # placeholder.  Low IoU is therefore a warning, not a failure.
                threshold = 0.35 if qa_mode == "final" else 0.45
                if iou < threshold:
                    issues.append(
                        {
                            "severity": "warning",
                            "category": "panel_detection_alignment",
                            "message": f"{fig_id} local replacement boundary and detected placeholder differ substantially (IoU={iou:.2f}).",
                            "location": f"spec._replacement_clear_boxes[{fig_id}]",
                            "suggested_fix": "Review the placeholder crop; rerun detection or choose another generated variant if the local panel is misidentified.",
                        }
                    )
    return issues


def _read_numeric_box(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) < 4:
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(item))) for item in value[:4]]
    except Exception:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _box_inside_canvas(box: tuple[int, int, int, int], canvas_size: tuple[int, int]) -> bool:
    return box[0] >= 0 and box[1] >= 0 and box[2] <= canvas_size[0] and box[3] <= canvas_size[1]


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ox0 = max(a[0], b[0])
    oy0 = max(a[1], b[1])
    ox1 = min(a[2], b[2])
    oy1 = min(a[3], b[3])
    if ox1 <= ox0 or oy1 <= oy0:
        return 0.0
    overlap = (ox1 - ox0) * (oy1 - oy0)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return overlap / max(1, area_a + area_b - overlap)


def _find_forbidden_lines(payload: Mapping[str, Any], forbidden: Sequence[str]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for location, text in _iter_strings(payload):
        lowered = text.lower()
        for phrase in forbidden:
            if phrase and phrase.lower() in lowered:
                hits.append({"location": location, "phrase": phrase})
    return hits


def _iter_strings(value: Any, path: str = "root"):
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "forbidden_phrases":
                continue
            yield from _iter_strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            yield from _iter_strings(item, f"{path}[{idx}]")
    elif isinstance(value, str):
        yield path, value


def _merge_qa_results(prechecks: Mapping[str, Any], llm_result: Mapping[str, Any]) -> dict[str, Any]:
    issues = [dict(item) for item in prechecks.get("issues") or []]
    deterministic_categories = {str(item.get("category") or "").lower() for item in issues}
    seen = {(item.get("severity"), item.get("category"), item.get("message"), item.get("location")) for item in issues}
    for item in llm_result.get("issues") or []:
        row = _normalize_llm_qa_issue(dict(item))
        _downgrade_speculative_visual_geometry_issue(row, deterministic_categories)
        key = (row.get("severity"), row.get("category"), row.get("message"), row.get("location"))
        if key not in seen:
            issues.append(row)
            seen.add(key)
    repairs = _dedupe_strings(list(prechecks.get("recommended_repairs") or []) + [str(item) for item in llm_result.get("recommended_repairs") or []])
    passes = bool(prechecks.get("passes", True)) and not any(str(issue.get("severity") or "").lower() == "critical" for issue in issues)
    checks = dict(prechecks.get("checks") or {})
    checks.update(dict(llm_result.get("checks") or {}))
    score = llm_result.get("score")
    if score is None:
        score = prechecks.get("score")
    summary = str(llm_result.get("summary") or prechecks.get("summary") or "QA completed.")
    if passes and _summary_claims_failure(summary):
        summary = "QA passed with nonblocking warnings after deterministic precheck reconciliation."
    return {
        "passes": passes,
        "summary": summary,
        "score": float(score if score is not None else 0.0),
        "issues": issues,
        "checks": checks,
        "recommended_repairs": repairs,
    }


def _summary_claims_failure(summary: str) -> bool:
    low = str(summary or "").lower()
    return any(marker in low for marker in ("qa fails", "qa failed", "fails because", "should not proceed", "does not pass"))


def _normalize_llm_qa_issue(row: dict[str, Any]) -> dict[str, Any]:
    category = str(row.get("category") or "").lower()
    message = str(row.get("message") or "").lower()
    suggested = str(row.get("suggested_fix") or "").lower()
    text = " ".join([category, message, suggested])
    nonblocking_label_markers = [
        "label is not exact",
        "not reliably exact",
        "line-broken",
        "hyphenated",
        "paraphras",
        "exact intended content label",
        "aspect-ratio text",
        "aspect ratio text",
        "math typography",
        "symbol substitutions",
        "all caps",
    ]
    blocking_markers = [
        "fake scientific",
        "fake plot",
        "axes",
        "curves",
        "heatmap",
        "thumbnail",
        "missing",
        "duplicated",
        "duplicate",
        "unreadable",
        "not square",
        "not a 1:1",
        "not 1:1",
        "not a clear 2.5",
        "overlap",
        "extends outside",
    ]
    if str(row.get("severity") or "").lower() == "critical":
        if any(marker in text for marker in nonblocking_label_markers) and not any(marker in text for marker in blocking_markers):
            row["severity"] = "warning"
    return row


def _downgrade_speculative_visual_geometry_issue(row: dict[str, Any], deterministic_categories: set[str]) -> None:
    """Trust deterministic replacement geometry over uncertain full-poster VLM QA.

    Final-mode visual QA is useful for spotting obvious mistakes, but it can
    misread a light figure mat or old dashed border as a containment violation.
    The prompt already tells the model not to invent containment failures when
    deterministic_prechecks are clean; enforce that policy here so a speculative
    VLM-only geometry concern becomes a warning rather than deleting a valid
    deterministic export.
    """

    if str(row.get("severity") or "").lower() != "critical":
        return
    category = str(row.get("category") or "").lower()
    text = " ".join(
        [
            category,
            str(row.get("message") or "").lower(),
            str(row.get("suggested_fix") or "").lower(),
        ]
    )
    placeholder_remnant_categories = {
        "public_text_cleanliness",
        "figure_replacement",
        "incomplete_final_figure_replacement",
        "final_cleanup",
    }
    placeholder_remnant_markers = ("[fig", "aspect-ratio", "aspect ratio", "placeholder label", "placeholder text")
    if category in placeholder_remnant_categories and "placeholder" in text and any(marker in text for marker in placeholder_remnant_markers):
        row["severity"] = "warning"
        note = " Deterministic replacement erased the approved placeholder region, so this visual-only remnant concern is nonblocking."
        row["message"] = str(row.get("message") or "").rstrip() + note
        return
    if category not in {"figure_containment", "figure_overlap", "panel_geometry"}:
        return
    blocking_categories = {
        "figure_containment",
        "figure_overlap",
        "panel_geometry",
        "panel_detection_alignment",
        "figure_visual_envelope",
        "figure_frame_oversized",
        "figure_inner_margin",
    }
    if deterministic_categories & blocking_categories:
        return
    row["severity"] = "warning"
    note = " Deterministic replacement-geometry prechecks were clean, so this visual-only concern is nonblocking."
    row["message"] = str(row.get("message") or "").rstrip() + note


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
