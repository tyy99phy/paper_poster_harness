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
    default_poster_spec,
    figure_selection_schema,
    normalize_assets_manifest,
    normalize_placeholder_id,
    placeholder_detection_schema,
    poster_qa_schema,
    poster_spec_schema,
)


SYSTEM_PROMPT = (
    "You are building internal poster-harness stage outputs for a scientific poster pipeline. "
    "Return JSON only. Keep text public-facing. Never invent scientific results that are not grounded in the provided text or assets."
)


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
            "Keep rendered text sparse: prefer short public bullets over paragraphs; avoid references, dense equations, and implementation prose.",
            "Each section should have at most one short body sentence plus 2-4 high-value bullets unless the source requires otherwise.",
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
            "source_text_excerpt": _truncate(text, 8000),
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


def select_figures(
    text: str,
    assets_manifest: Any,
    *,
    spec: Mapping[str, Any] | None = None,
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
            "The placeholder aspect must match the selected source asset aspect ratio; do not invent a more convenient poster ratio.",
            "If a source plot is very wide, allocate a wider card or full-width band rather than squeezing or changing its aspect.",
            "Give plots enough absolute size for axes, legends, and labels while preserving the source aspect ratio.",
            "Defer logos, contact sheets, decorative files, or clearly redundant variants unless they are uniquely necessary.",
            extra_instructions or "",
        ],
        context={
            "max_figures": limit,
            "poster_spec": target_spec,
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
    prechecks = _deterministic_qa_checks(normalized_spec, prompt=prompt, detected_placeholders=detected_placeholders)
    image_paths = [image_path] if image_path else None
    mode_instructions = _qa_mode_instructions(qa_mode)
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
            "detected_placeholders": detected_placeholders or {},
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


def _qa_mode_instructions(qa_mode: str) -> list[str]:
    if qa_mode == "placeholder":
        return [
            "This is the pre-replacement placeholder poster. Enforce the placeholder contract strictly.",
            "Every scientific figure/table/diagram area must be a blank neutral placeholder box with a visible exact label [FIG NN].",
            "Inside each placeholder, only the ID, intended content label, and aspect-ratio text are allowed.",
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
        normalized.append(
            {
                "placeholder_id": normalize_placeholder_id(idx),
                "asset": asset_name or str(asset.get("asset") or f"fig{idx:02d}.png"),
                "section": _coerce_section(row.get("section") or placeholder.get("section"), spec.get("sections") or []),
                "label": label,
                "aspect": aspect,
                "priority": int(row.get("priority") or idx),
                "rationale": str(row.get("rationale") or "LLM-selected figure for poster coverage."),
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
                            "location": f"spec.placements",
                            "suggested_fix": "Add a clear gutter between figure placeholders or reduce figure sizes.",
                        }
                    )

    passes = not any(issue["severity"] == "critical" for issue in issues)
    checks = {
        "public_text_clean": not offending_lines,
        "placeholders_accounted_for": not bool(placeholder_ids) or (not placements or all(pid in placements for pid in placeholder_ids)),
        "section_count": len(spec.get("sections") or []),
        "placeholder_count": len(spec.get("placeholders") or []),
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
    seen = {(item.get("severity"), item.get("category"), item.get("message"), item.get("location")) for item in issues}
    for item in llm_result.get("issues") or []:
        row = dict(item)
        key = (row.get("severity"), row.get("category"), row.get("message"), row.get("location"))
        if key not in seen:
            issues.append(row)
            seen.add(key)
    repairs = _dedupe_strings(list(prechecks.get("recommended_repairs") or []) + [str(item) for item in llm_result.get("recommended_repairs") or []])
    passes = bool(prechecks.get("passes", True)) and bool(llm_result.get("passes", True))
    checks = dict(prechecks.get("checks") or {})
    checks.update(dict(llm_result.get("checks") or {}))
    score = llm_result.get("score")
    if score is None:
        score = prechecks.get("score")
    return {
        "passes": passes,
        "summary": str(llm_result.get("summary") or prechecks.get("summary") or "QA completed."),
        "score": float(score if score is not None else 0.0),
        "issues": issues,
        "checks": checks,
        "recommended_repairs": repairs,
    }


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
