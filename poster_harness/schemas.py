from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

JSON_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_FORBIDDEN_PHRASES = [
    "replace placeholder",
    "validated CMS/PPTX",
    "validated CMS/PPTX plots",
    "generated design language",
    "preserving the generated design language",
    "next step:",
    "internal",
    "TODO",
    "draft",
    "to be replaced",
    "placeholder figures",
    "use a clean",
    "use CMS blue",
    "bottom strip should",
    "should read as",
    "caption should",
    "source text",
    "source document",
]


def default_poster_spec(title: str = "Untitled Scientific Poster") -> dict[str, Any]:
    return {
        "project": {
            "title": title,
            "topic": title,
            "subtitle": "",
            "authors": "",
            "identity": "",
            "audience": "academic conference audience",
        },
        "style": {
            "summary": "premium CERN/LHCC-inspired scientific poster, modern editorial HEP design, artistic but readable, not a collage",
            "aspect": "A0 vertical / 2:3 ratio",
            "top_band": "strong dark title banner with concise identity text and abstract detector/beam artwork",
            "body_layout": "4-6 large numbered modules with one dominant result region, generous gutters, varied card shapes, and light paper-like figure cards",
            "color_grammar": "primary result = blue; secondary result = warm red; use consistently; all figure surfaces remain warm-white or very pale neutral",
            "typography": "modern editorial sans-serif with bold title, crisp section headers, compact readable bullets, and disciplined type scale",
            "color_palette": "deep indigo atmosphere, blue/violet/gold accents, and warm-white content/figure cards",
            "figure_surface": "every scientific figure placeholder sits on a warm-white, pearl, or very pale neutral card/mat, never on a dark content block",
        },
        "forbidden_phrases": list(DEFAULT_FORBIDDEN_PHRASES),
        "sections": [
            {
                "id": 1,
                "title": "Motivation and physics target",
                "layout": "upper full-width, 3 columns",
                "text": [
                    {
                        "title": "Motivation",
                        "body": ["Summarize the central physics motivation in 1–2 public sentences."],
                        "bullets": [],
                    }
                ],
            },
            {
                "id": 2,
                "title": "Analysis strategy",
                "layout": "large middle card",
                "text": [
                    {
                        "title": "Method",
                        "body": ["Summarize the method, object selection, reconstruction, or experimental strategy."],
                        "bullets": [],
                    }
                ],
                "flowchart": ["Use a clean vector flowchart for the main analysis logic if applicable."],
            },
            {
                "id": 3,
                "title": "Samples and background estimation",
                "layout": "lower-left card",
                "text": [
                    {
                        "title": "Inputs",
                        "body": ["Describe datasets, simulation, controls, or background estimates."],
                        "bullets": [],
                    }
                ],
            },
            {
                "id": 4,
                "title": "Key results",
                "layout": "lower-right card",
                "text": [
                    {
                        "title": "Results",
                        "body": ["Describe one or two key result messages."],
                        "bullets": [],
                    }
                ],
            },
            {
                "id": 5,
                "title": "Summary and prospects",
                "layout": "bottom full-width card",
                "text": [
                    {
                        "title": "Summary",
                        "body": ["Summarize the takeaway and public prospects."],
                        "bullets": [],
                    }
                ],
            },
        ],
        "placeholders": [
            {"id": "FIG 01", "section": 1, "label": "Motivation/context figure", "aspect": "2:1 wide", "asset": "fig01.png"},
            {"id": "FIG 02", "section": 1, "label": "Signal or process diagram", "aspect": "4:3", "asset": "fig02.png"},
            {"id": "FIG 03", "section": 3, "label": "Samples/table or method sketch", "aspect": "4:3", "asset": "fig03.png"},
            {"id": "FIG 04", "section": 3, "label": "Background/method figure", "aspect": "2.4:1 wide", "asset": "fig04.png"},
            {"id": "FIG 05", "section": 4, "label": "Key result A", "aspect": "1:1 square", "asset": "fig05.png"},
            {"id": "FIG 06", "section": 4, "label": "Key result B", "aspect": "1:1 square", "asset": "fig06.png"},
            {"id": "FIG 07", "section": 5, "label": "Diagnostic/result C", "aspect": "1:1 square", "asset": "fig07.png"},
            {"id": "FIG 08", "section": 5, "label": "Diagnostic/result D", "aspect": "1:1 square", "asset": "fig08.png"},
        ],
        "placements": {},
        "conclusion": [
            "State the main public result in one sentence.",
            "State the robustness or validation message in one sentence.",
            "State future physics prospects in one sentence.",
        ],
        "closing": "Thank you for your attention!",
    }


def normalize_placeholder_id(index: int) -> str:
    return f"FIG {index:02d}"


def normalize_assets_manifest(assets_manifest: Any) -> list[dict[str, Any]]:
    if assets_manifest is None:
        return []
    items: list[Any]
    if isinstance(assets_manifest, Mapping):
        if isinstance(assets_manifest.get("assets"), Sequence) and not isinstance(assets_manifest.get("assets"), (str, bytes, bytearray)):
            items = list(assets_manifest.get("assets") or [])
        else:
            items = []
            for name, meta in assets_manifest.items():
                if name == "assets":
                    continue
                if isinstance(meta, Mapping):
                    row = dict(meta)
                    row.setdefault("name", str(name))
                    items.append(row)
                else:
                    items.append({"name": str(name), "source": meta})
    elif isinstance(assets_manifest, Sequence) and not isinstance(assets_manifest, (str, bytes, bytearray)):
        items = list(assets_manifest)
    else:
        items = [assets_manifest]

    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        if isinstance(item, Mapping):
            row = dict(item)
        else:
            text = str(item)
            row = {"name": Path(text).name, "path": text}
        name = str(row.get("name") or row.get("asset") or row.get("filename") or Path(str(row.get("path") or f"asset_{idx}")).name)
        path = row.get("path")
        if path is not None:
            row["path"] = str(path)
        row["name"] = name
        row.setdefault("asset", name)
        row.setdefault("path", str(row.get("path") or name))
        row.setdefault("label", str(row.get("label") or row.get("caption") or Path(name).stem.replace("_", " ")))
        row.setdefault("kind", str(row.get("kind") or row.get("type") or "figure"))
        row.setdefault("notes", [])
        normalized.append(row)
    return normalized


def poster_spec_schema() -> dict[str, Any]:
    return {
        "$schema": JSON_SCHEMA_DRAFT,
        "type": "object",
        "additionalProperties": True,
        "required": ["project", "style", "sections", "placeholders", "conclusion"],
        "properties": {
            "project": {
                "type": "object",
                "additionalProperties": True,
                "required": ["title", "topic", "authors", "audience"],
                "properties": {
                    "title": {"type": "string"},
                    "topic": {"type": "string"},
                    "subtitle": {"type": "string"},
                    "authors": {"type": "string"},
                    "identity": {"type": "string"},
                    "audience": {"type": "string"},
                },
            },
            "style": {
                "type": "object",
                "additionalProperties": True,
                "required": ["summary", "aspect", "top_band", "body_layout", "color_grammar"],
                "properties": {
                    "summary": {"type": "string"},
                    "aspect": {"type": "string"},
                    "top_band": {"type": "string"},
                    "body_layout": {"type": "string"},
                    "color_grammar": {"type": "string"},
                    "typography": {"type": "string"},
                    "color_palette": {"type": "string"},
                    "figure_surface": {"type": "string"},
                    "typography_rules": {"type": ["array", "string"], "items": {"type": "string"}},
                    "color_material_rules": {"type": ["array", "string"], "items": {"type": "string"}},
                },
            },
            "forbidden_phrases": {"type": "array", "items": {"type": "string"}},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": True,
                    "required": ["id", "title", "layout", "text"],
                    "properties": {
                        "id": {"type": "integer"},
                        "title": {"type": "string"},
                        "layout": {"type": "string"},
                        "text": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": True,
                                "required": ["title", "body", "bullets"],
                                "properties": {
                                    "title": {"type": "string"},
                                    "body": {"type": "array", "items": {"type": "string"}},
                                    "bullets": {"type": "array", "items": {"type": "string"}},
                                },
                            },
                        },
                        "flowchart": {"type": "array", "items": {"type": "string"}},
                        "caption": {"type": "string"},
                        "notes": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "placeholders": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": True,
                    "required": ["id", "section", "label", "aspect", "asset"],
                    "properties": {
                        "id": {"type": "string"},
                        "section": {"type": "integer"},
                        "label": {"type": "string"},
                        "aspect": {"type": "string"},
                        "asset": {"type": "string"},
                        "group": {"type": "string"},
                    },
                },
            },
            "placements": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "integer"},
                },
            },
            "conclusion": {"type": "array", "items": {"type": "string"}},
            "closing": {"type": "string"},
            "text_overlays": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
        },
    }


def figure_selection_schema() -> dict[str, Any]:
    return {
        "$schema": JSON_SCHEMA_DRAFT,
        "type": "object",
        "additionalProperties": True,
        "required": ["selected_figures", "selection_notes"],
        "properties": {
            "selected_figures": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": True,
                    "required": ["placeholder_id", "asset", "section", "label", "aspect", "priority", "rationale"],
                    "properties": {
                        "placeholder_id": {"type": "string"},
                        "asset": {"type": "string"},
                        "section": {"type": "integer"},
                        "label": {"type": "string"},
                        "aspect": {"type": "string"},
                        "priority": {"type": "integer"},
                        "rationale": {"type": "string"},
                        "source_path": {"type": "string"},
                        "notes": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "deferred_assets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": True,
                    "required": ["asset", "reason"],
                    "properties": {
                        "asset": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                },
            },
            "selection_notes": {"type": "array", "items": {"type": "string"}},
        },
    }


def placeholder_detection_schema() -> dict[str, Any]:
    return {
        "$schema": JSON_SCHEMA_DRAFT,
        "type": "object",
        "additionalProperties": True,
        "required": ["image_size", "placeholders"],
        "properties": {
            "image_size": {
                "type": "object",
                "required": ["width", "height"],
                "properties": {
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                },
            },
            "placeholders": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": True,
                    "required": ["id", "label", "bbox", "confidence"],
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                        "bbox": {
                            "type": "array",
                            "minItems": 4,
                            "maxItems": 4,
                            "items": {"type": "integer"},
                        },
                        "confidence": {"type": "number"},
                        "notes": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        },
    }


def poster_qa_schema() -> dict[str, Any]:
    return {
        "$schema": JSON_SCHEMA_DRAFT,
        "type": "object",
        "additionalProperties": True,
        "required": ["passes", "summary", "issues", "checks"],
        "properties": {
            "passes": {"type": "boolean"},
            "summary": {"type": "string"},
            "score": {"type": "number"},
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": True,
                    "required": ["severity", "category", "message"],
                    "properties": {
                        "severity": {"type": "string", "enum": ["critical", "warning", "info"]},
                        "category": {"type": "string"},
                        "message": {"type": "string"},
                        "location": {"type": "string"},
                        "suggested_fix": {"type": "string"},
                    },
                },
            },
            "checks": {
                "type": "object",
                "additionalProperties": True,
                "required": ["public_text_clean", "placeholders_accounted_for"],
                "properties": {
                    "public_text_clean": {"type": "boolean"},
                    "placeholders_accounted_for": {"type": "boolean"},
                    "section_count": {"type": "integer"},
                    "placeholder_count": {"type": "integer"},
                },
            },
            "recommended_repairs": {"type": "array", "items": {"type": "string"}},
        },
    }


def arxiv_resolution_schema() -> dict[str, Any]:
    return {
        "$schema": JSON_SCHEMA_DRAFT,
        "type": "object",
        "additionalProperties": True,
        "required": ["arxiv_id", "title", "abs_url", "pdf_url", "confidence", "sources"],
        "properties": {
            "arxiv_id": {"type": "string"},
            "title": {"type": "string"},
            "authors": {"type": "array", "items": {"type": "string"}},
            "abstract": {"type": "string"},
            "published": {"type": "string"},
            "abs_url": {"type": "string"},
            "pdf_url": {"type": "string"},
            "source_url": {"type": "string"},
            "confidence": {"type": "number"},
            "rationale": {"type": "string"},
            "sources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": True,
                    "required": ["title", "url"],
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                    },
                },
            },
        },
    }


def schema_skeleton(schema: Mapping[str, Any]) -> Any:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), schema_type[0] if schema_type else None)
    if schema_type == "object":
        props = schema.get("properties", {})
        required = schema.get("required") or props.keys()
        return {key: schema_skeleton(props.get(key, {})) for key in required}
    if schema_type == "array":
        items = schema.get("items") or {}
        return [schema_skeleton(items)] if items else []
    if schema_type == "string":
        enum = schema.get("enum")
        return enum[0] if enum else ""
    if schema_type == "integer":
        return 0
    if schema_type == "number":
        return 0.0
    if schema_type == "boolean":
        return False
    if "default" in schema:
        return copy.deepcopy(schema["default"])
    if "anyOf" in schema:
        return schema_skeleton(schema["anyOf"][0])
    if "oneOf" in schema:
        return schema_skeleton(schema["oneOf"][0])
    return None
