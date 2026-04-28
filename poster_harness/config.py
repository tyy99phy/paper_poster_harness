from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_HARNESS_CONFIG: dict[str, Any] = {
    "strict": True,
    "paths": {
        "runs_dir": "runs",
        "paper_subdir": "input",
        "assets_subdir": "assets",
    },
    "paper": {
        "query": "",
        "arxiv_id": "",
        "paper": "",
        "text_source": "",
        "assets_dir": [],
        "out": "",
    },
    "llm": {
        # Default and only supported route: local ChatGPT account auth JSON.
        "backend": "chatgpt_account",
        "model": "gpt-5.5",
        "timeout": 180,
        "account": {
            "account": "",          # optional email; empty means auto-discover best local codex-*.json
            "auth_dir": "~/.config/poster-harness/auth",
            "auth_file": "",        # optional explicit auth JSON path
            "endpoint": "https://chatgpt.com/backend-api/codex/responses",
            "min_token_seconds": 60,
            "proxy": "",            # optional HTTPS proxy URL; empty uses env HTTPS_PROXY if present
        },
        "web_search": {
            "model": "gpt-5.5",
            "tool_type": "web_search",
            "allowed_domains": ["arxiv.org"],
            "include_sources": True,
            "include": ["web_search_call.action.sources"],
            "reasoning_effort": "low",
        },
    },
    "image_generation": {
        # Use the same local ChatGPT account auth file route as the LLM backend.
        "backend": "chatgpt_account",
        "model": "gpt-5.5",
        "size": "1024x1536",
        "quality": "high",
        "variants": 1,
        "generated_scale": 4.0,
        "upscale_factor": 4.0,
        "account": {
            "account": "",
            "auth_dir": "~/.config/poster-harness/auth",
            "auth_file": "",
            "endpoint": "https://chatgpt.com/backend-api/codex/responses",
            "min_token_seconds": 60,
            "proxy": "",
        },
    },
    "autoposter": {
        "style": "cms-hep",
        "max_figures": 8,
        "max_assets": 48,
        "min_image_width": 96,
        "min_image_height": 96,
        "render_pages": True,
        "extract_pdf_images": True,
        "recursive_assets": True,
        "auto_assets": True,
        "keep_unselected_placeholders": False,
        "min_detection_confidence": 0.15,
        "normalize_placeholder_geometry": True,
        "redraw_normalized_placeholders": False,
        "pdf_render_dpi": 220,
        "max_pages": 12,
        "figure_layout_policy": (
            "Use the selected source asset aspect ratio as the placeholder aspect ratio. "
            "Do not warp square plots into wide slots or moderate wide plots into arbitrary poster ratios. "
            "Readability should come from allocating more absolute area, using wider/taller surrounding cards, "
            "or changing the card layout while preserving the source figure ratio."
        ),
    },
    "arxiv": {
        "enabled": True,
        "download_source": True,
        "download_pdf": True,
        "source_asset_roots": ["figures", "figure", "figs", "fig", "images", "plots", "graphics"],
        "render_pdf_figures_dpi": 240,
    },
    "styles": {
        "generic": {
            "project": {},
            "style": {
                "summary": "premium CERN/LHCC-inspired scientific poster, modern editorial HEP design, artistic but readable, not a collage",
                "aspect": "A0 vertical / 2:3 ratio",
                "top_band": "strong title banner with concise identity text and abstract scientific artwork",
                "body_layout": "4-6 large numbered modules on a light background with one dominant result region, generous gutters, and varied card shapes",
                "color_grammar": "one primary accent color for headline results and one secondary accent color for contrasts",
                "art_direction": "editorial scientific-magazine design with layered abstract geometry, luminous gradients, subtle depth, and a clear visual hierarchy",
                "layout_rhythm": "CERN/LHCC poster rhythm: strong header, numbered scan path, one hero result, supporting method cards, circular callouts and pill headers; avoid uniformly tiled boxes",
                "background_texture": "soft abstract scientific network/field texture that remains behind content and never becomes fake data",
            },
            "extras": {
                "decorative_art_constraints": [
                    "Decorative artwork must remain abstract.",
                    "Do not draw fake plots, fake tables, or unlabeled scientific diagrams outside placeholders.",
                ],
                "forbidden_phrases": ["internal workflow", "production workflow", "production-process", "replacement", "placeholder explanation"],
            },
        },
        "cms-hep": {
            "project": {
                "audience": "high-energy physics conference audience",
            },
            "style": {
                "summary": "premium high-energy-physics conference poster, CERN/LHCC-inspired, CMS-style detector aesthetic, luminous beamline abstraction, artistic but readable; not a collage",
                "aspect": "A0 vertical / 2:3 ratio",
                "top_band": "dark navy title band with identity area on left and a cinematic abstract detector/beam artwork on right",
                "body_layout": "five major modules on a pale scientific background: compact shaped motivation/strategy cards, one dominant key-result figure card that preserves source aspect, and a concise summary strip",
                "color_grammar": "primary result = CMS blue; secondary interpretation = magenta/purple; limits/results = gold and black accents",
                "art_direction": "cinematic CMS-inspired detector rings, beamline light trails, glassmorphism, depth, glow, and premium science-magazine polish; artwork must be abstract, not literal physics diagrams",
                "layout_rhythm": "asymmetric HEP hierarchy with a clear scan path, section ribbons, one hero result anchor, circular badges, pill headers, and smaller supporting cards; avoid equal-size plot mosaics",
                "background_texture": "subtle luminous particle-field/network texture with low contrast behind cards",
            },
            "extras": {
                "decorative_art_constraints": [
                    "The title/header artwork must be abstract detector or beamline art only.",
                    "Do not draw Feynman diagrams in decorative areas.",
                    "Do not put particle labels such as mu, nu, j, q, W, N in decorative header artwork.",
                    "Do not add Feynman diagrams, interaction vertices, or physics-process diagrams outside labeled placeholders.",
                    "Simple text-only analysis workflow flowcharts are allowed only when explicitly specified in the poster spec.",
                    "Only [FIG 01]-style placeholders may contain source-figure diagram slots.",
                ],
                "forbidden_phrases": [
                    "internal workflow",
                    "production workflow",
                    "production-process",
                    "replacement",
                    "validated source",
                    "placeholder explanation",
                ],
            },
        },
    },
}


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML configs. Install pyyaml or use JSON.")
        return yaml.safe_load(text) or {}
    return json.loads(text)


def dump_config(data: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML configs. Install pyyaml or use JSON.")
        p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_harness_config(path: str | Path | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_HARNESS_CONFIG)
    config_path = path or os.getenv("POSTER_HARNESS_CONFIG")
    if config_path:
        config = deep_merge(config, load_config(config_path))
    return config


def write_default_harness_config(path: str | Path) -> None:
    dump_config(copy.deepcopy(DEFAULT_HARNESS_CONFIG), path)


def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = copy.deepcopy(base)
        for key, value in override.items():
            merged[key] = deep_merge(merged.get(key), value)
        return merged
    return copy.deepcopy(override)


def cfg_get(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = config
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
