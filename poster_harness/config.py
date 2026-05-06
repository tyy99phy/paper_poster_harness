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
        "variants": 3,
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
        "placeholder_aspect_tolerance": 0.20,
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
                "body_layout": "4-6 large numbered modules on a light background with one dominant result region, generous gutters, varied card shapes, and light paper-like figure cards for every chart/diagram",
                "color_grammar": "one primary accent color for headline results and one secondary accent color for contrasts; all figure surfaces remain warm-white or very pale neutral",
                "art_direction": (
                    "Editorial scientific-magazine design with layered abstract geometry and luminous gradients. "
                    "Use atmospheric depth, subtle glow, frosted-glass surfaces, and carefully balanced negative space. "
                    "Create a cover-story feeling: authoritative, beautiful, contemporary, and clearly scientific. "
                    "Let abstract scientific forms sweep across the canvas to connect sections without becoming fake data. "
                    "Use refined typography, micro-shadows, metallic accents, and soft depth cues to avoid a PPT-like template. "
                    "Decorative artwork remains abstract; real plots, diagrams, tables, and images appear only in placeholders."
                ),
                "layout_rhythm": (
                    "CERN/LHCC poster rhythm with a strong header and a clear diagonal or Z-shaped scan path. "
                    "Anchor one hero result in the visual center of gravity, with supporting method and context modules around it. "
                    "Use section ribbons, numbered tabs, circular callouts, pill headers, wide bands, tall sidebars, and staggered panels. "
                    "Break the grid intentionally while keeping alignment disciplined and readable. "
                    "Vary card proportions and scale so the poster has hierarchy rather than a uniform matrix of boxes. "
                    "Generous gutters and overlapping translucent connector elements should create flow without clutter."
                ),
                "background_texture": (
                    "Soft abstract scientific network and field texture behind content cards, kept low contrast and atmospheric. "
                    "Use faint radial gradients, subtle particle trails, ghosted geometric forms, and margin-only network hints. "
                    "Texture should add depth and motion while never reading as a plot, table, equation, or data visualization. "
                    "Keep card interiors calm enough for text and placeholders to stay legible. "
                    "Let color temperature shift gently across the poster to support the reading path."
                ),
                "typography": (
                    "Modern editorial sans-serif system: Helvetica/Inter/Source-Sans-like, with a bold condensed title, "
                    "crisp section headers, compact readable bullets, and restrained scientific notation. "
                    "Use two weights and one accent style rather than many decorative fonts. "
                    "Create hierarchy through scale, weight, color, spacing, and numbered tabs."
                ),
                "color_palette": (
                    "Premium science palette: deep indigo or graphite atmosphere in the outer background, cobalt/cyan primary accents, "
                    "violet/magenta secondary accents, restrained amber/gold highlights, and warm-white or pearl content cards. "
                    "Text cards and figure cards should be light and calm; dark surfaces belong in the header, margins, frames, or accent rails."
                ),
                "figure_surface": (
                    "Every scientific figure placeholder must live on a warm-white, pearl, or very pale blue paper-like card/mat. "
                    "Never place a plot or diagram on a dark navy, black, purple, or saturated block. "
                    "Use shadows, outlines, halos, and side accents for drama while keeping the chart surface light."
                ),
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
                "body_layout": "five major modules on a pale scientific background: compact shaped motivation/strategy cards, one dominant warm-white key-result figure card that preserves source aspect, and a concise summary strip",
                "color_grammar": "primary result = CMS blue; secondary interpretation = magenta/purple; limits/results = restrained gold and black accents; every plot/diagram surface remains warm-white or very pale neutral",
                "art_direction": (
                    "Cinematic CMS-inspired detector rings rendered as luminous concentric forms in the background. "
                    "Beamline light trails should sweep diagonally across the canvas with soft particle-spray atmosphere. "
                    "Use glassmorphism cards with frosted translucency, micro-shadow depth, refined typography, and metallic accents. "
                    "The overall feel should evoke a Nature Physics or CERN Courier cover: authoritative, beautiful, and unmistakably particle-physics. "
                    "Use volumetric glow, atmospheric depth, cool CMS blues, magenta/purple contrast, and restrained gold highlights. "
                    "Decorative art is abstract only; no literal detector diagrams, Feynman graphs, fake plots, or particle-labeled schematics outside placeholders."
                ),
                "layout_rhythm": (
                    "Asymmetric HEP hierarchy with a clear diagonal scan path from top-left to bottom-right. "
                    "Use a strong title banner and anchor the hero result near the golden-ratio zone. "
                    "Section ribbons and numbered tabs should create wayfinding without forcing rigid PPT blocks. "
                    "Use circular badges, pill-shaped headers, wide bands, tall sidebars, L-shaped wraps, and staggered translucent panels to break the grid. "
                    "One dominant hero figure card should visibly outweigh supporting cards, while smaller diagnostic cards remain secondary. "
                    "Avoid equal-size plot mosaics, uniformly tiled white boxes, and slide-deck symmetry."
                ),
                "background_texture": (
                    "Subtle luminous particle-field texture with low contrast behind all content cards. "
                    "Use faint radial gradients emanating from the hero result region and ghosted detector-ring elements at very low opacity. "
                    "Network or graph-topology hints may live in the margins but must not cross into scientific figure placeholders. "
                    "The texture conveys atmospheric depth, not data, noise, or a fake event display. "
                    "Color temperature may transition from deep navy and indigo near the header to charcoal with restrained gold warmth near the bottom."
                ),
                "typography": (
                    "CMS/CERN editorial typography: a bold condensed sans-serif title, clear numbered module headers, "
                    "dark readable body text on light cards, and compact bullets. "
                    "Use a disciplined type scale with strong hierarchy; avoid decorative display fonts except for subtle numeric badges."
                ),
                "color_palette": (
                    "CMS-inspired premium palette: deep navy/indigo outer atmosphere, CMS cobalt blue, electric cyan glints, "
                    "violet/magenta interpretation accents, restrained amber/gold result highlights, and warm-white/pearl figure cards. "
                    "Dark colors should create cinematic depth in the header, background, frames, side rails, and badges, not inside figure blocks."
                ),
                "figure_surface": (
                    "All scientific figure areas, including the headline limit/result card, must use warm-white or pearl figure-card interiors. "
                    "Do not make the plot-containing block dark; use a light inset card/mat with subtle gray outline, soft shadow, and optional gold/CMS-blue outer frame. "
                    "This ensures white-background CMS plots blend into the poster instead of looking like pasted stickers."
                ),
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
