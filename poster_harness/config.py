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
        "variants": 2,
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
        "style": "generic",
        "domain_profile": "auto",
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
        "required_successes": 2,
        "max_candidate_batches": 3,
        "llm_stage_retries": 4,
        "layout_contract": {
            "enabled": True,
            "reject_misaligned": True,
        },
        "template_critic": {
            "enabled": True,
            "require_pass": True,
            "max_regen_rounds": 2,
            "min_overall_score": 0.72,
            "min_artistry_score": 0.65,
            "min_information_density_score": 0.65,
            "min_placeholder_contract_score": 0.75,
            "extra_instructions": "",
        },
        "pdf_render_dpi": 220,
        "max_pages": 12,
        "content_mode": "standard",
        "domain_classifier": {
            "enabled": True,
            "max_text_chars": 12000,
            "extra_instructions": "",
        },
        "content_outline": {
            "enabled": False,
            "max_sections": 6,
            "max_facts": 28,
            "max_formulas": 6,
            "extra_instructions": "",
        },
        "storyboard": {
            "enabled": True,
            "extra_instructions": "",
        },
        "physics_quiz": {
            "enabled": True,
            "max_questions": 20,
            "extra_instructions": "",
        },
        "copy_deck": {
            "enabled": True,
            "max_units": 34,
            "extra_instructions": "",
        },
        "flowchart_rewrite": {
            "enabled": True,
            "text_char_limit": 24000,
            "extra_instructions": "",
        },
        "micro_repair": {
            "enabled": True,
            "backend": "image_edit",
            "max_rounds": 2,
            "extra_instructions": "",
        },
        "figure_layout_policy": (
            "Use the selected source asset aspect ratio as the placeholder aspect ratio. "
            "Do not warp square plots into wide slots or moderate wide plots into arbitrary poster ratios. "
            "Readability should come from allocating more absolute area, using wider/taller surrounding cards, "
            "or changing the card layout while preserving the source figure ratio."
        ),
    },
    "content_modes": {
        "standard": {
            # Default/main-compatible route: keep the established prompt and stage graph.
            "autoposter": {
                "content_mode": "standard",
                "content_outline": {"enabled": False},
                "copy_deck": {"max_units": 34, "extra_instructions": ""},
            },
        },
        "hep_dense": {
            # Optional high-information-density expert HEP route inspired by the
            # successful regen2/P2P-density experiments.  This deliberately does
            # not replace the default route.
            "autoposter": {
                "content_mode": "hep_dense",
                "content_outline": {
                    "enabled": True,
                    "max_sections": 6,
                    "max_facts": 30,
                    "max_formulas": 6,
                    "extra_instructions": (
                        "Increase useful information density through paper-specific facts and smaller text tiers, "
                        "not by reducing whitespace, shrinking placeholders, or adding generic HEP boilerplate."
                    ),
                },
                "copy_deck": {
                    "enabled": True,
                    "max_units": 42,
                    "extra_instructions": (
                        "Prefer denser but still legible microcopy: preserve current generous whitespace and figure sizes, "
                        "but use smaller body/badge tiers for should/could units so more paper-specific facts survive."
                    ),
                },
                "physics_quiz": {
                    "enabled": True,
                    "max_questions": 20,
                },
            },
            "styles": {
                "generic": {
                    "style": {
                        "information_density": (
                            "Paper2Poster-rich editorial density without reducing whitespace: 18-30 concise public information units across the poster, "
                            "4-8 grounded fact badges, 4-6 section modules, and one compact conclusion strip. "
                            "Use a controlled smaller text tier for optional bullets/badges/micro-callouts, while keeping must-level facts clearly legible. "
                            "Prefer short claims and fact chips over paragraphs; omit lower-priority facts before sacrificing legibility or placeholder geometry."
                        ),
                        "text_density": [
                            "Compress prose into poster bullets: short noun phrases, ideally under 11 words each.",
                            "Maintain Paper2Poster-style information richness through smaller but legible body/badge tiers, not through crowded layouts or reduced whitespace.",
                            "Use three text scales: primary claims readable at a glance, compact bullets for specialist detail, and micro-badges for optional should/could facts.",
                            "Avoid microscopic paragraphs, footnote blocks, dense equations, and reference lists in the rendered poster.",
                            "Prefer 2-5 bullets/fact chips per non-hero card and at most one short sentence per text block; preserve meaning without adding new science.",
                            "Use public fact chips and numeric badges for explicitly grounded dataset/result facts; do not invent numbers.",
                            "If text competes with a result figure, shrink or omit lower-priority text first and enlarge/preserve the figure placeholder.",
                        ],
                    },
                },
                "cms-hep": {
                    "style": {
                        "information_density": (
                            "Paper2Poster-rich HEP density without reducing whitespace: preserve enough public facts for a viewer to answer what was measured/searched, "
                            "which dataset/channel/strategy was used, what the headline result says, and which figures support it. "
                            "Use 18-28 short bullets/fact chips total plus 4-8 grounded badges; increase density through a smaller but still legible specialist-detail tier, "
                            "never by shrinking figure placeholders, flattening the layout, or inventing CMS numbers."
                        ),
                        "text_density": [
                            "Compress prose into HEP poster microcopy: short noun phrases, ideally under 11 words each.",
                            "Maintain Paper2Poster-style information richness through smaller but legible specialist-detail text, not through crowded layouts or reduced whitespace.",
                            "Use three text scales: large must-level result/method claims, compact bullets for SR/CR/fit details, and micro-badges for optional uncertainties or category labels.",
                            "For HEP experts, prefer concrete SR/CR, fit, selection, and uncertainty chips over generic analysis prose.",
                            "Avoid paragraphs, footnote blocks, dense equations, reference lists, and tiny illegible tables.",
                            "Use public fact chips and numeric badges only for explicitly grounded luminosity, energy, channel, mass/limit, or CL facts.",
                            "If text competes with a result figure, omit could-priority units before shrinking placeholders or reducing gutters.",
                        ],
                    },
                },
            },
            "domain_profiles": {
                "hep": {
                    "style": {
                        "information_density": (
                            "Paper2Poster-rich HEP density without reducing whitespace: preserve enough public facts for a viewer to answer what was measured/searched, "
                            "which dataset/channel/strategy was used, what the headline result says, and which figures support it. "
                            "Use 18-28 short bullets/fact chips total plus 4-8 grounded badges; increase density through a smaller but still legible specialist-detail tier, "
                            "never by shrinking figure placeholders, flattening the layout, or inventing numbers."
                        ),
                        "text_density": [
                            "Compress prose into HEP poster microcopy: short noun phrases, ideally under 11 words each.",
                            "Maintain Paper2Poster-style information richness through smaller but legible specialist-detail text, not through crowded layouts or reduced whitespace.",
                            "Use three text scales: large must-level result/method claims, compact bullets for SR/CR/fit details, and micro-badges for optional uncertainties or category labels.",
                            "For HEP experts, prefer concrete SR/CR, fit, selection, and uncertainty chips over generic analysis prose.",
                            "Avoid paragraphs, footnote blocks, dense equations, reference lists, and tiny illegible tables.",
                            "Use public fact chips and numeric badges only for explicitly grounded luminosity, energy, channel, mass/limit, or CL facts.",
                            "If text competes with a result figure, omit could-priority units before shrinking placeholders or reducing gutters.",
                        ],
                    },
                },
            },
        },
    },
    "domain_profiles": {
        "generic": {
            "project": {"audience": "academic conference audience"},
            "style": {
                "domain_label": "general_science",
                "domain_grammar_heading": "SCIENTIFIC POSTER DESIGN GRAMMAR",
                "domain_poster_grammar": [
                    "Use a clear scientific story spine: question → approach → evidence → key result → implication.",
                    "Create one dominant hero evidence/result region and 3-5 supporting modules.",
                    "Use field-appropriate figure hierarchy: result plots, method schematics, visual examples, or tables only as blank placeholders.",
                    "Use numbered section tabs, concise claims, public fact badges, and generous gutters.",
                    "Avoid forcing HEP, CS, biology, or clinical visual language unless the detected domain supports it.",
                ],
                "figure_composition": [
                    "Allocate the largest placeholder to the paper's most important empirical result, method diagram, or visual evidence.",
                    "Every placeholder rectangle must match its selected source image aspect ratio; enlarge or reshape the surrounding block instead of stretching the placeholder.",
                    "Multi-panel figures need large absolute area; preserve their native ratio rather than forcing square slots.",
                    "Placeholders should align to the card grid and be easy to replace: rectangular, unobstructed, with visible margins.",
                    "All figure-containing cards must use a light paper/lab-white surface around the placeholder; dark or saturated fills may be outer accents only.",
                ],
            },
            "extras": {
                "decorative_art_constraints": [
                    "Decorative artwork must remain abstract and field-neutral.",
                    "Do not draw fake plots, fake tables, fake microscopy images, fake algorithms, or unlabeled scientific diagrams outside placeholders.",
                ],
            },
        },
        "hep": {
            "project": {"audience": "high-energy physics conference audience"},
            "style": {
                "domain_label": "high_energy_physics",
                "domain_grammar_heading": "HEP POSTER DESIGN GRAMMAR",
                "domain_poster_grammar": [
                    "Use a CERN/LHCC-style scientific story spine: motivation → dataset/selection → strategy/background → key result → interpretation/summary.",
                    "Use colored section ribbons or numbered tabs so the viewer can scan the poster from 2 meters away.",
                    "Create one dominant hero region for the headline result; do not divide the body into many equal tiny boxes.",
                    "Cards/blocks do not need to be square: use wide bands, tall sidebars, pill headers, circular callouts, curved connectors, and staggered panels when it improves hierarchy.",
                    "Let 40-55% of the body be reserved for figure placeholders; HEP posters are figure-led, not paragraph-led.",
                    "Use small badges for dataset, collision energy, luminosity, channel, or experiment only when present in the supplied text; never invent numbers.",
                ],
                "figure_composition": [
                    "Allocate the largest placeholder to the main result/limit/cross-section/significance plot.",
                    "Every placeholder rectangle must match its selected source image aspect ratio; enlarge or reshape the surrounding block instead of stretching the placeholder.",
                    "Method, detector, topology, or control-region placeholders should support the story, not dominate it unless the paper is instrumentation-focused.",
                    "Dense multi-panel HEP plots need large absolute area; preserve their native wide/tall ratio rather than forcing a square slot.",
                    "For wide post-fit/distribution plots, reserve enough vertical height for axes and legends; reduce nearby flowchart/text space before making the plot hard to read.",
                    "For professional HEP readers, dataset and strategy cards should show analysis-specific SR/CR, fit, and uncertainty details rather than a generic data-processing pipeline.",
                    "All figure-containing cards must use a light paper/lab-white surface around the placeholder; dark or saturated fills may be outer accents only.",
                ],
                "domain_text_guidance": (
                    "Prioritize concrete luminosity/energy/channel, object selections, SR/CR labels, binning variables, fitted observables, likelihood/CLs strategy, floating normalizations, and dominant uncertainties when present."
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
            },
        },
        "cs_ml": {
            "project": {"audience": "computer science / machine learning conference audience"},
            "style": {
                "domain_label": "computer_science_machine_learning",
                "domain_grammar_heading": "CS/ML POSTER DESIGN GRAMMAR",
                "domain_poster_grammar": [
                    "Use a CS/ML story spine: problem → method/architecture → experiments → quantitative results → limitations/impact.",
                    "Make the method/algorithm card visually prominent when the contribution is methodological.",
                    "Use placeholders for architecture diagrams, qualitative examples, learning curves, ablations, benchmark tables, and error analyses.",
                    "Prefer clean algorithmic flow, modular cards, and readable metric badges over dense paragraphs.",
                    "Do not draw fake network diagrams, fake tables, or fake plots outside placeholders.",
                ],
                "domain_text_guidance": "Prioritize task, dataset, model, training/evaluation protocol, baselines, metrics, ablations, and headline improvements when explicitly present.",
            },
            "extras": {
                "decorative_art_constraints": [
                    "Decorative artwork may use abstract graphs, tensors, networks, code-like grids, or data-flow motifs only as non-data background texture.",
                    "Do not draw fake benchmark tables, fake confusion matrices, fake architecture diagrams, or fake screenshots outside placeholders.",
                ],
            },
        },
        "bio": {
            "project": {"audience": "biology / biomedical research conference audience"},
            "style": {
                "domain_label": "biology_biomedicine",
                "domain_grammar_heading": "BIO/Biomed POSTER DESIGN GRAMMAR",
                "domain_poster_grammar": [
                    "Use a biological story spine: question/system → assay/cohort → mechanism/evidence → key phenotype/result → implication.",
                    "Use placeholders for microscopy, gels/blots, UMAPs, survival curves, heatmaps, pathway diagrams, or experimental workflows.",
                    "Keep biological imagery decorative and abstract unless it is a real source figure placeholder.",
                    "Use concise condition, perturbation, organism/cell-type, cohort, endpoint, and effect-size badges only when grounded.",
                ],
                "domain_text_guidance": "Prioritize organism/model system, perturbation/assay, sample or cohort, endpoint, mechanism, controls, and headline biological effect when explicitly present.",
            },
            "extras": {
                "decorative_art_constraints": [
                    "Decorative artwork may use abstract cellular, molecular, tissue, or pathway motifs only as background texture.",
                    "Do not draw fake microscopy panels, gels, blots, pathway diagrams, or clinical charts outside placeholders.",
                ],
            },
        },
        "astro": {
            "project": {"audience": "astrophysics / astronomy conference audience"},
            "style": {
                "domain_label": "astrophysics",
                "domain_grammar_heading": "ASTROPHYSICS POSTER DESIGN GRAMMAR",
                "domain_poster_grammar": [
                    "Use an astrophysics story spine: target/question → observations/simulation → analysis/model → result → cosmological/physical implication.",
                    "Use placeholders for sky maps, spectra, light curves, images, corner plots, simulations, or parameter constraints.",
                    "Use cosmic background art only as abstract atmosphere; never fake observational data outside placeholders.",
                    "Use badges for instrument/survey, wavelength, redshift, exposure, sample size, or simulation suite only when present.",
                ],
                "domain_text_guidance": "Prioritize target class, survey/instrument, wavelength/redshift, sample/simulation, model assumptions, constraints, and headline physical implication when grounded.",
            },
        },
        "math": {
            "project": {"audience": "mathematics / theoretical research conference audience"},
            "style": {
                "domain_label": "mathematics_theory",
                "domain_grammar_heading": "MATH/THEORY POSTER DESIGN GRAMMAR",
                "domain_poster_grammar": [
                    "Use a theory story spine: problem → definitions/setting → main theorem/result → proof idea → consequences/open questions.",
                    "Use placeholders for source diagrams, commutative diagrams, geometric sketches, numerical illustrations, or tables only when they exist.",
                    "Keep equations sparse and public-facing; do not typeset long fake derivations as decoration.",
                    "Use theorem/corollary badges and proof-roadmap cards only when grounded in the source.",
                ],
                "domain_text_guidance": "Prioritize assumptions, definitions, theorem statements, proof ideas, examples/counterexamples, and implications while keeping notation legible.",
            },
        },
        "chemistry": {
            "project": {"audience": "chemistry / materials science conference audience"},
            "style": {
                "domain_label": "chemistry_materials",
                "domain_grammar_heading": "CHEMISTRY/MATERIALS POSTER DESIGN GRAMMAR",
                "domain_poster_grammar": [
                    "Use a chemistry/materials story spine: system/synthesis → characterization → mechanism/property → performance/result → implication.",
                    "Use placeholders for spectra, microscopy, crystal structures, reaction schemes, phase diagrams, device curves, or tables.",
                    "Decorative molecular or lattice motifs must remain abstract and not substitute for real structures or data.",
                    "Use badges for composition, condition, catalyst/material, measurement, property, and performance only when grounded.",
                ],
                "domain_text_guidance": "Prioritize material/system, synthesis/condition, characterization method, property/performance metric, mechanism, and benchmark comparison when present.",
            },
        },
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
                "summary": "premium domain-adaptive scientific conference poster, modern editorial design, artistic but readable, not a collage",
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
                    "Editorial scientific-poster rhythm with a strong header and a clear diagonal or Z-shaped scan path. "
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
                    "Premium science palette: deep indigo or graphite atmosphere in the outer background, one disciplined primary accent, "
                    "one secondary contrast, restrained warm highlights, and warm-white or pearl content cards. "
                    "Text cards and figure cards should be light and calm; dark surfaces belong in the header, margins, frames, or accent rails."
                ),
                "figure_surface": (
                    "Every scientific figure placeholder must live on a warm-white, pearl, or very pale blue paper-like card/mat. "
                    "Never place a plot or diagram on a dark navy, black, purple, or saturated block. "
                    "Use shadows, outlines, halos, and side accents for drama while keeping the chart surface light."
                ),
                "information_density": (
                    "Paper2Poster-rich editorial density: 14-24 concise public information units across the poster, "
                    "3-6 grounded fact badges, 4-6 section modules, and one compact conclusion strip. "
                    "Prefer short claims and fact chips over paragraphs; omit lower-priority facts before sacrificing legibility or placeholder geometry."
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
                "information_density": (
                    "Paper2Poster-rich HEP density: preserve enough public facts for a viewer to answer what was measured/searched, "
                    "which dataset/channel/strategy was used, what the headline result says, and which figures support it. "
                    "Use 12-20 short bullets/fact chips total plus 3-6 grounded badges; never invent CMS numbers or shrink text to illegibility."
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


def load_autoposter_config(path: str | Path | None = None, *, content_mode: str | None = None) -> dict[str, Any]:
    """Load config for the one-command autoposter pipeline and apply a content mode.

    Mode overlays are applied between built-in defaults and user overrides:

    1. start from the built-in standard/main-compatible defaults;
    2. apply the selected mode's overlay (for example ``hep_dense``);
    3. merge the user's config file on top so explicit local overrides still win;
    4. apply a CLI ``--content-mode`` override last when provided.

    This keeps the default route identical to ``standard`` while making the
    regen2/P2P-density behavior opt-in instead of silently changing main.
    """
    base = copy.deepcopy(DEFAULT_HARNESS_CONFIG)
    config_path = path or os.getenv("POSTER_HARNESS_CONFIG")
    user_config = load_config(config_path) if config_path else {}
    requested_mode = (
        content_mode
        or cfg_get(user_config, "autoposter.content_mode")
        or cfg_get(base, "autoposter.content_mode", "standard")
    )
    config = apply_content_mode(base, requested_mode)
    if user_config:
        config = deep_merge(config, user_config)
    if content_mode:
        config = apply_content_mode(config, content_mode)
    else:
        # Normalize aliases even when the mode came from the config file.
        normalized = normalize_content_mode(str(cfg_get(config, "autoposter.content_mode", requested_mode)))
        config.setdefault("autoposter", {})["content_mode"] = normalized
    return config


def normalize_content_mode(mode: str | None) -> str:
    normalized = (mode or "standard").strip().lower().replace("-", "_")
    aliases = {
        "": "standard",
        "default": "standard",
        "main": "standard",
        "stable": "standard",
        "standard": "standard",
        "dense": "hep_dense",
        "hep_dense": "hep_dense",
        "p2p_dense": "hep_dense",
        "p2p_content_density": "hep_dense",
        "regen2": "hep_dense",
    }
    return aliases.get(normalized, normalized)


def apply_content_mode(config: dict[str, Any], mode: str | None) -> dict[str, Any]:
    normalized = normalize_content_mode(mode)
    modes = config.get("content_modes") or {}
    if normalized not in modes:
        available = ", ".join(sorted(str(key) for key in modes)) or "(none)"
        raise ValueError(f"unknown content_mode '{mode}'; available modes: {available}")
    merged = deep_merge(config, modes[normalized])
    merged.setdefault("autoposter", {})["content_mode"] = normalized
    return merged


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
