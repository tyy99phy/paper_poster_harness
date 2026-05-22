<p align="center">
  <img src="docs/assets/logo.svg" width="128" alt="Paper Poster Harness logo" />
</p>

<h1 align="center">Paper Poster Harness</h1>

<p align="center">
  Placeholder-first LLM + image-generation framework for turning papers into conference posters without fake scientific figures.
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="docs/i18n/README.ja.md">日本語</a> ·
  <a href="docs/i18n/README.es.md">Español</a>
</p>

<p align="center">
  <img src="docs/assets/selected_12_preview.jpg" alt="Selected 12-paper benchmark: P2P baseline vs Paper Poster Harness" />
</p>

## What is this?

Paper Poster Harness is a strict, auditable pipeline for producing academic posters from papers. The key idea is simple:

> Let the image model design the poster, but never let it invent scientific data.

The image-generation model creates an attractive poster template with **blank figure placeholders** such as `[FIG 01]`. The harness then detects those placeholders and deterministically replaces them with real figures extracted from the paper or its source package. If any stage fails QA, the run fails or regenerates a full candidate; it does not silently fall back to pasted poster screenshots.

## Why placeholder-first?

Image models can create beautiful scientific-looking charts, but those curves, error bars, histograms, event displays, and diagrams may be fabricated. That is unacceptable for scientific communication.

| Component | Producer | Rule |
|---|---|---|
| Layout, color, typography, atmosphere | image generation | use creative design ability |
| Scientific figures | deterministic replacement | only source figures from the paper |
| Poster copy | LLM planning + filters | grounded, public-facing, no workflow/internal text |
| Quality control | VLM + deterministic checks | reject bad placeholder geometry, bad replacement, or dirty public text |

## Quick start

```bash
# 1. Install
git clone https://github.com/tyy99phy/paper_poster_harness.git
cd paper_poster_harness
pip install -e .

# 2. Create a config and log in with a local ChatGPT/OpenAI account
poster-harness init-config --out poster_harness.yaml --login

# 3. Generate a poster from arXiv
poster-harness autoposter \
  --config poster_harness.yaml \
  --arxiv-id 2206.08956 \
  --out runs/ssww-demo

# Optional: HEP-dense mode for more expert-level HEP content
poster-harness autoposter \
  --config poster_harness.yaml \
  --arxiv-id 2206.08956 \
  --content-mode hep_dense \
  --out runs/ssww-demo-dense
```

Final posters are written under `exports/` inside the run directory.

## Pipeline

```text
paper PDF / arXiv ID / local source
  → extract text and source figures
  → infer domain profile (HEP, CS/ML, Bio, Astro, Math, Chemistry, Generic)
  → plan poster content, storyboard, copy deck, and figure roles
  → select real paper figures for [FIG NN] placeholders
  → build a strict image-generation prompt
  → generate a placeholder-only poster template
  → template critic checks design, text, information density, and placeholder contract
  → detect placeholder coordinates
  → placeholder QA and containment QA
  → insert real figures deterministically
  → upscale/export
  → final QA
  → optional micro-repair only for small public text/glyph issues, then re-detect/re-insert/re-QA
```

Every intermediate artifact is saved: prompts, specs, figure manifests, detections, and QA reports.

## Modes

| Mode | Default | Description | Use case |
|---|---:|---|---|
| `standard` | yes | stable general pipeline with moderate information density | routine generation across fields |
| `hep_dense` | no | higher-density HEP planning with analysis strategy, SR/CR, fits, systematics, limits | expert HEP posters and benchmark figures |

Examples:

```bash
poster-harness autoposter --config poster_harness.yaml --paper paper.pdf
poster-harness autoposter --config poster_harness.yaml --query "CMS W mass Nature 2024"
poster-harness autoposter --config poster_harness.yaml --arxiv-id 2309.03501 --domain-profile hep
poster-harness autoposter --config poster_harness.yaml --arxiv-id 1706.03762 --domain-profile cs_ml
```

## Configuration highlights

The full template lives at `templates/poster_harness_config.yaml`.

```yaml
llm:
  backend: chatgpt_account
  model: gpt-5.5
  account:
    auth_dir: ~/.config/poster-harness/auth

image_generation:
  backend: chatgpt_account
  model: gpt-5.5
  size: 1024x1536
  quality: high
  variants: 2

autoposter:
  required_successes: 2
  max_candidate_batches: 3
  content_mode: standard
  domain_profile: auto
  template_critic:
    enabled: true
    require_pass: true
  micro_repair:
    enabled: true
    backend: image_edit
```

Authentication is intentionally local. Users log in and store their own account JSON; the repository does not ship credentials.

## Selected 12-paper benchmark

A compact qualitative benchmark is included under [`benchmarks/selected_12`](benchmarks/selected_12):

- 6 HEP papers and 6 non-HEP papers.
- 24 poster PNGs: `ours.png` and corrected `p2p.png` for each paper.
- A manifest with arXiv IDs and selection notes.
- Contact sheets for quick visual inspection.

The corrected P2P baseline uses real figure caches for the affected papers instead of full-page PDF screenshot fallbacks.

## Repository layout

```text
poster_harness/                 core package
poster_harness/cli.py            CLI entry point
templates/poster_harness_config.yaml
docs/                            design notes and auth docs
benchmarks/selected_12/          curated qualitative benchmark posters
tests/                           unit tests for auth, arXiv, layout, QA, replacement
```

## Design principles

1. **No fake scientific plots.** Scientific figures must come from the paper/source.
2. **No silent fallback.** If a strict LLM/image stage fails, report the error or regenerate a full candidate.
3. **Keep layout creative.** The image model should control art direction, rhythm, typography, and atmosphere.
4. **Keep replacement deterministic.** Placeholder detection and figure insertion are auditable.
5. **Keep evidence.** Save prompts, specs, manifests, and QA files for every run.

## Development

```bash
pip install -e .
pytest
```

See also:

- [`docs/account_auth.md`](docs/account_auth.md)
- [`docs/prompt_contract.md`](docs/prompt_contract.md)
- [`docs/quality_policy.md`](docs/quality_policy.md)
- [`docs/paper2poster_lessons.md`](docs/paper2poster_lessons.md)
