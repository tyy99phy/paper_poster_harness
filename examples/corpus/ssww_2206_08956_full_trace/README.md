# Full-trace run-record corpus example

This directory is a compact corpus example for the `PosterHarness` run-record format. It complements the visual benchmark in [`benchmarks/selected_12`](../../../benchmarks/selected_12): the benchmark stores paired poster images, while this example stores the textual audit trail behind one complete harness run.

## Files

- [`run_record.json`](run_record.json): machine-readable corpus unit and source of truth.
- [`run_record.md`](run_record.md): human-readable rendering of the same record.
- [`README.zh-CN.md`](README.zh-CN.md): Chinese description of this example.

No PNG, PDF, PPTX, extracted source figure, or other binary run artifact is included here. The record keeps the workflow text only: stage summaries, prompts, structured planning outputs, placeholder detection, QA/critic judgments, provenance ledger, objective checks, model-judge scores, and the deterministic postprocess note.

## Source run

The example is derived from a PosterHarness run on arXiv `2206.08956` (CMS same-sign WW / heavy Majorana neutrino). It was chosen because the run contains a complete trace with LLM envelopes, image-generation prompt, template critic, placeholder QA, containment QA, final QA, figure provenance accounting, and a small deterministic FIG 01 placement adjustment.

The corpus files are sanitized before release:

- local absolute paths are replaced by placeholders such as `<run>`, `<source-run>`, and `<tmp>`;
- account identifiers are replaced by `<redacted-account>`;
- binary artifacts are referenced but not shipped.

## How to read it

A useful reading order is:

1. **Run card** and **Capability stages** — what the harness attempted and which stages passed.
2. **QA / failure accounting** — template critic, placeholder QA, final QA, and containment reports.
3. **Provenance ledger** — which source figure was assigned to each placeholder and whether the declared aspect contract matched the detected geometry.
4. **Complete workflow trace** — embedded prompts, planning YAML, LLM envelopes, detection YAML, placement specs, and QA judgments.
5. **Model-judge scores** — subjective scores with the explicit caveat that they are not ground truth.

## Scope and caveats

This is an example corpus unit, not an additional benchmark sample. It should not be counted as a thirteenth paper in the selected benchmark. Its purpose is to show what a complete auditable run record looks like and to provide a reusable text corpus for inspecting prompt contracts, model judgments, and deterministic checks.

To generate the same type of record for a new run:

```bash
poster-harness record runs/your-run-directory
```
