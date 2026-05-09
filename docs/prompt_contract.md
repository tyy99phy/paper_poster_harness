# Image-generation prompt contract

Every poster prompt should include these constraints:

- Create the whole poster as a polished design.
- Do not draw fake scientific plots, fake histograms, fake heatmaps, fake tables, fake microscopy images, or fake Feynman diagrams.
- Every figure/table/diagram area must be a clean labeled placeholder: `[FIG 01]`, `[FIG 02]`, etc.
- Each placeholder must include intended content name and aspect ratio.
- Placeholder aspect ratios should be explicit enough to drive layout, e.g. `4:3`, `2.5:1 wide`, or `1:1 square`.
- The poster must not include internal workflow notes, replacement instructions, TODOs, or draft comments.
- Conclusions must be public-facing scientific takeaways only.
- Section titles, placeholder IDs, and conclusion bullets are literal text; the model should copy them exactly instead of rewriting them.

The image model owns:

- art direction;
- background;
- typography style;
- card hierarchy;
- section layout;
- placeholder placement.

The deterministic harness owns:

- source text extraction;
- asset manifest construction;
- LLM-driven poster-spec drafting, figure selection, placeholder detection, and QA;
- public text filtering;
- real figure replacement;
- high-resolution export;
- optional text overlay repairs.

## Placeholder lifecycle

1. `poster_spec.yaml` defines `placeholders` with `id`, `section`, `label`, `aspect`, and source `asset`.
2. The image model draws only labeled placeholder boxes; it must not invent plot contents.
3. `llm-detect-placeholders` returns pixel boxes for those IDs.
4. The template critic checks the whole generated poster for information density, artistic quality, text legibility, and placeholder contract compliance.
5. If the critic rejects all candidates and regeneration rounds remain, the harness appends critic repairs to the prompt and regenerates a complete fresh poster.
6. The replacement stage inserts the real assets into those boxes.
7. QA checks that public text is clean and placeholders are accounted for.
