# Quality policy

Known issue: the current local image-generation route typically returns a 1024×1536 portrait poster. This is good enough for layout exploration but not enough for final poster printing when zoomed.

Recommended tiers:

1. **Layout exploration**: 1024×1536, `quality=high`, placeholders only.
2. **Review export**: replace real figures first, then upscale to 4096×6144 with `poster-harness upscale --factor 4`.
3. **Print-final rebuild**: reconstruct the selected image-generated design in deterministic HTML/SVG/PPTX. Keep image-generation art as background/texture only; make all text and vector layout native.

Do not use model-generated plots for scientific content at any tier.

## Automation quality gates

The `autoposter` command writes a `run_manifest.yaml` plus QA artifacts. A run is
not considered mature unless:

- `poster_spec.yaml` contains only public-facing text;
- all selected placeholders have real source assets;
- placeholder detections produce nonzero boxes with plausible confidence;
- replacement output exists before high-resolution upscaling;
- no conclusion bullet mentions placeholders, replacement, validation workflow, or internal production notes.

If a generated layout has a typo in public text, prefer regenerating with a
stronger prompt or rebuilding the typography deterministically. Avoid broad
"polish" passes that damage the image-generated design.
