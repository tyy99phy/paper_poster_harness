# Paper Poster Harness

[English](../../README.md) · [简体中文](../../README.zh-CN.md) · [日本語](README.ja.md) · [Español](README.es.md)

Paper Poster Harness es un flujo **placeholder-first** con LLM + generación de imágenes para convertir artículos científicos en pósteres de conferencia.

La idea principal: el modelo de imagen diseña la composición, pero no inventa figuras científicas. Primero genera un póster con marcadores vacíos como `[FIG 01]`; después el sistema detecta esos marcadores e inserta de forma determinista las figuras reales extraídas del artículo.

## Inicio rápido

```bash
git clone https://github.com/tyy99phy/paper_poster_harness.git
cd paper_poster_harness
pip install -e .
poster-harness init-config --out poster_harness.yaml --login
poster-harness autoposter --config poster_harness.yaml --arxiv-id 2206.08956 --out runs/demo
```

Modo denso para HEP:

```bash
poster-harness autoposter --config poster_harness.yaml --arxiv-id 2206.08956 --content-mode hep_dense --out runs/demo-dense
```

Consulte el README en inglés para la documentación completa.
