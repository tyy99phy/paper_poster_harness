# Paper Poster Harness

[English](../../README.md) · [简体中文](../../README.zh-CN.md) · [日本語](README.ja.md) · [Español](README.es.md)

Paper Poster Harness は、論文から学会ポスターを生成するための **placeholder-first** な LLM + 画像生成パイプラインです。

重要な方針は、画像生成モデルに美しいレイアウトを作らせても、科学的な図表を捏造させないことです。モデルは `[FIG 01]` のような空白プレースホルダーを含むテンプレートを作成し、ハーネスが論文由来の実図を決定的に差し替えます。

## Quick start

```bash
git clone https://github.com/tyy99phy/paper_poster_harness.git
cd paper_poster_harness
pip install -e .
poster-harness init-config --out poster_harness.yaml --login
poster-harness autoposter --config poster_harness.yaml --arxiv-id 2206.08956 --out runs/demo
```

HEP 向けの高密度モード:

```bash
poster-harness autoposter --config poster_harness.yaml --arxiv-id 2206.08956 --content-mode hep_dense --out runs/demo-dense
```

詳細は英語版 README を参照してください。
