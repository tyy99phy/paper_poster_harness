# 完整流程 run-record 语料示例

本目录是 `PosterHarness` run-record 格式的一个精简语料示例。它补充了 [`benchmarks/selected_12`](../../../benchmarks/selected_12) 中的视觉 benchmark：benchmark 保存配对海报图像，而这里保存一次完整 harness 运行背后的文本审计轨迹。

## 文件

- [`run_record.json`](run_record.json)：机器可解析的语料单元，也是事实源。
- [`run_record.md`](run_record.md)：由同一 JSON 渲染得到的人类可读审计日志。
- [`README.md`](README.md)：英文说明。

这里不包含 PNG、PDF、PPTX、提取出的源论文图表或其他二进制运行产物。保留的只是 workflow 文本：阶段摘要、提示词、结构化规划输出、占位符检测、QA/critic 判断、出处账本、客观检查、model-judge 分数，以及确定性后处理说明。

## 来源 run

该示例来自一轮针对 arXiv `2206.08956`（CMS same-sign WW / heavy Majorana neutrino）的 PosterHarness 运行。选择它是因为这轮 run 包含较完整的 trace：LLM envelope、图像生成 prompt、模板 critic、placeholder QA、containment QA、final QA、图表出处核算，以及一次局部的确定性 FIG 01 位置微调。

发布前已做脱敏与归一化：

- 本地绝对路径替换为 `<run>`、`<source-run>`、`<tmp>` 等占位符；
- 账户标识替换为 `<redacted-account>`；
- 二进制产物只保留引用，不随本目录发布。

## 建议阅读顺序

1. **Run card** 与 **Capability stages**：查看 harness 尝试了什么、哪些阶段通过。
2. **QA / failure accounting**：查看模板 critic、placeholder QA、final QA 和 containment 报告。
3. **Provenance ledger**：查看每个 placeholder 对应哪张真实源图，以及声明宽高比是否匹配检测几何。
4. **Complete workflow trace**：查看嵌入的提示词、规划 YAML、LLM envelope、检测 YAML、placement spec 和 QA 判断。
5. **Model-judge scores**：查看主观评分；注意这些评分不是 ground truth。

## 范围与注意事项

这是一个语料示例，不是额外 benchmark 样本；不要把它作为 selected-12 之外的第 13 篇论文统计。它的作用是展示完整、可审计的 run record 长什么样，并为检查 prompt contract、模型判断和确定性检查提供一个可复用文本语料。

对新的 run 生成同类记录：

```bash
poster-harness record runs/your-run-directory
```
