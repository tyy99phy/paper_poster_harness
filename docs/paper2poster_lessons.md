# Paper2Poster 吸收策略与后续架构

本项目不直接复刻 Paper2Poster 的 PPTX 生成路线。Paper2Poster 的核心优点在于：

1. **Parser / storyboard**：把论文压缩成章节角色、关键 claim、读者应能回答的问题。
2. **Asset matching**：把真实图片按语义分配给章节，而不是随便挑图。
3. **Painter–Commenter loop**：用视觉反馈检查 overflow、空白、对齐。
4. **PaperQuiz 思路**：海报不仅要好看，还要能回答论文核心问题。

我们当前采用的是 placeholder-first 生图路线：生图模型负责艺术模板和占位符，真实科学图由 harness 确定性贴回。进一步吸收 Paper2Poster 时，保留这一路线，只把它的信息规划能力内化到 prompt/spec/QA 中。

## 当前已经吸收的部分

- `storyboard_from_text()`：生成叙事主线、章节角色、阅读顺序、hero section、读者问题。
- `select_figures()`：根据真实素材选择 hero/result/method/context 图，并保持源图长宽比。
- placeholder QA + final QA：检查占位符、图片 containment、最终文本泄漏。
- 新增 `information_plan`：记录 `density_target`、`data_badges`、`display_facts`、`must_answer_questions`、`visual_story_units`。
- 生图 prompt 新增 **INFORMATION DENSITY TARGET**：要求不是稀疏封面图，而是信息丰富但可读的会议海报。
- `template_critic`：在贴真实图之前评价整张占位符模板的信息量、艺术性、文字质量和 placeholder 合规性；不合格时把 critique 注入 prompt 重新生成完整 poster。

## 为什么不建议直接多次生图后拼接整块 poster

多次调用 image generation 可以提高局部艺术性，但如果每次生成一个独立 poster block，再硬拼成整图，会带来几个问题：

- 全局风格、光源、透视、字体、边距不一致；
- 拼接边界会像 collage，违背当前“不拼贴”的核心目标；
- 每块图都可能生成自己的文字和占位符，placeholder detection/QA 难度显著上升；
- 多块之间的 reading order、hero hierarchy、gutter 和 figure containment 很难统一。

因此推荐的多次生图方式不是“分块生成整块内容”，而是：

1. **全局模板一次生成**：仍然生成完整海报 layout + placeholder，主标题、section 文案、badge、summary 等文字仍交给生图模型排版。
2. **LLM/视觉 QA 反馈再生图**：如果信息量不足、艺术性不足或 placeholder 不理想，用同一份 storyboard / information_plan / placeholder contract 生成下一版完整 poster，而不是把旧图裁成几块拼接。
3. **可选局部艺术素材**：只在需要时生成无文字、无科学内容的 header art、背景纹理、光效、detector ring 等透明或可遮罩装饰层；这些层不得承载科学文字或真实数据。

这样可以保留生图模型在文字—空间—艺术整体排版上的优势，同时不牺牲 placeholder 合约。

## 关于 PSD / 多图层

网页端可能展示或导出某种“可编辑/分层”体验，但稳定的框架不应依赖网页端专属 UI 能力。对仓库而言，更稳妥的做法是自己定义 layered artifact：

```text
exports/layered/
├── base_template.png          # 生图模型生成的全局底图
├── art_layers/                # 可选：header/background/ornament 透明装饰层
├── figures/                   # 真实论文图层
├── text_overlays.yaml         # 可确定性重绘的文字/徽章层
├── masks/placeholder_boxes.yaml
└── final.png
```

后续如果需要 Photoshop 友好格式，可以从这些 layer 导出 PSD/ORA/TIFF，但核心 harness 不应把 PSD 当成模型输出的前提。

重要原则：**主文字默认由生图模型原生排版**。确定性文字 overlay 只作为极端情况下的局部修复/打补丁工具，不作为推荐主路线；否则会损失生图模型在 typography、空间节奏、文字与艺术背景融合方面的优势。

## 推荐的下一阶段路线

1. **信息密度增强**：让 storyboard 生成 PaperQuiz-lite 信息计划，并注入 prompt；已完成第一版。
2. **信息量—艺术性迭代生图**：新增 critic→regen loop，让 VLM/LLM 评价“是否足够信息丰富、是否像高质量 HEP poster、placeholder 是否合规”，然后带着 critique 重新生成完整 poster。
3. **局部艺术层生成**：可选新增 `art_layers` 阶段，只生成无文字装饰图，再与模型原生文字模板轻量合成。
4. **Panel-level QA**：借鉴 Paper2Poster 的 zoom-in commenter，对每个 figure card、文本模块、badge 信息量做局部检查。
5. **Layer package export**：输出可编辑 layer package，必要时再转 Photoshop/ORA 兼容格式；但不依赖模型直接输出 PSD。
