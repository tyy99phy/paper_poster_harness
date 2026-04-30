# Paper Poster Harness

把一篇学术论文自动生成会议海报的框架。

核心思路：**LLM 只负责版式设计，不碰科学内容。** 生图模型先画出带空白占位符的海报，框架再把论文里的真实图片确定性替换进去——这样既享受了 AI 的设计能力，又杜绝了模型伪造科学图表的问题。

## 前提条件

- Python 3.10+
- 一个 OpenAI/ChatGPT 账号（用于 LLM 推理和图片生成）
- （可选）arXiv 论文需要网络访问

## 快速开始

```bash
# 1. 安装
git clone https://github.com/tyy99phy/paper_poster_harness.git
cd paper_poster_harness
pip install -e .

# 2. 登录 ChatGPT 账号（打开浏览器完成 OAuth）
poster-harness init-config --out poster_harness.yaml --login

# 3. 从 arXiv 论文一键生成海报
poster-harness autoposter \
  --config poster_harness.yaml \
  --arxiv-id 2206.08956 \
  --out runs/my-poster
```

跑完后在 `runs/my-poster/exports/` 下找最终海报。

## 为什么是"占位符优先"

AI 图片模型有一个致命问题：它会"发明"看起来很专业的科学图表——但这些图表的曲线、数据、误差棒全是假的。对于学术海报来说，这是不可接受的。

这个框架的做法是：

| 环节 | 谁来做 | 原则 |
|------|--------|------|
| 版式设计、配色、排版 | 生图模型 | 充分发挥 AI 的审美能力 |
| 科学图表（曲线、散点图、费曼图等） | 确定性替换 | 只插入论文原始图片，绝不让 AI 画数据 |
| 文字内容 | LLM + 过滤 | 从论文提取，LLM 压缩成海报文案，自动过滤内部/工作流用语 |

流程中每一步失败都会直接报错（严格模式），不会悄悄降级或拿旧结果拼贴。

## 工作流程

```text
arXiv / 本地论文
  → 提取文本 + 图片素材
  → LLM 起草 poster_spec（包含版式描述、章节、占位符规格）
  → LLM 从素材中挑选最有价值的图片，分配给 [FIG 01]、[FIG 02] ...
  → 组装完整 prompt，发给生图模型
  → 生图模型画海报（只含空白占位符，不画科学内容）
  → LLM 视觉检测每个占位符的像素坐标
  → 占位符 QA（检查是否空白、是否可读、比例是否正确）
  → 将真实论文图片按坐标插入
  → 4× 超分放大
  → 最终 QA（检查文字是否公开、图片是否正常）
```

每一步的产物（spec、manifest、prompt、QA 报告）都会保存到 `runs/<run>/`，方便审阅和调试。

## 配置说明

完整配置模板见 `templates/poster_harness_config.yaml`。主要板块：

### LLM 后端

```yaml
llm:
  backend: chatgpt_account    # 目前唯一支持的后端
  model: gpt-5.5
  timeout: 180
  account:
    auth_dir: ~/.config/poster-harness/auth   # 登录凭证存放目录
```

`account` 留空会自动发现本地认证文件，也可用 `POSTER_HARNESS_AUTH_FILE` 环境变量指定。

### 图片生成

```yaml
image_generation:
  backend: chatgpt_account
  model: gpt-5.5
  size: 1024x1536
  quality: high
  variants: 3                # 生成几个候选
  generated_scale: 4.0       # 生成后立即超分的倍数
  upscale_factor: 4.0        # 最终导出放大倍数
```

默认生成 3 个候选模板，只导出通过严格占位符 QA 的版本。

### 样式预设

```yaml
autoposter:
  style: cms-hep   # 或 generic
```

- **`cms-hep`**：CMS/CERN 风格，深色标题栏、探测器艺术抽象、非对称 HEP 版式
- **`generic`**：通用学术海报风格

可以在 `styles` 段下自定义或添加新的样式预设。

## 使用方式

### 方式一：一键生成（推荐）

```bash
# 按关键词搜索 arXiv
poster-harness autoposter --config poster_harness.yaml \
  --query "heavy Majorana neutrino CMS VBS same-sign WW"

# 已知 arXiv ID，跳过搜索
poster-harness autoposter --config poster_harness.yaml \
  --arxiv-id 2206.08956

# 本地论文
poster-harness autoposter --config poster_harness.yaml \
  --paper paper.pdf --assets-dir figures/ \
  --text-source main.tex
```

### 方式二：分步执行

当你想审阅或调整中间产物时，可以逐步运行：

```bash
# 第 1 步：登录（只需一次）
poster-harness login

# 第 2 步：定位 arXiv 论文
poster-harness resolve-arxiv --config poster_harness.yaml \
  --query "..." --out resolution.yaml

# 第 3 步：扫描本地图片素材
poster-harness manifest --assets-dir figures/ \
  --copy-to runs/my-poster/assets/ \
  --out runs/my-poster/specs/assets_manifest.yaml

# 第 4 步：生成 prompt（可在此编辑后再继续）
poster-harness prompt \
  --spec runs/my-poster/specs/poster_spec.yaml \
  --out runs/my-poster/prompts/poster_prompt.txt

# 第 5 步：生成海报
poster-harness generate --config poster_harness.yaml \
  --prompt runs/my-poster/prompts/poster_prompt.txt \
  --out-dir runs/my-poster/generated/

# 第 6 步：替换真实图片
poster-harness replace \
  --base-image runs/my-poster/generated/poster-placeholder-layout.png \
  --spec runs/my-poster/specs/poster_spec.yaml \
  --asset-dir runs/my-poster/assets/ \
  --out runs/my-poster/exports/poster-realfigures.png

# 第 7 步：超分导出
poster-harness upscale \
  --input runs/my-poster/exports/poster-realfigures.png \
  --out runs/my-poster/exports/poster-realfigures-4x.png \
  --factor 4
```

每一步的结果都写入文件，不会因为你跳步或重跑而丢失之前的数据。

## 输出文件结构

```text
runs/<run>/
├── input/                  # 下载/复制的论文原文
├── assets/                 # 复制的图片素材 + contact sheet
├── generated/              # 生图模型产出的占位符海报
│   ├── *-native.png        #   模型原生输出
│   └── *placeholder-layout.png  #   超分后的占位符海报
├── exports/                # 替换真实图片后的最终海报
│   ├── *realfigures.png    #   原始尺寸
│   └── *realfigures-4x.png #   4× 放大版
├── specs/                  # 各阶段的 spec/selection/manifest
├── prompts/                # 组装好的生图 prompt
├── qa/                     # 占位符 QA + 最终 QA 报告
├── scratch/                # 中间检测结果
└── run_manifest.yaml       # 完整运行记录
```

## 占位符规则

生图模型被严格要求：**凡是论文图片区域，必须画成空白矩形占位符**。每个占位符只能包含三样东西：

1. 精确编号：`[FIG 03]`
2. 内容标签（如 "Observed 95% CL exclusion limit"）
3. 目标宽高比（如 "1:1 square" 或 "2.5:1 wide"）

不允许画任何科学内容——曲线、坐标轴、图例、费曼线、热力图、表格、缩略图，统统不行。占位符 QA 会逐项检查，不通过就不会进入替换阶段。

## 常见问题

### 登录失败 / token 过期

认证文件在 `~/.config/poster-harness/auth/` 下。重新登录即可刷新：

```bash
poster-harness login --force
```

### 文本提取太短（"extracted text is too short"）

PDF 的文本层可能不完整。最可靠的做法是直接给 TeX 源码：

```bash
poster-harness autoposter --config poster_harness.yaml \
  --paper paper.pdf --text-source main.tex
```

### 占位符 QA 不通过

说明生图模型在占位符里画了不该画的东西，或者占位符编号不清晰。可以：
- 增大 `variants`（多生成几个候选）
- 调整 prompt（见 `docs/prompt_contract.md`）
- 降低 `min_detection_confidence`（但可能引入误检）

### 海报上的文字有内部/工作流用语

检查 `forbidden_phrases` 配置，把需要过滤的词加进去。`autoposter` 默认已经过滤了常见内部用语。

## 更多文档

- [prompt 合约说明](docs/prompt_contract.md) — 生图模型需要遵守的占位符规则
- [质量策略](docs/quality_policy.md) — 从初稿到打印级海报的质量分级
- [账号认证](docs/account_auth.md) — 认证文件的格式和刷新机制

## License

MIT
