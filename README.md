# Paper Poster Harness

把一篇学术论文自动生成会议海报的框架。核心流程是：先让 LLM/生图模型生成带占位符的海报版式，再把论文中的真实图片确定性替换进去，避免生图模型伪造科学图表。

## 流程

```text
读取 config
→ 用 LLM web_search 定位 arXiv / 或直接使用本地论文
→ 下载 PDF 和 source
→ 提取论文文本与图片素材
→ LLM 生成 poster_spec.yaml
→ LLM 选择真实图片并分配 FIG 占位符
→ 生图模型生成只含占位符的海报
→ LLM 检测占位符位置
→ 占位符 QA
→ 插入真实论文图片
→ 4× 放大
→ 最终 QA
```

框架默认是严格模式：任何 LLM/API/生图/QA 步骤失败都会直接报错，不会走 fallback，也不会用旧海报裁剪拼贴。

## 安装

```bash
git clone https://github.com/tyy99phy/paper_poster_harness.git
cd paper_poster_harness
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

也可以源码运行：

```bash
export PYTHONPATH=$PWD
python -m poster_harness --help
```

## 初始化配置

推荐使用本地 ChatGPT 账号登录：

```bash
poster-harness init-config --out poster_harness.yaml --login
```

这会打开浏览器完成 OpenAI/ChatGPT 登录，并在本地写入账号认证文件：

```text
~/.config/poster-harness/auth/chatgpt-<email>.json
```

后续运行会自动读取该文件。也可以手动指定：

```bash
export POSTER_HARNESS_AUTH_FILE=/absolute/path/to/account-auth.json
```

默认配置示例：

```yaml
llm:
  backend: chatgpt_account
  model: gpt-5.5
  account:
    auth_dir: ~/.config/poster-harness/auth
    account: ""
    auth_file: ""

image_generation:
  backend: chatgpt_account
  model: gpt-5.5
  size: 1024x1536
  quality: high
  variants: 1
  generated_scale: 4.0
  upscale_factor: 4.0
  account:
    auth_dir: ~/.config/poster-harness/auth
    account: ""
    auth_file: ""
```


## 生成海报

### 从 arXiv 查询生成

在 `poster_harness.yaml` 中填写：

```yaml
paper:
  query: "arXiv 2206.08956 heavy Majorana neutrino CMS"
  out: runs/majorana-neutrino-poster
```

运行：

```bash
poster-harness autoposter --config poster_harness.yaml
```

### 从已知 arXiv ID 生成

```bash
poster-harness autoposter \
  --config poster_harness.yaml \
  --arxiv-id 2206.08956 \
  --out runs/2206_08956
```

### 从本地论文和素材生成

```bash
poster-harness autoposter \
  --config poster_harness.yaml \
  --paper /path/to/paper.pdf \
  --assets-dir /path/to/source/figures \
  --out runs/my-paper
```

如果 PDF 文本提取不够好，可以额外指定 TeX：

```bash
poster-harness autoposter \
  --config poster_harness.yaml \
  --paper /path/to/paper.pdf \
  --text-source /path/to/main.tex \
  --assets-dir /path/to/figures \
  --out runs/my-paper
```

## 输出文件

```text
runs/<run>/generated/*placeholder-layout.png      # 生图模型生成的占位符海报
runs/<run>/exports/*realfigures.png               # 替换真实论文图片后的海报
runs/<run>/exports/*realfigures-4x.png            # 4× 放大版
runs/<run>/qa/*.qa.yaml                           # QA 结果
runs/<run>/run_manifest.yaml                      # 运行记录
```

## 占位符规则

生图模型不能绘制科学图表。所有论文图片区域必须是空白矩形占位符，只包含：

1. 精确编号，例如 `[FIG 03]`；
2. 图片内容标签；
3. 目标长宽比。

占位符 QA 通过后，框架才会把真实论文图片插入对应区域。

## 常用子命令

```bash
poster-harness login
poster-harness resolve-arxiv --config poster_harness.yaml --query "..." --out resolution.yaml
poster-harness manifest --assets-dir figures --copy-to runs/x/assets --out runs/x/specs/assets_manifest.yaml
poster-harness prompt --spec runs/x/specs/poster_spec.yaml --out runs/x/prompts/poster_prompt.txt
poster-harness generate --config poster_harness.yaml --prompt runs/x/prompts/poster_prompt.txt --out-dir runs/x/generated
poster-harness replace --base-image runs/x/generated/poster-placeholder-layout.png --spec runs/x/specs/poster_spec.yaml --asset-dir runs/x/assets --out runs/x/exports/poster-realfigures.png
poster-harness upscale --input runs/x/exports/poster-realfigures.png --out runs/x/exports/poster-realfigures-4x.png --factor 4
```
