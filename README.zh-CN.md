# video-course-rag

[![ci](https://github.com/yusiwen/video-course-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/yusiwen/video-course-rag/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)](#linux没有-apple-vision)
[![status](https://img.shields.io/badge/status-alpha-orange.svg)](docs/limitations.zh-CN.md)

[English](README.md) · **简体中文**

把**录播课**整理成学生双击一个 HTML 就能用的**课程包**：不用起服务、不用装 Python、不用账号。

状态：**alpha**。构建端只在 macOS 上验证过（OCR 走 Apple Vision）；语音识别中文优先。
打算用在自有素材上前，先看 `docs/limitations.zh-CN.md`。

## 产出什么

```
runs/<id>/                 一门课一个目录，所有中间产物都在里面
  asr_sent.json            逐句转写（绝对毫秒）
  profile.json             这门课实测出来的标题带几何
  chapters_fused.json      最终交付的目录
  chunks.jsonl             检索单元
  index/                   向量与页面元数据
  page/index.html          这门课单独的页面
dist/<包名>/index.html     交付物（所有课合成一个文件，6 小时约 1.5 MB）
```

发行包**不含视频**：页面按**内容指纹**（大小 + 首尾 4MB 的 SHA-1）匹配视频，
改名、移动都不影响；学生选一次目录，页面自己认出来。

## 工作原理（一段话）

音频转写；画面切成**单元**（一张幻灯片停住就算一个单元）；每个单元做 OCR；
标题取自某个**实测**出来的带（`vrag title` 会先标定，而不是假设位置）。
章节从逐句转写里切（LLM 只按句子编号提议边界，本地先修掉时长违规，修不动才重问），
再吸附到最近的画面切换点。检索是「讲师原话 BM25 + 章节标题 BM25」的
倒数排名融合；页面上加粗的是讲师原话，屏幕文字另起一行淡显。

## 快速开始

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[asr]"          # 需要转写才装 funasr + torch
swiftc -O native/ocr/ocr.swift -o native/ocr/ocr

make demo                        # 生成一段合成讲座，不需要真实素材
vrag --runs-root examples/runs title demo
vrag --runs-root examples/runs page
open examples/runs/page.html
```

用自己的录像：

```bash
vrag init lecture-1 /path/to/recording.mp4
vrag ingest lecture-1            # 音频 -> 转写 -> 画面单元
vrag title  lecture-1            # 标定标题带，再给单元命名
vrag chapter lecture-1           # 逐句转写切章，与画面融合
vrag index  lecture-1            # 检索单元、术语表、向量
vrag audit  lecture-1            # 目录体检：该不该发出去
vrag page                        # 或：vrag pack --out dist/course
```

`vrag audit` 是闸门：目录里中性标签过多、外壳文字过多、缺标题过多，都会让它失败。
体检不过的 run 不应该发出去。

### Linux（没有 Apple Vision）

macOS 上画面读字用 Vision（`swiftc` 现编一个二进制）；Linux 上用 RapidOCR，
模型随 wheel 附带，运行时不需要联网下载：

```bash
sudo apt install ffmpeg
python3 -m venv .venv && . .venv/bin/activate
uv pip install -e ".[asr,ocr-linux]"

# 有 NVIDIA 显卡时，torch 换成 CUDA 轮子（版本要和本机驱动匹配）
uv pip install --index-strategy unsafe-best-match \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  "torch==2.11.0+cu130" "torchaudio==2.11.0+cu130"
```

两条环境变量：

| 变量 | 作用 |
|---|---|
| `VIDEO_RAG_OCR` | `vision` / `rapidocr`，默认按平台选 |
| `VIDEO_RAG_ASR_DEVICE` | `cpu` / `cuda` / `mps`，默认 macOS 用 cpu、其余有 CUDA 就用 |
| `VIDEO_RAG_OLLAMA` | embedding 服务地址，默认 `localhost:11434` |

#### 更快的画面读字（可选）

同一套 PP-OCR 模型可以走 torch 而不是 onnxruntime：

```bash
export VIDEO_RAG_OCR_ENGINE=torch   # 默认 onnxruntime
export VIDEO_RAG_OCR_DEVICE=cuda
```

不用多装包——`asr` extra 里已经有 torch。代价是首次调用要把 torch 权重和字符字典
（约 32 MB）下到 `~/.cache/video-course-rag/ocr`（`VIDEO_RAG_OCR_MODELS` 可改路径，
`VIDEO_RAG_OCR_BATCH` 改识别批大小），因为 torch 权重里既没有字典，也没有 onnxruntime
能读的那些元数据。

同一门 92 分钟课程、同样的帧，实测：

| | onnxruntime / CPU | torch / CUDA |
|---|---|---|
| 画面分段（`states`） | 22 分钟 | **8.5 分钟** |
| 一张 121 行的 1080p 画面 | 2550 ms | 343 ms |
| 仅检测（det） | 317 ms | 130 ms |

差距由两个数字解释：识别占一张密集画面的 **87%**，而默认批大小 6 意味着 121 行要跑
约 21 次小前向。torch 引擎默认把批设成 32，**单这一项就能把 CPU 引擎从 2550 ms 降到
1495 ms**——所以即使没有 GPU，`VIDEO_RAG_OCR_BATCH` 也值得设。

换引擎会让边缘 OCR 文本变化（工具栏标签里多一个空格、把一个图标读成字符），而合并规则
正是"相邻单元标题相同则合并"：这门课的画面单元从 259 变成 249。但单元边界与模式没有
移动（243/243 在 ±2 秒内），而交付的章标题来自 LLM 切章表，不依赖这些单元标题。

实测（同一段 92 分钟课程，Mac 用 Vision+CPU，Spark 用 RapidOCR+CUDA）：

- 转写快 18 倍（92 分钟音频 19.7 秒），文字与 macOS 结果相似度 0.9998
- 切章 35 章全部对齐，标题 35/35 一致；前提是切章表带上了 `start_text`
- 画面分段换成 CUDA 后只慢 **1.1 倍**（同一门课 8.5 分钟 vs 7.5 分钟）；CPU 引擎上是
  2.9 倍（22 分钟）。这一阶段里 OCR 占 66%、ffmpeg 抓帧占 14%，它是最大开销，但不是唯一开销
- `chapter` 从 17 分钟降到 **21 秒**：原来每个单元都要扫最多 9 帧去找跳跃点，包括 246 个
  单元里那 231 个没有标题的——它们的匹配恒为 0，扫完还是返回原地
- Linux 侧校准常常找不到标题带，单元标题会退化成中性标题。这是有意的：
  给控件命名比承认没名字更糟；章标题主要由 LLM 切章表提供，不依赖它

## 已知限制（alpha）

- **两个平台都能构建**：macOS 用 Vision 读画面，Linux 用 RapidOCR；两者结果不逐字
  相同，画面分段在 Linux 上约慢 1.1 倍（CPU 引擎 2.9 倍），单元标题质量也更弱（见上面「Linux」一节）。
- **中文优先**：断句、标题启发式和界面文案都是按中文讲座设计的；英文素材未实测，
  当前标题过滤会丢弃非 CJK 行。
- **检索是词面匹配**：页面里没有查询编码器，只能匹配讲师说过的词和章节标题，
  换个说法就未必命中；`vrag ask`（填自己的 key）会读检索到的片段来回答。
- **页面就是一个文件**：这是刻意的——6 小时约 1.5 MB，120 小时首屏仍在 0.3 秒内。

## 许可

MIT。仓库不含任何录制素材：`runs/` 被忽略，CI 里有泄漏扫描。
