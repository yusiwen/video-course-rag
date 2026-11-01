# video-course-rag

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

## 已知限制（alpha）

- **构建端必须是 macOS**：OCR 助手是 Swift + Vision，Linux 标准栈里没有等价物；
  语音识别本身跨平台。
- **中文优先**：断句、标题启发式和界面文案都是按中文讲座设计的；英文素材未实测，
  当前标题过滤会丢弃非 CJK 行。
- **检索是词面匹配**：页面里没有查询编码器，只能匹配讲师说过的词和章节标题，
  换个说法就未必命中；`vrag ask`（填自己的 key）会读检索到的片段来回答。
- **页面就是一个文件**：这是刻意的——6 小时约 1.5 MB，120 小时首屏仍在 0.3 秒内。

## 许可

MIT。仓库不含任何录制素材：`runs/` 被忽略，CI 里有泄漏扫描。
