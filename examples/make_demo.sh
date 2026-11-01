#!/usr/bin/env bash
# Build a synthetic lecture, so the pipeline can be exercised without any real
# recording (and without shipping someone else's material).
#
#   bash examples/make_demo.sh [zh|en]
#
# Needs: ffmpeg, ImageMagick 7 (magick), macOS `say` for the narration.
# Produces examples/demo.mp4 — three slides, each with a spoken paragraph.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LANG_CODE=${1:-zh}
OUT="$ROOT/examples"
SLIDES="$OUT/slides"
mkdir -p "$SLIDES"
FONT_ZH="/System/Library/Fonts/STHeiti Light.ttc"

if [ "$LANG_CODE" = "en" ]; then
  VOICE=${VOICE:-Samantha}
  S1_TITLE="Chapter 3 · Data sources";   S1_A="A data source is a named connection to a database."
  S2_TITLE="Chapter 3 · Binding fields"; S2_A="Each field on the form is mapped to a column of the source."
  S3_TITLE="Chapter 3 · Verify";         S3_A="Open the page, submit a record, then check the list."
else
  VOICE=${VOICE:-Tingting}
  S1_TITLE="第三章 数据源";   S1_A="第三章讲数据源。数据源就是一条到数据库的命名连接。"
  S2_TITLE="第三章 字段绑定"; S2_A="表单上的每一个字段，都要映射到数据源里的某一列。"
  S3_TITLE="第三章 验证";     S3_A="打开页面提交一条记录，然后回到列表里检查结果。"
fi

slide() {  # slide <file> <title> <line>
  magick -size 1280x720 xc:white -font "$FONT_ZH" -pointsize 46 -fill '#1b2733' \
    -gravity north -annotate +0+70 "$2" \
    -pointsize 34 -fill '#33404d' -gravity northwest -annotate +90+260 "$3" "$1"
}
slide "$SLIDES/s1.png" "$S1_TITLE" "$S1_A"
slide "$SLIDES/s2.png" "$S2_TITLE" "$S2_A"
slide "$SLIDES/s3.png" "$S3_TITLE" "$S3_A"

# one narration file per slide, then concatenate: the timing is recorded as we go
i=1
: > "$OUT/parts.txt"
: > "$OUT/asr_parts.jsonl"
t=0.0
for line in "$S1_A" "$S2_A" "$S3_A"; do
  say -v "$VOICE" -o "$OUT/n$i.aiff" "$line"
  d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT/n$i.aiff")
  printf "file 'n%d.aiff'\n" "$i" >> "$OUT/parts.txt"
  python3 - "$OUT/asr_parts.jsonl" "$line" "$t" "$d" <<'PY'
import json, sys
path, text, t0, dur = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps({"start": round(t0*1000), "end": round((t0+dur)*1000), "text": text},
                       ensure_ascii=False) + "\n")
PY
  t=$(python3 -c "print($t + $d)")
  i=$((i+1))
done

ffmpeg -v error -y -f concat -safe 0 -i "$OUT/parts.txt" -c:a pcm_s16le "$OUT/narration.wav"
D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT/narration.wav")
# three slides spread evenly over the narration
SL=$(python3 -c "print($D/3)")
ffmpeg -v error -y -loop 1 -t "$SL" -i "$SLIDES/s1.png" \
  -loop 1 -t "$SL" -i "$SLIDES/s2.png" -loop 1 -t "$SL" -i "$SLIDES/s3.png" \
  -i "$OUT/narration.wav" \
  -filter_complex "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]" -map "[v]" -map 3:a \
  -c:v libx264 -pix_fmt yuv420p -r 10 -c:a aac -shortest "$OUT/demo.mp4"

# the transcript is known exactly here: it is the narration, with measured times
python3 - "$OUT/asr_parts.jsonl" "$OUT" <<'PY'
import json, sys, pathlib
parts, out = sys.argv[1], pathlib.Path(sys.argv[2])
items = [json.loads(l) for l in open(parts, encoding="utf-8")]
d = out / "asset_asr_sent.json"
d.write_text(json.dumps({"engine": "synthetic narration (demo)", "items": items},
                        ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{len(items)} sentences -> {d}")
PY

rm -f "$OUT"/n?.aiff "$OUT/parts.txt" "$OUT/asr_parts.jsonl"
echo "examples/demo.mp4 ($(du -h "$OUT/demo.mp4" | cut -f1))"
echo "next: vrag init demo examples/demo.mp4 && cp examples/asset_asr_sent.json runs/demo/asr_sent.json"
