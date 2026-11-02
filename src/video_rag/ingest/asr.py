#!/usr/bin/env python3
"""P0: FunASR (paraformer-zh + fsmn-vad + ct-punc) transcription with timestamps."""
from ..paths import run as _run, runs_root, mkdirs
import json
import os
import sys
import time


def fmt_ts(ms: float) -> str:
    h, rem = divmod(int(round(ms)), 3600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


def write_srt(items, path):
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(f"{i}\n{fmt_ts(it['start'])} --> {fmt_ts(it['end'])}\n{it['text']}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def device() -> str:
    """Where FunASR runs: ``VIDEO_RAG_ASR_DEVICE`` (cpu/cuda/mps), else automatic.

    macOS keeps CPU: FunASR's MPS path is unreliable and short clips transcribe
    fast enough. Anywhere else, use the GPU if torch can see one — a 90-minute
    lecture is a 15-minute CPU job versus a minute on a Blackwell card.
    """
    want = os.environ.get("VIDEO_RAG_ASR_DEVICE", "").strip().lower()
    if want:
        return want
    if sys.platform == "darwin":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    from funasr import AutoModel

    if len(sys.argv) < 2:
        raise SystemExit("usage: vrag ingest <run-id> [video]   (or call asr.py <run-id> [wav])")
    run = _run(sys.argv[1])
    wav = sys.argv[2] if len(sys.argv) > 2 else str(run.audio)
    out_prefix = str(run.root / "asr")

    dev = device()
    t0 = time.time()
    model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        punc_model="ct-punc",
        device=dev,
        disable_update=True,
        disable_pbar=True,
    )
    print(f"[funasr] device={dev} model loaded in {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    res = model.generate(input=wav, batch_size_s=300, hotword="")
    elapsed = time.time() - t1
    print(f"[funasr] generate done in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    with open(f"{out_prefix}_raw.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)

    r = res[0]
    sentences = r.get("sentence_info") or []
    items = []
    if sentences:
        for s in sentences:
            items.append(
                {
                    "start": float(s.get("start", 0)),
                    "end": float(s.get("end", 0)),
                    "text": str(s.get("text", "")).strip(),
                }
            )
    else:
        # fallback: single block, no sentence split
        items.append({"start": 0.0, "end": 0.0, "text": r.get("text", "")})

    # funasr timestamps are milliseconds -> srt writer expects ms, fine as-is
    write_srt(items, f"{out_prefix}.srt")
    with open(f"{out_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(
            {"engine": "funasr/paraformer-zh", "elapsed_s": round(elapsed, 1), "items": items},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[funasr] {len(items)} sentence segments, {len(r.get('text',''))} chars, {elapsed:.1f}s audio-time ratio {elapsed/900:.3f}")


if __name__ == "__main__":
    main()
