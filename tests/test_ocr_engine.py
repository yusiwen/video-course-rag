"""Which inference engine the Linux OCR path uses, decided from the environment.

The engine choice has to be readable without importing rapidocr: macOS runs
Vision and has no rapidocr installed, so these tests must pass there too.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_rag.ocr import rapid  # noqa: E402

VARS = ("VIDEO_RAG_OCR_ENGINE", "VIDEO_RAG_OCR_DEVICE",
        "VIDEO_RAG_OCR_BATCH", "VIDEO_RAG_OCR_MODELS")


def _clean():
    for v in VARS:
        os.environ.pop(v, None)


def test_defaults_are_the_offline_cpu_engine():
    _clean()
    try:
        assert rapid.options() == ("onnxruntime", "cpu", None)
    finally:
        _clean()


def test_torch_defaults_to_a_batch_that_actually_helps():
    _clean()
    os.environ["VIDEO_RAG_OCR_ENGINE"] = "torch"
    os.environ["VIDEO_RAG_OCR_DEVICE"] = "cuda"
    try:
        engine, device, batch = rapid.options()
        assert (engine, device) == ("torch", "cuda")
        # batch 6 measured 1076 ms on a dense frame, batch 32 measured 343 ms
        assert batch == 32, batch
    finally:
        _clean()


def test_batch_can_be_overridden_and_reaches_the_cpu_engine():
    _clean()
    os.environ["VIDEO_RAG_OCR_BATCH"] = "16"
    try:
        assert rapid.options() == ("onnxruntime", "cpu", 16)
    finally:
        _clean()


def test_unknown_values_fail_loudly():
    _clean()
    try:
        os.environ["VIDEO_RAG_OCR_ENGINE"] = "paddle"
        try:
            rapid.options()
        except SystemExit as e:
            assert "paddle" in str(e)
        else:
            raise AssertionError("an unknown engine must not be accepted")
        os.environ.pop("VIDEO_RAG_OCR_ENGINE")
        os.environ["VIDEO_RAG_OCR_BATCH"] = "many"
        try:
            rapid.options()
        except SystemExit as e:
            assert "many" in str(e)
        else:
            raise AssertionError("a non-numeric batch must not be accepted")
    finally:
        _clean()


def test_model_cache_lives_in_the_home_cache_by_default():
    _clean()
    try:
        assert rapid.models_dir() == Path.home() / ".cache" / "video-course-rag" / "ocr"
        os.environ["VIDEO_RAG_OCR_MODELS"] = "/tmp/models-elsewhere"
        assert rapid.models_dir() == Path("/tmp/models-elsewhere")
    finally:
        _clean()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} ok")
