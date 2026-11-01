#!/usr/bin/env python3
"""Zero-dependency test runner.

The suite exists to pin invariants, not to exercise a framework: each module
exposes ``test_*()`` functions, a failure prints where it happened and the runner
exits non-zero. ``pytest tests`` works too, if you prefer.

    python3 tests/run_all.py [-v]
"""
import importlib.util
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))


def load(path):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    verbose = "-v" in sys.argv
    files = sorted(f for f in os.listdir(HERE) if f.startswith("test_") and f.endswith(".py"))
    passed = failed = skipped = 0
    for f in files:
        mod = load(os.path.join(HERE, f))
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                out = fn()
            except Exception:
                failed += 1
                print(f"FAIL {f}::{name}")
                traceback.print_exc()
                continue
            if out == "skip":
                skipped += 1
                if verbose:
                    print(f"skip {f}::{name}")
            else:
                passed += 1
                if verbose:
                    print(f"ok   {f}::{name}")
    print(f"{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
