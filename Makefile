PY ?= python3
export PYTHONPATH := src

.PHONY: help test demo leak lint install clean
help:
	@echo "make test    run the test suite (no dependencies beyond python3)"
	@echo "make leak    scan for recorded-material leakage"
	@echo "make demo    generate a synthetic lecture and index it"
	@echo "make install install into the current venv (pip install -e .)"

test:
	$(PY) tests/run_all.py -v

leak:
	$(PY) tools/check_no_course_data.py

demo:
	bash examples/make_demo.sh

install:
	$(PY) -m pip install -e .

clean:
	rm -rf build dist *.egg-info examples/slides examples/*.mp4 examples/*.wav examples/runs
