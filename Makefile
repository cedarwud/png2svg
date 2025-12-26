.PHONY: help pytest regress-fast-render regress-fast-convert regress-fast-hard-sample \
	regress-hard regress-real rebuild-inputs build-hard-assets dataset-sanity check-png

PYTHON ?= venv/bin/python
REGRESS_DATASET ?= datasets/regression_v0
HARD_DATASET ?= datasets/regression_hard_v1
REAL_MANIFEST ?= datasets/real_regression_v1/manifest.yaml

help:
	@echo "Targets:"
	@echo "  pytest                  Run the test suite"
	@echo "  regress-fast-render     Run fast tier render regression"
	@echo "  regress-fast-convert    Run fast tier convert regression (input.png)"
	@echo "  regress-fast-hard-sample Run 3-case hard-input sample regression"
	@echo "  regress-hard            Run hard tier convert regression"
	@echo "  regress-real            Run real regression (requires REAL_PNG_DIR=...)"
	@echo "  rebuild-inputs          Rebuild FAST/HARD inputs from expected.svg"
	@echo "  build-hard-assets       Rebuild hard-tier expected assets"
	@echo "  dataset-sanity          Check dataset sanity"
	@echo "  check-png               Validate PNG magic headers"

pytest:
	$(PYTHON) -m pytest

regress-fast-render:
	$(PYTHON) tools/regress.py $(REGRESS_DATASET) --pipeline render --tier fast

regress-fast-convert:
	$(PYTHON) tools/regress.py $(REGRESS_DATASET) --pipeline convert --tier fast --input-variant fast

regress-fast-hard-sample:
	$(PYTHON) tools/regress.py $(REGRESS_DATASET) --pipeline convert --tier fast --input-variant hard --limit 3

regress-hard:
	$(PYTHON) tools/regress.py $(HARD_DATASET) --pipeline convert --tier hard

regress-real:
	@test -n "$(REAL_PNG_DIR)" || (echo "REAL_PNG_DIR is required"; exit 1)
	REAL_PNG_DIR=$(REAL_PNG_DIR) $(PYTHON) tools/regress.py --real-manifest $(REAL_MANIFEST)

rebuild-inputs:
	$(PYTHON) tools/rebuild_case_inputs.py $(REGRESS_DATASET) --variants fast,hard --overwrite

build-hard-assets:
	$(PYTHON) tools/build_hard_case_assets.py $(HARD_DATASET) --overwrite

dataset-sanity:
	$(PYTHON) tools/check_dataset_sanity.py $(REGRESS_DATASET)

check-png:
	$(PYTHON) tools/check_png_integrity.py
