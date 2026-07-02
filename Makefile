SHELL := /bin/sh

REPO_ROOT := $(CURDIR)
UV ?= uv
DIST_DIR := $(REPO_ROOT)/dist
FORMULA_DIR := $(REPO_ROOT)/Formula
FORMULA_FILE := $(FORMULA_DIR)/agefreighter.rb

.PHONY: release-artifacts sync build formula

release-artifacts: sync build formula

sync:
	$(UV) sync --extra test --group dev

build:
	rm -rf "$(DIST_DIR)"
	$(UV) build

formula:
	mkdir -p "$(FORMULA_DIR)"
	genformula \
		--source-subdir . \
		--pyproject "$(REPO_ROOT)/pyproject.toml" \
		--output "$(FORMULA_FILE)"
