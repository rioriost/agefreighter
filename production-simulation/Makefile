SHELL := /bin/sh

REPO_ROOT := ..
WORK := work

.PHONY: check clean-smoke smoke test validate

test:
	cd $(REPO_ROOT) && go test ./production-simulation/...

validate:
	./scripts/validate-assets.sh

check: validate
	cd $(REPO_ROOT) && git diff --check

smoke: clean-smoke
	cd $(REPO_ROOT) && ./production-simulation/scripts/generate-fixture.sh \
		tiny production-simulation/$(WORK)/tiny 4 2

clean-smoke:
	@test ! -e "$(WORK)/tiny" || { \
		printf 'Refusing to delete existing %s/tiny; remove it explicitly after review.\n' "$(WORK)" >&2; \
		exit 3; \
	}
