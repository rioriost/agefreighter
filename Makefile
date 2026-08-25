SHELL := /bin/sh

GO ?= go
GOVULNCHECK ?= $(shell $(GO) env GOPATH)/bin/govulncheck
COVERAGE_DIR ?= .coverage
COVERAGE_THRESHOLD ?= 90.0
AGEFREIGHTER_AGE_TEST_DSN ?= postgres://agefreighter:agefreighter_dev_only@127.0.0.1:55432/agefreighter?sslmode=disable
VERSION ?= dev
COMMIT ?= $(shell git rev-parse --short HEAD 2>/dev/null || printf unknown)
BUILD_DATE ?= unknown
LDFLAGS := -X github.com/rioriost/agefreighter/internal/version.Version=$(VERSION) \
	-X github.com/rioriost/agefreighter/internal/version.Commit=$(COMMIT) \
	-X github.com/rioriost/agefreighter/internal/version.BuildDate=$(BUILD_DATE)

.PHONY: build check check-full coverage dev-down dev-pull dev-reset dev-smoke \
	dev-status dev-up fmt install-tools test test-race tidy vet vuln

build:
	$(GO) build -trimpath -ldflags "$(LDFLAGS)" -o bin/agefreighter ./cmd/agefreighter
	$(GO) build -trimpath -ldflags "$(LDFLAGS)" -o bin/agefreighter-tools ./cmd/agefreighter-tools

fmt:
	@files="$$(gofmt -l .)"; \
	if [ -n "$$files" ]; then \
		printf 'Files require gofmt:\n%s\n' "$$files"; \
		exit 1; \
	fi

vet:
	$(GO) vet ./...

vuln:
	@command -v $(GOVULNCHECK) >/dev/null 2>&1 || { \
		printf 'govulncheck is required; run make install-tools\n' >&2; \
		exit 1; \
	}
	$(GOVULNCHECK) ./...

test:
	$(GO) test ./...

test-race:
	$(GO) test -race ./...

coverage:
	@mkdir -p "$(COVERAGE_DIR)"
	AGEFREIGHTER_AGE_TEST_DSN="$(AGEFREIGHTER_AGE_TEST_DSN)" \
		$(GO) test -covermode=atomic -coverpkg=./... \
		-coverprofile="$(COVERAGE_DIR)/unit.raw.out" ./...
	./scripts/coverage/filter.sh \
		"$(COVERAGE_DIR)/unit.raw.out" \
		"$(COVERAGE_DIR)/unit.out" \
		.coverage-exclude
	./scripts/coverage/check.sh "$(COVERAGE_DIR)/unit.out" "$(COVERAGE_THRESHOLD)"

check: fmt vet vuln test test-race

check-full: check coverage

dev-pull:
	./scripts/dev/dev.sh pull

dev-up:
	./scripts/dev/dev.sh up

dev-status:
	./scripts/dev/dev.sh status

dev-smoke:
	./scripts/dev/dev.sh smoke

dev-down:
	./scripts/dev/dev.sh down

dev-reset:
	./scripts/dev/dev.sh reset

tidy:
	$(GO) mod tidy

install-tools:
	$(GO) install golang.org/x/vuln/cmd/govulncheck@v1.1.4
