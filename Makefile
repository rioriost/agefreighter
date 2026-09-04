SHELL := /bin/sh

GO ?= go
GOVULNCHECK ?= $(shell $(GO) env GOPATH)/bin/govulncheck
ACTIONLINT ?= $(shell $(GO) env GOPATH)/bin/actionlint
COVERAGE_DIR ?= .coverage
COVERAGE_THRESHOLD ?= 90.0
BENCHTIME ?= 5x
BENCHFLAGS ?=
FUZZTIME ?= 3s
SCALE_ROWS ?= 200000
PERFORMANCE_ARTIFACTS ?= performance-artifacts
PGGRAPH_BENCH_PROFILE ?= small
PGGRAPH_BENCH_TRIALS ?= 1
PGGRAPH_BENCH_OUTPUT ?= performance-artifacts/pggraph-$(PGGRAPH_BENCH_PROFILE).txt
VERSION ?= dev
COMMIT ?= $(shell git rev-parse --short HEAD 2>/dev/null || printf unknown)
BUILD_DATE ?= unknown
LDFLAGS := -X github.com/rioriost/agefreighter/internal/version.Version=$(VERSION) \
	-X github.com/rioriost/agefreighter/internal/version.Commit=$(COMMIT) \
	-X github.com/rioriost/agefreighter/internal/version.BuildDate=$(BUILD_DATE)

.PHONY: bench-csv bench-csv-scale bench-pggraph bench-release build check check-full coverage dev-down dev-pull \
	dev-reset dev-smoke dev-status dev-up fmt fuzz-smoke install-tools release-check test \
	test-compatibility test-connectors-cosmos test-connectors-cosmos-pggraph \
	test-connectors-local test-diagnostics-race test-pggraph test-pggraph-apple \
	test-race test-recovery test-release-integration tidy vet vscode-check vscode-package \
	vscode-test-host vuln workflow-lint

build:
	$(GO) build -trimpath -ldflags "$(LDFLAGS)" -o bin/agefreighter ./cmd/agefreighter
	$(GO) build -trimpath -ldflags "$(LDFLAGS)" -o bin/agefreighter-tools ./cmd/agefreighter-tools

vscode-check:
	npm --prefix extensions/vscode ci
	npm --prefix extensions/vscode run check

vscode-test-host:
	npm --prefix extensions/vscode run test:host

vscode-package:
	npm --prefix extensions/vscode run package

bench-csv:
	AGEFREIGHTER_AGE_TEST_DSN="$(AGEFREIGHTER_AGE_TEST_DSN)" \
		$(GO) test -run '^$$' -bench '^BenchmarkLegacyCountriesLoad$$' \
		-benchtime="$(BENCHTIME)" -benchmem $(BENCHFLAGS) ./internal/app

bench-csv-scale:
	AGEFREIGHTER_AGE_TEST_DSN="$(AGEFREIGHTER_AGE_TEST_DSN)" \
		AGEFREIGHTER_BENCH_ROWS="$(SCALE_ROWS)" \
		$(GO) test -run '^$$' -bench '^BenchmarkGeneratedCSVLoad$$' \
		-benchtime="1x" -benchmem $(BENCHFLAGS) ./internal/app

bench-release:
	./scripts/bench/release-gate.sh "$(PERFORMANCE_ARTIFACTS)"

bench-pggraph:
	./scripts/bench/pggraph-load.sh "$(PGGRAPH_BENCH_PROFILE)" \
		"$(PGGRAPH_BENCH_TRIALS)" "$(PGGRAPH_BENCH_OUTPUT)"

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

test-diagnostics-race:
	@AGEFREIGHTER_AGE_TEST_DSN="$(AGEFREIGHTER_AGE_TEST_DSN)" \
	AGEFREIGHTER_POSTGRES_TEST_DSN="$(AGEFREIGHTER_POSTGRES_TEST_DSN)" \
		$(GO) test -race -count=1 -v \
		-run '^(TestReadOnlyDiagnosticsRaceIntegration|TestDoctorDegradedPostgreSQLIntegration)$$' \
		./internal/app

fuzz-smoke:
	$(GO) test -run '^$$' -fuzz '^FuzzParse$$' -fuzztime="$(FUZZTIME)" ./internal/config
	$(GO) test -run '^$$' -fuzz '^FuzzCypherLexer$$' -fuzztime="$(FUZZTIME)" ./internal/cypher
	$(GO) test -run '^$$' -fuzz '^FuzzReportDecode$$' -fuzztime="$(FUZZTIME)" ./internal/report
	$(GO) test -run '^$$' -fuzz '^FuzzAGEGraphNames$$' -fuzztime="$(FUZZTIME)" ./internal/age
	$(GO) test -run '^$$' -fuzz '^FuzzGraphIDRoundTrip$$' -fuzztime="$(FUZZTIME)" ./internal/age
	$(GO) test -run '^$$' -fuzz '^FuzzEncodeStringProperty$$' -fuzztime="$(FUZZTIME)" ./internal/age
	$(GO) test -run '^$$' -fuzz '^FuzzEncodeStringProperty$$' -fuzztime="$(FUZZTIME)" ./pkg/model

test-compatibility:
	@AGEFREIGHTER_NEO4J_TEST_URI="$(AGEFREIGHTER_NEO4J_TEST_URI)" \
	AGEFREIGHTER_NEO4J_TEST_USERNAME="$(AGEFREIGHTER_NEO4J_TEST_USERNAME)" \
	AGEFREIGHTER_NEO4J_TEST_PASSWORD="$(AGEFREIGHTER_NEO4J_TEST_PASSWORD)" \
	AGEFREIGHTER_NEO4J_TEST_DATABASE="$(AGEFREIGHTER_NEO4J_TEST_DATABASE)" \
	AGEFREIGHTER_AGE_TEST_DSN="$(AGEFREIGHTER_AGE_TEST_DSN)" \
		$(GO) test -count=1 \
		./internal/age ./internal/meta ./internal/source/neo4j ./internal/app ./internal/cli

test-connectors-local:
	@AGEFREIGHTER_AGE_TEST_DSN="$(AGEFREIGHTER_AGE_TEST_DSN)" \
	AGEFREIGHTER_POSTGRES_TEST_DSN="$(AGEFREIGHTER_POSTGRES_TEST_DSN)" \
	AGEFREIGHTER_NEO4J_TEST_URI="$(AGEFREIGHTER_NEO4J_TEST_URI)" \
	AGEFREIGHTER_NEO4J_TEST_USERNAME="$(AGEFREIGHTER_NEO4J_TEST_USERNAME)" \
	AGEFREIGHTER_NEO4J_TEST_PASSWORD="$(AGEFREIGHTER_NEO4J_TEST_PASSWORD)" \
	AGEFREIGHTER_NEO4J_TEST_DATABASE="$(AGEFREIGHTER_NEO4J_TEST_DATABASE)" \
		$(GO) test -count=1 -v ./internal/app \
		-run '^(TestCSVSourceModeMatrixIntegration|TestPostgreSQLSourceModeMatrixIntegration|TestNeo4jSourceModeMatrixIntegration)$$'

test-connectors-cosmos:
	@if [ "$(AGEFREIGHTER_REQUIRE_COSMOS_TESTS)" = "1" ]; then \
		test -n "$(AGEFREIGHTER_AGE_TEST_DSN)" || { \
			printf 'AGEFREIGHTER_AGE_TEST_DSN is required for the strict Cosmos release gate\n' >&2; exit 2; }; \
		test -n "$(AGEFREIGHTER_COSMOS_TEST_ENDPOINT)" || { \
			printf 'AGEFREIGHTER_COSMOS_TEST_ENDPOINT is required for the strict Cosmos release gate\n' >&2; exit 2; }; \
		test -n "$(AGEFREIGHTER_COSMOS_TEST_DATABASE)" || { \
			printf 'AGEFREIGHTER_COSMOS_TEST_DATABASE is required for the strict Cosmos release gate\n' >&2; exit 2; }; \
		test -n "$(AGEFREIGHTER_COSMOS_TEST_VERTEX_CONTAINER)" || { \
			printf 'AGEFREIGHTER_COSMOS_TEST_VERTEX_CONTAINER is required for the strict Cosmos release gate\n' >&2; exit 2; }; \
		test -n "$(AGEFREIGHTER_COSMOS_TEST_EDGE_CONTAINER)" || { \
			printf 'AGEFREIGHTER_COSMOS_TEST_EDGE_CONTAINER is required for the strict Cosmos release gate\n' >&2; exit 2; }; \
	fi
	@AGEFREIGHTER_AGE_TEST_DSN="$(AGEFREIGHTER_AGE_TEST_DSN)" \
	AGEFREIGHTER_COSMOS_TEST_ENDPOINT="$(AGEFREIGHTER_COSMOS_TEST_ENDPOINT)" \
	AGEFREIGHTER_COSMOS_TEST_DATABASE="$(AGEFREIGHTER_COSMOS_TEST_DATABASE)" \
	AGEFREIGHTER_COSMOS_TEST_VERTEX_CONTAINER="$(AGEFREIGHTER_COSMOS_TEST_VERTEX_CONTAINER)" \
	AGEFREIGHTER_COSMOS_TEST_EDGE_CONTAINER="$(AGEFREIGHTER_COSMOS_TEST_EDGE_CONTAINER)" \
		$(GO) test -count=1 -timeout=45m -v ./internal/app \
		-run '^(TestCosmosLiveIntegration|TestCosmosSourceModeMatrixIntegration)$$'

test-connectors-cosmos-pggraph:
	@if [ "$(AGEFREIGHTER_REQUIRE_COSMOS_TESTS)" = "1" ]; then \
		test -n "$(AGEFREIGHTER_PGGRAPH_TEST_DSN)" || { \
			printf 'AGEFREIGHTER_PGGRAPH_TEST_DSN is required for the strict Cosmos PostgreSQL property-graph gate\n' >&2; exit 2; }; \
		test -n "$(AGEFREIGHTER_COSMOS_TEST_ENDPOINT)" || { \
			printf 'AGEFREIGHTER_COSMOS_TEST_ENDPOINT is required for the strict Cosmos PostgreSQL property-graph gate\n' >&2; exit 2; }; \
		test -n "$(AGEFREIGHTER_COSMOS_TEST_DATABASE)" || { \
			printf 'AGEFREIGHTER_COSMOS_TEST_DATABASE is required for the strict Cosmos PostgreSQL property-graph gate\n' >&2; exit 2; }; \
		test -n "$(AGEFREIGHTER_COSMOS_TEST_VERTEX_CONTAINER)" || { \
			printf 'AGEFREIGHTER_COSMOS_TEST_VERTEX_CONTAINER is required for the strict Cosmos PostgreSQL property-graph gate\n' >&2; exit 2; }; \
		test -n "$(AGEFREIGHTER_COSMOS_TEST_EDGE_CONTAINER)" || { \
			printf 'AGEFREIGHTER_COSMOS_TEST_EDGE_CONTAINER is required for the strict Cosmos PostgreSQL property-graph gate\n' >&2; exit 2; }; \
	fi
	@AGEFREIGHTER_PGGRAPH_TEST_DSN="$(AGEFREIGHTER_PGGRAPH_TEST_DSN)" \
	AGEFREIGHTER_COSMOS_TEST_ENDPOINT="$(AGEFREIGHTER_COSMOS_TEST_ENDPOINT)" \
	AGEFREIGHTER_COSMOS_TEST_DATABASE="$(AGEFREIGHTER_COSMOS_TEST_DATABASE)" \
	AGEFREIGHTER_COSMOS_TEST_VERTEX_CONTAINER="$(AGEFREIGHTER_COSMOS_TEST_VERTEX_CONTAINER)" \
	AGEFREIGHTER_COSMOS_TEST_EDGE_CONTAINER="$(AGEFREIGHTER_COSMOS_TEST_EDGE_CONTAINER)" \
		$(GO) test -count=1 -timeout=45m -v ./internal/app \
		-run '^TestCosmosPostgreSQLPropertyGraphIntegration$$'

test-pggraph:
	@test -n "$(AGEFREIGHTER_PGGRAPH_TEST_DSN)" || { \
		printf 'AGEFREIGHTER_PGGRAPH_TEST_DSN is required\n' >&2; exit 2; \
	}
	@AGEFREIGHTER_PGGRAPH_TEST_DSN="$(AGEFREIGHTER_PGGRAPH_TEST_DSN)" \
		$(GO) test -count=1 -v ./internal/pggraph ./internal/app \
		-run '^(TestPropertyGraphIntegration|TestPropertyGraphOperationGuardsIntegration|TestPropertyGraphMutationLockIntegration|TestPropertyGraphSinkReplayAndAbortIntegration|TestPropertyGraphSinkFailureIntegration|TestPostgreSQLPropertyGraphCreateAndResumeIntegration|TestPostgreSQLPropertyGraphModeMatrixIntegration|TestPostgreSQLPropertyGraphIncrementalResumeIntegration|TestPostgreSQLPropertyGraphReplaceRecoveryIntegration|TestPostgreSQLPropertyGraphIncrementalAdmissionIntegration|TestPostgreSQLPropertyGraphCorruptionDetectionIntegration)$$'
	@AGEFREIGHTER_AGE_TEST_DSN="$(AGEFREIGHTER_PGGRAPH_TEST_DSN)" \
		$(GO) test -count=1 -v ./internal/meta \
		-run '^TestMetadataV14V17V18V19V20UpgradeToV21Integration$$'

test-pggraph-apple:
	./scripts/dev/pggraph-apple-container.sh test

test-release-integration:
	@AGEFREIGHTER_AGE_TEST_DSN="$(AGEFREIGHTER_AGE_TEST_DSN)" \
	AGEFREIGHTER_POSTGRES_TEST_DSN="$(AGEFREIGHTER_POSTGRES_TEST_DSN)" \
		$(GO) test -count=1 -v ./internal/meta ./internal/app \
		-run '^(TestMetadataV14V17V18V19V20UpgradeToV21Integration|TestDoctorDegradedPostgreSQLIntegration|TestDeepVerificationDetectsCorruptionIntegration)$$'

test-recovery:
	@AGEFREIGHTER_AGE_TEST_DSN="$(AGEFREIGHTER_AGE_TEST_DSN)" \
		$(GO) test -count=1 -v \
		-run '^(TestResumeAfterPreBatchFailureIntegration|TestRunningAndFailedBatchResumeIntegration|TestResumeAfterCommittedBatchIntegration|TestReplaceFailureResumeAndCleanupIntegration)$$' \
		./internal/app

coverage:
	@mkdir -p "$(COVERAGE_DIR)"
	@AGEFREIGHTER_NEO4J_TEST_URI="$(AGEFREIGHTER_NEO4J_TEST_URI)" \
	AGEFREIGHTER_NEO4J_TEST_USERNAME="$(AGEFREIGHTER_NEO4J_TEST_USERNAME)" \
	AGEFREIGHTER_NEO4J_TEST_PASSWORD="$(AGEFREIGHTER_NEO4J_TEST_PASSWORD)" \
	AGEFREIGHTER_NEO4J_TEST_DATABASE="$(AGEFREIGHTER_NEO4J_TEST_DATABASE)" \
	AGEFREIGHTER_POSTGRES_TEST_DSN="$(AGEFREIGHTER_POSTGRES_TEST_DSN)" \
	AGEFREIGHTER_PGGRAPH_TEST_DSN="$(AGEFREIGHTER_PGGRAPH_TEST_DSN)" \
	AGEFREIGHTER_AGE_TEST_DSN="$(AGEFREIGHTER_AGE_TEST_DSN)" \
		$(GO) test -covermode=atomic -coverpkg=./... \
		-coverprofile="$(COVERAGE_DIR)/unit.raw.out" ./...
	./scripts/coverage/filter.sh \
		"$(COVERAGE_DIR)/unit.raw.out" \
		"$(COVERAGE_DIR)/unit.out" \
		.coverage-exclude
	./scripts/coverage/check.sh "$(COVERAGE_DIR)/unit.out" "$(COVERAGE_THRESHOLD)"

release-check:
	./scripts/release/self-check.sh

workflow-lint:
	@command -v $(ACTIONLINT) >/dev/null 2>&1 || { \
		printf 'actionlint is required; run make install-tools\n' >&2; \
		exit 1; \
	}
	$(ACTIONLINT) -config-file .github/actionlint.yaml

check: fmt vet vuln workflow-lint test test-race fuzz-smoke release-check

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
	$(GO) install golang.org/x/vuln/cmd/govulncheck@v1.7.0
	$(GO) install github.com/rhysd/actionlint/cmd/actionlint@v1.7.12
