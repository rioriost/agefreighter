# agefreighter 2.0 Implementation Plan

**Status:** Reviewed and revised  
**Branch:** `2.0.0`  
**Architecture:** [agefreighter-2.0-research.md](agefreighter-2.0-research.md)

## 1. Delivery principles

1. The implementation is greenfield Go. No 1.x compatibility layer is created.
2. Every milestone ends in a working, tested repository and a Git commit.
3. Correctness and recoverability precede connector breadth and optimization.
4. Performance changes require an end-to-end benchmark or profile.
5. Normal CI enforces at least 90% merged statement coverage.
6. AGE physical-storage knowledge remains inside `internal/age`.
7. Azure resources are not generated or deployed until the separate
   `.azure/deployment-plan.md` is finalized and approved.
8. Destructive cleanup scripts operate only on explicit
   `agefreighter2-*` resources.

## 2. Initial technical choices

These choices minimize dependencies until a milestone proves another library is
necessary.

| Area | Choice |
| --- | --- |
| Go version | Go 1.27 |
| Module | `github.com/rioriost/agefreighter` |
| CLI | Cobra |
| Configuration | `go.yaml.in/yaml/v3` plus JSON Schema |
| Logging | standard `log/slog` |
| PostgreSQL | `github.com/jackc/pgx/v5` |
| Neo4j | official Neo4j Go Driver v6 |
| Cosmos DB | official Azure `azcosmos` and `azidentity` SDKs |
| IDs | UUID v7-compatible job IDs behind an internal generator interface |
| Unit tests | standard `testing`, `httptest`, fuzzing |
| Static analysis | `go vet`; additional pinned tools only after need is proven |
| Local containers | Apple Container 1.0 |
| CI containers | Docker on Linux |
| Initial AGE matrix | PostgreSQL 17 + Apache AGE 1.6.0, subject to image validation |
| Initial Neo4j | Neo4j 5.26.29 Community LTS, subject to image validation |

Dependency versions are resolved and committed in `go.mod` and `go.sum` during
the milestone that first uses them. Dependencies are not added speculatively.

## 3. Repository transition

After this plan is reviewed:

- remove tracked Python source, tests, packaging, generated artifacts, Homebrew
  formula, and 1.x-only documentation from the `2.0.0` branch
- move the reusable benchmark corpus from `data/` to a documented
  `testdata/legacy-baseline/` location until a deterministic generator replaces
  it
- retain `LICENSE`, Git history, the reviewed 2.x design documents, and the
  Azure deployment-plan skeleton
- replace the README, Makefile, ignore rules, and CI definitions with 2.x
  equivalents
- keep 1.x exclusively on `main`

The transition is committed separately before application functionality is
added. That commit is the rollback point between the 1.x repository shape and
the 2.x repository shape.

## 4. Planned package boundaries

```text
cmd/agefreighter
cmd/agefreighter-tools

internal/app
internal/age
internal/checkpoint
internal/config
internal/identity
internal/meta
internal/pipeline
internal/reject
internal/source/cosmos
internal/source/csv
internal/source/neo4j
internal/source/postgres
internal/telemetry
internal/transform

pkg/model
```

Dependency direction:

```text
commands -> app -> pipeline -> model
                         -> source interfaces
                         -> sink interfaces

source implementations -> source interfaces + model
AGE implementation     -> sink interfaces + model
```

All job-document types live in `internal/config`. Source packages and the AGE
adapter may import config types, but config never imports a connector package.
`internal/meta` owns all `agefreighter_meta` migrations and schema objects;
checkpoint, identity, rejection, and AGE packages depend on it in one
direction.

`internal/age` cannot be imported by source packages. `cmd` packages contain
only wiring and exit-code handling. `agefreighter-tools` cannot be imported by
the production loader.

## 5. Milestones

### Milestone 0: Greenfield repository baseline

**Deliverables**

- remove tracked 1.x files from this branch
- initialize `go.mod`
- create both command entry points
- add replacement README and contribution/testing instructions
- add `.editorconfig` and 2.x ignore rules
- add Makefile targets:
  - `fmt`
  - `vet`
  - `test`
  - `test-race`
  - `coverage`
  - `check`
- add `check-full` for container integration tests and the merged 90% coverage
  gate
- add a coverage-profile merge/filter script with an explicit allowlist format
- add minimal GitHub Actions workflow for format, vet, unit tests, race tests,
  and coverage
- add a pinned `govulncheck` gate
- add a dependency-boundary test using `go list -deps` to reject imports from
  source packages to `internal/age` and imports from production packages to
  tools packages
- add version package and `version` command to both binaries

**Test gate**

- `go test ./...`
- `go test -race ./...`
- unit-only coverage is 90% or higher while no integration-only package exists
- command tests assert output, exit status, and version injection

**Rollback commit**

`chore: bootstrap agefreighter 2.0 Go workspace`

### Milestone 1: Configuration and domain model

**Deliverables**

- versioned `LoadJob` model
- source and target discriminated configuration
- load modes and error policies
- duration, byte-size, and bounded-concurrency validation
- secret-reference model that rejects literal credentials
- JSON Schema generation or a checked-in schema validated against Go fixtures
- typed vertex, edge, endpoint, property, and source-position models
- `validate` and `plan` commands

The initial `plan` output is derived from static configuration. Connectors add
source consistency and capability sections in their respective milestones, so
the complete output contract is not frozen in this milestone.

**Test quality**

- table-driven semantic validation tests
- golden valid/invalid job files
- fuzz tests for YAML/JSON parsing
- tests proving redacted error output
- schema-to-Go fixture conformance tests

**Rollback commit**

`feat: add versioned load job configuration`

### Milestone 2: Bounded pipeline core

**Deliverables**

- source iterator interface
- sink transaction interface
- byte-accounted memory limiter
- fixed-capacity stage channels
- batching by rows and bytes
- cancellation propagation
- typed error classification
- rejection writer interface
- deterministic checkpoint state machine
- metrics snapshot interface

**Test quality**

- fake source and sink with injected failures at every batch boundary
- goroutine-leak tests
- cancellation tests
- property tests for checkpoint transitions
- tests proving memory permits are released on every error path
- benchmark for batching and transform overhead

**Rollback commit**

`feat: implement bounded migration pipeline`

### Milestone 3: Apple Container development environment and 1.x baseline

**Images**

- `apache/age:release_PG17_1.6.0`
- `postgres:17.6-alpine`
- `neo4j:5.26.29-community`

Exact digests are recorded after successful multi-architecture inspection and
pull.

**Deliverables**

- runtime-neutral lifecycle contract under `scripts/dev`
- matrix-entry parameters for name, image, host port, platform, and volume,
  initially enabling only `pg17-age160`
- Apple Container implementation for local development
- Docker implementation for Linux CI
- named local volumes
- localhost-only published ports
- readiness polling with timeouts
- log collection on startup failure
- Makefile targets:
  - `dev-pull`
  - `dev-up`
  - `dev-status`
  - `dev-down`
  - `dev-reset`
- test fixture initialization for AGE, PostgreSQL source, and Neo4j
- Apple Container system status/start precondition
- explicit `linux/arm64` platform locally and `linux/amd64` in CI
- a reproducible 1.x benchmark run from a clean `main` worktree using its
  locked Python environment, the pinned AGE image, and the preserved benchmark
  corpus
- `docs/benchmarks/1x-baseline.md` recording source commit, image digest,
  dataset manifest, Python version, hardware, throughput, and peak RSS

**Safety**

- scripts validate resource names
- reset deletes only the three documented named volumes
- no prune command is used
- tests use non-production fixed credentials that are isolated to localhost

**Test gate**

- clean `dev-up`
- service queries succeed
- repeated `dev-up` is idempotent
- `dev-down` preserves volumes
- `dev-reset` produces an empty deterministic environment
- the 1.x baseline can be reproduced from its recorded manifest

**Rollback commit**

`build: add local database integration environment`

### Milestone 4: AGE capability adapter

**Deliverables**

- pool initialization and capability probes
- conditional AGE library loading
- search-path setup on every connection
- supported-version validation
- graph and label lifecycle APIs
- catalog lookup for relation OID, label ID, kind, and sequence
- graphid encoding and validation
- AGE-name-safe shadow/backup names that satisfy the 3-63 byte limit and the
  AGE graph-name grammar
- property encoder and complete source-to-agtype mapping decision
- sequence block reservation
- label-scoped PostgreSQL locking primitive shared by incremental modes
- direct text COPY prototype
- staged binary COPY prototype
- graph and metadata transaction boundary
- post-load `ANALYZE` and verification queries
- fixture generation and benchmark dataset generation commands in
  `agefreighter-tools`

**Test quality**

- integration tests for every invariant listed in the architecture research
- fuzz tests for graph/label identifiers and graphid boundaries
- failure injection during COPY, sequence reservation, and rename
- test that Cypher writes after a load do not collide
- test as a restricted, non-superuser database role

**Decision gate**

Benchmark direct text COPY and staged binary COPY using:

- equivalent payload and indexes
- the same durability settings
- target CPU, WAL, elapsed time, and peak RSS
- a plain relational COPY reference using equivalent indexes and durability
- for fresh edge labels only, a third arm that drops the `start_id` and
  `end_id` indexes before COPY and recreates them before verification

Record the selected strategy in an architecture decision record.

**Decision outcome**

ADR 0001 selects staged binary COPY followed by a set-based cast into AGE
label tables. The pinned AGE 1.6.0 catalog creates no `start_id` or `end_id`
indexes on user edge labels, so the conditional index-drop/rebuild arm was
recorded as not applicable rather than benchmarking a nonexistent index set.

**Rollback commits**

- `feat: add Apache AGE capability adapter`
- `perf: select AGE bulk write strategy`

### Milestone 5: CSV `create` vertical slice

**Deliverables**

- streaming CSV/TSV connector
- compiled column mapping
- configurable delimiter, quote, escape, header, encoding, and null rules
- gzip streaming
- record-boundary source positions
- vertices-first load barrier
- external endpoint resolution without an unbounded Go map
- `load`, `status`, and `verify` commands
- target-side job and batch tables
- target-side vertex identity table, graph/label generation columns, indexes,
  and endpoint-join resolution
- independent diagnostic transaction
- quarantine output
- `resume --job`, including graph-generation admission checks

The initial missing-endpoint policies are `error` and `quarantine`. The bounded
`defer` policy is added with incremental staging in Milestone 7.

**Test quality**

- RFC edge cases and arbitrary delimiters
- quoted multiline records
- Unicode and malformed encodings
- self-loops, parallel edges, isolated vertices
- duplicate external IDs and missing endpoints
- restart after process termination at each batch boundary
- end-to-end fixture comparison through Cypher

**Performance gate**

- compare against the committed Milestone 3 baseline; the 2.x build has no
  runtime dependency on the 1.x worktree
- at least 2x 1.x CSV throughput
- no default temporary CSV
- bounded RSS as input size grows

**Rollback commit**

`feat: load CSV graphs into Apache AGE`

### Milestone 6: Atomic `replace`

**Implementation status:** Complete.

**Deliverables**

- shadow graph lifecycle
- verification before promotion
- graph rename and metadata-generation promotion in one transaction
- backup naming that satisfies AGE's 3-63 byte graph-name limit and
  `^[A-Za-z_][A-Za-z0-9_.-]*[A-Za-z0-9_]$`
- explicit old-graph retention and cleanup command
- resume of a partially loaded shadow graph

**Test quality**

- old graph remains queryable after every injected failure
- successful promotion exposes only the new graph
- metadata OIDs and generation match after rename
- cleanup cannot delete the active graph
- pooled sessions referring to the old shadow search path are closed or reset
  after promotion
- promotion without schema ownership fails before the old graph changes

**Rollback commit**

`feat: add atomic graph replacement`

### Milestone 7: `append` and `upsert`

**Implementation status:** Complete.

**Deliverables**

- target-side edge identity catalog
- graph and label catalog mismatch checks over the Milestone 5 generations
- append duplicate policies
- explicit vertex property merge policies
- edge external-ID requirement
- staging-table endpoint joins
- bounded deferred-edge table
- reconciliation diagnostics for unsupported external writes
- deferred-edge drain and capacity enforcement
- job- and label-scoped locks with a documented conflict error code

**Test quality**

- concurrent append jobs serialize safely or reject predictably
- retry does not duplicate records
- existing Cypher records advance ID allocation safely
- graph/label recreation is detected
- edge parallelism is preserved
- deferred-store limits fail deterministically

**Rollback commit**

`feat: support incremental AGE loads`

### Milestone 8: PostgreSQL source

**Implementation status:** Complete.

**Deliverables**

- `COPY (SELECT ...) TO STDOUT` streaming path
- cursor fallback
- keyset mode for explicit stable keys
- exported-snapshot coordinator for parallel reads
- source schema/type mapping

**Test quality**

- concurrent source mutation under shared snapshot
- exporter transaction remains open
- importer snapshot is first statement
- cancellation and connection loss
- unsupported consistent-parallel requests fail validation

**Rollback commit**

`feat: add PostgreSQL source connector`

### Milestone 9: Neo4j source

**Implementation status:** Complete.

**Deliverables**

- official Go driver adapter
- streaming session/result handling
- explicit stable source identity
- multi-label policy
- source type mapping for temporal and spatial values
- source consistency warning in the load plan

**Test quality**

- pinned Neo4j container contract tests
- driver interface tests for partial results and cancellation
- no eager whole-result materialization
- source mutation behavior is documented and tested

**Rollback commit**

`feat: add Neo4j source connector`

### Milestone 10: Cosmos DB for NoSQL source

**Prerequisites**

- finalize `.azure/deployment-plan.md`
- confirm subscription and region with the user
- inspect Azure Policy
- validate regional availability and account limits
- receive approval
- generate azd/Bicep artifacts
- invoke `azure-validate`
- deploy through `azure-deploy`

**Deliverables**

- `DefaultAzureCredential`
- official `azcosmos` adapter
- parameterized page queries
- multi-partition paging
- continuation diagnostics
- bounded/adaptive request concurrency
- request charge and 429 metrics
- consistency contract in the load plan
- disposable Azure integration fixture and cleanup procedure

**Test quality**

- injected HTTP transport contract tests in normal CI
- Entra ID integration test in Azure
- multi-page and multi-partition scan
- throttling and cancellation
- credentials and tokens never appear in logs

**Rollback commits**

- `feat: add Cosmos DB source connector`
- `infra: add Cosmos integration environment`

### Milestone 11: agefreighter-tools expansion

**Initial commands**

- source and target inspection
- benchmark report normalization

Tools are added only when required by a tested loader workflow. Cypher
translation and AI-assisted conversion are not copied from 1.x.

**Rollback commit**

`feat: add agefreighter diagnostic tools`

### Milestone 12: Release hardening

**Deliverables**

- OpenTelemetry opt-in export
- JSON logging contract
- SBOM
- vulnerability scan
- signed checksummed binaries
- macOS arm64/amd64, Linux arm64/amd64, and Windows amd64 builds
- Homebrew formula for 2.x
- installation and operations documentation
- benchmark and compatibility reports

**Release gate**

- all required CI and Azure integration tests pass
- merged coverage is at least 90%
- race and fuzz gates pass
- supported AGE matrix passes
- performance budgets pass
- disaster/recovery scenarios pass
- release artifacts install and run on each target

**Rollback commit**

`release: prepare agefreighter 2.0.0`

## 6. Coverage implementation

Normal CI produces coverage data for:

1. unit and interface-contract tests
2. race tests, used as a correctness gate but not merged because race coverage
   can distort timing and duplicate statements
3. Linux database integration tests

Coverage-emitting unit tests use `-coverpkg=./...`. Integration and end-to-end
test binaries emit binary coverage to `GOCOVERDIR`, including code reached
through command execution. Profiles are merged with `go tool covdata merge` and
converted once with `go tool covdata textfmt`. Only then does the reviewed
allowlist filter run. A script computes total statement coverage and fails
below 90.0%.

Allowed exclusions are limited to:

- generated schema bindings bearing a generated-code marker
- OS-specific command wrappers unreachable on the current CI platform
- `main` functions that only invoke a tested `Run` function and map its error to
  an exit code

The exclusion allowlist contains exact file paths and a reason. No glob may
exclude an entire functional package.

## 7. Commit and rollback policy

- Commit after every milestone or independently deployable sub-milestone.
- Never combine repository deletion, functional implementation, and formatting
  cleanup in the same commit.
- Commit messages describe behavior, not activity.
- Every commit must pass the smallest relevant test gate.
- `make check` runs formatting, vet, vulnerability, unit, and race gates without
  a database or a coverage threshold.
- `make check-full` adds container integration tests, merges coverage, and
  enforces the 90.0% threshold. Milestones containing integration-only code must
  pass `make check-full`; normal CI always runs it.
- Benchmark result documents are committed with the performance change they
  justify.
- Database schema changes include forward and rollback tests before commit.
- Do not amend published checkpoint commits.

## 8. Review checklist

The plan review must confirm:

- package dependencies have no cycles or AGE leakage into connectors
- every destructive operation has an explicit scope and rollback
- every load mode has interruption and resume semantics
- sequence allocation is safe under later Cypher writes
- diagnostics survive data rollback without becoming ambiguous
- 90% coverage is measurable in normal CI
- Apple Container is local-only and Linux CI remains equivalent
- Cosmos implementation cannot begin before Azure plan approval
- each milestone leaves a usable repository
- no milestone requires 1.x code or configuration

## 9. Recovery scenarios required before release

- process termination before, during, and after COPY
- target connection loss before checkpoint commit
- target PostgreSQL restart between committed batches
- failure while rebuilding fresh edge indexes
- failure before and during shadow graph promotion
- source PostgreSQL connection loss while the exported snapshot is active
- Neo4j partial-result failure
- Cosmos request cancellation and throttling
- diagnostic-connection failure while the data transaction rolls back
