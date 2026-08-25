# agefreighter 2.0 Architecture Research

**Status:** Reviewed and revised  
**Branch:** `2.0.0`  
**Date:** 2026-08-24

## 1. Executive decision

agefreighter 2.x is a greenfield implementation. It does not preserve the Python
API, the 1.x CLI, configuration files, package layout, or behavioral defaults.
The `main` branch remains the maintenance line for 1.x. The `2.0.0` branch is
the only development line for the new implementation.

The recommended product shape is:

- `agefreighter`: a Go CLI for validated, resumable graph migration into
  Apache AGE.
- `agefreighter-tools`: a separate Go CLI for diagnostics, fixture generation,
  inspection, and non-essential conversion utilities.
- A shared set of internal Go packages. There is no dynamically loaded plugin
  ABI in the initial release.
- Local PostgreSQL/Apache AGE and Neo4j environments managed with Apple
  Container.
- Azure Cosmos DB for NoSQL integration tests run against an azd-provisioned,
  disposable Azure environment.

The main performance gain is expected from a bounded streaming architecture,
not from the language change alone.

## 2. Scope and non-goals

### 2.1 Initial 2.x scope

- Sources:
  - Delimited text files, including CSV and TSV
  - PostgreSQL
  - Neo4j
  - Azure Cosmos DB for NoSQL
- Target:
  - Apache AGE on supported PostgreSQL versions
- Load modes:
  - `create`
  - atomic `replace`
  - `append`
  - `upsert`
- Operational features:
  - dry-run validation and load planning
  - bounded memory and bounded concurrency
  - durable target-side checkpoints
  - deterministic retry and resume
  - structured rejection records
  - row-count and referential verification
  - JSON logs and operational metrics

### 2.2 Explicit non-goals for 2.0.0

- Compatibility with any 1.x Python API, command, or JSON mapping
- Installing Python or Python packages at runtime
- A Python wheel that embeds the Go binary
- Cosmos DB Gremlin output
- Dynamic in-process plugins
- Parquet and Arrow as launch requirements
- Transparent synchronization of arbitrary Cypher changes made outside
  agefreighter
- A general graph transformation language
- A GUI or browser-based graph viewer

Parquet, Arrow, object storage, and an out-of-process connector protocol may be
added after the core loader meets its performance and recovery targets.

## 3. Why Go

The decision is based on the complete connector and distribution requirements,
not only parser microbenchmarks.

| Dimension | Go | Rust | .NET NativeAOT | Java/GraalVM | C++ |
| --- | --- | --- | --- | --- | --- |
| PostgreSQL bulk protocol | Excellent (`pgx`) | Excellent | Excellent | Excellent | Good |
| Official Neo4j driver | Yes | No | Yes | Yes | No |
| Official Cosmos NoSQL SDK | Yes | Preview/less mature | Yes | Yes | No |
| Static CLI distribution | Excellent | Excellent | Good | Moderate | Difficult |
| Bounded concurrency | Native channels | Tokio channels | Channels | Virtual threads | Manual |
| Memory safety | Yes | Yes | Yes | Yes | No |
| Team and contributor accessibility | High | Moderate | High | High | Moderate |
| Expected CPU ceiling | High | Highest | High | High | Highest |

Rust is the strongest alternative for local file decoding and Arrow/Parquet
processing. It is not the initial choice because official, maintained connector
support is more important than a possible CPU advantage in a pipeline whose
terminal bottleneck is normally PostgreSQL COPY, WAL, and index maintenance.

No Rust component should be introduced until an end-to-end profile shows that a
specific Go stage consumes enough CPU to prevent the acceptance target from
being met.

## 4. Product and repository boundaries

The repository will become a Go repository on the `2.0.0` branch. Python source,
Python packaging, 1.x fixtures, generated distributions, and 1.x documentation
will be removed from this branch. Git history and the `main` branch preserve
them.

Proposed top-level layout:

```text
cmd/
  agefreighter/
  agefreighter-tools/
internal/
  age/
  checkpoint/
  config/
  identity/
  pipeline/
  reject/
  source/
    cosmos/
    csv/
    neo4j/
    postgres/
  telemetry/
  transform/
pkg/
  model/
docs/
  design/
  operations/
  reference/
testdata/
  ...
scripts/
  dev/
.azure/
infra/
```

`internal/age` is the only package allowed to know AGE physical storage details.
Source connectors cannot import it. Source connectors emit target-independent
records.

`agefreighter-tools` may import stable shared packages, but the production
loader must not import tools packages. This keeps optional conversion and
fixture dependencies out of the loader.

## 5. Pipeline architecture

```text
source iterator
  -> decoder
  -> compiled mapping
  -> validator/normalizer
  -> bounded record channel
  -> bounded batcher
  -> identity resolver
  -> AGE sink
  -> checkpoint and verification
```

### 5.1 Backpressure

Every producer-consumer boundary uses a small fixed-capacity channel. A
byte-accounted weighted semaphore tracks records and batches across channel
boundaries and enforces the configured memory budget. Channel capacity is not
used as a proxy for byte size. No connector may launch work proportional to
input size.

Concurrency limits are independent:

- source request concurrency
- decoder workers
- transformation workers
- target connections
- labels loaded concurrently

The default target concurrency is conservative. Increasing source concurrency
cannot create additional target connections without an explicit setting.

### 5.2 Record representation

The core record types contain:

```text
Vertex:
  label
  namespace
  external ID
  encoded properties
  source position

Edge:
  label
  external edge ID (required for upsert)
  start vertex reference
  end vertex reference
  encoded properties
  source position
```

The hot path must avoid general-purpose `map[string]any` where column positions
and property types are known from the validated configuration. A mapping is
compiled once into typed column accessors and encoders.

Property encoding preserves distinctions that can be lost by a JSON
round-trip, including integer versus floating-point values. Each supported AGE
version receives type round-trip integration tests.

### 5.3 Vertex and edge phases

Vertices are loaded before dependent edges. The system has an explicit vertex
barrier for a full migration. Incremental jobs may process labels in parallel,
but an edge batch cannot commit until its endpoint identities resolve.

Missing endpoints are never silently discarded. The policies are:

- `error` (default)
- `quarantine`
- `defer`

`skip` is intentionally excluded for missing endpoints because an edge without
its endpoint is structural data loss. Deferred edges are written to a bounded,
target-side staging table rather than retained in Go memory. They are drained
after the relevant vertex barrier. The job configures a maximum deferred-row
count and fails when that limit is exceeded; it never silently degrades to
quarantine.

## 6. AGE compatibility boundary

AGE stores a graph in a PostgreSQL schema and labels in inheriting tables.
Direct DDL and DML in that namespace are not recommended by the AGE manual.
Managed PostgreSQL also prevents the generic client from relying on
server-visible data files. The implementation therefore needs a narrow,
tested compatibility layer.

### 6.1 Adapter responsibilities

The AGE adapter:

- detects PostgreSQL and AGE versions
- detects whether AGE is preloaded
- initializes every pooled session with the required search path; when AGE is
  not preloaded or the setting is hidden, invokes an installed AGE C function
  to load the library and run `_PG_init` without requiring the superuser-only
  `LOAD` command
- creates and renames graphs through AGE functions
- creates labels through AGE functions
- reads label ID, relation OID, kind, and actual sequence name from
  `ag_catalog.ag_label`
- encodes and validates `graphid`
- selects an `agtype` conversion path supported by the detected AGE version
- controls COPY and staging transactions
- reserves entry-ID blocks from each catalog-resolved label sequence before
  writing explicit graphids and advances the sequence high-water mark
- runs post-load `ANALYZE`
- validates catalog invariants before incremental writes

The code must never derive the sequence name as `<label>_id_seq`. PostgreSQL can
truncate or disambiguate identifiers. The catalog value is authoritative.

### 6.2 AGE version policy

2.0.0 will support a deliberately small, documented matrix of AGE and
PostgreSQL versions. Unsupported combinations fail during validation rather
than attempting a best-effort load.

The adapter is tested against every supported matrix entry in containers. Tests
cover:

- graph and label creation
- long and quoted label names
- sequence discovery
- graphid boundary behavior
- every property type
- COPY rollback
- graph rename
- append after Cypher-created rows
- Cypher `CREATE` after an agefreighter load without an ID collision
- non-superuser sessions on a managed-PostgreSQL-style preloaded AGE setup
- catalog mismatch detection

### 6.3 Fast and safe write paths

`create` and the shadow graph used by `replace` can use direct client-side COPY
after IDs are safely allocated. There are no concurrent application writers in
that new graph. Explicit graphids are allocated by reserving a contiguous block
from the actual label sequence under a label-scoped lock. Sequence advancement
is allowed to leave gaps after rollback; IDs are never reused. This prevents a
later Cypher `CREATE` from receiving an entry ID already copied by
agefreighter.

`append` and `upsert` use ordinary PostgreSQL staging tables and transactional
merge operations because identity conflict detection, endpoint joins, and
idempotency are required.

The exact direct-COPY versus staging choice remains a benchmark decision. AGE
binary output for `agtype` is versioned text rather than a compact property
encoding, and pgx would require per-database custom type registration. Binary
COPY is therefore expected to benefit ordinary staging columns more than
direct AGE property loading. Both paths must produce identical data and
validation results.

## 7. Load semantics

### 7.1 `create`

- Fail if the target graph already exists.
- Create labels through AGE APIs.
- Load and verify the new graph.
- Commit data in resumable batches with a target-side job record.
- On a new job, fail if the target graph already exists.
- On `resume`, accept the existing graph only when it belongs to the same
  resumable job and graph generation.
- Preserve committed batches and checkpoints after interruption; roll back only
  the uncommitted batch.

`create` is the default because it cannot destroy data implicitly.

### 7.2 `replace`

- Load into a uniquely named shadow graph.
- Verify counts, identities, and endpoint resolution.
- Rename the existing graph to a backup name and the shadow graph to the target
  name in one transaction.
- Retain the old graph until the replacement is committed and cleanup is
  explicitly allowed.

AGE graph rename behavior and metadata-table generation changes must be tested
for every supported AGE version. Shadow and backup names are generated by
length-aware, hash-suffixed naming that stays within PostgreSQL's identifier
limit. The graph rename and promotion of the shadow graph generation in
`agefreighter_meta` occur in the same transaction.

### 7.3 `append`

- Keep existing records unchanged.
- Add previously unseen external IDs.
- Apply an explicit duplicate policy: `error` or `ignore-identical`.
- Reject conflicting duplicates.

### 7.4 `upsert`

Vertex identity is:

```text
graph generation + label + source namespace + external ID
```

Property behavior is explicit:

- replace all properties
- merge input keys
- merge and delete keys whose input value is null

AGE supports parallel edges. An edge cannot be identified only by label and
endpoints. Edge upsert requires a source namespace and external edge ID.

## 8. Target-side metadata and recovery

Metadata is stored in a normal PostgreSQL schema outside the graph schema:

```text
agefreighter_meta.graph_generation
agefreighter_meta.label_generation
agefreighter_meta.vertex_identity
agefreighter_meta.edge_identity
agefreighter_meta.load_job
agefreighter_meta.load_batch
agefreighter_meta.reject_record
```

The identity rows include graph namespace OID, label ID, label relation OID,
and mapping generation. Before an incremental job, the adapter compares these
values with the AGE catalog and stops on mismatch.

The data batch, identity rows, and successful checkpoint are committed in the
same PostgreSQL transaction. A process crash can therefore lead only to:

- the complete batch and checkpoint being committed, or
- neither being committed.

Rejections, batch-attempt details, and failed job status must survive a batch
rollback. They are written through a second target connection and independent
transaction. Every diagnostic row is keyed by job, batch, and attempt so replay
is idempotent. A diagnostic can therefore describe an attempt whose data
transaction was rolled back; its status records that outcome explicitly.

Local files and SQLite are not authoritative checkpoint stores.

The identity catalog has material storage cost. A load that will never use
append or upsert may opt out and discard it after verification.

## 9. Connector decisions

### 9.1 Delimited files

- RFC 4180-compatible CSV behavior where applicable
- configurable single-rune delimiter, quote, escape, header, encoding, and null
  rules
- gzip input by streaming decompression
- checkpoint only at a parser-confirmed record boundary
- input fingerprint includes size, modification time, and content sample/hash
- parallel byte-range reading only after quoted-record boundaries are proven

### 9.2 PostgreSQL source

- use `pgx`
- prefer `COPY (SELECT ...) TO STDOUT` or a server-side cursor
- use keyset pagination only with a stable, unique ordering key
- never use large `OFFSET` pagination
- share a `pg_export_snapshot()` snapshot between parallel source sessions
- fail validation when requested parallelism cannot provide a consistent
  snapshot

The snapshot-exporting transaction uses `REPEATABLE READ` or `SERIALIZABLE` and
remains open for the complete parallel read. Every importing transaction uses
the same isolation level and executes `SET TRANSACTION SNAPSHOT` before any
query. If those preconditions cannot be established, validation rejects
parallel snapshot mode.

### 9.3 Neo4j source

- use the official Neo4j Go driver
- stream records rather than returning eager result slices
- require an explicit stable source key
- do not use deprecated or reusable Neo4j internal IDs as durable identity
- do not use `SKIP` as the primary large-export strategy
- document that several read transactions over a changing source are not a
  point-in-time graph snapshot
- require a mapping policy for Neo4j nodes with multiple labels

### 9.4 Cosmos DB for NoSQL source

- use the official `azcosmos` Go SDK
- reuse client instances
- use parameterized queries
- consume pages with bounded concurrency
- persist page continuation state as diagnostic recovery information
- handle 429 throttling using SDK retry information and adaptive concurrency
- record request charge and diagnostics
- use `DefaultAzureCredential` with developer Azure CLI credentials locally and
  managed identity in hosted CI

Cosmos query pagination does not provide a transactional snapshot across a
changing container. The job records its selected consistency level and this
limitation.

As of this research, the managed Change Feed Processor is documented for .NET
and Java, not Go. Change Feed is therefore not a 2.0.0 commitment. A later
milestone may implement the pull model if the Go SDK exposes the required
continuation APIs with supportable semantics.

## 10. Local development with Apple Container

The installed environment provides Apple Container 1.0 on Apple Silicon.
Apple Container does not implement Docker Compose. It is the local developer
runtime. The repository will use a Makefile and small shell scripts containing
explicit `container` commands.

Hosted CI runs the equivalent database matrix on Linux with Docker because
GitHub-hosted macOS runners do not support the nested virtualization required
by Apple Container. Lifecycle logic has a small runtime abstraction, while
using the same image tags, ports, fixtures, readiness semantics, and tests.

Required local images:

- a pinned PostgreSQL plus Apache AGE image
- a pinned Neo4j Community image
- optional PostgreSQL source image if isolation from the AGE target is needed

Rules:

- pin images by version and, in release validation, digest
- bind published ports to `127.0.0.1`
- use named volumes for database storage
- poll service readiness explicitly with a timeout, using `pg_isready`, a
  PostgreSQL query, and a Neo4j Bolt/HTTP probe; capture container logs on
  failure
- give every resource an `agefreighter2-` prefix
- stop and delete only known named containers
- never use broad volume or image prune commands
- expose deterministic `dev-up`, `dev-down`, `dev-reset`, and `dev-status`
  targets

Apple Container custom-network bare-hostname DNS is not Compose-equivalent.
Initial host-run tests connect through published localhost ports. This avoids a
dependency on global DNS configuration or custom-network IP discovery.

The supported AGE image matrix is constrained to tags available for both the
developer's arm64 environment and Linux CI, or to images that the project can
reproducibly build for both architectures.

## 11. Azure integration environment

Azure resources are exclusively for Cosmos DB integration tests in the initial
architecture.

Proposed profile:

- one disposable Cosmos DB for NoSQL account
- one database
- vertex and edge fixture containers with high-cardinality partition keys
- serverless capacity for low-frequency development tests, subject to regional
  availability and subscription policy
- session consistency unless a test explicitly selects another level
- local developer identity granted a least-privilege Cosmos data-plane role
- local authentication through Azure CLI and `DefaultAzureCredential`
- local key authentication disabled

Azure artifacts are not generated or deployed until
`.azure/deployment-plan.md` is finalized and approved. The workflow is:

```text
azure-prepare -> azure-validate -> azure-deploy
```

The deployment plan must record the confirmed subscription, region, policy
constraints, quota or account-count check, cleanup policy, and budget.

## 12. Configuration and CLI

The initial configuration format is YAML with a published JSON Schema. JSON is
accepted because it is a YAML subset only if the selected parser preserves the
same validation semantics.

Example command surface:

```text
agefreighter validate job.yaml
agefreighter plan job.yaml
agefreighter load job.yaml
agefreighter resume --job job.yaml <job-id>
agefreighter status --target target.yaml <job-id>
agefreighter verify --target target.yaml <job-id>

agefreighter-tools fixture ...
agefreighter-tools inspect ...
agefreighter-tools benchmark ...
```

Configuration has a versioned API such as:

```yaml
apiVersion: agefreighter.io/v2
kind: LoadJob
```

Secrets are referenced through environment variables or identity-based
authentication. They are not embedded in the job document.

## 13. Error handling and observability

Every rejected record contains:

- job and batch ID
- connector and source location
- label
- external ID when available
- stable error code
- redacted diagnostic message
- optional quarantined raw record under an explicit data-retention setting

Expected failures are typed and classified as configuration, authentication,
source, data, target, transient, conflict, or internal errors. Broad catches
that convert failures into success are prohibited.

Metrics include:

- records and bytes read per second
- records transformed and written per second
- batch latency
- channel utilization and backpressure duration
- source retries
- Cosmos request units and 429 responses
- COPY duration
- rejection and missing-endpoint counts
- target verification counts
- process RSS and Go runtime statistics

Human-readable logs are the default for an interactive terminal. JSON logs and
OpenTelemetry export are available for automated runs.

## 14. Testing strategy

The repository enforces at least 90% statement coverage for the Go modules.
Generated files, command bootstrap lines, and deliberately unreachable platform
guards may be excluded only through reviewed, narrow rules.

The normal required CI workflow runs unit, race, and Linux container integration
tests and merges their coverage profiles before enforcing the 90% whole-module
gate. Azure tests are a separately required release/scheduled gate because they
need identity and incur cost. Cosmos packages must still reach the normal gate
through SDK-interface and HTTP-transport contract tests.

Go has no source-level coverage exclusion directive. Any exclusions use
build-tagged generated/platform files plus a reviewed coverage-profile filter
script with an allowlist. The filter itself is tested, and the unfiltered
coverage report is retained as a CI artifact.

Coverage is not the primary quality measure. The test portfolio includes:

### 14.1 Unit tests

- configuration parsing and semantic validation
- mapping compilation
- every type conversion
- batch sizing and backpressure
- retry classification
- graphid operations
- identifier quoting
- checkpoint state transitions

### 14.2 Property and fuzz tests

- CSV parser and arbitrary delimiters
- property encoder round-trips
- graphid boundary values
- identifier and Unicode handling
- malformed configuration
- interrupted stream framing

Every discovered fuzz failure becomes a deterministic regression test.

### 14.3 Contract tests

Each connector is tested against a fake server or narrow interface contract for:

- pagination
- cancellation
- retries
- partial result errors
- token/checkpoint persistence
- bounded memory behavior

Cosmos tests use the Azure SDK's injectable HTTP transport where possible.
Neo4j does not get a hand-written fake Bolt server; interface-level seams cover
driver result handling, and protocol behavior is tested against the pinned
Neo4j container.

### 14.4 Container integration tests

- every supported PostgreSQL/AGE matrix entry
- Neo4j source extraction
- PostgreSQL source snapshot behavior
- transaction rollback after injected failure
- resume without duplicates
- replace failure preserving the old graph
- append conflict behavior
- endpoint rejection and quarantine

### 14.5 Azure integration tests

- Entra ID authentication
- multi-partition Cosmos scans
- continuation over multiple pages
- RU throttling behavior where practical
- cancellation and resume diagnostics

Azure tests are opt-in locally and scheduled in controlled CI because they incur
cost and require cloud identity.

### 14.6 End-to-end and performance tests

Correctness fixtures cover empty graphs, isolated vertices, self-loops,
parallel edges, long labels, Unicode, nulls, nested properties, duplicate IDs,
and missing endpoints.

Performance tests measure end-to-end throughput, peak RSS, target WAL, database
CPU, recovery time, and correctness. They do not fail normal CI on noisy timing
alone. A controlled benchmark job enforces release performance budgets.

## 15. Preliminary acceptance targets

- At least 90% statement coverage on every normal CI run
- `go test -race ./...` passes
- fuzz smoke tests pass in CI and longer fuzzing runs are scheduled
- static analysis and vulnerability checks pass
- configured memory limit is not exceeded by input-size growth
- no temporary CSV in the default path
- no unbounded goroutine or request creation
- failed `replace` leaves the previous graph usable
- resume creates no duplicate vertices or edges
- unresolved endpoints are fully accounted for
- all supported property types round-trip for every supported AGE version
- CSV `create` throughput is at least twice the measured 1.x baseline, with a
  stretch target of five times
- AGE write throughput reaches at least 70% of a plain relational COPY baseline
  using the same payload, equivalent indexes, and durability settings

These performance thresholds are hypotheses until the baseline and proof of
concept are measured.

## 16. Principal risks

| Risk | Mitigation |
| --- | --- |
| AGE direct table writes rely on implementation details | Isolate adapter, minimize version matrix, exhaustive integration tests |
| AGE version changes alter agtype or catalog behavior | Startup capability detection and round-trip tests |
| Identity metadata grows to graph scale | Explicit sizing, indexes, opt-out for non-incremental loads |
| External Cypher writes invalidate identity metadata | Single-writer contract and catalog consistency checks |
| Go GC increases RSS or stalls the pipeline | Reused buffers, bounded queues, `GOMEMLIMIT`, profiles before optimization |
| Cosmos scans observe live changes | Record consistency contract; add pull-model incremental support later |
| Apple Container behavior differs from CI runtime | Keep scripts OCI-compatible in image choice; test commands explicitly |
| Coverage target encourages shallow tests | Mutation/fuzz/contract/integration requirements and defect regression policy |

## 17. Decisions requiring later confirmation

These decisions are intentionally deferred until the proof of concept:

- exact supported PostgreSQL and AGE version matrix
- direct text COPY versus staged binary COPY per load mode
- source-type to agtype mapping, including Neo4j temporal/spatial values,
  nested Cosmos documents, lossy-conversion policy, and integer overflow
- whether the external identity is also copied into AGE properties
- metadata retention defaults after `create` and `replace`
- initial YAML library and CLI framework
- re-evaluate `go.yaml.in/yaml/v4` only after a stable v4 release
- serverless Cosmos regional availability and selected region
- whether Parquet belongs in 2.1 or a later release
- arm64 image availability for every supported local AGE matrix entry

## 18. Sources

- PostgreSQL COPY:
  <https://www.postgresql.org/docs/current/sql-copy.html>
- PostgreSQL synchronized snapshots:
  <https://www.postgresql.org/docs/current/functions-admin.html#FUNCTIONS-SNAPSHOT-SYNCHRONIZATION>
- Apache AGE graph storage:
  <https://age.apache.org/age-manual/master/intro/graphs.html>
- Apache AGE graph implementation:
  <https://github.com/apache/age/blob/master/src/backend/commands/graph_commands.c>
- Apache AGE label implementation:
  <https://github.com/apache/age/blob/master/src/backend/commands/label_commands.c>
- Apache AGE agtype coercions:
  <https://github.com/apache/age/blob/master/sql/agtype_coercions.sql>
- pgx:
  <https://pkg.go.dev/github.com/jackc/pgx/v5>
- Neo4j Go Driver:
  <https://neo4j.com/docs/go-manual/current/>
- Azure Cosmos DB Go SDK:
  <https://learn.microsoft.com/azure/cosmos-db/sdk-go>
- Cosmos DB pagination:
  <https://learn.microsoft.com/azure/cosmos-db/nosql/query/pagination>
- Cosmos DB role-based access:
  <https://learn.microsoft.com/azure/cosmos-db/how-to-connect-role-based-access-control>
- Cosmos DB Change Feed Processor support:
  <https://learn.microsoft.com/azure/cosmos-db/change-feed-processor>
- Apple Container command reference:
  <https://github.com/apple/container/blob/main/docs/command-reference.md>
- Apple Container networking:
  <https://github.com/apple/container/blob/main/docs/networking.md>
- Apple Container volumes:
  <https://github.com/apple/container/blob/main/docs/volumes.md>
