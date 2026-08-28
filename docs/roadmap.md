# agefreighter 2.1 implementation plan

- Status: proposed
- Baseline: agefreighter 2.0.0
- Target compatibility: PostgreSQL 17.x and Apache AGE 1.6.x

## Release theme

2.1 focuses on **migration assurance and operational readiness**. It extends
the durable job, checkpoint, identity, and verification foundation delivered in
2.0 instead of introducing another migration path.

The release should answer four questions:

1. Is the target ready before a migration starts?
2. What was migrated, and can the result be audited?
3. Does the target contain the expected graph and property data?
4. What safe, evidence-based changes would improve the migrated graph?

## Candidate review

| Candidate | Decision for 2.1 | Rationale |
|---|---|---|
| Post-load optimization | Include a bounded, recommendation-first MVP | `ANALYZE` support already exists internally, but post-load statistics and property-index recommendations are not exposed. |
| Migration validator | Extend `verify`; do not add another `validate` command | `validate` already means offline job validation, while `verify` already checks committed jobs, configuration fingerprints, graph generations, label catalogs, row counts, and graph IDs. |
| Cypher compatibility analyzer | Include a static-analysis MVP in `agefreighter-tools` | It reduces application-cutover risk and can later provide workload evidence to the optimizer. Full parsing and automatic rewriting are out of scope. |
| Workload advisor | Merge the offline portion into `optimize` | Query-file analysis overlaps index recommendation. Collection from `pg_stat_statements` is deferred because query normalization and attribution need a separate design. |
| Schema and constraint migration | Defer; inventory before translation in 2.2 | Automatic creation of uniqueness, full-text, vector, or existence constraints can change write semantics and requires a broader compatibility contract. |
| `doctor` | Include | It is read-only by default and valuable for support, but requires a new degraded PostgreSQL probe so it can diagnose missing or unsupported AGE installations. |
| Graph profiler | Include a bounded source-profile MVP | Existing Neo4j and Cosmos discovery finds mappings, not complete counts, cardinality, size, or migration estimates. |
| Target graph benchmark | Do not add a new public command in 2.1 | `benchmark-age-copy`, benchmark reports, and the release CSV benchmark already cover loader regression. A representative query benchmark requires user workload definitions. |
| Source-versus-AGE benchmark | Defer | Cross-engine plans, caches, consistency, and query semantics make a generic comparison easy to misrepresent. |
| Incremental migration and live cutover | Defer beyond 2.1 | CDC, ordering, replay, deletion semantics, schema changes, and cutover coordination form a separate large release. |
| Migration resume and checkpoint | Remove from the candidate list | Already implemented for 2.0 with durable jobs, batches, source tokens, configuration fingerprints, and connector-specific resume contracts. |
| Migration manifest | Add a report over existing metadata | Most manifest data already exists in `agefreighter_meta` and command JSON. 2.1 should export it rather than create a second state store. |
| Graph and vector migration | Defer | Requires a pgvector compatibility matrix, embedding type conversion, distance semantics, and index lifecycle management. |

## Scope

### P0: required for 2.1

1. Versioned migration report
2. Target health check with `doctor`
3. Deep post-load count and integrity verification
4. Post-load optimization report with opt-in `ANALYZE`

### P1: include after P0 contracts are stable

5. Bounded source graph profile
6. Static Cypher compatibility analysis
7. Query-file evidence for index recommendations

### Explicit non-goals

- CDC or continuous synchronization
- automatic cutover orchestration
- automatic index or constraint creation
- automatic Cypher rewriting
- arbitrary Cypher execution during verification
- persisted source-property samples or sample property verification
- live workload capture from `pg_stat_statements`
- source-versus-target latency claims
- vector property or vector-index migration
- support beyond the published PostgreSQL and AGE compatibility matrix

## Command contract

All production commands continue to accept a load-job document rather than raw
credentials or connection strings. JSON remains the default machine-readable
output. Markdown is an explicit presentation format.

```text
agefreighter doctor --target JOB [--format json|markdown]

agefreighter report --target JOB JOB_ID
  [--format json|markdown]
  [--output FILE]
  [--include-counts]
  [--limit-batches N]

agefreighter verify --target JOB JOB_ID
  [--level catalog|counts]

agefreighter profile JOB
  [--mode sample|exact]
  [--sample-size N]
  [--format json|markdown]

agefreighter optimize --target JOB
  [--queries FILE ...]
  [--apply-analyze]
  [--format json|markdown]

agefreighter-tools check-cypher FILE ...
  [--target-age 1.6]
  [--format json|markdown]
```

Command rules:

- `doctor`, `report`, `profile`, `verify`, and `optimize` are read-only by
  default.
- `optimize --apply-analyze` is the only mutating operation among the new
  diagnostic and advisory commands.
- `verify` without `--level` preserves the 2.0 catalog-verification behavior.
- `profile --mode sample` is the default. Exact source scans require an
  explicit flag.
- Query inputs and generated reports have fixed file-count, byte, query-count,
  property-count, and execution-time limits.
- Reports never contain resolved secrets, raw credentials, or unrestricted
  source records.

## Target access and metadata upgrade policy

The current 2.0 target-opening path runs metadata migrations before returning a
connection. It cannot be reused unchanged by diagnostics that must detect
missing AGE, unsupported versions, or stale metadata without modifying them.

2.1 must add:

- a plain PostgreSQL degraded probe that does not require AGE to be installed
  or loadable
- a read-only AGE target path that validates capabilities without calling
  metadata migration code
- a read-only metadata-version probe that reports pending or unsupported
  migrations
- explicit separation between capability detection, metadata inspection, and
  metadata mutation

`doctor`, `report`, `profile`, `verify`, and recommendation-only `optimize` use
the read-only paths. They must never create a schema, apply a metadata
migration, or repair the target implicitly.

Only `load` and `resume` may apply 2.1 metadata migrations. The upgrade is
one-way: after a target is upgraded beyond the 2.0 metadata schema, 2.0
binaries reject it as newer than supported. Operators must upgrade all writers
before the first 2.1 `load` or `resume`. This policy must be documented and
covered by mixed-version integration tests.

## Milestone 1: report foundation

### Deliverables

- Add a shared versioned report envelope containing:
  - schema version
  - command and agefreighter version
  - generation time
  - job ID and configuration fingerprint where applicable
  - target PostgreSQL and AGE versions
  - warnings, errors, and incomplete checks
- Add deterministic JSON and Markdown renderers.
- Add metadata read APIs for jobs, graph generations, label generations,
  batches, reject summaries, and retained replacement backups.
- Persist bounded connector telemetry summaries for jobs created by 2.1.
- Implement `agefreighter report`.
- Keep PostgreSQL metadata as the source of truth; do not write a parallel
  `.agefreighter` manifest file.

### Report contents

- job status and timestamps
- source type and load mode
- committed and rejected row counts
- batch attempts and last committed source positions
- graph and label generation identities
- stored per-label counters when available
- replacement and backup state
- persisted source telemetry for 2.1 jobs

### Acceptance criteria

- Repeated report generation over unchanged metadata is byte-stable except for
  the envelope generation time.
- JSON output has a checked-in schema and golden compatibility tests.
- Missing optional metadata is represented as unavailable, not as a successful
  zero value.
- Output is bounded for jobs with many batches; detailed batch output requires
  `--limit-batches`.
- Default output does not scan identity or graph tables. Exact identity counts
  require `--include-counts` and explicit statement timeouts.
- Reports for 2.0 jobs mark new counters and telemetry as unavailable.
- Permission and metadata-version failures are reported without partial
  success being presented as a passing report.

## Milestone 2: `doctor`

### Deliverables

Implement a target-focused health report that checks:

- PostgreSQL version and supported compatibility range
- AGE extension installation and version
- AGE loadability and required search path
- `agefreighter_meta` schema version and migration health
- target graph, namespace, label, relation, and sequence catalog consistency
- required identity and metadata indexes
- active, loading, retired, and retained-backup generations
- database size, free-storage visibility, and WAL visibility where permissions
  allow
- autovacuum settings, dead-tuple indicators, and last analyze timestamps
- long-running or conflicting agefreighter jobs

The command first uses the degraded PostgreSQL probe, then progressively adds
AGE and agefreighter metadata checks when each capability is available. A
missing AGE extension, unsupported version, or stale metadata schema is a
diagnostic result rather than a connection-path failure.

### Safety constraints

- The default check must not scan all vertex or edge rows.
- Optional checks use statement and lock timeouts.
- Lack of permission yields `unknown` with remediation guidance, not `healthy`.
- Recommendations must include the evidence and catalog object that triggered
  them.

### Acceptance criteria

- A healthy pinned development target produces a passing report.
- Missing AGE, unsupported versions, stale metadata, changed label catalogs,
  and concurrent writers are covered by integration tests.
- The command makes no persistent database changes.
- Every database operation has a statement timeout, and the default command
  performs no sequential scan of an AGE label relation.

## Milestone 3: deep verification

### Existing behavior to preserve

The 2.0 verifier already checks:

- committed job state
- unchanged configuration fingerprint
- active graph generation and expected graph name
- graph and label catalog identity
- AGE label-table rows against durable identity counts
- graph IDs against the expected AGE label ID

These checks remain `--level catalog` and the default.

### New verification level

`--level counts` adds:

- accepted, committed, and attributable rejected totals by vertex and edge
  label, plus an `unclassified` reject bucket
- identity graph IDs missing from physical label tables
- physical label rows with no matching identity
- edge-identity endpoint IDs versus vertex-identity rows
- physical edge endpoints that differ from the durable edge identity
- generation ownership and retained-backup isolation

### Metadata changes

- Store per-label accepted, rejected, and committed counters aggregated per
  batch rather than written per record.
- Store separate fingerprints for the submitted job and the resolved discovery
  mappings.
- Persist the resolved mapping summary required to verify 2.1 jobs without
  reconnecting to a mutable source.
- Version metadata migrations and retain compatibility with 2.0 jobs.

For 2.1 jobs, `verify` uses the stored resolved-mapping fingerprint and does not
rerun Neo4j or Cosmos discovery. For 2.0 discovery jobs, source rediscovery may
remain as an explicit compatibility fallback; the output must state that
source access was required and that it is not an original migration snapshot.

### Acceptance criteria

- All four connectors pass create, replace, append, and upsert count
  verification.
- Deliberate row deletion, wrong graph IDs, missing identities, dangling edges,
  endpoint changes, orphan physical rows, and backup contamination are
  detected.
- Existing 2.0 jobs remain verifiable at `catalog` level.
- Verification limits and timeouts fail explicitly rather than degrading to a
  shallower successful check.

Semantic comparison by executing arbitrary source and target Cypher is deferred.
Neo4j Cypher and AGE openCypher do not have a sufficiently equivalent execution
contract for 2.1 to compare arbitrary results safely.

## Milestone 4: source graph profile

### Deliverables

Implement a common profile model for CSV, PostgreSQL, Neo4j, and Cosmos DB:

- source version and connector mode
- resolved vertex labels and relationship types
- exact or estimated counts with the method recorded
- property presence and observed type distribution
- bounded null and distinct-value estimates
- endpoint-label combinations
- input bytes where the connector exposes them
- estimated target rows, staging space, and a documented storage range

Connector behavior:

- CSV uses file metadata and bounded parsing; exact mode scans configured files.
- PostgreSQL wraps configured source queries only where a safe count query can
  be generated; otherwise it streams and counts in exact mode.
- Neo4j reuses automatic discovery and issues bounded count/profile queries.
- Cosmos reuses NoSQL/Gremlin discovery, records request charge, and requires
  `--mode exact` as explicit approval for cross-partition exact scans.

### Estimation rules

- Estimates are ranges, never single-value capacity promises.
- Every estimate records sample size, source timestamp, and method.
- Migration-time estimates are based only on a user-selected or recorded
  throughput baseline.
- Disk recommendations include graph rows, identity metadata, staging, WAL,
  replacement shadow graphs, and retained backups.

### Acceptance criteria

- Sample mode enforces row, byte, property, label, request-charge, and timeout
  limits.
- Exact mode is opt-in and clearly identifies source consistency assumptions.
- The same fixture produces equivalent label and relationship summaries across
  connectors.
- No profile operation writes to the source or target.

## Milestone 5: post-load optimizer

### Deliverables

Implement `agefreighter optimize` as an evidence-producing advisor:

- report label row counts and edge density
- report last analyze time and statistics freshness
- verify required AGE and agefreighter identity indexes
- detect exact duplicate indexes
- report unused-index statistics with the statistics-reset timestamp
- sample property presence, type, null rate, and cardinality within configured
  limits
- recommend B-tree expression indexes proven by integration tests; recommend a
  GIN strategy only when the target capability probe finds a supported AGE 1.6
  operator class
- emit quoted, reviewable SQL without executing it
- run catalog-safe `ANALYZE` only with `--apply-analyze`

Recommendation confidence:

- `high`: supported query predicate plus adequate selectivity evidence
- `medium`: supported query predicate with incomplete statistics
- `low`: data-only observation without workload evidence
- `not-recommended`: low selectivity, unsupported type, or insufficient
  evidence

### Query evidence

`--queries` consumes the bounded output model from `check-cypher` and recognizes
only:

- label references
- equality predicates
- range predicates
- ordering properties
- supported containment patterns

Unknown syntax must not produce an index recommendation.

### Deferred optimizer behavior

- automatic property-index creation or deletion
- automatic unique-constraint creation
- hypothetical-index claims without a supported facility
- live `pg_stat_statements` workload collection
- source index translation and application

### Acceptance criteria

- Default execution performs no DDL or statistics mutation.
- Suggested SQL uses PostgreSQL identifier quoting and parameter-safe literal
  handling.
- Existing, duplicate, low-selectivity, and unsupported-property cases have
  deterministic tests.
- `--apply-analyze` validates graph and label catalogs immediately before each
  operation and reports partial completion explicitly.
- Profiling remains bounded on large labels.

## Milestone 6: Cypher compatibility MVP

### Deliverables

Implement `agefreighter-tools check-cypher` with a versioned rule catalog for
the supported AGE release:

- split bounded query files without executing them
- identify common Neo4j-only procedures and functions
- identify supported, unsupported, and unknown syntax
- recognize predicates needed by `optimize --queries`
- report file, line, rule ID, severity, evidence, and remediation text
- produce summary counts and a compatibility score that excludes unknown
  queries from false-success claims

Classifications:

- `compatible`
- `compatible-with-manual-change`
- `unsupported`
- `unknown`

### Constraints

- Use a deterministic lexer and bounded structural analysis, not an incomplete
  full Cypher parser presented as authoritative.
- Do not send application queries to OpenAI automatically.
- Do not rewrite source files in 2.1.
- OpenAI-assisted conversion remains an explicit
  `agefreighter-tools convert-gremlin` operation.

### Acceptance criteria

- A checked-in corpus covers supported AGE syntax, Neo4j-specific procedures,
  nested strings/comments, parameters, and malformed queries.
- Rules are tied to an AGE version and fail closed for an unknown target.
- Repeated analysis is deterministic and performs no network access.
- Unsupported and unknown queries cause a nonzero exit code under an explicit
  `--strict` flag.

## Cross-cutting implementation requirements

### Security

- Reuse load-job secret references; never accept literal secrets in reports.
- Redact DSNs, API keys, tokens, source values, and query parameters.
- Treat report paths and query files as untrusted input.
- Put size and time limits on every source, database, and file operation.
- Keep all recommendation commands read-only unless the user supplies the
  specific mutating flag.

### Compatibility

- Preserve all 2.0 command behavior and JSON fields.
- Add new JSON through versioned report schemas.
- Support existing 2.0 metadata without destructive migration.
- Extend `age.Capabilities` with the catalog and permission probes needed by
  diagnostics; keep version-dependent behavior behind this capability model.

### Testing

- Unit tests for report schemas, renderers, redaction, limits, rule catalogs,
  estimators, and recommendation confidence.
- PostgreSQL/AGE integration tests for metadata migrations, doctor failures,
  deep verification corruption, index inspection, and `ANALYZE`.
- Connector matrix coverage for source profiles and per-label counts.
- Golden JSON and Markdown reports with deterministic ordering.
- Fuzz targets for Cypher splitting, report decoding, and identifier handling.
- Race tests for parallel read-only diagnostics.

### Performance gates

- The calibrated CSV create floors remain passing: 109,190 rows/s on the M4 Max
  calibration host and 50,000 rows/s on hosted release runners.
- Per-label metrics are aggregated per batch and do not add per-row metadata
  writes.
- No input-size-dependent client-memory growth in default profile and
  diagnostic modes.
- `doctor` and default `report` must avoid full graph scans.
- Every exact or potentially expensive operation must expose progress through
  structured diagnostics and obey cancellation and statement timeouts.

## Delivery order

| Phase | Work | Dependency |
|---|---|---|
| 1 | Read-only target access, degraded probes, report schemas, renderers, and metadata read APIs | None |
| 2 | `report` and `doctor` | Phase 1 |
| 3 | Per-label metrics and deep `verify` | Phases 1–2 |
| 4 | Common source profile and connector implementations | Phase 1 |
| 5 | Read-only optimizer and opt-in `ANALYZE` | Phases 1–2 |
| 6 | Cypher compatibility MVP and optimizer query evidence | Phases 1 and 5 |
| 7 | Connector E2E matrix, corruption tests, performance gates, and docs | Phases 2–6 |

Phases 3 and 4 can proceed independently after the report contracts are
stable. The release is complete only when all P0 items and their cross-connector
tests pass. P1 items may be moved to 2.2 rather than weakening limits,
determinism, or compatibility guarantees.

## Deferred roadmap

### 2.2 candidates

- Neo4j index and constraint inventory with AGE translation reports
- deterministic digest-based property sample verification for replace-mode
  migrations
- live workload advice from `pg_stat_statements`
- representative target graph benchmarks driven by user query suites
- reviewed application of supported property indexes
- optional manual Cypher rewrite suggestions
- pgvector property and index compatibility research

### 3.0 or separate major initiative

- source change-data capture
- ordered replay with inserts, updates, and deletions
- continuous synchronization and lag reporting
- schema-change handling during synchronization
- cutover readiness and rollback orchestration
- source-versus-target semantic and latency comparison

These items require new consistency, recovery, and compatibility contracts and
must not be added incrementally to the 2.1 bulk-migration pipeline.
