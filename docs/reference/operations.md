# Operations guide

## Preflight

1. Keep the load-job file and referenced secret files readable only by the
   service account.
2. Run `agefreighter validate JOB` and review `agefreighter plan JOB`.
3. Confirm the target reports an exact PostgreSQL and Apache AGE pairing listed
   in the [compatibility matrix](compatibility.md).
4. Confirm that the configured memory, batch, concurrency, and timeout limits
   fit the host and database capacity.
5. For `replace`, verify that enough database storage exists for the active,
   shadow, and retained backup graphs at the same time.
6. Save the command's JSON output and job ID in the deployment record.

## Run and monitor

```sh
agefreighter load job.yaml
agefreighter status --target job.yaml JOB_ID
agefreighter verify --target job.yaml JOB_ID
agefreighter report --target job.yaml JOB_ID
```

Standard output is reserved for command results. Diagnostic logs use standard
error and are opt-in; see [observability](observability.md). Never place a
literal password, token, or connection string in the job document or command
arguments.

Source-specific consistency and resume constraints are part of the
[configuration contract](configuration.md). In particular, PostgreSQL
snapshot modes, Neo4j per-mapping transactions, Cosmos paging, and CSV file
stability have different restart boundaries.

## Bounded source profiles

`agefreighter profile JOB` validates and resolves the configured source
mappings, then reads only the source. It never resolves the target credential,
opens the target, creates metadata, writes rejects, or changes the load-job
document. JSON is the default; `--format markdown` uses the shared deterministic
report renderer.

```sh
agefreighter profile job.yaml
agefreighter profile --sample-size 25000 job.yaml
agefreighter profile --mode exact --format markdown job.yaml
```

Sample mode defaults to 10,000 rows and is capped at 100,000 requested rows,
64 MiB each of raw and decoded input, 100,000 connector pages, 1,000 Cosmos
request units, 64 resolved mappings, 256 properties, 1,024 distinct hashes per
property, and the job's operation timeout. Discovery and sampling share these
cumulative limits; every successful connector page response is charged once
(including an empty terminal PostgreSQL cursor/keyset response), and malformed
or unmapped input is included. A reached bound is `unknown` and the report outcome
is `incomplete`; observed counts are never extrapolated into a false total.
Exact mode is explicit approval to stream each configured mapping, including
Cosmos cross-partition queries. It still fails closed at 1,000,000 rows, 1 GiB
each of raw and decoded input, 1,000,000 pages, 10,000 request units, and the
operation timeout, so it is exact only when discovery and every mapping reach
end of input.

CSV uses the configured delimiter, quote, escape, header, encoding, and null
rules and reports cumulative input-byte telemetry. PostgreSQL executes the configured
queries unchanged in the connector's repeatable-read, read-only snapshot; it
does not add `COUNT(*)`, `ANALYZE`, or write statements. Neo4j and Cosmos reuse
their bounded discovery and read-only iterator paths. Cosmos Gremlin profile
discovery uses one bounded catalog scan rather than separate large vertex and
edge scans. Connector retry and rate-limit behavior is unchanged.

Reports contain only aggregate facts: resolved labels and endpoint-label
combinations, sampled rows and bytes, missing identity/endpoint/property
signals, nulls, observed value kinds, hashed distinct-value indicators, value
widths, and non-binding storage ranges. No raw value, record identity, source
position, query, DSN, credential reference, secret, or continuation token is
rendered. Storage ranges are capacity indicators, not promises: graph data is
modeled at 2–4 times observed logical bytes, identity metadata at 128–384 bytes
per observed row, staging at 1–2 times logical bytes, and WAL at 1–2 times the
graph high range. Replace-mode ranges also include a shadow and retained
backup. Migration time remains unavailable unless a trustworthy recorded or
user-selected throughput baseline exists.

Source versions not exposed by the existing iterator contract and other
unavailable facts remain explicitly unavailable, preventing a false `pass`.
The report contract is
[`source-profile.schema.json`](source-profile.schema.json).

## Recommendation-first target optimization

`agefreighter optimize --target JOB` opens the target through the read-only
diagnostic path and produces evidence and recommendations only. It does not run
`ANALYZE`, DDL, arbitrary SQL, `VACUUM`, `REINDEX`, configuration changes, or
metadata migrations. JSON is the default; Markdown and exclusive mode-`0600`
output files use the shared report renderer.

```sh
agefreighter optimize --target job.yaml
agefreighter optimize --target job.yaml --format markdown --output optimizer.md
agefreighter optimize --target job.yaml \
  --queries service-a.cypher --queries service-b.cypher
```

The report covers PostgreSQL, AGE, and metadata versions; migration and
connector counters; estimated graph and label sizes and edge density; database,
WAL, dead-tuple, analyze, and statistics-reset visibility; required and exact
duplicate indexes; and zero-scan indexes. Lists are deterministic and bounded
to 64 active labels, 64 indexes per relation, 1,000 batch attempts, and 128
index and recommendation entries. Truncation, permission failures, and
timeouts produce `unknown` or `unavailable` evidence and an incomplete
outcome. Values, graph IDs, external IDs, source positions, records, query
text, SQL error bodies, DSNs, and secrets are never reported.

Live AGE property parsing and cardinality inspection are disabled for this
milestone. Apache AGE 1.6 serializes and fully detoasts an `agtype` value before
a text substring can cap the result, so the optimizer cannot safely pre-bound
property serialization. The report explicitly marks property statistics and
data-only property-index recommendations unavailable and emits no property
recommendation from absent evidence. When `--queries` provides a compatible,
structurally proven label plus equality/range/order property pattern, the
report may emit a deduplicated, review-only AGE expression-index candidate at
`medium` confidence because selectivity remains unavailable. Containment-only,
ambiguous-label, unsupported, and unknown patterns remain evidence only. The
target capability check may still
report whether an allowlisted AGE 1.6 `agtype` GIN operator class exists, but
that fact alone does not produce an index recommendation.

Exact duplicate and zero-scan evidence is advisory: statistics-reset time and
application requirements must be reviewed before any manual DDL. The optimizer
makes no source-versus-target performance claim. Required AGE indexes must be
valid, ready, non-partial B-tree indexes with exact keys and ordering; `id`
must be a unique primary key, while `start_id` and `end_id` must be non-unique,
non-primary edge indexes. Required-index checks use separate targeted,
uncapped catalog probes, so a capped alphabetical index-display list is never
used to infer that an index is missing. Recoverable evidence probes use
savepoints inside the shared repeatable-read snapshot.

`--apply-analyze` is the sole optimizer mutation opt-in:

```sh
agefreighter optimize --target job.yaml --apply-analyze
```

It requires PostgreSQL 17, AGE 1.6, a current v17 metadata schema, a complete
allowlisted metadata catalog, and a non-truncated active graph label catalog.
Each operation revalidates relation ownership and catalog identity, quotes the
identifier, and runs in its own transaction with the configured command
deadline plus local statement and lock timeouts. Before revalidation it takes
and retains a `SHARE UPDATE EXCLUSIVE` lock on the quoted relation, so a
concurrent drop, rename, or replacement either waits or causes exact OID and
ownership validation to fail safely. Only the exact current agefreighter
metadata relations and active graph label relations are eligible.
The report gives attempted, succeeded, and failed counts and a sanitized result
per relation; partial completion is incomplete. Cancellation stops further
work and propagates to the caller. No optimizer history is persisted.

Automatic schema or index changes and live `pg_stat_statements` collection are
intentionally deferred. The JSON contract is
[`optimizer-report.schema.json`](optimizer-report.schema.json).

## Static Cypher compatibility

`agefreighter-tools check-cypher FILE ... [--target-age 1.6]
[--format json|markdown]` analyzes regular local UTF-8 files.
It never reads standard input, follows symlinks, executes a query, opens a
database, uses the network, invokes an LLM, or rewrites input.
Each filesystem operation has a fixed two-second limit, and the complete
analysis has a fixed 30-second limit. At most 64 filesystem workers can remain
occupied by uninterruptible operating-system calls.

```sh
agefreighter-tools check-cypher app.cypher \
  --format markdown
agefreighter-tools check-cypher app.cypher \
  --format json --strict
```

The AGE 1.6 rule catalog reports each query as `compatible`,
`compatible-with-manual-change`, `unsupported`, or `unknown`. Findings contain
stable rule codes, file/query/line/column locations, sanitized bounded
evidence, and remediation. Strings, numbers, parameter names/values, comments,
and uncataloged identifiers are not copied into evidence snippets. An unknown
query is never counted as supported, and the compatibility percentage is
withheld whenever unknown queries exist.

Analysis is capped at 64 files, 1 MiB per file, 8 MiB total, 1,024 queries,
8,192 tokens per query, nesting depth 128, 4,096 findings, and 4 MiB output.
Cancellation is checked during reads, lexing, output, and between queries.
Reports use basenames, with deterministic opaque IDs for duplicate basenames;
directories and absolute paths are never emitted. Inputs and
findings/patterns are canonically sorted.
`--target-age` fails closed for anything other than the cataloged `1.6`.
The JSON contract is
[`cypher-compatibility-report.schema.json`](cypher-compatibility-report.schema.json).

Exit status is zero for compatible queries, warnings/manual changes, and
non-strict unsupported or malformed/unknown reports. `--strict` returns
nonzero after writing the report when any query is unsupported or unknown.
Flag errors, unsafe/special files, limit violations, cancellation, I/O errors,
and output errors always return nonzero. Query manifests and automatic
rewrites are not part of the 2.1 input contract; pass each local file directly.

## Failure and recovery

Do not start a second writer for the same target graph. First inspect the
durable status:

```sh
agefreighter status --target job.yaml JOB_ID
```

For a failed or interrupted job whose source still satisfies its documented
resume contract:

```sh
agefreighter resume --job job.yaml JOB_ID
agefreighter verify --target job.yaml JOB_ID
```

Resume rejects a changed configuration fingerprint, incompatible graph
generation, or committed job. It resumes from the last committed batch and
does not infer success from process exit alone.

An interrupted `replace` leaves the prior graph active until atomic promotion.
After a committed replacement has been verified and its rollback retention
window has elapsed, remove only that job's retained backup:

```sh
agefreighter cleanup --target job.yaml JOB_ID
```

`cleanup` is intentionally scoped to a committed replacement job. Do not use
manual `DROP SCHEMA`, `drop_graph`, metadata deletion, or broad container
cleanup as a recovery shortcut.

## Migration reports

`agefreighter report --target JOB JOB_ID` reads durable metadata without
running metadata migrations or scanning graph and identity tables. JSON is the
default; use `--format markdown` for review output. The report includes the
job and graph generations, a bounded label catalog and reject summary, the
latest checkpoint, replacement-backup state, target versions, and connector
telemetry recorded by 2.1 loads.

Batch details are opt-in and bounded:

```sh
agefreighter report --target job.yaml --limit-batches 20 JOB_ID
```

Exact per-label identity counts require `--include-counts`. Every count has
both a client deadline and PostgreSQL statement timeout; timeout or permission
failure is reported as incomplete rather than as zero. `--output FILE` creates
a new regular file with mode `0600` and refuses existing paths and symlinks.
The JSON contract is
[`migration-report.schema.json`](migration-report.schema.json).

## Deep verification

The default `verify` remains the read-only catalog-level check and keeps its
2.0 JSON job output. Opt in to 2.1 persisted-versus-live counts and bounded
identity/endpoint consistency checks:

```sh
agefreighter verify --target job.yaml --counts JOB_ID
agefreighter verify --target job.yaml --integrity --limit 100 JOB_ID
agefreighter verify --target job.yaml --counts --integrity \
  --format markdown --output verification.md JOB_ID
```

`--level counts` is equivalent to `--counts`. Integrity selection is ordered
by graph ID and capped at 1,000 rows per identity and physical-label window;
the default is 100. A clean truncated sample is `incomplete`, never a pass.
Counts compare persisted committed-row counters with exact live physical-label
totals, using client deadlines and PostgreSQL statement timeouts. The resolved
mapping snapshot records whether each label guarantees an external identity
for every accepted row. Full-coverage labels additionally require exact
physical/identity count equality and check both identity-to-physical and
physical-to-identity integrity. Edge labels whose mappings intentionally omit
external IDs still check every persisted identity row, its physical edge, and
its endpoints, but report reverse physical-to-identity coverage as unavailable
and the outcome as incomplete. Snapshot version 1 edge labels receive the same
conservative treatment because their coverage capability was not recorded.
Missing legacy counters, permission failures, and timeouts are
unavailable/unknown rather than zero. Reports contain only aggregate counts and
statuses—never raw graph IDs, external IDs, properties, or source records.
`--output` uses the same exclusive mode-`0600` writer as migration reports. The
JSON contract is
[`verification-report.schema.json`](verification-report.schema.json).

Deep verification validates the versioned resolved-label snapshot and its
fingerprint, including identity coverage derived from the actually resolved
source mappings after Neo4j discovery or Cosmos Gremlin interpretation. It
checks only those exact label generations and ignores unrelated labels already
present in an append/upsert target. More than 128 expected labels yields an
explicit incomplete result without scanning a truncated subset.

Metadata schemas v14 through v16 remain readable by `report`, `status`,
`verify`, and `cleanup`. Reports mark v15 connector telemetry and unstored
per-label counters as unavailable where appropriate. These read-only commands
do not upgrade metadata. The next 2.1 `load` or `resume` applies the
non-destructive migrations through v17; older writers must be upgraded first.
Schemas newer than the binary supports are rejected.

Per-label counters carry explicit completeness and provenance. A v14-v16 job
resumed after migration keeps incomplete, null aggregate values because its
historical accepted, rejected, and byte attribution cannot be reconstructed.
Fresh create/replace jobs begin with a known-zero lifecycle baseline.
Append/upsert jobs do not scan or count pre-existing identities; when no exact
baseline exists, their aggregate is explicitly incomplete while per-batch
counter records remain atomic and idempotent.

## Target doctor

`doctor` starts with a plain PostgreSQL degraded probe, then progressively
checks AGE, metadata, the configured graph and bounded catalog/operational
state:

```sh
agefreighter doctor --target job.yaml
agefreighter doctor --target job.yaml --format markdown --output doctor.md
```

The default command is strictly read-only and emits a report even when AGE is
missing, unloadable, unsupported, not preloaded, or metadata is absent, stale,
invalid, or not visible. `fail` and `incomplete` are report outcomes, not
command failures. Connection, cancellation, and rendering failures still
return command errors. Catalog and metadata lists are bounded and filter the
configured graph or health-critical job states before applying limits; a
truncated check is `unknown`, never `pass`. Identity and graph rows are never
scanned. Required metadata indexes are checked for their table, readiness,
validity, uniqueness, exact key/include columns, ordering, and predicate.
Permission failures are `unknown`, never `pass`. Files written with `--output`
use the same exclusive mode-`0600` writer as migration reports. The contract is
[`doctor-report.schema.json`](doctor-report.schema.json).

`--persist` is the only doctor write path. It stores the final bounded JSON
report and a small set of typed summary fields. It requires load or resume to
have already upgraded metadata to current v17 and requires a compatible,
loadable AGE target. Persistence takes the metadata migration lock and
revalidates exact v17 while holding it; doctor never migrates or repairs
metadata.

Persisted summaries are read newest first:

```sh
agefreighter doctor history --target job.yaml
agefreighter doctor history --target job.yaml --limit 100 --format markdown
```

History defaults to 20 records and is capped at 100. On v14/v15 it reports
history as unavailable without migrating. Newer-than-supported schemas fail
closed.

## Operational checks

- Alert on failed commands and repeated source retry/throttle telemetry.
- Preserve job output and quarantine files for the configured retention period.
- Monitor PostgreSQL storage and WAL growth during large loads.
- Keep source credentials and target DSNs in environment variables or
  permission-restricted files.
- Run `verify` before declaring a migration complete.
- Test resume and replace rollback with production-like sizes before a release.
- Upgrade only to combinations in the
  [compatibility matrix](compatibility.md).

## Local integration environment

The development database commands affect only the fixed resources documented
in [`scripts/dev/README.md`](../../scripts/dev/README.md):

```sh
make dev-up
make dev-status
make test-compatibility
make test-recovery
make dev-down
```

`make dev-reset` deletes the named development volumes and is destructive.
It must not be used against production resources.
