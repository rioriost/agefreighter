# Operations guide

## Preflight

1. Keep the load-job file and referenced secret files readable only by the
   service account.
2. Run `agefreighter validate JOB` and review `agefreighter plan JOB`.
3. Confirm the target reports PostgreSQL 17 and Apache AGE 1.6.x.
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

Metadata schemas v14 (2.0) and v15 remain readable by `report`, `status`,
`verify`, and `cleanup`. Reports mark v15 connector telemetry and unstored
per-label counters as unavailable where appropriate. These read-only commands
do not upgrade metadata. The next 2.1 `load` or `resume` applies the
non-destructive migrations through v16; older writers must be upgraded first.
Schemas newer than the binary supports are rejected.

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
have already upgraded metadata to current v16 and requires a compatible,
loadable AGE target. Persistence takes the metadata migration lock and
revalidates exact v16 while holding it; doctor never migrates or repairs
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
