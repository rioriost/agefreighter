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
