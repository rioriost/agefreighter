# Review-gated runbook

## 1. Repository review

From the repository root:

```sh
make -C production-simulation check
make -C production-simulation smoke
git diff --check
git status --short
```

Review every file under this directory. Confirm that generated data, raw logs,
credentials, SSH keys, private hostnames, and connection strings are absent.

## 2. Azure selection and preview

Choose a region and numeric availability zone only after confirming capacity
for PostgreSQL 18, Apache AGE 1.7, the requested Flexible Server SKU, Premium
storage, and all VM SKUs in the same zone. Record the subscription ID out of
band; do not commit it.

Copy `infra/main.parameters.example.json` outside the repository and fill the
approved non-secret values. Supply administrator passwords and SSH keys through
the deployment environment or an approved secret store, never through a
tracked parameter file.

Run template compilation and Azure `what-if`. Save the redacted `what-if`
result for review. Do not deploy during this step.

## 3. Live-operation approval

Live scripts require:

```sh
export PRODUCTION_SIMULATION_APPROVAL='reviewed-p1'
```

Use `reviewed-p0`, `reviewed-p1`, `reviewed-p2`, or `reviewed-p3` matching the
phase. This is an interlock, not a credential or substitute for human review.
P2 and P3 approval must be granted only after the prior phase report passes.

## 4. Fixture generation and import

Generate the fixture once, store it on durable Azure storage, and verify it
before either Neo4j import:

```sh
production-simulation/scripts/generate-fixture.sh p1 /data/fixtures/p1 64 8
go run ./production-simulation/cmd/fixturegen verify \
  --manifest /data/fixtures/p1/manifest.json
```

Use `scripts/import-neo4j.sh` to verify the fixture and execute the
version-specific offline import. It pins the official image digests already
qualified by the compatibility workflow, uses the different 4.4/5.26 command
forms explicitly, and refuses a non-empty Neo4j data directory. Review the
fully resolved mount paths and available disk capacity before invocation.

After import:

1. create planner-usable indexes for `source_key` on all labels and
   relationship types (B-tree on Neo4j 4.4, range on Neo4j 5.26);
2. wait for every index to become ONLINE;
3. verify exact counts against the manifest;
4. capture store and index sizes;
5. run `EXPLAIN` for each discovered keyset mapping;
6. set the source operationally read-only.

## 5. PostgreSQL target preflight

Before every phase, create two new isolated databases with
`scripts/prepare-target-databases.sh`. Point `AGEFREIGHTER_ADMIN_DSN` at the
`postgres` administration database; the script refuses to reuse either name.
Before every timed run, use the matching source-specific DSN, a new empty graph,
and record:

```sql
SELECT version();
SELECT name, default_version, installed_version
FROM pg_available_extensions
WHERE name = 'age';
SELECT extname, extversion FROM pg_extension WHERE extname = 'age';
```

The run is blocked unless PostgreSQL reports major version 18 and AGE reports
the 1.7 release line (`1.7` or `1.7.0`). Capture storage, IOPS, throughput, HA mode, zone, and relevant
server parameters. Keep durability settings at their production values. Do not
treat the Azure control-plane `Ready` state as SQL readiness: the preflight
retries the data-plane version query because managed `pg_hba` propagation can
lag a stop/start operation.

## 6. Migration

On the dedicated agefreighter VM, install the checksum-verified v2.1.0 release
binary. Export the two secret references only in the process environment:

```sh
export AGEFREIGHTER_NEO4J_PASSWORD='...'
export AGEFREIGHTER_TARGET_DSN_NEO4J44='...dbname=agefreighter_p1_neo4j44...'
export AGEFREIGHTER_TARGET_DSN_NEO4J526='...dbname=agefreighter_p1_neo4j526...'
```

Validate and plan the selected static job, start monitoring, and then invoke
`scripts/run-load.sh`. Record the emitted job ID immediately. Fault injection
is performed only according to the approved phase run sheet and only after a
committed checkpoint.

## 7. Verification and reporting

The measured load interval ends when the load commits. Afterwards, run
`scripts/post-load-maintenance.sh` to capture optimization advice, refresh
planner statistics with `ANALYZE`, and capture the advice again. Do not include
this maintenance time in migration throughput.

Run built-in report, doctor, count verification, integrity verification, and
the independent verifier:

```sh
production-simulation/scripts/verify-range-digest.sh \
  p1 neo4j-4.4.48 /data/fixtures/p1/manifest.json JOB_ID /data/results/RUN/digest
```

The verifier independently streams the fixture and committed AGE generation,
canonicalizes every property and relationship endpoint, compares fixed
100,000-record leaves, and then compares the root. Compare the recovery root to
the clean-run root. Copy raw artifacts into `results/raw/<run-id>/` and prepare
a redacted summary from `results/summaries/TEMPLATE.md`.

## 8. Cleanup

Cleanup is a separately approved destructive operation. First resolve and
print the exact resource group and all tagged resources. Do not use wildcards,
an unresolved variable, a subscription-wide delete, or a repository script
that silently deletes the previous run. Retain summaries and required raw
evidence before deleting data-plane resources.
