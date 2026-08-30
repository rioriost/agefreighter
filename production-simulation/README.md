# Production simulation

This directory owns the review-gated qualification of agefreighter migrations
at production scale. The target workload contains exactly 160,000,000 vertices
and 400,000,000 edges. It is loaded from both Neo4j 4.4.48 and Neo4j 5.26.30
into one approved Azure Database for PostgreSQL Flexible Server configuration:
PostgreSQL 18 with Apache AGE 1.7.

No large run is automatic. The repository assets must be reviewed, the 1% and
10% capacity gates must pass, and a human must approve the Azure deployment and
each migration run. The scripts reject live operations unless
`PRODUCTION_SIMULATION_APPROVAL` has the exact value documented in the
[runbook](docs/runbook.md).

## Directory map

| Path | Purpose |
| --- | --- |
| [`docs/test-plan.md`](docs/test-plan.md) | Scope, phases, topology, and test matrix |
| [`docs/dataset.md`](docs/dataset.md) | Supply-chain schema and generation rules |
| [`docs/fixture-calibration.md`](docs/fixture-calibration.md) | Synthetic representativeness decision and limits |
| [`docs/acceptance.md`](docs/acceptance.md) | Metrics, correctness checks, stop and pass criteria |
| [`docs/runbook.md`](docs/runbook.md) | Review, deployment, execution, recovery, and cleanup procedure |
| `cmd/fixturegen` | Deterministic sharded CSV generator and manifest verifier |
| `cmd/rangedigest` | Independent fixture/AGE canonical range-digest verifier |
| `internal/fixture` | Tested dataset model and streaming generator implementation |
| `configs` | Static agefreighter jobs for both Neo4j versions |
| `infra` | Review-only Azure deployment contract and parameters |
| `scripts` | Safe entry points for validation, generation, and live runs |
| `results/summaries` | Small reviewed result summaries suitable for Git |

Completed live-phase reports:

- [P0 result](results/summaries/p0-20260829.md)
- [P1 result](results/summaries/p1-20260830.md)

Generated CSV, secrets, database dumps, and raw telemetry must not be committed.
They belong under `work/` or `results/raw/`, both of which are ignored.

## Local review commands

These commands create only a temporary local fixture and do not contact Azure:

```sh
make -C production-simulation check
make -C production-simulation smoke
```

`check` runs unit tests, validates both load jobs, checks shell scripts, and
builds the Azure template when the required local tools are present. `smoke`
generates and verifies a 160-vertex/400-edge fixture.

## Review gate

Before any Azure resource is deployed, reviewers must approve:

1. the exact region, availability zone, subscription, SKU, storage, and budget;
2. PostgreSQL 18 and Apache AGE 1.7 availability in that selected region;
3. both Neo4j image digests and the version-specific offline import commands;
4. the generated P1 manifest and its exact counts;
5. the monitoring, abort, cleanup, and artifact-retention settings;
6. the absence of production values or credentials in the fixture and results.

P0, P1, and P2 are complete. P3 was separately authorized on 2026-08-31 under
the frozen resource, time, cost, and stop envelope in `docs/p3-run-sheet.md`.
A technical promotion decision never substitutes for the next phase's
resource, budget, and live-operation review.
