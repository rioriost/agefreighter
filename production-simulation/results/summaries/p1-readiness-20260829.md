# P1 readiness investigation — 2026-08-29

## Outcome

The P1 technical prerequisites are complete. P1 itself was not started: no P1
fixture, source store, target database, or load job was created. The retained
P0 resources were used only for read-only profiling and correctness probes.

The only open administrative gate is posted Azure cost. Cost Management still
returned an empty result immediately after this work and then rate-limited the
follow-up query. P1 must not start until the posted P0 cost is captured, the P1
budget is approved, and `PRODUCTION_SIMULATION_APPROVAL=reviewed-p1` is granted.

## Changes under test

- Branch: `codex/production-simulation`
- Read-only source enforcement: `727d6fd`
- AGE edge-label primary keys: `7a5b6b5`
- Focused AGE 1.7 integration coverage: `611df53`
- Isolated target databases, SQL readiness retry, maintenance procedure, and
  independent range verifier: `489a5d3` through `b84529d`

All production-simulation tests, static job validation, shell checks, Bicep
compilation, and `git diff --check` passed.

## Source immutability

Both retained source VMs were re-bootstrapped with the same P0 stores and
fixture root. Neo4j 4.4.48 uses `dbms.databases.default_to_read_only=true` and
Neo4j 5.26.30 uses `server.databases.default_to_read_only=true`. Each bootstrap
waited for indexes, rechecked exact source counts, and confirmed that the
`neo4j` database reports `read-only` access.

- Fixture root: `ea4cf0de87ea5730a3823dd605d8af6957a323fb20e0ba37790d90dfb41b4516`
- Source counts on each VM: 100,000 vertices and 358,000 relationships

## P0 elapsed-time investigation

Source-only exact profiling showed no meaningful version-specific regression:

| Source | Cold | Warm |
| --- | ---: | ---: |
| Neo4j 4.4.48 | 19.48 s | 14.22 s |
| Neo4j 5.26.30 | 15.80 s | 14.46 s |

Reversing the load order in the already-used `agefreighter_p0` database moved
the slowdown to the second load: 5.26 first completed in 90.44 seconds and 4.4
second completed in 130.24 seconds. Both jobs committed and passed count
verification.

Repeating the same loads in separate, newly-created databases on the same
Flexible Server removed the bias: 5.26 completed in 19.69 seconds and 4.4 in
21.02 seconds. The original 4.3x difference therefore came from accumulated
agefreighter identity/catalog state in a shared database, not Neo4j 5.26.

P1 and later phases now require two databases on one reviewed Flexible Server:
`agefreighter_<phase>_neo4j44` and `agefreighter_<phase>_neo4j526`. Configuration
files use different DSN environment variables, and the preparation script
refuses database reuse. The two temporary probe databases (333 MiB and 336 MiB)
were removed after retaining their result artifacts.

## PostgreSQL/AGE readiness corrections

Azure reported the Flexible Server control plane as `Ready` before its managed
`pg_hba` state accepted the same private SSL connection. The target preflight
now retries the SQL version query for up to five minutes. It still blocks unless
the data plane reports PostgreSQL 18 and AGE 1.7.x.

The nine required-index warnings in P0 were the missing `id` primary keys on
the nine AGE edge-label tables. Graph creation now adds those primary keys.
Focused integration coverage passed against the retained Azure PostgreSQL 18 /
AGE 1.7 target. Planner statistics are deliberately refreshed after, not
during, the timed load; `post-load-maintenance.sh` captures optimization advice
before and after `ANALYZE`.

## Fixture calibration decision

No privacy-approved aggregate statistics from the historical full-scale
migration are present in the repository or P0 evidence. The fallback model was
therefore retained rather than calibrated from memory. Its cardinalities,
degree skew, property types, nullable values, text-width buckets, Unicode, and
known deviations are reviewed in `docs/fixture-calibration.md`.

This is accepted for synthetic correctness and capacity qualification. P1
throughput must not be presented as a customer runtime commitment, and P2 must
remeasure storage, WAL, compression, and throughput before sizing P3.

## Independent full-record proof

`cmd/rangedigest` independently streamed the deterministic P0 fixture and each
committed AGE generation. It canonicalized integer and float types, nulls,
arrays, Unicode, all properties, identities, and edge endpoints. Both source
paths matched all 458,000 records and all 59 fixed ranges:

- Canonical root: `34169b7c7907cef2e0348909d3dabeb8df72b40ee6fb60e25976944bd06f4a9a`
- Neo4j 4.4.48 -> AGE: `pass`
- Neo4j 5.26.30 -> AGE: `pass`

The proof also caught two hosted/import semantics during development: Azure
Flexible Server rejects manual `LOAD 'age'`, and Neo4j preserves the generated
empty status field as a null property. Both cases now have explicit handling
and regression coverage.

Successful evidence is retained on the loader disk under
`/var/lib/agefreighter-p0/results/p1-readiness/range-digest-b84529d/`.
Comparison artifact SHA-256 for each source is
`5e2536f9ed7c295eca09d2c4378b35558442b80b5a0bce060177c0b3ed5f0b69`.

## Gate status

| Gate | Status |
| --- | --- |
| Source database operationally read-only | Pass |
| Fixture fallback reviewed and limitations recorded | Pass |
| P0 version/time discrepancy explained | Pass |
| Source paths isolated at the target database boundary | Pass |
| Required AGE edge indexes created by the loader | Pass |
| SQL data-plane readiness retry implemented | Pass |
| Independent canonical verifier proven on both P0 paths | Pass |
| P0 posted Azure cost captured | Waiting for Azure billing data |
| P1 budget and live-operation approval | Not granted; P1 not started |

No P1 promotion decision is implied by this report. After the cost and approval
rows close, create the two P1 databases, generate/import the P1 fixture once,
verify both read-only sources, and begin the first reviewed timed run.

## Subsequent disposition

The user subsequently authorized P1 execution. P1 completed successfully on
2026-08-30; its results, evidence hashes, capacity coefficients, retained-cost
status, and P2 gate are recorded in [`p1-20260830.md`](p1-20260830.md). This
section preserves the original readiness decision while linking its closure.
