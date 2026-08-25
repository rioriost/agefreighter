# ADR 0001: Use staged binary COPY for Apache AGE

- Status: Accepted
- Date: 2026-08-25
- Scope: agefreighter 2.x Apache AGE bulk writes

## Context

Apache AGE 1.6.0 label tables use `graphid` and `agtype`, while pgx binary COPY
cannot encode those extension types directly without AGE-specific codecs.
The adapter therefore prototyped two write paths:

1. direct PostgreSQL text COPY into the AGE label table
2. binary COPY of `bigint` and `text` into a temporary table, followed by one
   set-based `INSERT ... SELECT` that casts to `graphid` and `agtype`

A plain PostgreSQL binary COPY is the reference. Its table has the same data
columns, durability, and number of indexes as the AGE user label. Catalog
inspection on AGE 1.6.0 showed that user labels inherit the AGE base tables but
have no indexes of their own. The parent ID indexes do not index child-table
rows. In particular, no `start_id` or `end_id` indexes exist to drop and
rebuild, so that planned conditional benchmark arm is not applicable to this
version.

## Method

The benchmark used:

- Apple M4 Max, arm64
- macOS 26.6.2
- Go 1.27.0
- Apple Container CLI 1.0.0
- `apache/age:release_PG17_1.6.0` pinned by index digest
  `sha256:fe8b33905a61549a067f8512808b88011bdcaa82ab65d8788f39fb07a04aa5be`
- PostgreSQL 17.7 and AGE 1.6.0
- a 1 GiB target-container memory limit
- 64 payload bytes per property object
- 100,000 vertex rows
- 500,000 edge rows with 10,000 preloaded endpoint vertices
- five trials per arm, with the AGE container restarted before every trial
- the 1 GiB cgroup memory limit validated before every trial
- the median of five trials as the decision statistic

ID reservation and row materialization occur before the timed write. Elapsed
time and WAL cover COPY, the target insert where applicable, `ANALYZE`, row
verification, and commit. Target CPU and memory cover the complete isolated
tool invocation, including equivalent graph/table setup and cleanup. CPU usage
is read immediately before and after the workload without in-cgroup polling.
Target memory uses the kernel-maintained cgroup v2 `memory.peak`; it is total
cgroup memory, including anonymous memory, page cache, and kernel memory, not
process RSS. The table reports that peak above the cold pre-run
`memory.current` baseline. A trial is rejected unless the workload establishes
a new peak above the pre-run startup peak. Peak client RSS comes from `time -l`.
AGE and relational arms materialize rows in the same typed Go structures.

The benchmark is reproducible with:

```sh
AGEFREIGHTER_AGE_TEST_DSN='postgres://agefreighter:agefreighter_dev_only@127.0.0.1:55432/agefreighter?sslmode=disable' \
  ./scripts/bench/age-copy.sh \
  edges staged-binary 500000 5 .local/benchmarks/edges-staged.jsonl
```

## Results

### Vertices, 100,000 rows

| Strategy | Elapsed | Rows/s | Target CPU | WAL | Peak target cgroup memory above baseline | Peak client RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct text COPY | 1.268 s | 78,879 | 0.466 s | 9.67 MiB | 42.4 MiB | 19.1 MiB |
| Staged binary COPY | 0.119 s | 837,527 | 0.146 s | 14.56 MiB | 73.7 MiB | 24.8 MiB |
| Plain relational reference | 0.081 s | 1,239,415 | 0.090 s | 9.66 MiB | 34.8 MiB | 24.9 MiB |

### Edges, 500,000 rows

| Strategy | Elapsed | Rows/s | Target CPU | WAL | Peak target cgroup memory above baseline | Peak client RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct text COPY | 6.413 s | 77,961 | 2.070 s | 56.05 MiB | 152.0 MiB | 39.7 MiB |
| Staged binary COPY | 0.568 s | 879,727 | 0.630 s | 80.37 MiB | 315.6 MiB | 65.9 MiB |
| Plain relational reference | 0.258 s | 1,937,490 | 0.283 s | 56.00 MiB | 143.9 MiB | 64.4 MiB |

Staging is 10.6 times faster for vertices and 11.3 times faster for edges than
direct text COPY. It reaches 68% of the plain PostgreSQL vertex reference and
45% of the edge reference. Both staged workloads exceed the 109,190 rows/s
agefreighter 2.x release target by more than seven times.

The cost is approximately 51% more WAL for vertices, 43% more WAL for edges,
74% more target cgroup-memory growth for vertices, and 108% more for edges than
direct text COPY. Target CPU falls by approximately 69% for both workloads.
The benchmark intentionally uses one large 100,000-vertex or 500,000-edge batch
to expose strategy costs; production memory remains bounded by the configured
batch size. The WAL and peak-memory costs are accepted in exchange for the
order-of-magnitude throughput improvement.

## Decision

Use staged binary COPY as the default AGE bulk-write path:

1. create or truncate a transaction-local temporary staging table
2. binary COPY ordinary `bigint` and `text` columns with pgx
3. execute one set-based insert into the AGE label table using
   `id::text::graphid` and `properties::agtype`
4. verify and commit in the same target transaction

Keep direct text COPY as a tested diagnostic/reference path, not as an
automatic fallback. A staged-write failure is surfaced and rolled back; it is
never silently retried through a different write path.

## Consequences

- Milestone 5 must wire the sink to staged binary COPY.
- Batch limits must account for typed records plus staging materialization.
- Temporary staging data remains transaction-local and is dropped on commit or
  rollback.
- WAL capacity planning must use the staged measurements rather than the direct
  text measurements.
- Endpoint-index drop/rebuild must remain capability-gated. It cannot be
  enabled for AGE 1.6.0 because the relevant indexes do not exist.
- New PostgreSQL or AGE versions require rerunning this matrix before changing
  the default.
