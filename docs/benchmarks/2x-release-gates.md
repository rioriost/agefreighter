# agefreighter 2.x release performance gates

This report records the release-gate calibration performed on 2026-08-27. The
measurements used Apple M4 Max, macOS arm64, Go 1.27.0, and the repository's
pinned PostgreSQL 17 / Apache AGE 1.6.0 development image.

## AGE staged-binary versus relational COPY

The release benchmark ran three independent samples of 100,000 rows with
64-byte generated properties for each workload and strategy.

| Workload | Strategy | Median rows/s | Ratio to relational |
|---|---|---:|---:|
| Vertices | staged-binary | 899,938 | 48.7% |
| Vertices | plain-relational | 1,848,214 | 100% |
| Edges | staged-binary | 775,438 | 49.6% |
| Edges | plain-relational | 1,561,588 | 100% |

The preliminary 70% hypothesis did not account for AGE's required conversion
from staging `bigint` and `text` values into `graphid` and `agtype`. The release
floor is therefore calibrated at **40%**. This retains an 8.7–9.6 percentage
point margin below the measured medians while still rejecting a material
regression.

## End-to-end CSV create

The preserved 1.x countries baseline is 54,595 rows/s. A three-iteration 2.x
run measured 118,832 rows/s, or **2.18x** the baseline. The release floor
remains 109,190 rows/s, exactly twice the 1.x baseline.

## Reproducing the gate

With the pinned development database running:

```sh
AGEFREIGHTER_AGE_TEST_DSN='postgres://...' \
  ./scripts/bench/release-budget.sh performance-artifacts
```

The script emits raw AGE COPY samples, a normalized version 1 JSON report, and
the Go benchmark transcript. It fails when either workload's staged-binary
median falls below 40% of the equivalent relational median or when CSV create
throughput falls below 109,190 rows/s.

The 109,190 rows/s floor is retained as the default for the calibrated M4 Max
environment. GitHub-hosted Linux runners use one benchmark execution averaging
three complete fixture loads and an explicit 50,000 rows/s floor. Independent
release runs on AMD EPYC 7763 and Intel Xeon Platinum 8573C measured 72,343,
68,836, and 59,283 rows/s. The hosted-runner floor leaves a 15.7% margin below
the slowest observation while still detecting a material end-to-end regression.
The hardware-normalized staged-binary ratio remains 40% in every environment.

The `Release performance` workflow runs this gate for 2.x tags and on explicit
dispatch. Performance is intentionally excluded from pull-request CI because
shared runners are noisy.
