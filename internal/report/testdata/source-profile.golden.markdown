# agefreighter profile report

- Schema version: 1
- agefreighter version: dev
- Generated at: 2026-08-28T00:00:00Z
- Outcome: **incomplete**

## Checks

| Check | Status | Summary | Detail |
|---|---|---|---|
| source-read | unknown | source profile was truncated by a configured bound | limit=rows |
| source-version | unavailable | source version is not exposed by the connector iterator |  |

## Warnings

- **PROFILE\_TRUNCATED:** reported counts and statistics are lower-bound observations from a bounded prefix

## Incomplete checks

- source-profile

## Source

| Field | Value | Status |
|---|---|---|
| connector | csv | pass |
| mode | sample | pass |

## Vertex labels

| Field | Value | Status |
|---|---|---|
| 001 | label=Person,sampledRows=2,countRange=2..unknown,countMethod=observed-bounded-prefix,configuredProperties=1 | pass |
