# agefreighter migration report

- Schema version: 1
- agefreighter version: dev
- Generated at: 2026-08-27T21:00:00Z
- Outcome: **incomplete**
- Job ID: `aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee`
- Configuration fingerprint: `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`

## Target versions

| Component | Version | Status |
|---|---|---|
| PostgreSQL | 17.9 | pass |
| Apache AGE | 1.6.0 | pass |

## Checks

| Check | Status | Summary | Detail |
|---|---|---|---|
| metadata | pass | metadata is current |  |
| telemetry | unavailable | not recorded by metadata schema v14 |  |

## Job

| Field | Value | Status |
|---|---|---|
| sourceType | csv | pass |
| telemetry | requires schema v15 | unavailable |
