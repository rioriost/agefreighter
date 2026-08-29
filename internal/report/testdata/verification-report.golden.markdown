# agefreighter verify report

- Schema version: 1
- agefreighter version: dev
- Generated at: 2026-08-27T21:00:00Z
- Outcome: **pass**
- Job ID: `aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee`
- Configuration fingerprint: `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`

## Checks

| Check | Status | Summary | Detail |
|---|---|---|---|
| metadata | pass | metadata is current |  |

## Bounded integrity

| Field | Value | Status |
|---|---|---|
| e.KNOWS | limit=100,identityCoverage=full,identityRowsChecked=1,physicalRowsChecked=1,reversePhysicalCoverage=checked,missingPhysicalRows=0,orphanPhysicalRows=0,missingEndpointRows=0,changedEndpointRows=0,identityTruncated=false,physicalTruncated=false | pass |

## Per-label counts

| Field | Value | Status |
|---|---|---|
| v.Person | counterCompleteness=complete,counterProvenance=v17-lifecycle,identityCoverage=full,acceptedRows=2,committedRows=2,livePhysicalRows=2,liveIdentityRows=2,storedPhysicalComparison=verified,physicalIdentityEquality=verified,committedBytes=unavailable,rejectedRows=0 | pass |
