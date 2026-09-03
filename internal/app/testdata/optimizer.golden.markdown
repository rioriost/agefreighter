# agefreighter optimize report

- Schema version: 1
- agefreighter version: dev
- Generated at: 2026-08-28T09:00:00Z
- Outcome: **incomplete**

## Target versions

| Component | Version | Status |
|---|---|---|
| PostgreSQL | 17.6 | pass |
| Apache AGE | 1.6.0 | pass |

## Checks

| Check | Status | Summary | Detail |
|---|---|---|---|
| active-graph | pass | active graph generation ownership was inspected | one active agefreighter-owned graph generation was selected |
| bounded-catalog-visibility | pass | bounded graph relation and index catalogs were inspected | labels=1 relation\_visibility\_problems=0 labels\_truncated=false indexes\_truncated=false |
| duplicate-indexes | pass | exact catalog-equivalent indexes were inspected | duplicate\_groups=0 |
| index-usage | pass | bounded index usage was inspected with its statistics-reset timestamp | zero\_scan\_indexes=0 |
| metadata-schema | pass | metadata schema is current |  |
| property-statistics | unavailable | live AGE property statistics and index recommendations were not produced | Apache AGE cannot pre-bound agtype serialization before detoast; live property parsing, cardinality inspection, and property-index recommendations are disabled |
| required-age-indexes | pass | required AGE label indexes were inspected | missing\_or\_invalid\_required\_indexes=0 unknown\_relations=0 |
| required-metadata-indexes | pass | version-compatible agefreighter metadata indexes were inspected | invalid\_or\_missing\_indexes=0 missing\_relations=0 schema\_version=19 |
| statistics-freshness | pass | bounded analyze and dead-tuple evidence was inspected | never\_analyzed=0 stale\_indicators=0 unknown=0 |
| storage-wal-visibility | pass | database storage and WAL counters were inspected | filesystem free capacity is not exposed by portable PostgreSQL catalogs |

## AGE property evidence

| Field | Value | Status |
|---|---|---|
| cardinalityAndIndexRecommendations | Apache AGE cannot pre-bound agtype serialization before detoast; live property parsing, cardinality inspection, and property-index recommendations are disabled | unavailable |

## Bounded relation statistics

| Field | Value | Status |
|---|---|---|
| 001.vertex.Person | kind=v estimatedRows=100 liveRows=100 deadRows=0 totalBytes=8192 indexBytes=4096 seqScans=0 indexScans=0 lastAnalyze=2026-08-28T08:00:00Z lastAutoAnalyze=unknown | pass |

## Graph size and density

| Field | Value | Status |
|---|---|---|
| activeLabelsInspected | 1 | pass |
| estimatedEdgeDensity | 0.000000 | pass |
| estimatedEdgeRows | 0 | pass |
| estimatedVertexRows | 100 | pass |
| labelIndexBytes | 4096 | pass |
| labelRelationBytes | 8192 | pass |

## Index evidence

| Field | Value | Status |
|---|---|---|
| summary | no exact duplicate or missing required indexes were detected in the bounded catalog | pass |

## Metadata relation statistics

| Field | Value | Status |
|---|---|---|
| 001.load\_job | estimatedRows=1 liveRows=1 deadRows=0 totalBytes=8192 indexBytes=4096 seqScans=0 indexScans=0 lastAnalyze=unknown lastAutoAnalyze=2026-08-28T08:00:00Z | pass |

## Migration counters and telemetry

| Field | Value | Status |
|---|---|---|
| batchAttemptsObserved | 1 | pass |
| batchAttemptsTruncated | false | pass |
| committedBatchCount | 1 | pass |
| committedBytes | 8192 | pass |
| committedRows | 100 | pass |
| connectorPages | 1 | pass |
| failedRequestAttempts | 0 | pass |
| labelCounterCommittedBytesKnown | 8192 | pass |
| labelCounterCommittedRows | 100 | pass |
| labelCounterRejectedRows | 0 | pass |
| labelCountersComplete | 1 | pass |
| labelCountersIncomplete | 0 | pass |
| latestBatchAttempt | 0 | pass |
| latestBatchBytes | 8192 | pass |
| latestBatchRejectedRows | 0 | pass |
| latestBatchRows | 100 | pass |
| latestBatchStatus | committed | pass |
| loadMode | create | pass |
| rejectedRows | 0 | pass |
| requestCharge | 0 | pass |
| sourceRejectedRows | 0 | pass |
| sourceType | csv | pass |
| status | committed | pass |
| throttledRequests | 0 | pass |

## Optimizer target evidence

| Field | Value | Status |
|---|---|---|
| databaseBytes | 1048576 | pass |
| databaseStatisticsReset | 2026-08-27T00:00:00Z | pass |
| evidencePhase | captured before any explicitly requested ANALYZE | pass |
| filesystemFreeBytes | unknown; use platform storage monitoring | unavailable |
| ginAgtypeOperatorClass | no supported allowlisted AGE operator class detected | pass |
| metadataInstalledVersion | 19 | pass |
| metadataSupportedVersion | 19 | pass |
| mode | recommendation-only | pass |
| walBytesSinceReset | 4096 | pass |
| walStatisticsReset | 2026-08-27T00:00:00Z | pass |

## Per-label row evidence

| Field | Value | Status |
|---|---|---|
| 001.vertex.Person | kind=v catalogEstimatedRows=100 statisticsLiveRows=100 storedCommittedRows=100 counterCompleteness=complete | pass |

## Recommendations

| Field | Value | Status |
|---|---|---|
| summary | no recommendation met the bounded evidence rules | pass |
