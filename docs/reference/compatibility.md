# Compatibility matrix

agefreighter 2.x deliberately has a narrow target compatibility boundary.
Unsupported versions fail during the Apache AGE capability probe before any
graph is changed.

| Component | Supported | Release qualification |
|---|---|---|
| Go | 1.27.x | Linux and macOS CI |
| PostgreSQL target | 14.x through 18.x, in the exact pairings below | Weekly and manually runnable target matrix |
| Apache AGE target | 1.6.x through 1.8.x, in the exact pairings below | Weekly and manually runnable target matrix |
| PostgreSQL source | 17.x | PostgreSQL 17.6 pinned image |
| Neo4j source | 4.4.48 and 5.26.30 | Fourteen-pair migration matrix using pinned official Community images |
| Cosmos DB for NoSQL | Current Azure service API supported by Azure SDK v1.5 | Controlled Azure integration environment |
| Cosmos DB for Apache Gremlin backing documents | Current `_isEdge`, `_vertexId`, `_sink`, and `_value` document layout through the NoSQL API | Controlled Azure integration environment |

As of 2026-08-28, PostgreSQL majors 14 through 18 are supported upstream.
agefreighter qualifies only the following Apache AGE release pairings; a
supported PostgreSQL major and a supported AGE series do not form a supported
target unless the exact pair appears here.

Neo4j support status and migration-source compatibility are deliberately
separate. Neo4j 4.4 is upstream end-of-life, but remains an important migration
source. agefreighter qualifies the final 4.4 patch and the current 5.26 LTS
patch using the following official Community image indexes.

| Neo4j source | Lifecycle | Qualification artifact | Version observed in qualification |
|---|---|---|---|
| 4.4 | Upstream support ended 2025-11-30; migration source only | `neo4j@sha256:5098db94262985f26a71d4ff573116cf893bce636e879bceb8ec9ba02a5a1553` | Neo4j Community 4.4.48 |
| 5.26 LTS | Upstream LTS receives hotfixes through 2028-06-06 | `neo4j@sha256:037cf5756f0135cbfd66b739b6df7c7c4bb100f9ce11602f6f9538e17e02c74d` | Neo4j Community 5.26.30 |

| PostgreSQL target | Apache AGE target | Qualification artifact | Version observed in qualification |
|---|---|---|---|
| 14.x | 1.6.x | `apache/age@sha256:d840e88fdc3ba9f60b6beeccec4576f66b73081c69d1f4d794bbde67c69bcb57` | PostgreSQL 14.20, AGE 1.6.0 |
| 15.x | 1.6.x | `apache/age@sha256:bd653255f4b9449c3f52bac3849524069cf83a6d867a25b1fc6d8535171498b7` | PostgreSQL 15.15, AGE 1.6.0 |
| 16.x | 1.6.x | `apache/age@sha256:16aa423d20a31aed36a3313244bf7aa00731325862f20ed584510e381f2feaed` | PostgreSQL 16.10, AGE 1.6.0 |
| 17.x | 1.6.x | `apache/age@sha256:fe8b33905a61549a067f8512808b88011bdcaa82ab65d8788f39fb07a04aa5be` | PostgreSQL 17.7, AGE 1.6.0 |
| 17.x | 1.7.x | `apache/age@sha256:92a5d223965bc2e436f9eee436e0bd2c0d81f3b59124b3d197ec94706f3450a8` | PostgreSQL 17.11, AGE 1.7.0 |
| 18.x | 1.7.x | `apache/age@sha256:e7de1717e487dac7c1be93a1cd5360a2cf07ff4170342c2af2ac4713c21baf00` | PostgreSQL 18.1, AGE 1.7.0 |
| 18.x | 1.8.x | `apache/age@sha256:c7255d32a10de6b5d84daa12346aa545e062b223fa3e574fe96a5c428f249894` | PostgreSQL 18.6, AGE 1.8.0 |

Apache AGE 1.4.x and 1.5.x release artifacts for currently supported
PostgreSQL majors were also evaluated. They are not supported migration
targets: their label lifecycle and incremental-write behavior do not satisfy
the current adapter contracts.

The upstream source of truth is the
[PostgreSQL versioning policy](https://www.postgresql.org/support/versioning/)
and the
[Apache AGE release list](https://github.com/apache/age/releases). PostgreSQL
14 reaches upstream end of life on 2026-11-12; this matrix must be revised by
that date rather than silently continuing support.

The weekly and manually runnable `AGE compatibility` workflow executes the full
cross-product of the two Neo4j sources and seven PostgreSQL/AGE targets: fourteen
migration configurations in total. Each configuration exercises Neo4j streaming,
explicit mappings, schema discovery, create, replace, append, upsert, metadata,
and target recovery contracts. Official release images are pinned by
multi-architecture digest.

The 2.1 metadata schema is v17. Read-only lifecycle and report commands accept
compatible v14 through v16 metadata without migration; `load` and `resume`
upgrade it through v17. Version 15 stores one bounded, non-secret connector
telemetry summary per completed job. Version 16 adds bounded diagnostic history,
written only by explicit `doctor --persist`. Version 17 adds migration-snapshot
fingerprints and aggregate per-batch/per-label verification counters.
Resolved-mapping snapshot version 2 adds per-label external-identity coverage;
version 1 snapshots remain readable, with edge reverse-coverage checks reported
as unavailable rather than inferred.
Per-label rows record completeness and provenance; migrating a v14-v16 job
does not synthesize missing historical counters.
`doctor history` marks v14/v15 history
unavailable rather than migrating. Newer-than-supported metadata fails closed.
Only `load` and `resume` invoke metadata migration. Diagnostic and lifecycle
read paths inspect v14-v17 without changing it. Once either writer upgrades a
target to v17, an unmodified 2.0 binary (whose maximum is v14) rejects that
target as newer than supported; upgrade every writer before the first 2.1
`load` or `resume`.

Compatibility does not imply support for arbitrary Neo4j patches or combinations
within other PostgreSQL or AGE major/minor lines. Adding a source requires a
pinned official multi-architecture Neo4j image. Adding a target requires a
pinned official multi-architecture AGE release image. Both changes require a
complete cross-product compatibility run and an update to this document.
