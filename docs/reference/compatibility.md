# Compatibility matrix

agefreighter 2.x deliberately has a narrow target compatibility boundary.
Unsupported versions fail during the selected Apache AGE or native SQL/PGQ
capability probe before any graph is changed.

| Component | Supported | Release qualification |
|---|---|---|
| Go | 1.27.x | Linux and macOS CI |
| PostgreSQL for Apache AGE | 14.x through 18.x, in the exact pairings below | Weekly and manually runnable target matrix |
| Apache AGE target | 1.6.x through 1.8.x, in the exact pairings below | Weekly and manually runnable target matrix |
| Native PostgreSQL property-graph target | 19 Beta 3, experimental and digest-pinned | Apple Container arm64 and protected Linux amd64 SQL/PGQ suites |
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

The experimental native target was qualified on Apple Container 1.0.0 with
`postgres:19beta3@sha256:a48b19841e04b35b72a25e9a94314ac80546d32b5e2e3cd9279390cbd8a99572`
(`linux/arm64` manifest
`sha256:d2803db84af749f279166b231e05a92c7d5ef991540cb292a76fb41af997ebd4`),
PostgreSQL `server_version_num` 190000. The suite covers create, replace,
append (`error` and `ignore-identical`), all three upsert property modes,
concurrent-writer exclusion, interrupted replacement and same-job resume,
relational constraints, directed and undirected `GRAPH_TABLE`, metadata
v14-to-v21 upgrade, and nineteen target-corruption cases. The authenticated
Cosmos qualification additionally covers NoSQL create, replace, append, and
upsert plus Gremlin backing documents, with the existing Cosmos-to-AGE path as
a control. It is not a supported
production target while PostgreSQL 19 is pre-release, and PostgreSQL 19 GA
requires a fresh digest-pinned qualification before this status can change.

The 2.2 metadata schema is v21. Read-only lifecycle and report commands accept
compatible v14 through v21 metadata without migration; `load` and `resume`
upgrade it through v21. Version 15 stores one bounded, non-secret connector
telemetry summary per completed job. Version 16 adds bounded diagnostic history,
written only by explicit `doctor --persist`. Version 17 adds migration-snapshot
fingerprints and aggregate per-batch/per-label verification counters.
Version 18 records the target backend and schema, version 19 adds native
property-graph generation and label mappings, version 20 stores ranged digest
baselines, and version 21 adds active, loading, superseded, and retained-backup
lifecycle states with a single-active-generation constraint.
Resolved-mapping snapshot version 2 adds per-label external-identity coverage;
version 1 snapshots remain readable, with edge reverse-coverage checks reported
as unavailable rather than inferred.
Per-label rows record completeness and provenance; migrating a v14-v16 job
does not synthesize missing historical counters.
`doctor history` marks v14/v15 history
unavailable rather than migrating. Newer-than-supported metadata fails closed.
Only `load` and `resume` invoke metadata migration. Diagnostic and lifecycle
read paths inspect v14-v21 without changing it. Once a 2.2 writer upgrades a
target to v21, older binaries whose maximum metadata version is lower reject
that target as newer than supported; upgrade every writer before the first 2.2
`load` or `resume`.

Compatibility does not imply support for arbitrary Neo4j patches or combinations
within other PostgreSQL or AGE major/minor lines. Adding a source requires a
pinned official multi-architecture Neo4j image. Adding a target requires a
pinned official multi-architecture AGE release image. Both changes require a
complete cross-product compatibility run and an update to this document.
