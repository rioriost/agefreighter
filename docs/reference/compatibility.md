# Compatibility matrix

agefreighter 2.x deliberately has a narrow target compatibility boundary.
Unsupported versions fail during the Apache AGE capability probe before any
graph is changed.

| Component | Supported | Release qualification |
|---|---|---|
| Go | 1.27.x | Linux and macOS CI |
| PostgreSQL target | 17.x | PostgreSQL 17 from the pinned AGE image |
| Apache AGE target | 1.6.x | AGE 1.6.0, pinned multi-architecture image digest |
| PostgreSQL source | 17.x | PostgreSQL 17.6 pinned image |
| Neo4j source | 5.26.x | Neo4j 5.26 pinned image |
| Cosmos DB for NoSQL | Current Azure service API supported by Azure SDK v1.5 | Controlled Azure integration environment |
| Cosmos DB for Apache Gremlin backing documents | Current `_isEdge`, `_vertexId`, `_sink`, and `_value` document layout through the NoSQL API | Controlled Azure integration environment |

The weekly and manually runnable `AGE compatibility` workflow executes the AGE
adapter, metadata, CSV load, incremental load, and replace/recovery contracts
against every entry listed in its matrix. The matrix contains one target pair
for 2.0.0: PostgreSQL 17 with AGE 1.6.0.

The 2.1 metadata schema is v16. Read-only lifecycle and report commands accept
compatible v14 and v15 metadata without migration; `load` and `resume` upgrade
it through v16. Version 15 stores one bounded, non-secret connector telemetry
summary per completed job. Version 16 adds bounded diagnostic history, written
only by explicit `doctor --persist`. `doctor history` marks v14/v15 history
unavailable rather than migrating. Newer-than-supported metadata fails closed.

Compatibility does not imply support for arbitrary combinations within other
PostgreSQL or AGE major/minor lines. Adding a matrix entry requires a pinned
multi-architecture image, complete compatibility run, and an update to this
document.
