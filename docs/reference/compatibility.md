# Compatibility matrix

agefreighter 2.0 deliberately has a narrow target compatibility boundary.
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

The weekly and manually runnable `AGE compatibility` workflow executes the AGE
adapter, metadata, CSV load, incremental load, and replace/recovery contracts
against every entry listed in its matrix. The matrix contains one target pair
for 2.0.0: PostgreSQL 17 with AGE 1.6.0.

Compatibility does not imply support for arbitrary combinations within other
PostgreSQL or AGE major/minor lines. Adding a matrix entry requires a pinned
multi-architecture image, complete compatibility run, and an update to this
document.
