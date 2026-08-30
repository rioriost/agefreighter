# Test plan

## Objective

Demonstrate that agefreighter can correctly and recoverably migrate a realistic
property graph containing 160 million vertices and 400 million edges. This is a
post-release qualification of v2.1.0, not a retroactive release gate.

## Fixed migration paths

1. Neo4j 4.4.48 Community -> PostgreSQL 18 / Apache AGE 1.7
2. Neo4j 5.26.30 Community -> PostgreSQL 18 / Apache AGE 1.7

Both paths use the same generated logical graph, VM sizes, zone, load settings,
and target SKU. Each path uses a dedicated database on the same Flexible Server
and starts from an empty target graph. Database isolation prevents identity and
catalog growth from the first path from biasing the second path. The sources remain
read-only from the first discovery query through final verification because
Neo4j mappings do not share one point-in-time transaction.

## Azure topology

All resources are created in one selected Azure region and one explicitly
selected availability zone in the same subscription:

```text
VNet
  vm subnet
    Neo4j 4.4.48 VM  --+
    Neo4j 5.26.30 VM --+--> agefreighter VM
                              |
  delegated PostgreSQL subnet | private DNS
    PostgreSQL 18 + AGE 1.7 <--+
```

There are no VM public IP addresses. PostgreSQL uses private networking. NSGs
permit Bolt only from the agefreighter VM and administrative traffic only
through the separately approved operator path. If the production-equivalent
target uses high availability, the test uses `SameZone` HA; otherwise HA is
disabled and that fact is recorded in the result.

Initial sizing, to be corrected after P1 and P2:

| Role | Initial candidate |
| --- | --- |
| Neo4j source VM | 32-64 vCPU, 128-256 GiB RAM, dedicated durable data disk |
| agefreighter VM | 16-32 vCPU, 64 GiB RAM |
| Flexible Server | Memory Optimized, 32-64 vCore |
| Target storage | 4-8 TiB, provisioned IOPS and throughput |

P1 used the upper/conservative part of this envelope and showed substantial
headroom: source CPU stayed below 10%, target CPU below 4%, and each 4 TiB
source disk stayed below 1% used. The measured P1 result and coefficients are
recorded in [`../results/summaries/p1-20260830.md`](../results/summaries/p1-20260830.md).
P2 must use a separately reviewed right-sized configuration; the initial P1
SKUs are not P2 defaults.

The two Neo4j sources are run sequentially to prevent target and network
contention. They use identical VM definitions, separate version-specific data
disks, and `agefreighter_<phase>_neo4j44` / `agefreighter_<phase>_neo4j526`
target databases. Local ephemeral disks may hold regenerable CSV scratch data,
but never the only copy of a Neo4j store or test result.

## Phases

| Phase | Vertices | Edges | Purpose |
| --- | ---: | ---: | --- |
| tiny | 160 | 400 | Local generator, import, and config smoke test |
| P0 | 100,000 | 358,000 | Reproduce the existing v2.1.0 baseline and check the Azure harness |
| P1 | 1,600,000 | 4,000,000 | 1% correctness and first capacity coefficient |
| P2 | 16,000,000 | 40,000,000 | 10% tuning, fault injection, and P3 forecast |
| P3 | 160,000,000 | 400,000,000 | Full clean and recovery qualification |
| P4 | P3 plus 0.1-1% | proportional | append/upsert operational qualification |

P0 through P3 run against both source versions. P1 uses the default production
settings. P2 changes one of `fetchRows`, `batchRows`, or `batchBytes` at a time,
then freezes the winning setting before P3. P3 has at least one uninterrupted
clean run and one interrupted/resumed run per Neo4j version.

Full `replace` is outside the mandatory P3 scope unless it is an intended
production operation. `replace` remains mandatory at P2 because a P3 replace
can require the active graph, shadow graph, retained backup, indexes, staging,
and WAL concurrently.

## Fault schedule

The recovery run injects one controlled fault only after a committed checkpoint:

1. stop agefreighter near 25%, then resume the recorded job;
2. interrupt Bolt or restart Neo4j near 40%, then resume;
3. interrupt PostgreSQL connectivity near 60%, then resume;
4. reboot the agefreighter VM near 75%, then resume.

P2 rehearses every fault separately. P3 may combine them in one recovery run
only after P2 proves that each individual failure is understood. Source writes
are not a fault-injection mechanism; source immutability is a precondition.

## Stage promotion

- P0 starts only after repository and Azure `what-if` review.
- P1 starts only after the P0 harness and exact-count checks pass.
- P2 starts only after P1 exact-count and digest verification passes.
- P3 starts only after P2 supplies measured storage, WAL, throughput, runtime,
  and cost forecasts, and the projected P3 values fit the approved limits.
- P4 starts only after both P3 source paths pass.

Every promotion is recorded in a small result summary. A failed phase is not
silently rerun with changed settings; it receives a new run ID and the change is
documented.
