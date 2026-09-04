# Azure HorizonDB + Apache AGE P2 feasibility study plan

## Status and decision to be made

This document is an execution plan. It does not authorize deployment, live
testing, resource deletion, or spending.

Execution was authorized on 2026-09-03. The live HorizonDB capability response
then showed PostgreSQL 18 as available in Australia East, although the public
extension catalog still documented AGE only for PostgreSQL 17. The study will
therefore attempt PostgreSQL 18 / AGE 1.7 first, matching the historical P2
engine and extension pair. If the service rejects AGE on PostgreSQL 18 or
reports a different AGE version, no timed P2 run begins until the comparison
matrix is amended and reviewed.

The study will determine whether Azure HorizonDB with Apache AGE is a viable
target for the production-simulation workload at P2 scale and how its migration
performance compares with Azure Database for PostgreSQL Flexible Server when
the primary compute allocation is matched as closely as the services allow.

The final decision will be one of:

- **GO**: functionally correct and recoverable, with acceptable performance and
  operational characteristics;
- **CONDITIONAL**: functionally viable, but with a documented performance,
  preview-service, cost, observability, or operations condition; or
- **NO-GO**: an extension, correctness, recovery, replacement, or sustained
  performance gate fails.

## Important comparability constraints

The study must not present the result as a pure storage-architecture benchmark.
As of 2026-09-03, the services cannot be made identical in all material ways:

1. Azure HorizonDB is in preview and is not available in Japan East. Its only
   documented Asia Pacific region is Australia East. The existing P2 result was
   collected in Japan East, so a fresh Flexible Server control must be deployed
   in Australia East for the primary comparison.
2. The live HorizonDB capability response exposes PostgreSQL 17 and 18, while
   the public HorizonDB extension catalog currently documents AGE 1.6.0 only
   for PostgreSQL 17. PostgreSQL 18 / AGE 1.7 is the preferred exact match, but
   it remains an execution gate rather than an assumed service capability.
3. HorizonDB storage grows automatically and does not expose a provisioned
   1,024 GiB capacity or user-selected IOPS. Logical data volume can be held
   constant, but physical storage capacity and I/O provisioning cannot.
4. HorizonDB high availability uses shared, zone-resilient storage and a
   standby compute replica in another availability zone. The P2 Flexible Server
   baseline used SameZone HA. The study will match the number and size of
   compute nodes used for HA, while recording this topology difference.
5. HorizonDB compute includes a local NVMe read cache and offloads durability,
   checkpointing, and several WAL tasks to its storage layer. CPU equality is
   therefore allocation equality, not hardware or work-placement equality.

These differences are part of the product feasibility result. Any performance
delta must be reported as an end-to-end service delta, not attributed solely to
HorizonDB's database-as-a-log architecture or to Apache AGE.

## Reference configuration

### Existing P2 evidence

The historical reference is
[`../results/summaries/p2-20260830.md`](../results/summaries/p2-20260830.md):

| Item | Frozen P2 value |
| --- | --- |
| Dataset | Supply-chain fixture, seed `20260829`, 64 shards, 1,170 files |
| Logical root | `561f3de625ab4ff26ac0c38f0c01bb2747a71de2ac8a096239212caea637e780` |
| Vertices / relationships | 16,000,000 / 40,000,000 |
| Total records / properties | 56,000,000 / 455,834,776 |
| Bytes committed per clean run | 95,752,780,229 |
| Source versions | Neo4j 4.4.48 and 5.26.30 Community |
| Source VM, each | `Standard_E8bds_v5`, 8 vCPU / 64 GiB |
| Source disk, each | 512 GiB Premium SSD v2, 10,000 IOPS, 750 MB/s |
| Neo4j memory | 24 GiB heap / 28 GiB page cache |
| Loader VM | `Standard_D8ds_v5`, 8 vCPU / 32 GiB |
| Loader settings | `fetchRows: 5000`, `batchRows: 20000`, `batchBytes: 64MiB` |
| Loader concurrency | source 1, transform 1, target connections 4 |
| Flexible Server | PostgreSQL 18.6 / AGE 1.7.0, `Standard_E8ds_v5` |
| Flexible primary compute | 8 vCore / 64 GiB |
| Flexible storage / HA | 1,024 GiB, autogrow disabled / SameZone HA |
| Clean load, Neo4j 4.4 | 48:04.79 |
| Clean load, Neo4j 5.26 | 44:42.64 |
| Flexible CPU | 10.69% average / 30.27% maximum over the final monitor window |
| Flexible memory | 33.34% maximum over the final monitor window |
| Target footprint | approximately 42.46 GB per graph generation |

The historical result is a continuity check only because its region and
PostgreSQL/AGE versions differ from the HorizonDB candidate.

### Controlled head-to-head matrix

Both targets will be created in Australia East and reached from the same loader
subnet. Availability-zone placement, service build, host generation, and quota
availability must be captured before the first timed run.

| Dimension | Flexible Server control | HorizonDB candidate |
| --- | --- | --- |
| Region | Australia East | Australia East |
| PostgreSQL major | 18 | 18, subject to live AGE preflight |
| Apache AGE | 1.7.0, subject to preflight observation | 1.7.x required for the primary comparison |
| Primary compute | `Standard_E8ds_v5`, 8 vCore / 64 GiB | 8 vCore / 64 GiB |
| HA compute | SameZone standby, same SKU | One standby replica, same 8 vCore / 64 GiB allocation |
| Storage | 1,024 GiB, autogrow disabled | Service-managed automatic growth |
| Network | Private access | Private Link; public access disabled |
| Workload endpoint | Primary/read-write endpoint | Primary/read-write endpoint |

The HorizonDB cluster must use two compute replicas in total so that it has one
primary and one HA/read standby. No migration or digest traffic will be sent to
the read-only endpoint. If the service or subscription cannot provision exactly
8 vCores per replica, the study stops before a timed run.

If HorizonDB PostgreSQL 18 cannot expose AGE 1.7.x, retain the failed preflight
as evidence. A PostgreSQL 17 comparison remains possible, but its AGE version
would not match Flexible Server and requires an explicit matrix amendment
before resources are resized or a P2 timed run begins.

## Scope

### Included

- Exact P2 fixture regeneration or reuse after manifest and SHA-256 validation.
- Both frozen Neo4j source versions and the corrected Neo4j 4.4 B-tree
  source-key indexes established during the later P3 investigation.
- agefreighter target compatibility, clean create, resume, replacement, and
  independent canonical digest verification on HorizonDB.
- Fresh, alternating clean-create runs on HorizonDB and Flexible Server for the
  controlled performance comparison.
- Primary CPU and memory, transaction/WAL, latency, storage use, network,
  loader, source, elapsed-time, and cost evidence.
- A comparison against both the fresh Flexible Server control and the
  historical Japan East P2 evidence.

### Excluded

- P3 or P4 scale.
- Graph-query workload benchmarking beyond the existing doctor, optimizer,
  count, bounded-integrity, and digest operations.
- Read scaling through HorizonDB replicas.
- Cross-region disaster recovery.
- Production cutover or customer workload claims.
- Infrastructure deletion, which requires separate authorization after
  evidence review.

## Execution stages

### Stage 0: Freeze access, budget, and test identity

1. Obtain HorizonDB preview access and confirm Australia East quota for two
   8-vCore replicas.
2. Confirm Australia East capacity for the VM SKUs, Premium SSD v2 settings,
   and `Standard_E8ds_v5` Flexible Server with SameZone HA.
3. Record a cost estimate and approve a 36-hour live test window. Because the
   HorizonDB preview pricing page directs customers to obtain a quote, do not
   infer its cost from the historical P2 charge.
4. Use a new resource group, immutable run ID, and tags identifying this as a
   P2 HorizonDB feasibility study. Preserve the current production-simulation
   resource groups and evidence.
5. Pin the tested git commit, binary SHA-256, container digests, fixture root,
   Azure CLI version, deployment API versions, and the retrieval time of all
   preview-service documentation.

### Stage 1: Build a reviewable, parallel target topology

Extend the infrastructure as a separate feasibility-study deployment rather
than replacing the existing Flexible Server resources in `infra/main.bicep`.
The deployment should contain:

- the same two Neo4j VM definitions, disks, memory settings, and private names;
- the same loader VM definition and a dedicated evidence disk;
- one PostgreSQL 17 Flexible Server control with AGE enabled;
- one PostgreSQL 17 HorizonDB cluster with two 8-vCore replicas;
- a private endpoint and private DNS for HorizonDB, because HorizonDB does not
  currently support VNet injection;
- no public IP on a VM and no public database ingress;
- a Key Vault or equivalent approved secret source, with no credential written
  to the repository or evidence bundle; and
- Azure Monitor collection for both targets using the finest common supported
  interval, plus 15-30 second guest and agefreighter sampling.

Run Bicep/ARM validation and `what-if`, inspect every resource identity and
SKU, and obtain review before deployment. A preview API or CLI behavior change
is a review stop, not an invitation to patch the live environment ad hoc.

### Stage 2: Prove HorizonDB and AGE compatibility

Create and attach a HorizonDB parameter group containing `age` in both
`azure.extensions` and `shared_preload_libraries`, allow the automatic cluster
restart, and then run `CREATE EXTENSION IF NOT EXISTS age CASCADE` in each test
database. Do not issue `LOAD 'age'`; HorizonDB preloads the library and the
documented behavior is to reject a manual load for lack of privilege.

Before any P2-scale load, capture and gate on:

```sql
SELECT version();
SELECT extname, extversion FROM pg_extension ORDER BY extname;
SHOW shared_preload_libraries;
SHOW search_path;
SHOW max_connections;
SHOW shared_buffers;
SHOW work_mem;
SHOW maintenance_work_mem;
SHOW effective_cache_size;
SHOW max_wal_size;
SHOW checkpoint_timeout;
```

Then run the agefreighter doctor and capability probe, a tiny fixture, and the
existing P0 smoke path. Confirm that:

- PostgreSQL 17 and AGE 1.6.x pass the repository's compatibility contract;
- every session can use `ag_catalog` through the configured search path;
- graph creation, label creation, binary load, count verification, digest
  export, metadata persistence, resume, and replacement primitives work with
  the managed-service privilege model; and
- no SQL path attempts to write local server storage or call a prohibited
  superuser-only operation.

Any extension version outside the repository's qualified matrix, failure to
preload AGE, or managed-service privilege incompatibility blocks P2 execution.

### Stage 3: Prepare and freeze the P2 sources

1. Generate the P2 fixture with seed `20260829`, 64 shards, and the existing
   deterministic allocation rules, or attach the retained fixture read-only.
2. Verify all 1,170 file hashes, the manifest root, exact per-label/type counts,
   and the expected 16,000,000 vertices and 40,000,000 relationships.
3. Import the same logical fixture into Neo4j 4.4.48 and 5.26.30 and verify the
   pinned image digests.
4. Use 24 GiB heap and 28 GiB page cache. Verify all required source-key
   indexes are online and that the Neo4j 4.4 query plans use B-tree indexes.
5. Freeze both sources read-only from the first discovery query through final
   verification. Record before/after source profiles and canonical roots.

The loader configuration is frozen at `fetchRows: 5000`, `batchRows: 20000`,
`batchBytes: 64MiB`, a 4 GiB memory limit, and 1/1/4 source, transform, and
target concurrency. No HorizonDB-specific tuning is permitted in the primary
comparison.

### Stage 4: Run the controlled clean-load comparison

For each Neo4j version, execute at least three paired repetitions. Each target
run must use a new empty database and graph. Alternate target order between
pairs and record it, for example `H-F`, `F-H`, `H-F`; randomize the first order
before execution. Do not run the targets concurrently.

For every repetition:

1. Confirm target identity, compute allocation, primary role, extension
   version, parameters, source root, and zero existing graph data.
2. Allow a fixed 10-minute idle period and record starting CPU, memory, active
   connections, storage used, WAL counters, and loader/source state.
3. Run one uninterrupted `create` load with the frozen P2 configuration.
4. Run report, exact counts, bounded integrity, doctor, optimizer, and the full
   56,000,000-record canonical digest.
5. Record the clean root and retain the database until the pair is reviewed.
6. Wait for target metrics to settle before starting the other member of the
   pair.

If the coefficient of variation for elapsed load time exceeds 5% on either
target/source combination, investigate the outlier and add two paired
repetitions without changing configuration. Failed runs remain evidence and
are never silently replaced.

### Stage 5: Complete HorizonDB P2 functional qualification

After the clean comparison, run the existing P2 recovery schedule against
fresh HorizonDB databases with the frozen settings:

1. terminate agefreighter near 25%, then resume the durable job;
2. restart the Neo4j 4.4 container near 40%, then resume;
3. block and restore HorizonDB connectivity near 60%, then resume; and
4. reboot the loader VM near 75%, then resume.

Each fault is separate and occurs only after a committed checkpoint. Recovery
must begin within 15 minutes after the fault clears. The resumed database must
have 56,000,000 records, monotonically advancing checkpoints, no unresolved
failed batch, and the same canonical root as its HorizonDB clean run.

Then run one uninterrupted full `replace` against a reviewed HorizonDB clean
target. Prove that the old graph remains active while the shadow graph loads,
promotion is atomic, the old generation is retained as a backup, and the new
generation matches the clean root. Do not run cleanup.

Optionally perform one planned HorizonDB failover during a separate recovery
run after the mandatory P2 schedule. Label this as HorizonDB-specific
operational evidence, not part of the direct Flexible Server performance
comparison.

## Measurements and normalization

### Common end-to-end measures

Collect the following per run and separately for vertex and relationship
phases where applicable:

- load elapsed time and total verification elapsed time;
- committed records and bytes per second;
- committed batches, batch latency p50/p95/p99, and p99/median ratio;
- first-to-final throughput-quintile ratio;
- loader CPU, RSS, GC, major faults, disk/network activity, and checkpoint age;
- Neo4j CPU, heap, GC, page cache, Bolt latency, disk/network activity, and
  executed query plans;
- target primary CPU and memory average, p95, and maximum;
- active/failed connections, transactions, deadlocks, lock waits, temporary
  file count and bytes, and buffer-pool hit ratio;
- network ingress/egress, storage used, relation/index/metadata sizes, and WAL
  bytes generated;
- commit latency and write/WAL latency using the closest semantically common
  metrics; and
- target service cost and total study cost, normalized per successful
  56,000,000-record load and per billion committed bytes.

Azure Monitor for HorizonDB provides CPU, memory, connections, transaction,
network, storage-used, WAL-bytes, WAL-writes, commit-latency, and write-latency
metrics at one-minute grain. Capture SQL statistics in parallel where the
platform metric is absent or not exportable. Do not manufacture a direct
HorizonDB equivalent for Flexible Server provisioned IOPS, disk queue depth,
or IOPS-consumption percentage; report those as service-specific diagnostics.

### Comparison method

For each source version, calculate the paired HorizonDB/Flexible ratios for:

- elapsed clean-load time;
- sustained records/s and bytes/s;
- p95 and p99 batch latency;
- primary CPU-seconds per million records, when the metric resolution permits;
- peak and average memory percentage;
- WAL bytes and final physical bytes per logical record; and
- target cost per completed load.

Report every repetition, median, range, and median paired ratio. With only
three planned pairs, avoid claims of statistical significance. Also show the
historical P2 result in a separate table clearly labeled **non-controlled**.

## Gates and stop conditions

### Hard correctness and operations gates

All existing P1-P4 correctness gates in [`acceptance.md`](acceptance.md) apply.
In addition, the study stops and preserves evidence if:

- the observed service, region, primary vCores, memory, replica count,
  PostgreSQL major, or AGE version differs from the reviewed manifest;
- HorizonDB cannot create or preload AGE using supported service controls;
- a managed-service privilege restriction breaks a required agefreighter path;
- any canonical range or final Merkle root differs;
- a reject, duplicate, missing endpoint, source mutation, or unresolved failed
  batch occurs;
- no checkpoint commits for 15 minutes while work remains;
- any host swaps, reaches OOM, loses durable data, or target storage behavior
  prevents safe completion;
- primary CPU exceeds 90% or memory exceeds 85% for five consecutive minutes;
  or
- the approved live window or cost ceiling is reached.

### Feasibility decision thresholds

Subject to reviewer approval before deployment:

- **GO** requires every correctness, resume, and replace gate to pass; median
  HorizonDB clean-load elapsed time no more than 10% above the fresh Flexible
  control for either source; no paired run more than 20% slower without an
  explained external event; p99/median batch latency at most 3; final/first
  throughput at least 80%; and normalized target cost no more than 20% above
  the control.
- **CONDITIONAL** applies when all functional gates pass but HorizonDB is
  10-25% slower, costs 20-35% more, needs a service-specific workaround, or a
  preview limitation prevents production equivalence.
- **NO-GO** applies on any hard functional failure or when median clean-load
  elapsed time is more than 25% above the control after outlier investigation.

A result can be performance-GO and still be operationally CONDITIONAL because
HorizonDB remains a preview service or lacks a required production feature.

## Evidence and deliverables

Each run must produce the artifacts required by `acceptance.md`, plus:

- reviewed deployment manifests and `what-if` output for both target services;
- HorizonDB parameter-group, replica-role, private-endpoint, and failover state;
- observed service/engine/extension versions and full relevant parameter dumps;
- raw one-minute Azure Monitor exports and aligned 15-30 second loader/source
  samples with UTC timestamps;
- one machine-readable row per run containing target, source, order, elapsed,
  correctness, resource, storage, WAL, latency, and cost fields;
- a redacted feasibility report with controlled and historical comparison
  tables, limitations, anomalies, and the GO/CONDITIONAL/NO-GO decision; and
- hashes for every committed summary and retained raw evidence bundle.

At the end of the authorized window, stop or scale to the approved retained
state without deleting databases, disks, logs, metrics, or manifests. Resource
deletion is a separate reviewed action.

## Execution checklist

- [ ] Preview access, quotas, region, SKUs, two HorizonDB replicas, and budget approved.
- [ ] Current HorizonDB region, version, AGE, networking, metrics, and pricing facts revalidated.
- [ ] Infrastructure changes reviewed with validation and `what-if` evidence.
- [ ] Exact 8-vCore / 64-GiB primary allocation proven on both targets.
- [ ] AGE preload, extension creation, doctor, tiny, and P0 compatibility gates passed.
- [ ] P2 fixture root, source images, counts, indexes, and source immutability frozen.
- [ ] At least three alternating clean-load pairs completed per Neo4j version.
- [ ] Every clean target passed exact counts and the 56,000,000-record digest.
- [ ] Four HorizonDB recovery scenarios passed against clean roots.
- [ ] HorizonDB full replacement and atomic promotion passed.
- [ ] Metrics, physical sizes, WAL, cost, and all service differences normalized and reported.
- [ ] Final decision reviewed; resources stopped and evidence retained.

## Current Microsoft references

These preview facts must be rechecked immediately before execution:

- [Azure HorizonDB overview and current regions](https://learn.microsoft.com/en-us/azure/horizondb/overview)
- [HorizonDB compute replicas and 8 GB memory per vCore](https://learn.microsoft.com/en-us/azure/horizondb/configure-maintain/concepts-compute-replicas)
- [Apache AGE on Azure HorizonDB](https://learn.microsoft.com/en-us/azure/horizondb/graph/age-overview)
- [HorizonDB extensions by PostgreSQL version](https://learn.microsoft.com/en-us/azure/horizondb/extensions/concepts-extensions-versions)
- [HorizonDB high availability and failover](https://learn.microsoft.com/en-us/azure/horizondb/high-availability/concepts-high-availability-failover)
- [Azure Monitor metrics for HorizonDB](https://learn.microsoft.com/en-us/azure/azure-monitor/reference/supported-metrics/microsoft-horizondb-clusters-metrics)
- [Flexible Server extensions by PostgreSQL version](https://learn.microsoft.com/en-us/azure/postgresql/extensions/concepts-extensions-versions)
