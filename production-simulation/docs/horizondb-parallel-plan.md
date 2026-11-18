# HorizonDB qualification and parallel execution plan

Prepared 2026-11-18. **Plan only for P0-P3; no large run is authorized.**
The user approved tiny migrations and their charges, with publication of
2.4.2 conditional on successful actual HorizonDB tiny qualification.

Read with the existing [test plan](test-plan.md),
[acceptance criteria](acceptance.md) and [dataset contract](dataset.md).
Those assertions remain required unless an explicit source-specific exception
is documented here; the historical Flexible Server result is not HorizonDB
evidence.

## Current execution status

This change has created no HorizonDB resources and has run no HorizonDB migration.
Azure CLI resource discovery failed with `AADSTS135010`. The alternative MCP
account did not expose the intended subscription, so no other subscription was
used. The existing controlled Azure GitHub runner is offline. Authentication,
resource inventory, capacity, current-month spend and network access remain
unverified. The configurations below are **conditional cost scenarios, not
validated deployment recommendations**.

Restore the intended Azure CLI tenant/subscription session without logging out
other shared sessions. Then inspect only the approved subscription and confirm
the current account before any write. Do not copy tokens or passwords into
logs, jobs, plans or Git.

## Coverage and gates

Six source configurations each target both HorizonDB PostgreSQL 17/AGE 1.8
and PostgreSQL 18/AGE 1.8: twelve independent routes.

| Source configuration | Source qualification |
|---|---|
| CSV | Deterministic generated supply-chain fixture |
| PostgreSQL | PostgreSQL 17, typed relational fixture |
| Neo4j 4.4 | Pinned official 4.4.48 image |
| Neo4j 5.26 | Pinned official 5.26.30 image |
| Cosmos NoSQL | Immutable, partitioned documents through the NoSQL API |
| Cosmos Gremlin backing documents | `_isEdge`/`_vertexId`/`_sink`/`_value` layout through the NoSQL API, not the native Gremlin protocol |

HorizonDB is GA, but its documentation still marks PostgreSQL 18 and AGE as
Preview. Record effective SQL versions and extension versions, not just
control-plane versions. See the [extension matrix][extensions].

The existing production harness is Neo4j-to-Flexible-Server-specific and
hardcodes PostgreSQL 18/AGE 1.7 in its preflight. It is not a HorizonDB runner.
Before live execution, adapt the source seeders, identity mappings, independent
verification and managed-service preflight. Preserve the old qualified harness
and its checks; do not bypass its version interlock.

| Phase | Base vertices | Base edges | Required route scenarios |
|---|---:|---:|---|
| tiny, approved | 160 | 400 | Actual generated load on all twelve routes; existing smaller mode, type, paging, identity and committed-batch resume fixtures within the approved cap |
| P0, proposed | 100,000 | 358,000 | Clean create, full replace, 1% append, 1% upsert, one interrupted/resumed full load, exact-count harness checks |
| P1, proposed | 1,600,000 | 4,000,000 | Same modes and recovery; full canonical ranged digest and capacity/RU calibration |
| P2, proposed | 16,000,000 | 40,000,000 | Clean create, full replace, 1% append/upsert; four separate full-load recovery cases where applicable; freeze tuning and P3 forecast |
| P3, proposed | 160,000,000 | 400,000,000 | One clean and one combined-fault recovery load per route; full independent digests on both |

Append adds 1% of the base vertices and edges; upsert changes 1% of existing
records. The expected final graph after append is 101% of the base, not 102%.
Prepare immutable, separately selectable delta fixtures before discovery.
Tiny functional subcases must not silently import the 20,200-row countries
fixture or use benchmark defaults. A small fixture pass alone does not count
as the actual 160/400 load.

P3 full replace is not mandatory under the existing test plan: it remains a P2
gate because replacement temporarily retains active, shadow and backup
generations. P4 and P3-scale incremental operations require a separate scope.
Retain every applicable existing assertion; do not silently count unimplemented
Gremlin mode/recovery coverage as passing.

Every route follows its own dependency chain:

```text
tiny -> P0 -> P1 -> P2 -> approved P3
```

Different routes may progress concurrently. No route skips its previous-phase
correctness, capacity or cost gate. A shared implementation defect pauses all
affected routes; an isolated source failure need not discard other evidence.
Human approval is still required for each larger phase and its exact envelope.

## Parallel layout

Use a bounded queue of **four isolated lanes**, not twelve always-on targets.
The following grouping is the cost model; a ready route can fill a free lane
when its prerequisites and the global RU/budget tokens permit.

| Group | Four concurrent routes |
|---|---|
| A | Neo4j 4.4 and 5.26, each to PG17 and PG18 |
| B | CSV and PostgreSQL source, each to PG17 and PG18 |
| C | Cosmos NoSQL and Gremlin backing documents, each to PG17 and PG18 |

Each active lane has one loader VM, one dedicated HorizonDB cluster and fresh
run-scoped databases. Do not measure multiple writers in a shared cluster as
independent performance qualifications. Separate source VMs/disks are required
for fault isolation in A and for the two PostgreSQL routes in B. Cosmos routes
can share immutable source datasets and a governed aggregate RU allocation;
source-outage faults must be scoped to each loader, not the whole account.

Use fixed, individually identified VMs, not autoscaling or automatic VM
replacement: same-VM reboot, exact job resumption and retained state are part
of the evidence. No load balancer is needed for these batch workers.

Cost-scenario resources, based on the earlier P3 engineering baseline:

| Role | Conditional configuration | Quota status |
|---|---|---|
| HorizonDB, each lane | One `Standard_E8ds_v6` writer, 8 vCores/64 GiB; no reader/HA node | Unverified; blocked |
| Loader, each lane | `Standard_D8ds_v5`, 8 vCPU/32 GiB | Unverified; blocked |
| Neo4j 4.4 source, each target route | `Standard_E8bds_v5`, 8 vCPU/64 GiB | Unverified; blocked |
| Neo4j 5.26 source, each target route | `Standard_E16bds_v5`, 16 vCPU/128 GiB | Unverified; blocked |
| PostgreSQL source, each target route | `Standard_E16bds_v5` capacity placeholder, to be calibrated at P0/P1 | Unverified; blocked |
| Stateful source disk | 1,024 GiB Premium SSD v2, 10,000 IOPS/750 MB/s | Capacity and policy unverified |
| Large Cosmos source | Separate single-region manual-provisioned account, up to 200,000 RU/s aggregate | Account/partition throughput quota unverified |

Single-writer HorizonDB keeps this a migration correctness/recovery qualification,
not an HA/failover qualification. Adding one equal-sized reader per cluster
doubles the HorizonDB compute component; reapprove that scenario. Do not use
Spot or temporary disks for the sole copy of state/evidence.

Keep all data-plane traffic in Japan East with private endpoints, private DNS,
TLS and minimum scoped identities. Use no public VM IP or public database
exposure. HorizonDB uses Private Link, not Flexible Server delegated-subnet
network injection. Its resource provider is `Microsoft.HorizonDb`; the
[current quickstart][quickstart] uses REST API `2026-10-01-preview`, with no
dedicated Azure CLI command. Do not deploy the Flexible Server template under
a HorizonDB name.

The largest VM group needs 80 regional vCPUs: 48 in the source VM family and 32
in the loader family, plus current usage. Four targets need 32 managed-service
vCores separately. Discover actual quota names and SKU/zone restrictions using
the intended subscription; neither a public price nor an earlier successful
deployment proves current capacity. Do not provision until both regional and
family quotas, HorizonDB capacity and Cosmos partition limits are verified.

## Faults, verification and evidence

Retain the existing checkpoint-relative schedule: process termination near 25%,
source connectivity interruption near 40%, target connectivity interruption near
60%, and loader reboot near 75%. P2 rehearses each independently; P3 combines
them only after the applicable P2 cases pass. Use the same job ID, immutable
source, target generation and configuration fingerprint on resume.

Restart only a lane-owned Neo4j/PostgreSQL source. Cosmos source interruption
means a lane-local connectivity fault, not deletion or restart of the shared
service. CSV has no database restart: record that fault as not applicable and
test applicable process/read/target/reboot interruptions instead.

Require exact total/per-label/type counts; zero rejects, duplicates, failed
committed batches and missing endpoints; monotonic checkpoints; and matching
clean/resumed canonical range digests and roots. Keep P0's documented bounded
integrity exception distinct from P1-P3 full digest requirements.

Capture build/version/hash, actual PG/AGE versions, source image digests, seed,
manifest, route/phase/run IDs, scoped resource IDs, all mode reports, fault
timeline, source before/after invariance, RSS/CPU/IO/latency/throughput and Cosmos
RU/throttle telemetry. A skipped case, counts-only result or unavailable digest
is not a pass. Raw evidence stays outside Git; commit redacted summaries only.
Retain the existing 2-GiB loader RSS bound before P3 and 4-GiB P3 bound, 24-hour
provisional clean-P3 objective, throughput/latency stability criteria and
15-30-second metric sampling. P2 must freeze the actual P3 objective.

## Public price inputs

USD, Japan East, pay-as-you-go, retrieved 2026-11-18 from the
[Azure Retail Prices API][retail]. No free tier, reservation, Spot or private
contract discount is assumed. These are unit prices, not account quotes.

| Meter | Unit price |
|---|---:|
| HorizonDB compute, one vCore | $0.206406/hour |
| HorizonDB data storage | $0.301724/GB-month |
| HorizonDB chargeable backup storage | $0.12/GB-month |
| Linux `Standard_D8ds_v5` VM | $0.584/hour |
| Linux `Standard_E8bds_v5` VM | $0.812/hour |
| Linux `Standard_E16bds_v5` VM | $1.624/hour |
| Premium SSD v2 capacity | $0.000132/GiB-hour |
| Premium SSD v2 IOPS above 3,000 | $0.000008/IOPS-hour |
| Premium SSD v2 throughput above 125 MB/s | $0.000066/MBps-hour |
| P15 LRS managed OS disk | $43.720714/month |
| Cosmos single-write-region manual throughput | $0.009/100 RU/s-hour |
| Cosmos data storage | $0.2825/GB-month |

HorizonDB retail products are under service **Azure Database for PostgreSQL**,
not service name "Azure HorizonDB". Select the HorizonDB product/meter, not a
Flexible Server compute price. Exclude zero-price free-tier meters. VM
`Standard_D8ds_v5` also appears as a Flexible Server SKU: select the **Virtual
Machines** service for the loader.

Example reproducible public API filters:

```text
contains(productName, 'Horizon') and armRegionName eq 'japaneast'
serviceName eq 'Virtual Machines' and armSkuName eq 'Standard_D8ds_v5'
  and armRegionName eq 'japaneast' and priceType eq 'Consumption'
serviceName eq 'Azure Cosmos DB' and productName eq 'Azure Cosmos DB'
  and meterName eq '100 RU/s' and skuName eq 'RUs'
  and armRegionName eq 'japaneast'
```

Also exclude Windows, Spot, low-priority and reservation VM prices. Respect API
pagination. Monthly examples use 730 hours. Preserve meter units: the retail
API reports GB-month for service storage and GiB-hour for SSD v2; convert
measured GiB to GB before applying a GB meter and confirm effective billing.

## Cost model and sensitivity

Four targets and four loaders cost
`4 * (8 * 0.206406 + 0.584) = $8.940992/hour`, before sources and storage.
Group A adds $4.872/hour for four source VMs; group B adds $3.248/hour for the
two PostgreSQL source VMs. The 1-TiB/10k-IOPS/750-MBps source disk costs
`1024*.000132 + 7000*.000008 + 625*.000066 = $0.232418/hour`.

| Conditional concurrency option | Target/loader compute only | Trade-off | Capacity status |
|---|---:|---|---|
| 2 isolated lanes | $4.470496/hour | Lower peak quota; longer queue and retention, not half the total cost | Unverified; blocked |
| 4 isolated lanes, model below | $8.940992/hour | Balance source/version parallelism with bounded peak capacity | Unverified; blocked |
| 12 isolated lanes | $26.822976/hour | Higher quota/idle exposure and shared-RU contention; no phase barriers removed | Unverified; blocked |

The complete mode/recovery scenario count, not just twelve clean copies,
drives the volume estimate. With `N0=458000`, `N1=5600000`, `N2=56000000`,
`N3=560000000` total base records:

```text
source records read per route =
  3.02*N0 + 3.02*N1 + 6.02*N2 + 2*N3 = 1,475,415,160
Cosmos source records across four Cosmos routes = 5,901,660,640
two immutable Cosmos layouts plus deltas seeded across phases =
  2*1.02*(N0+N1+N2+N3) = 1,268,998,320 documents
```

The small fraction after an interrupted uncommitted checkpoint, discovery,
profiles, query/index costs and retries must be included in the measured
effective RU/record, not ignored. The following **unmeasured assumptions**
are inputs to replace at P0/P1:

- 10 RU per seeded document including indexes; 3 RU per source record read in
  the central case, 5 RU in the higher-cost case, including additional reads.
- 200,000 provisioned RU/s at 70% effective utilization; $18/hour while active.
  Partition skew, sequential iterator limits or throttling can reduce that
  utilization and invalidate both the time and cost forecast.
- Non-Cosmos P3: 22 hours per four-route group including preparation, both
  loads and verification. Scale that group cost by
  `1,475,415,160 / 1,120,000,000` for P0-P2 extra scenarios. This is a forecast
  from earlier Neo4j/Flexible Server timings, not a HorizonDB or PG-source
  performance result.
- Cosmos targets are created after bulk seed completion. Keep all four active
  for `source-read RU / (200000*.70*3600) + 8` hours; those eight hours are
  provisional setup/digest allowance, to be replaced by measurements.

| Complete P0-P3 cost component | 3 RU/read central case | 5 RU/read higher-cost case |
|---|---:|---:|
| Eight non-Cosmos routes, target/loader/source VM compute | $753.57 | $753.57 |
| Cosmos seed and read throughput, including all phases | $1,085.53 | $1,507.08 |
| Four Cosmos routes, target and loader compute | $385.62 | $595.01 |
| Storage, backup, disks, private networking, fixture generation/artifacts/logging allowance | $500.00 | $500.00 |
| Subtotal | $2,724.72 | $3,355.66 |
| 25% uncertainty reserve | $681.18 | $838.92 |
| Planning total | **$3,405.90** | **$4,194.58** |

These are not hard upper bounds. The $500 allowance assumes prompt cleanup of
owned live resources after at most 24 hours of post-verification review, at most
four active source data disks, approximately 4 TiB peak HorizonDB data, 1 TiB
Cosmos data, and seven-day artifact retention in Blob rather than keeping live
databases for a month. Backups and additional billed network/logging usage may
exceed it; measure and reprice. If actual sizes or retention exceed these
assumptions, block promotion rather than consuming the reserve silently.

At the central assumptions Cosmos throughput is active for about 60.31 hours,
with its target group active about 43.13 hours. At 5 RU/read those increase to
83.73 and 66.55 hours. With two other groups at roughly 29 hours each including
smaller phases, plan approximately 5-7 active days for the complete queue, plus
approval delays. This is not a promised completion date.

## Monthly budget and stop envelope

The **$3,750/month budget is not all known to be available**: current spend and
unposted accrual could not be queried. Provisionally allocate $3,000 to test
consumption and retain $750 as monthly reserve, subtracting unrelated spend
first. Do not treat the budget or alerts as automatic enforcement.

The central estimate fits only if these assumptions are validated. The higher
case exceeds the monthly budget: split remaining waves across billing months,
reduce measured costs without dropping required cases, or obtain a larger
explicit envelope. P0/P1 RU and storage measurements decide this before P2/P3.
Reducing concurrency alone does not divide total billed work or Cosmos RU.

Four always-on target/loader lanes alone cost **$6,526.92 per 730-hour month**,
before any source or Cosmos charges. Bounded execution windows and authorized
cleanup are essential; do not leave four clusters running all month. Twelve
simultaneous lanes multiply peak capacity and contention without removing any
dependency or reducing required RU, so they are not the baseline scenario.

Use an external controller with a strict four-lane semaphore, per-route leases
and scoped identities. Account for observed spend, accrued resource-hours,
Cosmos provisioning, retained storage and remaining-work forecasts. Stop
admitting new work before the smaller of the approved phase envelope and
remaining monthly budget would be exceeded. If cost/monitoring is unavailable,
do not promote. Alerts at 50/75/90% complement, not replace, this controller.

Retain the existing immediate stops: any reject/missing endpoint/duplicate,
unexpected source mutation, OOM/swap, storage at 80%, or no committed checkpoint
for 15 minutes. Retain the existing five-minute sustained-throttling stop,
twice-forecast runtime/cost stop, durable-volume loss stop and reviewed-manifest
identity/version/SKU interlock as well. A deadline must actively stop lane-owned
compute; a tag or approval timestamp is not an automatic shutdown. Do not assume
HorizonDB supports Flexible Server's stop/start semantics.

The larger-run approval must explicitly include lifecycle actions and exact
owned resource IDs: retain manifests/reports/digests, review results, then
delete disposable clusters/accounts/disks and archive raw artifacts to the
approved retention store. Do not delete earlier P3 resources, shared source
accounts, or unrelated evidence. Without cleanup authorization, include all
continuing storage and compute in the forecast and do not start an otherwise
unfunded run.

## Tiny and release resumption

After authentication and capacity are restored, use two isolated 2-vCore,
single-writer HorizonDB clusters and a bounded loader/source environment.
The compute-only six-hour scenario for both clusters plus one D8ds v5 worker
is approximately $8.46; allow roughly $15 including small storage/network
usage, subject to the actual resource quote and lifecycle. This is a cost
forecast for the approved tiny work, not permission to run P0.

Run all twelve actual tiny routes and the applicable smaller regression
fixtures. Retain SQL version observations and full outcomes. Only after every
required tiny route passes may the release proceed: finalize the compatibility
evidence/release notes and installation examples, merge the reviewed change,
then create the v2.4.2 tag and use the existing signing, notarization, quality,
Cosmos, packaging and provenance gates. Restore the offline protected Azure
runner rather than skipping its gate. No tag or published release has been
created while these prerequisites are blocked.

[extensions]: https://learn.microsoft.com/azure/horizondb/extensions/concepts-extensions-by-engine
[quickstart]: https://learn.microsoft.com/azure/horizondb/configure-maintain/quickstart-create-cluster
[retail]: https://learn.microsoft.com/rest/api/cost-management/retail-prices/azure-retail-prices
