# HorizonDB qualification and parallel execution plan

Prepared 2026-11-18. **P0-P3 execution is approved; technical and cost
promotion gates remain mandatory. No P0-P3 migration has started.**
After successful tiny qualification and publication of 2.4.2, the user
approved all four larger phases on 2026-11-18. Existing tiny HorizonDB clusters
may be reused, or deleted if replacing them with new clusters is preferable.
That conditional replacement approval is not unrestricted teardown authority.

Read with the existing [test plan](test-plan.md),
[acceptance criteria](acceptance.md) and [dataset contract](dataset.md).
Those assertions remain required unless an explicit source-specific exception
is documented here; the historical Flexible Server result is not HorizonDB
evidence.

## Current execution status

The initial Azure CLI discovery failed with `AADSTS135010`. After the user
reauthenticated on 2026-11-18, the intended subscription was accessible.
Historical test resources and registered GitHub runners were absent.
Japan East advertises PG17/PG18 and two-vCore pools; the tiny D8ds v5 has
unrestricted availability and sufficient regional/family quota.

An isolated tiny-only environment was provisioned using
[`horizondb-tiny.bicep`](../infra/horizondb-tiny.bicep), after successful ARM
validation and a Create-only what-if. It contains two two-vCore single-writer
clusters, one private runner VM and a private-endpoint serverless Cosmos
fixture account with local authentication disabled. HorizonDB Private Link
creation failed because the network provider called the incompatible
`2026-01-20-preview` cluster API for a `2026-10-01-preview` cluster. The explicit
`allowTinyNATFallback=true` alternative permits only the runner NAT's single
static IPv4 in each pool firewall, with verified TLS and Entra authentication.
It never adds an all-Azure or Internet-wide firewall rule. The public IP remains
outbound-only for the VM, which has no public IP or inbound access.
The default template still requests private-only HorizonDB access.
All twelve actual tiny routes and all 24 selected existing top-level
mode/type/recovery scenarios subsequently passed without skips. The
[qualification evidence](horizondb-tiny-20261118.json) records exact counts,
versions and canonical roots; raw failed and successful attempts were retained.
Public HorizonDB access was disabled after qualification. These tiny results
do not validate the larger configurations below, which remain
**conditional cost scenarios, not validated deployment recommendations**;
their complete SKU restrictions and remaining monthly budget are unverified.
The protected 2.4.2 release gates subsequently passed. After the live Cosmos
gate completed, the one-job Azure runner was removed and its VM was
deallocated, retaining its disks and qualification evidence.

SQL preflight observed PostgreSQL 17.11 and 18.6, AGE 1.8.0 available,
the AGE library already preloaded and `pending_restart=false`. However,
pool restart and the actual post-qualification stop requests returned HTTP 501
`PowerOperationNotEnabled`, explicitly stating that start, stop and restart are
unavailable for these clusters.
The emergency timer therefore **cannot cap HorizonDB charges on these
clusters**. Their four total vCores continue at approximately $0.825624/hour,
plus storage/networking, until supported lifecycle control or explicitly
approved cleanup is available. The six-hour figure below is not an enforced
upper bound. Do not start any larger phase without a verified lifecycle path.

### Approved scale preflight

The authenticated subscription's month-to-date cost query now succeeds; its
actual cost response is retained privately rather than published here.
Japan East advertises the required 8-vCore/64-GiB Intel generation-1
provisioned HorizonDB profile. The D8ds v5, E8bds v5 and E16bds v5 VM SKUs have
no subscription/zone restrictions. Regional VM quota is 101 vCPUs with 8 in
use, and the loader/source family quotas are 100 each. The largest proposed
group plus the retained tiny VM would use 88/101 regional vCPUs, above the 80%
advisory threshold but below the limit. These checks do not prove HorizonDB
allocation capacity, Cosmos throughput capacity or the larger-run forecast.

Private Link remains blocked by the live provider. Both a manual-approval
private endpoint through Azure CLI and an explicit request using the latest
advertised Network API, `2026-07-01`, fail with `ApiVersionMismatch`: the
network provider calls HorizonDB with `2026-01-20-preview`, incompatible with
the current clusters. The original automatic-approval attempt also failed.
The failed probes created no private endpoint resource. Both existing
HorizonDB clusters remain `Ready` with public access `Disabled`; no cluster
was deleted, no replacement was deployed, and no migration load was started.

An 8-vCore profile is a candidate for reusing the existing targets. There is
no verified evidence that replacing them would resolve the Private Link
incompatibility. Before proceeding, either restore working Private Link or
approve extending the exact fixed-NAT-IPv4, verified-TLS, managed-identity
fallback beyond tiny. The approved larger-run lifecycle must also cover the
owned resources needed to enforce deadlines and budget limits, including
reused targets; unsupported stop operations cannot enforce those limits.

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
| P0, approved | 100,000 | 358,000 | Clean create, full replace, 1% append, 1% upsert, one interrupted/resumed full load, exact-count harness checks |
| P1, approved | 1,600,000 | 4,000,000 | Same modes and recovery; full canonical ranged digest and capacity/RU calibration |
| P2, approved | 16,000,000 | 40,000,000 | Clean create, full replace, 1% append/upsert; four separate full-load recovery cases where applicable; freeze tuning and P3 forecast |
| P3, approved | 160,000,000 | 400,000,000 | One clean and one combined-fault recovery load per route; full independent digests on both |

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
The four phases are approved, but changes beyond the approved execution
envelope still require review; approval does not waive a failed promotion gate.

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

The **$3,750/month budget is not all available**: month-to-date reported spend
is now queryable, but unposted accrual must still be estimated. Provisionally
allocate $3,000 to test consumption and retain $750 as monthly reserve,
subtracting unrelated spend first. Do not treat the budget or alerts as
automatic enforcement.

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

Use two isolated 2-vCore, single-writer HorizonDB clusters and a bounded
loader/source environment.
The compute-only six-hour scenario for both clusters plus one D8ds v5 worker
is approximately $8.46; allow roughly $15 including small storage/network
usage, subject to the actual resource quote and lifecycle. This is a cost
forecast for the approved tiny work, not permission to run P0.

Run all twelve actual tiny routes and the applicable smaller regression
fixtures. The strict live suite is
`go test -count=1 -v ./production-simulation -run '^TestHorizonDBTinyIntegration$'`
with `AGEFREIGHTER_REQUIRE_HORIZON_TINY=1`. It refuses missing settings, existing
fixture output, non-HorizonDB targets or a generated dataset outside 160/400.
It seeds isolated PostgreSQL/Neo4j/Cosmos fixtures, verifies all twelve exact
loads and retains full target canonical comparisons. Offline configuration
checks do not count as live results.
The first live attempt passed ten routes; the two explicit Cosmos queries
failed because the Go SDK cannot merge cross-partition `ORDER BY` responses.
Use supported unordered cross-partition queries and bounded, source-key-sorted
canonical verification, not a row-count-only substitute. A deliberate retry
may set `AGEFREIGHTER_HORIZON_TINY_REUSE_FIXTURE` to the earlier output directory:
its verified fixture hash must match, no sources are reseeded or deleted, and
the new output directory and target graph names remain unique.
The immutable-source retry passed all twelve complete canonical comparisons.
The smaller existing source assertions initially attempted superuser-only
`LOAD 'age'`; switching their setup to the existing hosted AGE session
initializer preserved all assertions and allowed all 24 top-level scenarios
(66 including subtests) to pass on both actual targets.

The HorizonDB API requires default parameters during pool creation. Deploy
the template once with its default `configureAGE=false`, then with
`configureAGE=true` and the observed runner `administratorObjectId`. Check SQL
preload settings and `pending_restart`. Restart only if required and supported;
the current resources applied the immediate association without a pending
restart despite the unavailable power-operation API. Do not redeploy the
first-pass settings over initialized pools.
Source bootstrap refuses non-tiny VM metadata and existing credentials.
The live suite also fences sources to the dedicated local PostgreSQL/Neo4j
ports and run-scoped Cosmos account naming. Pool resources must omit tags;
the live resource provider rejects them despite the published generic schema.
Poll the actual authentication operation to completion and verify its persisted
state is `Enabled`; an accepted response or a previous `Succeeded` resource
state is insufficient. A documented `PATCH` enabled Entra authentication on
the current resources when the initial `PUT` had not left it enabled.
If Private Link fails with the API mismatch above, explicitly select the
NAT-only fallback and verify the sole firewall rule matches the owned NAT IP.
Cosmos must remain private and the VM must retain deny-all inbound rules.
Install a guarded pool-stop deadline only after actual stop/start support has
been verified for the intended resource model. The current HTTP 501 means the
installed five-hour timer is not an effective HorizonDB cost control.
The VM has a separate UTC auto-shutdown schedule. Neither timeout nor a failed
stop attempt authorizes deleting resources or evidence.

SQL version observations and full outcomes were retained. The `v2.4.2` tag
was created only after all required tiny routes passed and the qualified
change was merged. The protected
[release workflow](https://github.com/rioriost/agefreighter/actions/runs/44221531718)
then passed signing, notarization, quality, live Cosmos, packaging and
provenance gates and published 2.4.2 on 2026-11-18. The protected Azure runner
was restored for its gate rather than bypassing it. P0-P3 execution was
subsequently approved, but remains blocked on the scale preflight above.
Deletion of the existing tiny clusters is approved only if replacement is
preferable; broader budget/deadline teardown still needs an approved scope.

[extensions]: https://learn.microsoft.com/azure/horizondb/extensions/concepts-extensions-by-engine
[quickstart]: https://learn.microsoft.com/azure/horizondb/configure-maintain/quickstart-create-cluster
[retail]: https://learn.microsoft.com/rest/api/cost-management/retail-prices/azure-retail-prices
