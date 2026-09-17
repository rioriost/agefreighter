# Gremlin source preparation — live execution

September 17, 2026. **First preparation failed before writes; corrected retry
being prepared. Not a qualified migration.**
Continuation of [candidate/source preparation](gremlin-live-preparation-20260917.md).

## Authorization and fresh gates

The user explicitly approved creation of `p1/graph-gremlin-p1-20260917`, a
temporary writer scoped only to that new container, and the pinned preparation
loader on retained VM `af-bd3b66801e184d788f36`, for at most eight hours.
After preparation, remove that writer and deallocate the VM. Preserve all
source/target/failed-run evidence. No permission to modify the accepted
`p1/graph` is inferred.

- USD 800 and outer deadline `2026-09-20T07:14:35.311Z` are unchanged.
  Cost Management succeeded at this check: USD **126.342306949342** for this
  resource group, queried for September 12–17. This is billed-to-date data and
  can lag; it is not the full historical trial bill or a guaranteed final cost.
- Keep the existing USD 600 conservative reserve. Fresh Azure Retail Prices
  responses show Linux B2s_v2 USD 0.109/hour and Japan East Cosmos autoscale
  USD 0.0135 per 100 RU/hour. At a 4,000-RU maximum, the additional container
  is at most USD 0.54/hour for throughput, approximately USD 38.3 through the
  outer deadline; eight VM hours are USD 0.872. Reserving another USD 10 for
  this preparation's storage/network/margin gives a planning envelope under
  USD 650 including the USD 600 reserve. This is not an invoice guarantee.
  Later migration VM/target costs require their own refreshed gate.
- Initial read-only checks: nine VMs deallocated, 17 Flexible Servers Stopped,
  no RG lock, no delete/deny/deallocate/lock activity returned by the bounded
  recent activity query. Observed external policy/security actions are retained
  and not disabled or attributed to an unobserved fault.
- Transfer storage remains HTTPS-only, TLS 1.2, public HTTPS enabled with
  anonymous/shared-key access disabled and its already authorized exact-account
  exception unchanged. The VM's existing Blob Reader is limited to its original
  transfer container; no Blob access grant was added.

## Exact new resources and pinned inputs

Cosmos account: `afcosmosp120260907`; existing database: `p1`.
New container: **`graph-gremlin-p1-20260917`**, `/partitionKey`, autoscale
maximum 4,000 RU/s. The accepted `graph` container remains separate at 4,000
RU/s. Account public networking remains Disabled and local/key auth disabled.

Temporary role assignment:
`879de73d-ab1b-4d10-97b6-b822ce8c5a55`, Built-in Data Contributor ending `0002`,
principal `b7269277-f2bd-4660-8301-7bb24789f997`, exact scope ending
`/dbs/p1/colls/graph-gremlin-p1-20260917`. A subsequent GET confirms that binding.
No account-wide writer and no account key are used. Container-specific scopes
are supported by the [official Cosmos RBAC guidance](https://learn.microsoft.com/en-us/azure/cosmos-db/how-to-connect-role-based-access-control).

The fixture loader now supports `-require-empty`: drain empty continuation
pages to EOF before any writes; deny existing documents, denied/incomplete
reads and cancellation. Unit/race tests and vet pass. This is a preflight,
not a distributed writer lock; no other writer is to use this new fixture.
Historical loader behavior without the opt-in flag is unchanged.

| Input | Binding |
|---|---|
| Loader source | `6c382c2` (clean Linux amd64, CGO disabled build) |
| Loader SHA-256 | `93570ccebac744503e94775614cc50befb9ef2fea032d79a851a6fa6afd0cfb9` |
| Loader local path | `production-simulation/work/gremlin-loader.BPKxhs/cosmosfixtureload` |
| Loader bytes / Blob ETag | 11,502,980 / `0x8DF14956E2BE85D` |
| Source archive SHA-256 | `f01f2044429b3a8cb2f1d123d0b2e41ec20bc1610c022e949ae5ec087219dcc6` |
| Source archive bytes / Blob ETag | 246,433,349 / `0x8DF1495719D8266` |
| Guest wrapper source | `0c62fb4`, `prepare-p1-gremlin.sh` |
| Guest wrapper SHA-256 | `ab6aa67a040b9f67be46777de8438c6c688d5ed2dba2fbdf9f188d5493675fb0` |

Uploads used authenticated, create-only Blob writes beneath the new
`gremlin-p1-20260917/<sha256>/` prefix in existing account
`afbd3b66801e184d788f36f4`, container
`af-bd3b6680-1e18-4d78-8f36-f43467a09a0a`. No prior Blob was replaced.
The wrapper authenticates downloads with IMDS, refuses redirects, verifies
length/hash/archive membership/every document hash and count, and uses a
create-only guest directory. All partial files are retained. It does not
install an additional package or run the historical destructive wrapper.

## Guest dispatch and stop controls

Only the retained B2s_v2 preparation VM was started. Fresh guest evidence at
`08:27:45Z`: boot `bddc0474-fae9-4b42-b11f-157589f051e1`, disk 4%, memory
available 7,428 MiB, no swap/OOM, no existing loader/service, new staging path
absent. Python is available; jq is absent, so the reviewed wrapper uses the
Python standard library rather than installing a package.

Azure Run Command acknowledged dispatch of unit `af-gremlin-p1-20260917.service`.
The unit sets RuntimeMaxSec=8h, MemoryMax=2G, MemorySwapMax=0 and Restart=no.
The loader additionally has a seven-hour timeout. An enabled Azure VM shutdown
schedule targets this exact VM at **16:00 UTC** (September 18, 01:00 JST),
earlier than the eight-hour bound from this dispatch. Verify/deallocate after
completion rather than treating a shutdown schedule as deallocation evidence.

Guest evidence/staging: `/var/lib/agefreighter-gremlin-p1-20260917`.
Successful dispatch alone proves neither a source write nor completed source
preparation. The next observation must establish service/loader progress.

At `08:30:29.071Z`, all 66 installed-extension workflow/report JSON files still
retain aggregate SHA-256
`819cdcd7c1bae30b57cb669fcd0f7a2d45172c912072e05219f9197b33c28e9f`.
No GUI workflow, accepted target, production runner or verifier was changed.

## First attempt: gateway query refusal, no source writes

Unit started at `08:29:30Z`, verified the pinned archive/binary and all documents
at `08:30:19Z`, then exited 1 at `08:30:21Z`. Disk remained 9%. The empty
preflight's `SELECT TOP 1 VALUE 1 FROM c` received HTTP 400: cross-partition TOP
cannot be directly served by the gateway. This is not an RBAC failure. The
failure occurs before any writer goroutine is started; the result file is zero
bytes. Retained progress log SHA-256:
`9f618de4222380fe55850cf44841c946fec69843517db36f5bea34547e00ba6f`.

The correction reuses `SELECT VALUE 1 FROM c`, the existing remotely validated
count projection, for both empty preflight and final count. It still drains
empty continuation pages and refuses the first observed document before writes.
Race/unit tests and vet pass. No permission expansion or source change is needed.
An explicit, reviewed retry will use a newly pinned corrected loader and fresh
`-r2` guest directory/unit; the failed first attempt is not restarted or replaced.
The original budget, outer deadline, 16:00 UTC shutdown and maximum preparation
window remain unchanged. The first binary in the table is historical and must
not be used for the retry.

## Remaining acceptance

1. Observe unit health, progress, swap/OOM and storage. Do not rerun the
   create-only wrapper or silently retry a failed/partial source preparation.
2. Require successful unit exit, checksummed load report, all 18 file counts
   and an independently drained remote total of 5,600,000 documents.
3. Remove only role `879de73d-ab1b-4d10-97b6-b822ce8c5a55`, verify absence,
   then deallocate the preparation VM and retain its disk/evidence.
4. Proceed to a fresh installed-GUI Gremlin migration and all 64 target ranges;
   B05/B10 remain open until their actual acceptance evidence is complete.
