# Gremlin installed-GUI migration — fresh draft

September 18, 2026 JST. **Approved transfer-storage exception applied; authenticated
access and pinned runner upload pass. VM provisioned and Linux readiness verified;
Cosmos read access verified; first inventory failed and evidence is retained.
Discovery-limit correction required before retry. No migration submitted.**

The [source preparation](gremlin-source-execution-20260917.md) passed separately.
Its 5.6M Gremlin-shaped NoSQL documents are not a GUI/target qualification.

## Fresh gates

- USD 800 and `2026-09-20T07:14:35.311Z` remain unchanged. Billing refresh
  returned 429; do not label a stale billed total current. Keep the existing
  USD 600 extended-retention reserve and the separately reserved preparation
  increment. Runner/target execution requires a bounded incremental estimate.
- All nine VMs are deallocated and all 17 Flexible Servers are Stopped. No RG
  lock was returned. Cosmos remains private with local/key auth disabled; the
  isolated container still uses `/partitionKey`. Its temporary writer is absent.
- An external disk-write activity at September 17 15:56 UTC targeted the retained
  preparation OS disk. The disk still exists, Reserved/Succeeded, 64 GiB. No
  restart, modification or attribution was inferred from that event.
- The installed extension bundle remains
  `073232d2528ed59271ed33d5a43dba9e77235a4a5757e1555fefdee1b980e618`.
  Both [pinned runner/verifier archives](gremlin-target-preflight-20260917.md#pinned-local-artifacts)
  retain their recorded hashes. Neither was rebuilt or executed; the runner
  archive was subsequently uploaded as recorded below.

## Actual GUI steps

The installed VS Code opened a **new** guided workflow, without a project-folder
prompt or desktop CLI. Its existing Azure session loaded the authorized
subscription and listed the dedicated resource group. Cosmos discovery returned
the expected `afcosmosp120260907` candidate. Selected Japan East, zone 1,
B2s_v2 and the existing `vnet-af-vscode-p1/runner` subnet.

The initial paste did not populate the subnet field; preflight rejected the
empty value. After setting and visibly verifying the full ARM ID, normal release
preflight correctly refused the unpublished AGEFreighter 2.4.0 Linux release.
No deployment was submitted. Continue via the existing explicitly gated pinned
development-artifact path, not by weakening release verification.

New local draft: `4043e008-b86e-47b8-8722-1efe637ae12a`.
In its source form, reviewed migration name `p1-gremlin-20260918`, namespace
`migration`, host `afcosmosp120260907.documents.azure.com`, database `p1`,
container `graph-gremlin-p1-20260917`, Gremlin-over-NoSQL format,
`partitionKey`, and `score=float64,distance_km=float64`. No source assessment
has started. The accepted explicit-document workflow was not reused.

The initial native approval dialog requested **new** transfer account
`af4043e008b86e47b887221e`, Japan East / Standard LRS, with a Blob Data Contributor
grant to the signed-in user on that new account only. Anonymous/shared-key access
is disabled; the HTTPS endpoint is network-public, not a private endpoint. A
specific action-time approval was requested before creating the access grant.
At that initial checkpoint `storageDeployment` was absent. The later approved
deployment and access recovery are recorded below.

## Next gates

Storage, pinned upload, VM readiness and approved Cosmos Reader have passed as
recorded below. Correct and qualify complete Gremlin discovery without suppressing
its safety bounds, then explicitly retain/retry the failed inventory. Next, privately deploy/review the
target, resize the same runner, load, verify counts, and compare all 64 target
ranges with the Gremlin root. Include active-operation reload/no-replay evidence
for B10. Each approval remains bound to its actual artifact/resource/scope.
B05/B10 remain open; this checkpoint is not source assessment or migration PASS.

## Approved storage deployment — 21:50–21:54 UTC

The user approved the new account and scoped Blob access. The installed GUI
submitted deployment `af4043e008b86e47b887221e-transfer`, which ARM reports
Succeeded at `21:50:49Z`. The local workflow retains `submitted` until its
normal reconciliation and safety checks succeed; it has not been edited by hand.
The new container's ownership metadata matches the workflow and anonymous access
is None. The user-only assignment is `f1794e35-cebd-4298-9070-5b28908816de`.

Fresh ARM inspection found **publicNetworkAccess=Disabled**, although the reviewed
template requested Enabled. TLS 1.2, HTTPS-only, shared-key disabled and anonymous
access disabled remain the intended controls. A read-only authenticated Blob
listing failed with a network-rules error. Activity Log separately records a
successful `Microsoft.Authorization/policies/modify/action` on this exact account
at `21:50:49.4292566Z`; timing is recorded without asserting the unreturned policy
definition or modification payload.

No network setting or tag was changed in response. Requested action-time approval
for this exact trial account's official `SecurityControl=Ignore` tag and public
HTTPS enablement, preserving existing ownership tags, TLS and data authentication.
No exception for a different account is reused. Fixed runner upload, VM creation,
Cosmos Reader grant, assessment and target migration remain pending. Existing
sources, accepted graphs, preparation evidence and stopped compute are untouched.

## Approved account-only exception and pinned upload

The user explicitly authorized adding the tag and continuing. Applied
`SecurityControl=Ignore` to **only** `af4043e008b86e47b887221e`, preserving its
application/purpose/workflow ownership tags, and enabled public HTTPS access.
Readback confirmed Enabled, HTTPS-only, minimum TLS 1.2, anonymous access disabled
and shared keys disabled. An authenticated Blob listing succeeded. No source
firewall, Cosmos networking or other account was changed. The tag is not treated
as a guarantee against future policy changes; actual resource state was checked.

The installed source-assessment GUI reconciled transfer storage to `ready`.
Its pinned-development-artifact command selected the exact workflow and reviewed
the retained manifest, archive size, commit and SHA-256. The immutable upload
completed and the GUI displayed “Pinned development archive is prepared”.
The saved workflow records `developmentUpload.phase=ready`:

- Commit: `e70e02068c6865cd701e7ef99afb150dd64ca01f`.
- SHA-256: `1746ef42794468c90e034cbbe527c1489cb7f8df42eb7646d254929b9f0d4cd8`.
- Blob length: **37,124,976 bytes**, independently read back through authenticated
  Blob metadata; its SHA metadata matches the reviewed archive.
- ETag: `0x8DF1506C741116B`.

This proves an authenticated write and subsequent metadata read, not an independent
full-byte download or Linux execution. Reconnected to this draft in the installed
GUI and requested a fresh runner preview. The future VM's scoped Blob Reader and
unpublished executable remain separate action-time approval gates. No VM, Cosmos
Reader grant, assessment, target, or migration has yet been submitted.

Fresh GUI preflight passed at `2026-09-17T21:59:17.778Z`: exact VM
`af-4043e008b86e47b88722`, Japan East / zone 1 / B2s_v2, compute
**USD 0.109/hour** plus disk/network costs. Its identity would receive only Blob
Reader on this workflow's transfer container. Requested action-time confirmation
for that grant and execution of the pinned development binary. No deployment
approval button was pressed. At this rate even retaining this VM until the outer
deadline adds less than USD 6.3 compute (under 58 hours); disk/network remain
additional and covered by the trial reserve, not silently treated as free.
The 15-minute preview expiry is a preflight freshness limit, not a VM shutdown
timer. Refresh the preview and safety gates if it expires before approval.

## Approved runner deployment — 22:08 UTC

The user explicitly approved the reviewed VM, pinned development executable and
container-only Blob Reader. Before submission, the RG still had no lock; the
transfer account retained its exact workflow tags, public HTTPS enabled and both
anonymous/shared-key access disabled. Activity Log includes a signed-in user
storage write at `22:04:35Z` and policy audits; no other changes were inferred.
The preview remained within its 15-minute validity period and existing cost and
deadline limits remained unchanged.

The installed GUI's network/cost checkboxes and matching native confirmation were
accepted. Durable workflow state records `deployment-submitted` at
`2026-09-17T22:08:18.659Z`, with deployment name `af-4043e008b86e47b88722`.
ARM initially reports Running with no error; the GUI refresh retained this same
deployment rather than replaying it. Linux readiness is still pending. No source
Data Reader grant, assessment or target migration has been started.

ARM deployment completed successfully at `22:08:55.898809Z`; the installed GUI
reconciled it to `provisioned`. The actual NIC has no public IP and uses the
reviewed existing runner subnet. The new system identity is
`5abb7ef3-ee29-4bdc-abf5-dc667cc1d7e7`; its only returned Azure role assignment is
Storage Blob Data Reader on the exact workflow container.

The first GUI readiness check ran before bootstrap finished. Retained RunCommand
`af-2fcabb93-4578-442b-8919-f80c9dd55922` exited 127 at `22:09:32Z` because
`/usr/local/bin/agefreighter-tools` was not yet present. A separate read-only
diagnostic at `22:10:34Z` found cloud-init done without errors, both expected
binaries and `bootstrap.complete` present, root disk 4% used, about 288 MiB used
memory and zero swap. No installer, VM or source operation was restarted. After
reviewing that evidence, explicitly requested a fresh GUI readiness check; the
failed initial RunCommand remains retained.

The fresh GUI readiness command `af-4695a9bb-5d89-4019-9736-b4156d107ffc`
finished successfully. The GUI displays “Pinned Linux guest verified” and the
durable readiness evidence matches the exact approved commit, version and archive
SHA-256. Health: idle, disk **3.5095%**, swap **0**, OOM events **0**.
Boot ID: `df841056-5cdf-4cce-b528-24977fc5b793`.

Opened the source form and reviewed the next permission gate. Proposed Cosmos
assignment `5311b64f-1c2c-4f8f-b2f7-485b01e1bc44` remains **previewed only**.
Its role is Built-in Data Reader for the new VM identity on the **whole trial
Cosmos account**, not just the Gremlin container. Requested explicit action-time
approval for that broader read scope; no grant was submitted. The configured
assessment remains bound to `p1/graph-gremlin-p1-20260917`, with no source writes,
keys or network exposure. The VM remains running at the approved USD 0.109/hour
while awaiting this next gate; no new deadline or automatic shutdown is implied.

## Approved Reader and complete source inventory — 22:53–22:56 UTC

The user approved the account-scoped Reader. On resuming, the saved GUI workflow
already recorded the exact assignment submitted at `22:53:49.638Z`; no duplicate
PUT was performed. GUI reconciliation and independent ARM GET agree on assignment
`5311b64f-1c2c-4f8f-b2f7-485b01e1bc44`, VM identity
`5abb7ef3-ee29-4bdc-abf5-dc667cc1d7e7`, Built-in Data Reader and the reviewed
trial account. Cosmos remains network-private with local/key auth disabled;
the RG has no lock. The queried activity-log interval returned no additional
successful non-audit entries; this is a time-specific observation, not assurance
against ingestion lag. Budget/window and bounded compute reserve are unchanged.

Fresh GUI readiness at `22:54:59.463Z` preserved the same boot and pinned
installation: idle, disk 3.5095%, swap/OOM zero. Reviewed the exact Gremlin form
and explicitly approved complete inventory, bounded to **30 minutes / 4 GiB /
no swap**, reading only `p1/graph-gremlin-p1-20260917` without writes or target.
The GUI submitted operation `e2d85f21-96e0-40a0-a73c-b529e91016be` at
`22:56:05.313Z`, RunCommand `af-690e3787-26d6-4edd-896a-564e6e80510f`.
It reconciled to `accepted`; successful dispatch is not inventory completion.
Configuration SHA-256: `ccfef301d9431c039ce3cbe8cbe2e1e776e721be4f26d6c0009024f87a1f598c`;
guest configuration SHA-256: `a6d11000837a6f1cbe37ba38fa123024d2e84c436e61441950023e3217aa8048`.
The report and active-operation reload evidence remain pending.

## Failed inventory and bounded read-only diagnosis

The retained worker started at `22:56:11.564828809Z` and failed at
`22:56:12.727945990Z`, exit 1. GUI and the retained status RunCommand
`af-976bbb09-e466-4a4b-8d93-5214868dc782` agree on failure for the same operation,
boot and configuration. Guest `stderr.log` contains only:
`inventory: network inventory mapping resolution failed`.
`internal/app/inventory.go` deliberately removes the underlying resolution error;
the precise initial cause cannot be recovered from that message. Do not attribute
it conclusively to RBAC propagation or the discovery limit below. No retry or
active-operation reload was performed, and no target exists.

Read-only diagnostics on the same VM used its identity, TLS and the same source
container, without logging tokens or source values:

- One page of the generated vertex-label query returned HTTP 200, 100 rows,
  continuation present, **5.63 RU**. This proves current data-plane read access,
  not that it was already propagated at the earlier failing worker's instant.
- A separate **101-page / 60-second** bounded probe of that same query returned
  **10,100 rows**, only **2 distinct labels**, continuation still present,
  **524.61 RU**, in **3.904 seconds**. It stopped without draining the source.
- The reviewed GUI configuration has `maxDiscoveryDocuments=10000`. The current
  unsampled discovery query does not deduplicate on the server; its visitor counts
  every returned document before deduplicating labels locally. Thus that path
  necessarily exceeds the configured limit on this fixture, independently of
  whether it caused the first opaque failure. The configured maximum is not a
  sample that can safely be promoted to complete discovery.
- Existing local `TestInterpretGremlin*` tests pass, including deliberate rejection
  of repeated labels beyond the document cap. That is a guard regression result,
  not evidence that the current implementation supports this P1 size.

### Required correction and acceptance before another run

Preserve the failed operation and fix exact catalog discovery rather than raising
or removing its cap, accepting an incomplete label sample, or replacing the GUI
with manually prepared mappings. Prefer a bounded, fully drained distinct-label
and edge-endpoint catalog with pagination proved on the real service. Do not
assume plain DISTINCT or GROUP BY is continuation-safe: the
[official pagination documentation](https://learn.microsoft.com/ja-jp/cosmos-db/query/pagination)
requires ORDER BY with DISTINCT for continuation support. Review index/query
requirements without silently mutating source indexing or broadening access.
Tests must cover duplicate-heavy input above 10,000 records, late labels/endpoint
combinations, multiple pages, exhausted/unchanged tokens and cap/error handling.
Retain typed, non-sensitive resolution failure categories rather than returning
raw SDK errors with credentials or values. Any changed runner requires new pinned
artifact review; no alternative executable has been deployed.

Requested deallocation of **only** `af-4043e008b86e47b88722` after diagnostics to
avoid idle compute charges. Its OS disk, original failed readiness command, failed
inventory, read-only role and workflow evidence are preserved. No storage, source
document, graph or permission was deleted. Final ARM instance-view readback
confirmed `ProvisioningState/succeeded` and **`PowerState/deallocated`**.
B05 and active-operation B10 remain unqualified.

## Local discovery correction and review — September 18 JST

Implemented an exact catalog enumeration that requests a one-item page and
excludes previously observed entries with a parameterized `NOT ARRAY_CONTAINS`
filter. Vertex labels and complete `(label, startLabel, endLabel)` combinations
are enumerated separately. Query values are parameters, not interpolated source
text. Continuation is reset only when the exclusion predicate changes; empty
pages are drained, and completion requires an exhausted query. Existing
source-immutability requirements remain in force.

The earlier DISTINCT proposal was rejected during review: pinned `azcosmos`
1.5.0 `ContainerClient.NewQueryItemsPager` documents gateway cross-partition
support for projections/filtering only, not distributed sorting/aggregation.
The new approach needs no index, data, permission or network change. It bounds
returned discovery rows, total pages and catalog entries (including ignored
endpoint combinations). It does **not** establish a bound on server-side scanned
rows or request units; real-service latency/RU behavior remains to be measured
within the existing timeout/budget. The bounded profiling path is unchanged and
is not relabeled as complete inventory.

Review covered missed late labels, endpoint combinations sharing one edge label,
empty-page EOF confusion, continuation reuse after query changes, parameter
aliasing, repeated tokens, ignored-entry memory growth and secret-bearing SDK
errors. Added fixed resolution-error categories for limits, cancellation,
deadlines and recognized HTTP status classes; raw errors, response bodies,
endpoints and credentials are never included or exposed as wrapped causes.
The original opaque failure remains unattributed.

Local validation:

- Duplicate-heavy modeled source: 30,300 vertex and 30,300 edge records with
  late catalog entries behind 10,100 duplicates; all three labels and all three
  endpoint combinations discovered in eight requests including EOF queries.
- Empty-page draining, fresh predicate continuations, frozen parameters,
  row/page/catalog limits, invalid JSON, visitor/request failures, cancellation,
  token cycles and secret-canary redaction tests pass.
- `go test ./... -count=1` passes; focused Cosmos/app race tests pass.
- Extension typecheck, all **300** unit tests and compilation pass. No extension
  source change is required; its existing pinned idle-runner upgrade flow retains
  old binaries and failed-operation evidence.

During this September 17 23:20 UTC local-validation checkpoint, fresh ARM readback again showed the exact runner
`af-4043e008b86e47b88722` **deallocated**. This local correction has not restarted
it or changed any cloud resource. New pinned Linux artifact approval, real
service inventory, GUI migration, all 64 target digest ranges and active-operation
reload remain pending. B05/B10 are **not** qualified by these tests.

### Replacement candidate (local only, not yet approved or installed)

Built from the clean committed fix
`290c6efa3b3e08423d35bf004485dfd6e0f1fd98` using the existing local packaging script.
Both executables are Linux x86-64 ELF. Retained manifest:
`production-simulation/work/vscode-runner-build.gR5UPC/manifest.json`.

- Version: `2.4.0-dev.290c6efa3b3e`.
- Archive: `agefreighter-2.4.0-dev.290c6efa3b3e-linux-amd64.tar.gz`.
- Size: **37,132,538 bytes**.
- Independently rehashed SHA-256:
  `00b66081e7629ecd4b7ad378cc4b134987d0b892d91b751b789354158e63432e`.

Next approval is for upgrading only the existing idle runner with this candidate
through the installed GUI and explicitly retrying source inventory. Preserve the
old installation, workflow and failed operation. No new resource, target,
permission or network change is included. Proposed runner bound: two hours,
including the existing 30-minute inventory limit; compute at 0.109 USD/hour,
within the unchanged USD 800/September 20 deadline. Refresh cost/health/governance
and guest-readiness gates before mutations. No live inventory outcome is claimed.
