# PostgreSQL catalog GUI qualification preparation

September 19, 2026, approximately 09:00–09:03 UTC. **B04 remains partial**.
This checkpoint qualifies installation and the retained-workflow negative gate,
not a remote catalog read, recommendation adoption, inventory or migration.

## Reviewed artifacts

Built from clean commit `d40d6ccc9a4ddf6e2ca626392cd7bf83140ed6c7` using the
existing development-runner builder and VSIX packaging workflow. All 428
extension unit tests, typechecking and compilation passed again.

- VSIX `agefreighter-2.4.0.vsix`, SHA-256:
  `8d70b7b76137a5d168e7d541c34e4bb3d12d8c4c8508d2d00998c87027480539`.
- Extension JavaScript SHA-256:
  `548a0cf23fa6a72216b9e56a2b5fa17376aff5a1c5bda69354f4ebe970b3250e`.
- Linux AMD64 archive version `2.4.0-dev.d40d6ccc9a4d`, 37,197,546 bytes,
  SHA-256 `2321022975f85c73068a54fd21a287e2802d5bc33fa21eb9323c142dd7262ff6`.
  Its local manifest remains under
  `production-simulation/work/vscode-runner-build.HwWiUz/manifest.json`.
  This archive has **not** been uploaded, installed or executed on Azure.

The user specifically approved installing this unpublished VSIX on MacStudio.
Installation succeeded and the installed JavaScript SHA matches the build.
Actual installed VS Code is **1.138.0, arm64**, commit
`7debcd0e2acdea1c52de81bf9ee1620444407dda`; older 1.136.1 results remain historical.
No Marketplace publication occurred.

## Installed GUI observations

Launched the operator's VS Code and opened New Guided Migration. Reconnected
through the actual picker to retained AZ-PGFS workflow
`29558917-403e-4a76-aaa0-de07122ea9c6`, without replaying any operation.
Opened Configure source & assessment and checked its accessibility tree and
screenshot:

- The PostgreSQL schema discovery section is visible, including explicit schemas,
  discovery, status, report transfer and selected adoption controls.
- All four catalog actions are disabled for this old, already-assessed workflow.
  Its pinned runner predates `postgresql-catalog-v1`; the UI explains that a
  matching reviewed Linux artifact is required. Both the missing capability and
  existing assessment apply here, so this is not an isolated test of either gate.
- The saved connection, nine vertex and nine edge mappings, retained inventory
  operation and imported report remain visible. Nothing was edited or submitted.
- The 70-file operator store aggregate filename/content SHA-256 is unchanged:
  `fcc85c6021b1191d9207a9fe93eb661cd1665edebbf83e5d33a4602d87f248ef`.

No passwords were read or entered. No worker, inventory, target deployment,
migration, source write or report export was dispatched.

## Read-only Azure preflight

Scope: existing subscription `67c417f3-5a13-446c-afb9-40cd87f2fdb7`, resource
group `rg-af-vscode-p1-20260905-a`. Fresh reads show all **10 retained VMs
deallocated** and all **18 Flexible Servers Stopped**; no RG locks were returned.
No resource was started, stopped, resized, tagged or otherwise changed here.

The cost query for this RG, September 5–20, returned **HTTP 429**. It did not
produce a new cost total. Historical USD 220.808334109599 and USD 650 planning
reserve are not current billing. The USD 800 ceiling and outer deadline
`2026-09-20T07:14:35.311Z` remain unchanged; the previous Gremlin 06:00 UTC
September 18 session deadline is expired and is not authority for a new session.

The 24-hour activity query (bounded to 200 events) returned external writes,
including failed PostgreSQL TLS-configuration writes around 08:34 UTC September
19, successful writes to the two on-premises-simulation Neo4j VMs, and a write to
the retained Gremlin runner's OS disk on September 18. These observations do not
prove their initiator, intent or effective changes. No governance setting was
reversed. A read of the PostgreSQL source's `require_secure_transport` parameter
returned `ServerStoppedError`; the source was **not** restarted just to inspect it.

## Next bounded live trial

Before any paid session, recheck cost/time reserve, exact resource ownership and
the effective relevant governance changes. Define a fresh catalog workflow and
its exact private runner/storage scope, approve the pinned Linux artifact, and
set a short explicit shutdown bound within the outer deadline. Preserve all
accepted workflows; the old AZ-PGFS workflow cannot be repurposed as a fresh
catalog operation. Start only the selected PostgreSQL source and trial runner;
no target is needed for the catalog-only phase.

Then use the installed GUI for explicit `p1` schema discovery, operation/status
reconciliation across reconnect, sealed report import, selection/adoption and
manual editing. Source credentials must be entered privately. Complete inventory
is a separate approved read and remains necessary before any sizing claim.
No remote/catalog/adoption PASS is inferred from this installation checkpoint.

## Fresh local draft and budget gate — September 19 follow-up

Created a separate **local-only** workflow through the installed GUI:
`66a26571-953f-4221-9659-a4b35460ffc4`. Placement was explicitly reselected as
Japan East / zone 1 / `Standard_B2s_v2`, in the existing trial group and runner
subnet, with Azure source `afpg-p1-source-20260907`. The proposed VM identity is
`af-66a26571953f42219659`; **that VM has not been created**. The saved draft has
no artifact, storage deployment, assessment, catalog operation or target.

The open source form contains `pg-catalog-p1-20260919`, database `p1source`,
read-only user `agefreighter_reader`, the source's verified-TLS hostname and
explicit schema `p1`. These connection fields are currently unsaved webview
inputs, not a persisted reviewed source mapping; do not assume they survive
closing or reloading the window. No password was entered and no source read
was submitted. Excluding the new draft, all original **70 files** still match
`fcc85c6021b1191d9207a9fe93eb661cd1665edebbf83e5d33a4602d87f248ef`.

Read-only follow-up narrowed the source's two failed configuration changes
(`require_secure_transport`, `ssl_min_protocol_version`) to `ServerIsBusy`
under `ResourceOperationFailure`. This does not establish the attempted values
or the effective current parameters. ARM still reports Stopped, public access
Disabled, and the expected delegated subnet/private DNS. Its retained expiry
tag still says September 16; it has not been rewritten here. Reconcile that tag
with the actual approved outer window before starting a later paid session.

Both the subscription-filtered cost query and the separate resource-group
MonthToDate query returned **429**. No current billed total is available. Asked
the user to provide the portal's September RG cumulative cost and its update
date before any new resource creation/start. This is a budget-check gate, not
authorization to exceed USD 800, extend September 20, use stale cost as current,
or override governance. No live experiment is running while waiting.

### Current portal cost supplied — September 19, approximately 09:43 UTC

The user reports the September cumulative cost as **USD 265.69**, with the
portal updated currently. This clears the missing-current-cost gate for the
bounded trial; it is user-supplied portal evidence, not a successful Cost
Management API refresh. The USD 800 ceiling, conservative USD 650 overall
planning reserve (not an additional charge), and September 20 07:14:35.311 UTC
outer deadline remain unchanged. Actual current billed headroom is USD 534.31
before delayed/unbilled charges. A separate short shutdown bound is still
required before starting compute.

The installed GUI now displays the exact new-account confirmation for
`af66a26571953f42219659a4`: Japan East, Standard LRS, signed-in-user Storage
Blob Data Contributor scoped only to this new account. Its HTTPS endpoint is
network-public; anonymous access and shared keys remain disabled. No source
firewall is changed. The agent has requested action-time confirmation for this
access grant and has **not** accepted the dialog. The fresh RG lock query
returned no locks. No storage, VM, source start, catalog read or Linux upload
has occurred at this checkpoint.

### P1 schema expectations and qualification limits

Inspection of the retained fixture preparation script
`prepare-p1-postgresql-flexible-server.sh` shows 18 tables with a `source_key`
primary key, plus unique text graph IDs, **no declared foreign keys**. This is
an expectation from the preparation code, not a fresh live catalog observation.
Consequently the metadata engine should not invent the nine graph relationships
from table or column names. It can propose primary-key vertices even for tables
used as edge tables by the accepted manual graph mapping.

The live trial must distinguish that relational recommendation from the accepted
P1 graph model. Select only intended vertex candidates; explicitly review/edit
their labels, stable graph IDs and property projections, and retain manual edge
mappings/endpoints. Do not adopt every table and then claim the original P1
graph was preserved. Compare the resulting reviewed configuration with the
accepted 18-map fixture before full inventory. Foreign-key recommendation
coverage remains in the earlier local dedicated fixture; this unmodified P1
source cannot establish a live FK-adoption PASS. Do not add constraints or change
the accepted source merely to obtain such a result.
