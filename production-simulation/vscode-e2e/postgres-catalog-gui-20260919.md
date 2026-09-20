# PostgreSQL catalog GUI qualification preparation

Latest checkpoint — September 20, approximately 02:08 UTC: approved extension
fix installed; same-workflow preview renewal, VM provisioning and pinned Linux
readiness pass in the operator GUI. **B04 remains partial**: the single catalog
attempt failed during source authentication (SQLSTATE 28P01), before schema
collection. Evidence is retained; no automatic retry. The approved
reader credential recovery now succeeded: corrected helper exit 0, committed
rotation, and a separate verified-TLS/read-only login passed. The named Keychain
item below is now the current PGFS reader password. Source Stopped and runner
deallocation are verified; guest evidence is preserved. B04 still requires a fresh, separately
reviewed catalog workflow; the earlier failure is not retried or converted to PASS.
Earlier checkpoints below are historical, including their not-installed claims.

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

### Transfer account created; networking gate — September 19, 09:58–10:01 UTC

The user approved the scoped account/role creation. On the next GUI observation,
the native dialog was already dismissed and the operation was `submitted`;
only status reconciliation was performed, without resubmission. ARM confirms
deployment `af66a26571953f42219659a4-transfer` **Succeeded** at
`2026-09-19T09:58:54.401392Z`. Account ownership tags match this workflow.
The exact account-scoped role assignment
`a0be79e6-522b-4a1b-ab36-117aaea8e2a2` grants Storage Blob Data Contributor to
the approved signed-in user; no wider role scope was returned.

The installed GUI now correctly reports
`ready — public network: Disabled (provisioning is not transfer readiness)`.
ARM confirms HTTPS-only, minimum TLS 1.2, anonymous blob access disabled and
shared-key access disabled. Public network access is **Disabled**, despite the
reviewed deployment requesting Enabled. The bounded activity reads returned
no matching events yet, so the actor/cause is not established by those reads.
No authenticated upload has passed and no Linux archive has been uploaded.

Requested explicit confirmation to apply `SecurityControl=Ignore` and enable
authenticated HTTPS public access **only on this new account**, keeping
anonymous/shared-key access disabled. Neither change has been applied at this
checkpoint. No VM or source server has been started. The previous 70 operator
files still match the retained aggregate SHA; only the new workflow changed.
The source form's database/user reset to defaults during refresh; re-enter
`p1source` / `agefreighter_reader` before a future approved source operation.

### Transfer ready and Linux archive uploaded — September 19, 11:51–11:56 UTC

After the user's instruction to continue in response to the exact-account
exception request, merged `SecurityControl=Ignore` only onto
`af66a26571953f42219659a4` and enabled its public network access. Existing
ownership tags were preserved. ARM confirms Enabled, HTTPS-only, TLS 1.2,
anonymous access false and shared-key access false. Authenticated Blob listing
and container metadata reads succeeded; the container matches this workflow
and does not allow anonymous access. The GUI refreshed to Enabled without
replaying storage deployment. No source or RG-wide networking setting changed.

Through the installed extension's pinned-development-artifact command, selected
this draft and the reviewed `vscode-runner-build.HwWiUz/manifest.json`. The local
archive was rehashed and matches the earlier SHA-256. The GUI successfully
uploaded the immutable archive and displayed that it was prepared. The retained
`developmentUpload.phase` is **ready**, and the remote Blob properties report
37,197,546 bytes with metadata SHA-256
`2321022975f85c73068a54fd21a287e2802d5bc33fa21eb9323c142dd7262ff6`.
This is upload evidence, **not guest installation or execution**.

Reconnected to the same draft through the installed picker and completed its
fresh prerequisite/what-if preview: Japan East, zone 1, `Standard_B2s_v2`,
USD 0.109/hour compute plus other charges, matching private runner subnet and
pinned artifact. The exact VM remains `af-66a26571953f42219659`. Its template
has no public IP, preserves the OS disk on VM deletion, and grants the VM
identity Blob Reader only on this workflow's container. The native final
creation dialog is open; requested action-time approval for that grant and
unpublished guest installation/execution. **No VM creation was submitted.**

The proposed session is at most two hours after start, stopped sooner on idle
input/approval wait, and always within the unchanged outer deadline. Before
accepting any delayed approval, refresh expired preview and time/governance
checks and establish the exact shutdown bound. The source remains Stopped
with public network access Disabled and its expected subnet/private DNS;
no source start or target creation occurred. Fresh RG lock query returned none.

### Expired-preview renewal defect and local fix — September 19, 13:11–13:15 UTC

At the user's next continuation, the retained preview had expired at 12:09:55Z.
The installed GUI displayed the stale-preview rejection; ARM independently
returned ResourceNotFound for the proposed VM. No deployment was submitted.
Refreshing the preview then incorrectly attempted to fetch the unpublished
2.4.0 release/checksums instead of retaining the already-uploaded development
archive, and failed closed again.

Root cause: the webview retained `draftId` only in phase `draft`, dropping it
on `previewed`. The controller and evidence-retention helper also rejected
renewal of an unsubmitted `previewed` record. Fixed all three layers to preserve
the same workflow/artifact for explicit renewal. Source and full placement must
still match, submitted/provisioned/failed/unknown records cannot be renewed,
the latest record is rechecked under the exclusive lock, and a changed pinned
artifact is rejected. The original preview expiry check is unchanged; new
preflight, pricing, what-if and user consent are still required. No operator
state file was edited to bypass a guard.

Validation: **430/430 unit tests**, typecheck/build, **13 isolated VS Code 1.105.0
Extension Host tests**, and VSIX packaging pass. New regressions cover renewing
an expired preview, preserving artifact/CA/file evidence, blocking submitted or
changed placement, retaining the ID in the actual view script, and clearing
deployment checkboxes before re-review. The original 70 operator files still
match their retained aggregate SHA. The signed-in VS Code extension is not yet
updated; installed-GUI renewal and B04 catalog qualification remain unproven.

New local VSIX SHA-256:
`ff1d1770c3738ba1ec8c892f954ebbaa9133fbd81d951218f6d93f6b48859f55`;
JavaScript SHA-256:
`9f3dd3114ba2734364f80bf152f82719a07e794d5b87365d095e9e4a105a1ced`.
This is an extension-only fix: the reviewed guest archive remains the exact
`d40d6cc` bytes already uploaded. No VM/source start, guest install or target
creation occurred. Storage remains authenticated-public HTTPS with anonymous
and shared-key access disabled. A 12:01 storage write and policy audit events
were observed; current ownership/security fields remain as approved, with no
reversal performed. RG locks remain absent. Refresh all time-dependent gates
again after installing the reviewed fix before any paid session.

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

### Fixed installed GUI, deployment and readiness — September 20, 00:48–00:59 UTC

Installed the user-approved extension fix from `d85965c`; VSIX and installed
JavaScript match the SHA-256 values recorded above. The user manually performed
Reload Window after initial remote input attempts did not take effect. The
actual installed GUI then renewed the expired preview for the same workflow
`66a26571-953f-4221-9659-a4b35460ffc4`, retaining the uploaded `d40d6cc` Linux
archive. Fresh preview and consent succeeded without release-download fallback.

Submitted the exact private discovery VM `af-66a26571953f42219659` once at
`2026-09-20T00:53:04.566Z`. ARM and installed-GUI reconciliation both reached
provisioned. Placement is Japan East / zone 1 / Standard_B2s_v2 in the existing
runner subnet, with no public IP or SSH ingress and retained OS disk. Its
identity has Blob Reader only on this workflow's transfer container. No target
database, source firewall change or migration was created.

The installed GUI dispatched and reconciled readiness operation
`5ca5a966-80b0-421b-967e-2cca68c83e91` successfully. Evidence at
`2026-09-20T00:56:35.910Z` reports boot
`85fb7508-aa1c-451c-9c21-733f30b9e312`, exact reviewed archive hash/version/commit,
and capability `postgresql-catalog-v1`. Guest is idle, disk 3.4877% used,
swap 0 and OOM events 0. No catalog worker or source read has run; readiness
does not establish catalog/recommendation qualification.

Entered the reviewed source host, database `p1source`, user
`agefreighter_reader` and schema `p1` in the actual editor. The native catalog
approval correctly lists those values and the exact new VM, with a two-minute
read-only metadata limit, no row values/counts and required TLS validation.
Action-time approval and private password entry remain pending. No credential
was read, copied or entered. Before continuing, reconcile that dialog and obtain
fresh guest readiness after restart; do not submit using stale boot evidence.

Safety/cost ledger: fresh RG locks were absent. The September MonthToDate RG
ActualCost query succeeded at approximately 00:52 UTC, returning
**USD 284.682696744039** (billing can lag), below the unchanged USD 800 ceiling.
The outer deadline remains `2026-09-20T07:14:35.311Z`. An enabled UTC 02:45
auto-shutdown schedule targets only the new VM, establishing this session's
earlier **11:45 JST** bound; it does not authorize any extension of that bound.
After reaching the approval/input wait, explicitly deallocated this exact VM
and verified `PowerState/deallocated` at approximately 00:58 UTC, preserving
all evidence and disk. Source `afpg-p1-source-20260907` was not started.

During fresh fleet inspection, retained completed target
`afpg-09bb0173608f4ab1bf62` was unexpectedly Ready. Verified its ownership,
finished migration/P1-pass record and retired VM before stopping it; ARM then
confirmed Stopped. The start cause is unproven, not attributed to an operator
or automatic restart. All other 17 Flexible Servers were already Stopped and
all 10 previous VMs deallocated. No resources, disks or data were deleted.
The 70 pre-existing operator files still match the retained aggregate SHA;
only the new workflow acquired deployment/readiness evidence.

### First catalog attempt: authentication failure — September 20, 01:00–01:09 UTC

The user continued the reviewed read-only trial. Fresh RG locks were absent,
the recent activity query showed the expected trial actions and policy audit,
and the established budget/session bounds were unchanged. Started only the
exact new runner and retained PostgreSQL source. The source's old September 16
expiry tag was observed, not rewritten; explicit subsequent authorization
extends the trial to September 20. Public access remained Disabled. Effective
parameters were `require_secure_transport=on`, minimum `TLSv1.2`, with no
pending restart; no TLS setting was changed.

Cancelled the old password prompt before submitting anything, then refreshed
Linux readiness in the installed GUI after restart. The new boot is
`7df7db0f-6e35-436a-947f-258bb5c921d7`; readiness at 01:01:46.413Z verified the
same pinned artifact, idle guest, 3.4886% disk, zero swap and OOM events.
Reopened the same reviewed catalog confirmation. The user approved in VS Code
and entered the source password privately. The local staging password-file
existence and mode 0600 had been checked without displaying its value; that
alone did not establish current validity for the reader account.

Catalog operation `6f425d11-fba7-4af1-a876-38374138e614` was submitted once at
`2026-09-20T01:04:29.448Z`, with configuration SHA-256
`d801f9817d9e01f6240ffa6869f700004125fb98376208d4d8989ad66b704cc3`.
Installed GUI reconciliation progressed from submitted to accepted to failed.
Sanitized read-only guest diagnostics confirm execution 01:04:32.458111284Z
through 01:04:34.812840036Z, exit 1, stage `connect PostgreSQL catalog`, SQLSTATE
**28P01**. No sealed catalog report exists. Retained stderr is 51 bytes, SHA-256
`ae75535c68de66ac0248073dec36b64861380174d4201f9ad28298d32f245c13`;
raw stderr and credentials were not exported. The operation directory retains
state, worker claim, job and stderr; the transient secrets file is absent.

This proves authentication rejection, not a catalog-query defect, changed
password, or user-input mistake. No row collection, mapping adoption, inventory,
target creation, migration or source-data modification occurred. Do not retry
this retained operation or erase its evidence. Establish the current credential
or obtain explicit credential-reset approval before a separately reviewed fresh
catalog workflow. Source stop was submitted after failure; runner deallocation
was verified after bounded diagnostics. No permissions/network exposure changed.
Final ARM reconciliation at approximately 01:12 UTC confirms the source is
Stopped as well. Both exact resources are now stopped with all evidence retained.

### Credential provenance correction and approved rotation — September 20

After explicit user approval to reset `agefreighter_reader`, inspection of the
original preparation invocation established the earlier file guidance was
incorrect: `work/az-pgfs-staging/postgres-password` supplied **adminPassword**
for `afsourceadmin`; **sourcePassword** for `agefreighter_reader` was supplied
from `work/az-pgvm-staging/postgres-password`. The operator was incorrectly
directed to the administrator file. No secret values were displayed in this
investigation. Neither historical file is overwritten or relabelled as the new
reader credential, and the administrator password is not reset.

The approved rotation is limited to the existing reader role on
`afpg-p1-source-20260907` / `p1source`. A create-only Keychain item was saved and
read back locally, using service/label
`agefreighter-afpg-p1-source-20260907-agefreighter_reader-20260920` and account
`agefreighter_reader`. This name explicitly identifies the source and role;
the generated 32-byte random password is not stored in a repository file,
command argument, chat or report. Both secret parameters travel only through
the protected Managed Run Command body on stdin. Raw Azure submission output
is suppressed to prevent accidental request disclosure.

The helper is scoped to the exact existing runner, runs at most 420 seconds,
checks disk/swap/OOM/idle state, and uses verified TLS. It changes only the
reader password in a bounded transaction; equality of `pg_roles` snapshots
guards all non-password role attributes. A separate reader connection must
verify the expected database/user, read-only default and active TLS before
the completion marker is emitted. No schema, grants, row data, network or
administrator credential changes are included. Script SHA-256:
`804b3bd6d285419174b00593e48ef6b5388d75963ab4d4747aa0c0a7ba6f51ce`.
Submission succeeded and execution ran `2026-09-20T01:18:21Z`–01:18:42Z,
then exited 3. Sanitized guest diagnostics identified `invalid command
\\getenv`: the installed PostgreSQL client does not implement that psql command.
The retained transaction log ends after BEGIN, SET, SET, SELECT 1, DO, with no
ALTER ROLE or COMMIT. Thus no password rotation committed; no completion marker
or reader login verification exists. Stderr SHA-256:
`fd53df161d2ea689a6f33302c2a7f7283a6a71e9c7c7715dfbaa1e89422f7396`.
The original catalog operation and helper evidence remain intact.

Corrected the private helper to stream a strictly validated 64-character hex
password through psql stdin, not arguments/files or unsupported client commands.
The revised script passes shell syntax validation; Swift submitter compiles.
Revised script SHA-256:
`5d68e86955300e96ddbd1f7a2f80f8f7fea20c57579d7a66f034a91198ed4db8`.
It reuses the exact pending Keychain item without generating/replacing another
secret. The revised executable triggered macOS Keychain authorization, which
requires the user's local action; no bypass was attempted. At the idle wait,
terminated only that exact local helper (exit 143) before any Azure submission.
ARM returned ResourceNotFound for the proposed r2 command. Do not describe
the saved Keychain password as applied or ask the user to use it for login yet.

Deallocated only the trial VM and requested source stop to avoid paid idle
waiting. Removing the failed helper's protected Managed Run Command transport
was attempted, but Azure rejected deletion because the VM was already stopped.
That protected control record therefore remains; guest evidence is retained.
Do not restart solely for cleanup. On the next authorized session, reconcile
and remove that exact transport after preserving its terminal metadata, then
complete the separately named r2 rotation. Obtain Mac Keychain authorization
before starting paid resources where possible. The same USD 800, 02:45 UTC
session bound and 07:14:35.311 UTC outer bound remain; no automatic extension.

A replacement local `submit-r2-gated` process now requests only Keychain access
while cloud resources remain off. After that authorization it waits for explicit
`DISPATCH` on stdin; it cannot submit an Azure request without that gate and
rejects dispatch at/after 02:35 UTC. Recheck resource readiness, governance and
time before releasing the gate. Do not start a second helper or expose its
in-memory credential. The requested local authorization is still pending.
Final ARM reconciliation confirms both VM deallocated and source Stopped.

### Reader rotation complete — September 20, 02:04–02:09 UTC

The previous gated helper had exited without obtaining Keychain data or
submitting Azure work. At the user's continuation, reopened the same helper;
the user explicitly confirmed the Mac permission. Credential readback then
succeeded without printing its value. Fresh reads confirmed both resources
stopped, source public access Disabled, no RG locks, and only expected prior
stop/deallocate actions in the bounded activity query. Cost refresh returned
429; latest delayed RG total remains USD 284.682696744039, not a new total.
Started only the exact runner and source, within unchanged budget/time bounds.

After source Ready and VM running, removed the old protected reset command and
verified its absence. The corrected r2 command was absent before explicit
dispatch. It ran **02:07:31Z–02:07:32Z, Succeeded, exit 0**, and emitted the
completion marker. Independent read-only guest evidence retrieval confirmed
COMMIT, both reader checks true (expected role/database/read-only default and
active TLS), empty stderr and empty kernel-OOM evidence. The same pending
Keychain value is now **applied and login-verified**. The administrator password,
role attributes, grants, source data and network settings were not changed.

Use Keychain service/label
`agefreighter-afpg-p1-source-20260907-agefreighter_reader-20260920`, account
`agefreighter_reader`, for this PGFS source from now on. Neither historical
staging password file was overwritten; the old PGVM reader file must no longer
be used as this PGFS reader credential. No other source credential was rotated.

Both exact temporary protected reset command resources were removed while the
VM was running, and a fresh command list proves their absence. This removes
the reset control/secret transport only; both failed and successful guest
evidence, disks, and the original failed catalog remain. Source stop and runner
deallocation were submitted after evidence capture. See the redacted
[rotation receipt](evidence/pgfs-reader-rotation-20260920.json).
Credential recovery is not B04 catalog/import/adoption qualification; no catalog
operation was retried, and a fresh reviewed workflow is still required.
Final reconciliation confirms source Stopped and runner deallocated before
the 02:45 UTC session bound. No cloud compute remains active for this trial.

### Fresh catalog draft — September 20, approximately 02:20 UTC

At the user's continuation, the installed GUI created fresh local workflow
`ae952310-5eba-42b6-9fe3-9db00e93cdac`. It targets the same retained PostgreSQL
source, authorized subscription/group, Japan East zone 1 and existing runner
subnet, with B2s_v2 selected. The source form specifies `p1source`, reader
`agefreighter_reader` and schema `p1`; no password was entered or exported.
The prior failed workflow and operation remain untouched.

The user approved the native confirmation for new transfer account
`afae9523105eba42b69fe39d` and Storage Blob Data Contributor for the signed-in
user on that account only. Deployment succeeded at 02:21:09.189586 UTC;
read-only ARM confirmed the exact account-scoped role
`c092fa18-772e-4374-99b7-20df8d75746a`. Anonymous access and shared keys remain
disabled. The installed GUI reconciled `ready — public network: Disabled`
and explicitly distinguished provisioning from transfer readiness. No archive
upload, VM creation/start, source start or catalog submission occurred.
Requested action-time approval for the established `SecurityControl=Ignore`
exception and authenticated public HTTPS access on this new account only;
that approval is pending. No policy tag or network setting was changed.
The source form's database/user reverted to defaults on storage reconciliation;
re-entered and visually verified `p1source` / `agefreighter_reader`. Verify
the exact reviewed connection again before any read; no password was entered.
Read-only ARM reconciliation confirms the old runner deallocated, source Stopped
with public access Disabled, and no RG locks. The USD 800 ceiling, latest delayed
cost and 02:45 UTC session shutdown bound are unchanged. Recheck remaining time
and governance before any subsequent paid run; do not extend the session.
