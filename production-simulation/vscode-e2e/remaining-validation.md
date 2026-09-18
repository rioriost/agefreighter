# Remaining guided migration qualification

Updated: 2026-09-18 JST. Status: **running; not release-qualified**.

September 18 next local follow-up: added explicit single-readiness-control removal
admission, archive-before-intent-before-DELETE ordering and GET-only recovery.
Only already-deallocated VMs and fresh successful ARM evidence are admitted;
Pending/Updating blocks. This code is **not installed or live-qualified** and no
Azure resource was removed/restarted. B08/B09 live lifecycle and B10 real forced
Extension Host crash remain open. See the stage 2 limits in the
[lifecycle record](command-receipt-lifecycle-20260918.md).

September 18 local follow-up: successful readiness controls now retain sealed
receipts, with a native local-only archive action and negative tests. This is
stage 1 of command lifecycle work, **not slot reclamation or live qualification**.
No cloud operation or installed extension change was performed. B08/B09's
archive-before-removal lifecycle gate remains open, as does B10 forced Extension
Host crash. See [scope and remaining stages](command-receipt-lifecycle-20260918.md).

September 18 05:11 UTC follow-up: **B05 Gremlin-shaped NoSQL full PASS** in the
installed GUI and independent Mac validation: 5.6M records / 18 labels / all 64
ranges and canonical root agree. Actual active-verifier Reload Window/reconnect
preserved the same operation through completion/export/import without replay.
B10 forced Extension Host crash remains separate/open. Exactly three historical
readiness control records were specifically approved, archived/pushed then removed;
all guest evidence/data/disks remain. At **05:13:54 UTC**, exact VM **deallocated**
and target **Stopped** are verified; scoped safety monitor paused.
[Full receipt](evidence/gremlin-full-pass-20260918.json).

Historical Gremlin checkpoint, September 18 03:32 UTC: installed-GUI migration
and complete counts **PASS** for 5.6M rows / 18 labels, zero rejects. Full
64-range Gremlin canonical digest is not run; see the latest receipt-cap blocker
above. B10 active migration Reload Window/reconnect preserved the
same job through successful completion without replay; active verification
reload and forced Extension Host crash remain open. The idle exact VM/target
are verified deallocated / Stopped by 03:34:36 UTC with evidence retained;
the scoped monitor is paused. The approved exact VM/target
session ends at **06:00 UTC / 15:00 JST September 18**, earlier than the outer
deadline. Continuation requires the verifier approval and fresh health/bounds. See
[execution evidence](gremlin-gui-execution-20260918.md).

The nine base P1 routes are [qualified](progress.md). This ledger covers the
additional requirements in [the original plan](plan.md#branch-coverage-beyond-the-base-paths).
It does not reopen or replace their evidence. Unit tests, actual CLI contracts,
isolated Extension Host tests and live installed-GUI/Azure tests are different
evidence levels. A test of a simulated failure is not a live recovery pass.

## Authorization and preservation

Use only the already authorized dedicated trial environment. Current renewed
ceiling: USD 800 (no additional budget); deadline extended by the user for
96 hours from the previous deadline to `2026-09-20T07:14:35.311Z`
(September 20, 16:14 JST). The extended-retention planning reserve is USD 600;
current actual billing is unavailable (HTTP 429). See the latest
[execution record](recovery-execution-20260915.md) for inventory and assumptions.
Recheck conservative remaining cost, ownership,
health, governance, quota and time before any cloud mutation. The nine accepted
targets and failure evidence must remain intact. Recovery trials require a
separate job/graph; do not fault or overwrite a qualified graph. As observed
after base-route completion, all 17 VMs were deallocated and all 13 Flexible
Servers stopped. Local regression work has not restarted them. Storage and
Cosmos charges continue; stopped PostgreSQL servers eventually auto-start.

The September 16 quota blocker is **resolved by user-authorized retirement**,
not a quota increase. Preflight originally refused `cores` 100/101 before any
target creation. Thirteen retired runner VMs were then deleted with their OS
disks, NICs and archived evidence preserved; six reusable VMs remain stopped.
Regional usage is now 48/101 and DSv5 44/100. No increase request was submitted
or is currently needed. See [retirement evidence](vm-retirement-20260916.md).
The current recovery inventory and 18 verified transfers remain reusable,
subject to fresh guest checks. Target deployment/recovery still need execution;
this does not close B02/B09 or the recovery qualification.

After the user requested continuation, the current runner was restarted and
fresh GUI readiness passed. The exact same private-target preflight now passes
with sufficient quota and USD 0.736/hour combined compute. The native final
target/SecretStorage confirmation was approved by the user. After refreshing
guest health, the installed GUI submitted the exact reviewed target; ARM
reports successful completion. AGE preload/restart and the same-VM D4s_v5
resize are complete, with fresh post-resize health at `2026-09-15T19:59:47Z`.
The installed GUI started the CSV recovery load, retained a real SIGTERM at
1,465,000 rows and a loader reboot at 3,405,000 rows, and explicitly resumed
the same job after each fault. The final continuation and complete counts
passed in the installed GUI (1.6M vertices, 4M edges, no rejects). Full
64-range canonical verification also **PASS**: all properties, identities and
endpoints agree, with independent local recomputation of both roots. See
[CSV recovery r2 evidence](evidence/csv-recovery-r2-p1-pass-20260916.json).
This qualifies the CSV process/reboot recovery path, not network-source recovery.
See the execution record for immutable operation and evidence identities.
No other surviving VM or retained target was started.

The September 16 independent Neo4j network attempt completed normal migration
and installed-GUI counts verification (5.6M records, zero rejects), but its
fault watcher failed closed with `ValueError` before injection. No network
rule or applied-fault evidence exists. Python 3.10 fractional-timestamp parsing
was reproduced and fixed (23 local tests plus guest precision probes pass).
The completed graph and original watcher evidence are preserved; B11 remains
partial and requires a fresh, separately bound fault trial, explicit same-job
recovery and canonical verification. See [network execution record](network-recovery-20260916.md).

R2 subsequently reached private-target readiness but its initial load was
rejected by Neo4j authentication before graph/metadata creation. The installed
GUI reconciled failure and a read-only diagnostic proved the target empty.
No network fault was injected; the watcher is stopped and evidence preserved.
The evidence-bound retry subsequently authenticated, received the actual
one-job network fault at 1,405,000 rows, restored connectivity and explicitly
resumed the same job from its final 1,415,000-row checkpoint. The continuation
finished and installed-GUI complete counts passed with zero rejects. Following
the user's exact-verifier approval, full 64-range canonical verification also
passed at 12:47:13 UTC and was imported through the installed GUI. Independent
local recomputation of both roots and every leaf agrees with the frozen P1
fixture. B11 now passes for its defined CSV process/reboot and Neo4j network
fault scenarios; other fault timings/sources are not implied. All initial
authentication/watcher failures are retained. See
[network recovery evidence](evidence/network-r2-p1-pass-20260916.json).

## Branch-to-evidence ledger

"Not-run" means additional live qualification is not established by this audit,
not that no related unit tests exist. Reconcile older route artifacts where
they prove a precise choice before scheduling redundant infrastructure work.

| ID | Required branch | Existing evidence / current limitation | Remaining acceptance | State |
|---|---|---|---|---|
| B01 | Default/separate migration RG; independent network RG | Retained workflow audit binds all accepted base paths to same source/runner/network RG | Separate migration/network RG GUI choices remain unrepresented | partial |
| B02 | Region/zone defaults, overrides, unknown zone; invalid region/SKU/subnet/quota | All retained paths bind Japan East/zone 1; placement/preflight tests and missing SKU/quota resize rejection pass | Defaults versus overrides, unknown zone and invalid choices still need precise GUI/no-write evidence | partial |
| B03 | Private Azure and IP-only discovery | Nine base routes cover private Azure and IP-only on-premises; signed-in location GUI audit passes; endpoint-only preflight request traces omit source ARM; equivalent other-cloud configurations pass real CLI validation | Other-cloud choice has selection/local-contract evidence, not an additional live end-to-end migration | partial |
| B04 | Neo4j versions; PostgreSQL recommendations/review; CSV typed mapping | Relevant base routes and full P1 canonical results pass | Map recommendation acceptance/edit branches and CSV choices to retained GUI evidence | running |
| B05 | Cosmos explicit and Gremlin documents | Explicit-document base P1 passed; Gremlin-shaped NoSQL installed-GUI inventory, private target/resize, migration and complete counts pass; independent full digest generated 05:08:24 UTC and GUI-imported, all 5.6M records / 18 labels / 64 ranges and root match, including typed properties, composite IDs and endpoints | Complete for the defined document-format branch; not native Gremlin API qualification | pass |
| B06 | Supported Cosmos authentication and RBAC propagation | Fixed managed-identity GUI scope documented; existing live account-scoped Reader/security rechecked; trust, approval binding and VM principal guards corrected with simulated denial/no-retry coverage; corrected candidate installed and wording verified in signed-in GUI | Live trust/principal-change and data-plane denial/propagation evidence still missing; do not claim other CLI modes GUI-tested | partial |
| B07 | Same-VM resize; active-job/incompatible resize denied | Live identity seal match on three surviving resized VMs; installed-GUI completed-migration refusal; 18 added resize rejection regressions | Active-job and incompatible-layout live GUI denial remain untested; do not mutate qualified resources to fabricate them | partial |
| B08 | CSV multi-file/reconcile, changed/hash mismatch, transfer/folder cancellation | Base CSV P1, picker/transfer Cancel, changed-manifest refusal and full retry/readback pass; Linux corruption/receipt/no-replay checks pass; installed-GUI committed response loss reconciles with HEAD only, unchanged ETag and full readback; normal candidate restored/reconnected | Defined cases complete; injected client response loss is not an actual Azure outage or a new graph migration | pass |
| B09 | Approval cancellation, expired preview, duplicate windows, lost ARM reply, bootstrap/artifact/quota failure | Approval inventory recorded; Cosmos cross-window/trust guards corrected and tested; controller tests cover stale previews, locks, persist-before-PUT and GET-only reconciliation; native resize, checkpoint inspection, target diagnosis, empty-target archival and report-transfer Cancel preserve prior records | Other approval surfaces and unrepresented faults still require bounded GUI/no-write evidence | partial |
| B10 | Close/reload during assessment/load/verification; no replay | Persisted-state/process-exit tests pass; installed panels closed/reopened; actual active Gremlin inventory, migration and full verifier Reload Window/reconnect preserve exact operations through successful report import without replay; verifier full canonical GUI PASS and independent Mac recomputation agree | Forced Extension Host crash remains distinct and unqualified | partial |
| B11 | Loader/network interruption; explicit same-job recovery | CSV r2 actual SIGTERM + VM reboot; Neo4j network r2 actual connection fault; explicit same-job GUI resumes, complete counts and all 64 canonical ranges PASS in both trials | Complete for these defined faults; not every timing or source | pass |
| B12 | Invalid verification must never be PASS | Counts/isolated-host negative panels pass; 20 production P1 controller tests cover full-digest identity/coverage/forgery, transfer failures, retained identity changes and corrupted reopen; signed-in installed candidate refuses retained P1 failure without retry/new PASS | Other invalid-import/transfer cases still lack installed-GUI evidence; synthetic reports and inert adapters are not live GUI/Azure fault evidence | partial |

## First local regression batch

September 17 07:42–07:50 UTC: [Gremlin live preparation](gremlin-live-preparation-20260917.md)
rechecked stopped compute/private Cosmos and all 18 fixture hashes, then packaged
the isolated source capsule. Billing refresh returned 429; no cloud mutation
was made. After explicit approval, installation/reload and actual Gremlin field
display passed with the existing Azure session; no source settings were saved.
All 66 saved JSON files remain unchanged. B05/B10 are still open.

September 17 06:50–07:12 UTC: [Gremlin target/profile preflight](gremlin-target-preflight-20260917.md)
adds the read-only target verifier and binds the source, artifact, persisted
operation and imported report to the partition-preserving profile. A 560-record
contract passed on actual AGE 1.7/PostgreSQL 18.1 with corruption negatives;
full Go checks, 300 extension unit tests and ten CLI contracts pass. This is
not a migration or live GUI qualification. Local test databases are retained
in stopped containers; cloud resources and all 66 saved reports are unchanged.

September 17 06:31–06:47 UTC: [Gremlin offline P1 preparation](gremlin-offline-p1-20260917.md)
completed the create-only export and independently derived partition-preserving
oracle. All 5.6M records and 64 ranges agree through the production Cosmos
decoder, including a read-only rerun simulating integral-float spelling loss.
The original raw-ID CSV root is unchanged. Tiny corruption negatives, full Go
tests and simulation race/vet checks pass. No cloud/GUI mutation or new installed
candidate; nine VMs and 17 Flexible Servers remain stopped. B05 remains not-run
for the live Gremlin path; target composite-identity verification is next.

September 17 06:12–06:24 UTC: [Gremlin type-preservation prerequisite](gremlin-types-preflight-20260917.md)
reproduced integral-float inference loss and added explicit, bounded Gremlin
types in the CLI and source form. Full Go tests, Cosmos/config race tests,
295 extension unit tests and ten actual CLI contracts pass. No new candidate
was installed or deployed. The existing signed-in GUI also cancelled cost
renewal and refused new-load/resume requests on the accepted retained Cosmos
workflow; all 66 saved JSON files are unchanged, all compute remains stopped.
Gremlin's partition-preserving P1 oracle and live trial remain open.

September 17 06:01–06:10 UTC: the [installed-candidate GUI follow-up](cosmos-guard-gui-20260917.md)
completed the authorized installation/reload, corrected Cosmos wording check,
four native execution cancellations and retained P1 failure refusal. All 66
saved JSON files remained byte-identical; no role, compute or guest mutation.
Nine VMs are deallocated and 17 Flexible Servers stopped. B06/B09/B12 remain
partial, and the terminal-state reload does not close active-operation B10.

September 17 05:44–05:54 UTC: [Cosmos access and approval audit](cosmos-access-approval-audit-20260917.md)
reproduced and fixed missing trust, reviewed-grant binding and live VM-principal
guards. Nineteen added tests bring the suite to 292/292 PASS; build/package PASS.
Existing Cosmos scope/security were read-only rechecked. Candidate installation
approval requested; no Azure mutation or new live data-plane/GUI pass claimed.

September 17 05:34–05:41 UTC: [placement/mapping/resize audit](placement-resize-audit-20260917.md)
bound existing coverage to exact retained records, verified unchanged identities
of three surviving resized VMs through live ARM reads, and exercised installed
GUI completed-migration resize refusal and native resize approval cancellation.
No compute start or cloud write; prior 64 artifacts unchanged. Eighteen new
resize regressions bring the full unit suite to 273/273 PASS. Missing live
placement/active-job/layout branches remain open; these are partial cases.

September 17 follow-up: [remaining-work sequence and B12 controller results](remaining-next-batch-20260917.md).
Twenty additional tests exercise the actual full-P1 import controller with
production download/hash/canonical checks and inert external adapters. Invalid
evidence never creates a success panel or new PASS state; failed transfers do
not retain invalid bytes or retry automatically. Installed candidate and Azure
resources are unchanged. Signed-in interaction remains open in B12.

September 17 05:10–05:14 UTC: user-approved temporary installed extension
completed the [GUI lost-acknowledgement trial](csv-lost-ack-20260917.md).
Real commit 201, visible uncertain error and saved `prepared` state were
followed by explicit GUI retry: one HEAD 200, zero PUTs, same ETag and
`uploaded` state. Independent full readback matched before/after retry.
The normal candidate was restored and reloaded; GUI reconnection retained the
result. Prior four Blob ETags and 64 other saved artifacts were unchanged.
All 9 retained VMs and 17 Flexible Servers are stopped. B08's defined cases
pass; the other ledger cases and overall release qualification remain open.

September 17 04:00 UTC: the [lost-acknowledgement transport trial](csv-lost-ack-20260917.md)
used the actual production uploader against the existing isolated Azure account.
After an actual commit 201, the injected adapter dropped only its response.
An explicit second invocation issued HEAD only, no PUT, and independently
matched the complete Blob with unchanged ETag. GUI state and compute are
unchanged. The later approved installed-GUI result above closes the remaining
GUI interaction case; this earlier transport result alone did not.

September 17: the separate [Linux CSV negative trial](csv-guest-negative-20260917.md)
passes live same-size hash-mismatch rejection, absence of final file/seal,
capability removal, positive full-hash control, mapping receipt gate and no
implicit retry. The new VM is deallocated and all evidence retained; no target,
assessment or migration was created. The obsolete CSV-upload warning was
corrected in source; typecheck and all 235 unit tests pass. The installed
candidate was not changed during that Linux trial. The later lost-acknowledgement
trial and normal-candidate restoration are recorded above.

CSV upload cancellation is now implemented and locally regression-tested:
**235/235 unit tests**, typecheck/build and **13/13 isolated VS Code 1.138.0 host
tests** pass. The installed signed-in candidate subsequently passed real Cancel,
changed-manifest refusal and explicit same-destination retry. All five blobs
(1,840,125,623 bytes) were independently streamed and hashed; prior blob ETags
and all 64 earlier JSON artifacts are unchanged. No assessment/migration started.
That September 16 transfer-only trial did not establish guest receipt/hash
behavior; the September 17 trial above now supplies it. The separate September 17
GUI trial closes lost committed-upload acknowledgement. See [B08 transfer record](csv-interruption-20260916.md).

Latest non-mutating GUI audit: [source locations and CSV cancellation](branch-gui-audit-20260916.md).
All eight allowed source/location selections were inspected in signed-in VS Code
1.138.0. Neo4j/PostgreSQL endpoint-only selections hide source ARM discovery;
Cosmos is Azure-only and CSV local-only. Four new preflight request-trace tests
exclude source ARM reads, two prove equivalent endpoint-only configurations,
and one checks new-wizard CSV cancellation. All 227 unit tests and nine real
CLI-validator contracts pass. Three native file-dialog cancellations preserve
all saved workflow/report bytes. This is not an additional cloud migration,
interrupted-upload qualification or complete B03/B08 acceptance.

September 16 network-trial preparation also reproduced and fixed a stale
workflow-selection defect when reopening the new wizard. The signed-in Mac
GUI now starts a genuinely independent panel, while explicit reconnect still
works and all 14 retained workflow records remain byte-identical. Unit tests
are 218/218 PASS; current isolated host tests are 13/13 PASS. See
[panel isolation evidence](panel-isolation-20260916.md). This is not a completed
network fault or desktop crash-recovery qualification.

The subsequent September 16 B12 presentation batch found and corrected an
unconditional verified-tab title for failed/incomplete reports. Unit tests are
211/211 PASS; isolated VS Code 1.105.0, 1.136.1 and current 1.137.0 hosts each
pass 13 tests.
See [scope, negative cases and review](verification-presentation-20260916.md).
This is not a signed-in profile installation or an Azure fault result.

Independent network-source recovery now has a separate Neo4j 5.26 draft,
confirmed transfer storage and a hash-verified copy of the pinned Linux build.
Its new runner preview passed, but creation/identity access is awaiting the
exact native confirmation. No source or migration is active. See
[network-recovery preparation](network-recovery-20260916.md); B11 remains partial.

Code baseline: `73aa6d690cad947a5d5d6a7371dc7adf7f191627`, plus the new
`extensions/vscode/src/test/unit/runnerReconnect.test.ts` in this change.
No new Linux artifact is installed by these tests.

- TypeScript typecheck: PASS.
- Extension unit suite: **199/199 PASS**, including four new reconnect cases.
- Real Go configuration-validator contracts: **9/9 PASS** (Azure/on-premises/
  other-cloud Neo4j and PostgreSQL, explicit/Gremlin Cosmos, local CSV).
- Isolated installed VS Code **1.137.0** Extension Host: **3/3 PASS** (four source editors,
  command registration, workspace-free wizard). This is a development-extension
  test in a disposable profile, not the installed signed-in Azure profile.
- Targeted Go tests: runner, app, config and all source packages PASS (cached).
- Go race-enabled runner tests: PASS (cached).

The reconnect tests reopen persisted uncertain state without changing its
bytes, reconcile with GET only, retain failed/interrupted identity without
claiming verification, and simulate a lock owner's abrupt process exit. A
retained crash lock deliberately blocks action; this proves fail-closed
behavior, not user-facing crash recovery. Safe stale-lock reconciliation needs
review alongside B10/B11 rather than automatic lock deletion.

## Next execution order and gates

### Recovery inspection implementation — 2026-09-15

Added the allowlisted `inspect-resume` request and native GUI inspection action.
The request accepts only the current checked boot and protected target connection;
the original configuration is read from its hash-checked guest file. The database
transaction explicitly uses `READ ONLY, REPEATABLE READ`; no source is connected,
worker launched, lease removed or metadata changed. The result binds job, graph,
generation and original submitted configuration. Missing or inconsistent evidence
is rejected. Stale checkpoints and retained leases are review reasons, not silently
repaired state. All results keep `canResume: false`. Lossless decimal counters and
generation IDs are validated on both boundaries, and an uncertain dispatch is
reconciled by GET only. Accepted inspection evidence is retained in local workflow
metadata separately from migration verification.

Validation: extension **204/204 unit tests PASS**, typecheck/build and host-test
compilation PASS; Go runner/tools tests including `-race` PASS. New Go tests cover
protocol restrictions, identity/mapping/generation mismatches, reject counts,
stale/running checkpoints and local-file tampering without start/lease removal.
New extension tests cover lossless IDs, invalid/false-success responses, incapable
runner rejection, protected-only credentials, persisted intent and GET-only
reconciliation. These are local tests, not installed-GUI/Azure qualification of
the new control. The previously qualified Linux build is unchanged and does not
advertise the new capability; it must reject inspection until a reviewed pinned
candidate is installed. Actual remote resume, stale-lease/lock reconciliation and
P1 fault/recovery execution are still prerequisites for closing B11.

### Explicit continuation implementation — 2026-09-15

The follow-on change adds an explicit `resume-migration` boundary and native GUI
action. The new operation references the previous operation and original job,
configuration hash, fingerprint, generation and committed-row checkpoint. It
accepts no replacement configuration. Admission is kernel-lock serialized with
other guest submissions; the previous systemd service must be loaded but inactive
or failed, with no main/control PID. Existing operation state must be failed or
boot-interrupted. Current health, other workflow leases and target evidence are
checked before a one-use continuation claim is written. Old logs/configuration/
state are not overwritten; only the proven predecessor lease is atomically
replaced, and a fresh operation directory retains the exact original job bytes.
Partial admission/lost start responses remain fail-closed for reconciliation.

The worker rechecks the target binding before calling the actual CLI `resume`
command, skips AGE preparation/new graph creation, and independently requires the
same committed graph generation before counts verification. The verifier report
must retain the original fingerprint. Host admission binds the source draft/CA,
target/private placement, VM and preserved disk/NIC/identity, and original pinned
artifact. Old unbound jobs are intentionally unsupported; the nine qualified
base targets have not been upgraded or resumed. No cloud resources were started.

Local validation: **206 extension unit tests PASS**, typecheck/build and host-test
compilation PASS; Go runner/tools tests including race detection PASS. Controlled
child-CLI worker tests cover preserved job/configuration, no preparation, refused
replay, parallel continuation claims, unsafe gates, lost start acknowledgement and
post-load generation mismatch. These use synthetic execution/metadata seams and
do not establish live PostgreSQL, Neo4j, Cosmos or CSV recovery correctness.
Actual installed-GUI fault/recovery/P1 digest qualification is still pending.
Desktop crash-lock recovery remains a distinct B10 gap; no automatic lock removal
was added. The installed qualifying VSIX/Linux archive is unchanged.
Linux/amd64 CLI and tools cross-builds and `go vet` also passed. The isolated
installed VS Code host smoke suite passed 3/3 after recompilation; it verifies
activation/editor behavior, not the new resume action against Azure.

### Candidate installation and live handoff — 2026-09-15

The reviewed commit was packaged with 206/206 tests passing, cross-built into a
pinned Linux archive, and installed in the normal Mac VS Code profile. Installed
JavaScript matches the package hash. Existing-host reload has not been confirmed:
GUI input did not advance and View commands were disabled. The user was asked to
unlock/focus/reload VS Code. All 17 VMs and 13 Flexible Servers remain stopped;
the Linux candidate has not been deployed. Cost API returned 429, so the prior
conservative reserve remains in force, not a new actual-spend claim. See the
[candidate seals and reviewed live fault sequence](recovery-execution-20260915.md).
B10/B11 remain open; no live recovery qualification is claimed.

After the user reloaded VS Code, installed-GUI preparation resumed with a fresh
CSV workflow. File-selection cancellation and storage-approval cancellation
returned without transfer/deployment; ARM independently confirmed the proposed
storage does not exist. All 18 P1 CSVs are selected. The new storage/account-only
role dialog now awaits action-time approval. A stale folder-error message after
successful selection was observed and remains to fix/retest. See the execution
record above for exact workflow and scope; B08/B09 are only partially exercised.

Subsequent user approval created the dedicated storage/account-scoped user role.
The existing trial-only tag exception restored authenticated HTTPS transfer while
anonymous/shared-key access stayed disabled. Upload-confirmation cancellation
left zero transfer records; the next upload completed all 18 P1 CSVs (1,168,576,671
bytes). The pinned Linux archive is also uploaded. Fresh VM preflight passed;
native installation/container-reader approval was subsequently received. ARM
provisioning and separate pinned Linux readiness now pass, including both
recovery capabilities, idle health, 3.4791% disk use and zero swap/OOM. All 18
GUI mappings independently match the portable P1 schema. Guest CSV imports have
completed; all eighteen full-hash receipts are verified (1,168,576,671 bytes).
Complete inventory now passes with exact 1.6M vertices / 4M edges, unchanged
source files and a hash-verified imported report. Closing/reopening the source
panel during its retained submission restored the same operation without a
duplicate inventory dispatch. This is not an Extension Host crash/reload test.
Private target preflight passed; its native deployment/credential approval is
pending. Migration and live fault/recovery tests have not started. B08/B09 remain
partial and B10/B11 open.

The subsequent CSV trial reached the per-VM 25 managed Run Command ceiling
after ten verified imports. An operator archived and independently reconciled
24 completed command definitions/results, then removed only those ARM command
resources; guest CSVs and evidence remain intact. The current controller
reference was preserved, and GUI import resumed after fresh readiness. See the
execution record for the archive seal. Before release, B08/B09 must address
command lifecycle without requiring routine manual cleanup: never remove an
outstanding/uncertain or currently referenced request, preserve verified durable
evidence before removal, and reconcile interrupted cleanup without replay.
The existing fail-closed limit remains enabled during pinned-candidate testing.

### CSV recovery private target checkpoint (September 15)

The new CSV recovery workflow completed all 18 guest imports and full inventory
(1.6 million vertices / 4 million edges). Its private PostgreSQL 18 target is
GUI-reconciled as provisioned; independent Azure reads confirm AGE allowlist
and preload applied after the approved restart. After GUI capture recovered,
restart reconciliation and same-VM resize completed. The new load committed
all 5.6 million records with zero rejects, and its transferred counts report
passes independently checked integrity. The fault observer missed its safe
window and refused to signal; no fault or resume occurred. Full canonical
verification now passes in the installed GUI: all 64 ranges / 5.6 million
records and the frozen root independently match after report transfer. See the
[execution record](recovery-execution-20260915.md) for exact boundaries and seals.
This closes no live fault/recovery acceptance case. B10/B11 remain open.

### Second CSV recovery trial checkpoint (September 15)

Fresh workflow `54da6ddd-27d2-45e0-bb68-cf5f352801db` now has all 18 Linux CSV
hash receipts and a GUI-imported complete source inventory PASS: 1.6 million
vertices / 4 million edges, zero rejected records, unchanged before/after files.
Mac-side rehashing independently matches the receipts and original P1 mappings.
The new private runner is ready on the unchanged pinned candidate; no target
or migration has been created. Target review paused when computer use reported
the Mac locked. Resume after manual unlock and fresh target preflight/approval.
Two manual archive-first ARM command-capacity interventions were necessary;
the extension's production command lifecycle remains a gap. The initial
bootstrap/readiness race is retained separately. These are preparation and
reconciliation observations, **not** B11 fault/recovery acceptance. See the
[execution record](recovery-execution-20260915.md) for scope, seals and next steps.

### Remaining sequence

1. Complete the existing-evidence mapping and local/host negative tests (B01–B10,
   B12). Preserve separation between mocked and actual service evidence.
2. Review and exercise B11 remote resume, including allowlisted guest protocol, explicit
   GUI approval, durable intent, credential handling, config and identity checks,
   no automatic replay, and safe reconciliation of interrupted control actions.
   Test both source and CSV behavior; unsupported combinations must block clearly.
3. Review/test/package a pinned candidate before installing it for live trials.
   Recheck remaining time and conservative cost; do not start an experiment that
   cannot finish inside the current envelope.
4. Run B10/B11 on a fresh P1 target/job with precise fault points and retained
   checkpoints; after recovery require complete counts and the frozen canonical
   root, not merely job completion. Run B05 Gremlin and remaining independent
   choices through the installed GUI as their own evidence permits.
5. Close each case with artifact IDs/hashes and screenshots/action evidence.
   Stop owned compute after retention. Release readiness still requires all
   mandatory cases and a matching tested/installed VSIX; this ledger is not a waiver.
