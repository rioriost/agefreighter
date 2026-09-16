# Remaining guided migration qualification

Updated: 2026-09-16 JST. Status: **running; not release-qualified**.

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

## Branch-to-evidence ledger

"Not-run" means additional live qualification is not established by this audit,
not that no related unit tests exist. Reconcile older route artifacts where
they prove a precise choice before scheduling redundant infrastructure work.

| ID | Required branch | Existing evidence / current limitation | Remaining acceptance | State |
|---|---|---|---|---|
| B01 | Default/separate migration RG; independent network RG | Private base paths passed; `runner.test.ts` validates resource placement | Bind each selectable RG choice to an exact GUI artifact; exercise unrepresented choices | not-run |
| B02 | Region/zone defaults, overrides, unknown zone; invalid region/SKU/subnet/quota | Placement and preflight unit tests pass | GUI defaults/override evidence; invalid choices produce no write request | not-run |
| B03 | Private Azure and IP-only discovery | Nine base routes cover private Azure and IP-only on-premises; other-cloud generated configuration passes real CLI validation | Audit other-cloud selectable path; no inferred source ARM lookup | running |
| B04 | Neo4j versions; PostgreSQL recommendations/review; CSV typed mapping | Relevant base routes and full P1 canonical results pass | Map recommendation acceptance/edit branches and CSV choices to retained GUI evidence | running |
| B05 | Cosmos explicit and Gremlin documents | Explicit-document base P1 passed; both generated formats pass CLI validation | Prepare equivalent Gremlin P1 representation, then full installed-GUI migration and canonical verification | not-run |
| B06 | Supported Cosmos authentication and RBAC propagation | Guided source currently fixes `default-azure` managed identity; base route passed | Document fixed GUI authentication scope; audit propagation/denial handling; do not claim other CLI modes GUI-tested | running |
| B07 | Same-VM resize; active-job/incompatible resize denied | Base routes resize same VM; resize/preflight unit tests pass | Bind identity preservation artifacts; exercise missing live denial branches without destructive resize | not-run |
| B08 | CSV multi-file/reconcile, changed/hash mismatch, transfer/folder cancellation | Base CSV P1 passed; transfer tests cover changed files, verified receipts and matching existing blobs | Installed-GUI cancellation/partial transfer/reconciliation, with no load before complete receipts | not-run |
| B09 | Approval cancellation, expired preview, duplicate windows, lost ARM reply, bootstrap/artifact/quota failure | Controller unit tests cover stale previews, locks, persist-before-PUT and GET-only reconciliation | Enumerate every actual approval surface and safely inject unrepresented faults; capture zero unauthorized writes | not-run |
| B10 | Close/reload during assessment/load/verification; no replay | Persisted-state/process-exit tests pass; installed panels closed/reopened during inventory, load and full P1 verification, preserving operation IDs | Finish verification reconciliation and retain no-replay evidence; actual Extension Host crash/reload is still distinct and untested | partial |
| B11 | Loader/network interruption; explicit same-job recovery | CSV r2 live SIGTERM + VM reboot, both same-job GUI resumes, counts and 64-range digest PASS; network-source fault not yet qualified | Qualify the separate network-source interruption and explicit same-job recovery with full P1 verification | partial |
| B12 | Invalid verification must never be PASS | Unit tests plus isolated real VS Code Extension Host panels cover complete counts, wrong-job, stale, missing counts, incomplete coverage, count mismatch, rejects, failed checks, truncated and hash-mismatched evidence; misleading verified tab title fixed | Signed-in installed-candidate retest and remaining digest/controller import failures; synthetic reports are not Azure fault evidence | partial |

## First local regression batch

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
