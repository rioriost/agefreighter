# Remaining guided migration qualification

Updated: 2026-09-15 UTC. Status: **running; not release-qualified**.

The nine base P1 routes are [qualified](progress.md). This ledger covers the
additional requirements in [the original plan](plan.md#branch-coverage-beyond-the-base-paths).
It does not reopen or replace their evidence. Unit tests, actual CLI contracts,
isolated Extension Host tests and live installed-GUI/Azure tests are different
evidence levels. A test of a simulated failure is not a live recovery pass.

## Authorization and preservation

Use only the already authorized dedicated trial environment. Current renewed
ceiling: USD 800; retained reserve: USD 400; deadline:
`2026-09-16T07:14:35.311Z`. Recheck conservative remaining cost, ownership,
health, governance, quota and time before any cloud mutation. The nine accepted
targets and failure evidence must remain intact. Recovery trials require a
separate job/graph; do not fault or overwrite a qualified graph. As observed
after base-route completion, all 17 VMs were deallocated and all 13 Flexible
Servers stopped. Local regression work has not restarted them. Storage and
Cosmos charges continue; stopped PostgreSQL servers eventually auto-start.

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
| B10 | Close/reload during assessment/load/verification; no replay | New persisted-state/process-exit tests pass; no actual live window-close evidence yet | Close/reopen installed GUI in all three phases; retain same operation/job and prove no duplicate dispatch | not-run |
| B11 | Loader/network interruption; explicit same-job recovery | Guest continuation and GUI explicit resume implemented; local tests pass, not live-qualified | Review/package candidate, run actual faults and unchanged database/graph/job/generation/fingerprint recovery, then full P1 verification | running |
| B12 | Invalid verification must never be PASS | Verification/report unit tests cover mismatch, rejects, incomplete, wrong-job, stale, changed/truncated evidence | Installed-host presentation/interaction tests for representative rejected results; no forged success in retained base workflows | not-run |

## First local regression batch

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
