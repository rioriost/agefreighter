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
| B11 | Loader/network interruption; explicit same-job recovery | CLI supports resume, but remote guided protocol and execution panel do not expose it | Implement and review remote resume; test unchanged database/graph/job/generation/fingerprint, then complete P1 counts and canonical verification | blocked |
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

1. Complete the existing-evidence mapping and local/host negative tests (B01–B10,
   B12). Preserve separation between mocked and actual service evidence.
2. Implement B11 remote resume, including allowlisted guest protocol, explicit
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
