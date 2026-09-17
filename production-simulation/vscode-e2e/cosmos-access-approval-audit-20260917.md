# Cosmos access and approval-surface audit

September 17, 2026, approximately 05:44–05:54 UTC. B06/B09 follow-up.
No Azure resource, permission, network, credential or accepted graph was changed.
The budget/deadline were not renewed. This is not another P1 migration pass.

## Findings reproduced and corrected

The production source-panel handler was exercised with inert VS Code/store/ARM
adapters; the actual Cosmos access controller was used, not a mocked verdict.
Before the fix, five panel cases and four controller cases failed:

- An untrusted workspace could reach the grant handler; trust was not checked
  again after waiting for approval either.
- A different window could change the reviewed principal, account/scope or VM
  while the native approval dialog was open. The handler reread the latest
  record but did not bind the submitted grant to what the dialog had shown.
- Grant submission did not reread the VM's ownership/system-assigned principal.
  A replaced/removed VM or changed ownership could leave a stale grant preview.
  Source admission also accepted the old role while the VM principal changed.

Corrections: check workspace trust before preview/reconciliation and again
under the workflow lock before granting; compare the complete reviewed
subscription/VM/source/access binding after the dialog; reread the owned VM
principal before a grant or a new Cosmos source operation. Reject changes
before persisting a submission or sending a role PUT. These are additional
preflight checks, not an atomic lock on external Azure governance changes.

The source UI now states the important distinction: `ready` verifies the ARM
role assignment, **not data-plane propagation**. Only a successful source
assessment proves actual reads. Authentication is fixed to the Linux runner's
managed identity; Cosmos account keys and alternate desktop credentials are
not GUI choices. No new authentication or permission path was added.

## Validation and scope

Nineteen added tests, **292/292 unit tests PASS**, no skips; TypeScript check,
normal production build and VSIX packaging PASS.

- Eight panel tests: untrusted workspace, trust lost during approval, Cancel,
  panel disposal, three concurrent identity/scope changes, and one-time approved
  submission followed by reconciliation without replay.
- Ten controller tests: three changed/unavailable VM identities, replaced VM
  before source admission, role-read HTTP403/404, role-PUT HTTP401/403/429, and
  denied assignment reconciliation. A failed/uncertain PUT remains `unknown`;
  explicit subsequent checks are GET-only, not another grant or key fallback.
- One rendered-source-view regression states the authentication/propagation
  distinction. The existing remote-role-change test now changes the simulated
  remote response rather than changing the retained approved principal.

The simulated HTTP statuses above are ARM responses, **not** live Cosmos
data-plane propagation/denial experiments. No new grant was created or revoked
to manufacture a failure. B06/B09 remain partial pending their live GUI cases.

Live read-only ARM checks around 05:53 UTC confirmed the existing source
`afcosmosp120260907`: Japan East, public network Disabled, local/key auth
disabled. The previously approved assignment
`0caf9405-337c-4b6f-879c-b40900c9c03b` still has principal
`0132a1ea-492d-41d3-b910-b5d978a5a06c`, Built-in Data Reader ending `0001`, and
scope exactly that account. No current guest-health or fresh data-plane access
claim is made. The historical complete-read/migration proof remains the
[explicit-document qualification](az-cosmos-r2-execution-20260915.md).

## Native approval inventory for B09

This is a source/control-flow inventory, not a claim that every cancellation
below was tested through the installed GUI. Selection dialogs and credentials
also have Cancel paths; legacy desktop-CLI approvals are separate from this
runner-first matrix.

| Controller | Approval surfaces | Remaining cancellation / binding work |
|---|---|---|
| `runnerMigration` | Create reviewed Linux VM | Fresh/expired preview and duplicate-window installed-GUI no-write evidence |
| `developmentRunner` | Prepare pinned executable; upgrade idle runner | Cancel both artifact approvals without uploads/install/replay |
| `runnerSourcePanel` | Retain failed assessment; grant Cosmos Reader; upload CSV; import CSV; create transfer storage/user role; transfer assessment report; sampled/complete source reads | CSV defined cancellation cases already pass; Cosmos controller cancellation/binding tests pass in this batch, installed-GUI refusal/Cancel still pending; other surfaces remain open |
| `runnerTargetPanel` | Save plan only versus approve deployment; repair failed preload | Native form/folder cancellation, unchanged evidence and no deployment/repair |
| `runnerExecutionPanel` | New cost authorization; preload restart; resize; start; resume; checkpoint inspection; archive empty-target failure; diagnose target; transfer migration report | Resize Cancel previously observed; remaining distinct surfaces need exact evidence, including no implicit authorization renewal |
| `p1QualificationPanel` | Independent full P1 verifier | Cancel before protected guest dispatch; invalid import/display cases remain B12 |
| `p1DiagnosticPanel` | Read-only failure diagnosis | Cancel before protected guest dispatch; never qualification PASS |

Persisting a local preview before approval is distinct from an Azure write.
Reconcile buttons may persist observations but must not replay an uncertain
mutation. ARM role existence must never be used as proof of source-read success.

## Prepared candidate; installed GUI still pending

The normal candidate was packaged, not published or installed in this batch:

- VSIX SHA-256: `ceb7cc74ca1f60723fb693b659b0ead1d4e2b25631979375252cc3a7db1a5165`.
- Bundle SHA-256: `34dc9fc71284d7a356aa52f40b03772a06197b40887dfd0ade79b4216707d91e`.
- Existing installed bundle is unchanged:
  `3061aea43c315113f47df3ad1da31dc68ad73c27b47a18571ddfb66bf742e929`.

Action-time approval was requested to install this unpublished candidate and
reload VS Code for display/refusal/Cancel checks only. That request does not
authorize a new Azure role, VM start or migration. Until installation and the
specific UI observations finish, do not call the fixes installed-GUI qualified.
