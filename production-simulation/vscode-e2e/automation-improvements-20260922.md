# Guided test orchestration improvements — 2026-09-22

## Scope

Reduce interactive waits without bypassing authentication or treating service acceptance as qualification. Existing B03 inventory and all accepted routes remain retained. Local implementation happens with compute stopped. The current live continuation still ends at 10:58 UTC; local development does not extend it.

1. Opt-in, workflow/endpoint/account/CA-bound encrypted source credential reuse, maximum eight hours or the earlier target deadline. Explicit forget/replace; no secrets in JSON, logs, webviews or AI prompts.
2. Retain target inputs and output-folder choice before live preflight. Allow offline plan saving; refresh readiness after user input and before deployment. No automatic deadline extension or new deployment approval.
3. Bounded monitoring of existing operations, cancellation/reconnect, exact-operation binding, no write replay. Complete an approved sealed report transfer in one invocation. Approval of transfer is not approval of source reread or migration.
4. Automated unit/adapter tests for expiry, binding changes, cancellation, failure, report identity and duplicate-dispatch prevention. Installed-GUI/live acceptance remains a separate gate.

MFA, native OS consent, changes to security scope, new resource deployment and verifier installation remain explicit. No promise of unattended execution through those boundaries. Pending or uncertain operations must be reconciled, never restarted automatically.

## Review

Fail-closed boundaries: deadline and budget gates remain in the execution core; source passwords cannot authorize target writes; draft persistence grants no deployment permission; each status step reads the latest store under its own lock; errors stop monitoring. Cancellation stops monitoring only, never an already running migration. Polls are bounded to protect the Azure command receipt limit.

Implementation and installed-GUI/live outcomes are recorded below only after verification.

## Implementation verification — 09:32 UTC

Implemented encrypted opt-in credential reuse/prepare/forget, source-connection and
retained-failure invalidation, per-field target draft persistence and stopped-VM
save-only planning, post-input readiness refresh, bounded exact-operation watches,
single-approved sealed report export/import, and a 20-minute exact-scope resize
sequence grant. Reconnect does not start new source reads or migrations. The
source watcher invalidates credentials conservatively on any terminal failure;
it never attempts authentication recovery or replays a load.

`npm run check`: typecheck, 502 unit/adapter tests and bundle build pass.
Isolated real Extension Host: 25 pass on the baseline 1.105.0 and 25 pass on the
installed macOS arm64 VS Code 1.138.0 (`7debcd0e2acdea1c52de81bf9ee1620444407dda`).
Both host runs used disposable profiles, not the operator's Azure/SecretStorage.
The first installed-host launch used the outdated executable name `Electron`
and failed before tests; resolving the actual executable `Code` produced the
successful run above. These are synthetic/local regression results, not a new
Azure migration qualification. Production B03 resources remain stopped and its
663-byte imported inventory is preserved. Development VSIX install and actual
signed-in GUI/live continuation remain the next gates.

## Handoff — packaged, production-profile installation pending

Implementation committed/pushed as `5f93f3c`. Packaged development VSIX:
`extensions/vscode/dist/agefreighter-2.4.0.vsix`, SHA-256
`3a82dfceaefc5534911e3c1e7f994bef5d84102fab60de8c8eeb2dd3117662de`.
Requested action-time consent for installing this unpublished build and reloading
the operator's window. Do not start compute while waiting. Read-only ARM refresh
confirmed the exact B03 source and runner are still deallocated with their expected
disks/tags; recent activity contains the authorized deallocation and health events.
No new target, migration, source read or credential change occurred in this turn.

After installation, prepare the optional workflow credential and target inputs
while stopped; reuse inventory `731f5d8f-b99c-4c6e-9c0c-59044ac6a217` rather than
repeating source investigation. Reactivate the exact-scope safety monitor before
any authorized restart. Current hard stop is still **2026-09-22 10:58 UTC**,
begin stop at10:53; do not reset the clock. If insufficient time remains, retain
the prepared inputs and request a new bounded live window without starting VMs.
New target/subnet/credential creation and the dedicated verifier remain separate
action-time gates. The B03 branch remains partial until actual installed-GUI load,
all labels and64 canonical ranges/root have been verified.

## Operator-profile installation — 2026-09-22 11:51–11:53 UTC

User explicitly requested installation of the corrected build. Rechecked the
approved VSIX SHA above, installed with the official VS Code CLI, and observed
successful installation. Installed bundle SHA-256
`03e64421e06ed338b7d5a9997f4c2cc7ebcec2933b2d0498a520324f6799b3aa`
matches the bundle inside the approved VSIX. Actual CUA `Developer: Reload Window`
completed. Executing the new `Prepare, Replace or Forget Source Credential`
command opened the retained-workflow picker including B03; cancelled before
credential retrieval, entry or storage. This verifies activation, not live migration.

Read-only local evidence check: B03 inventory remains finished/imported,663bytes,
SHA `4e0efa9b18b6b1a985046a80599405687d1b7d7b9e82bdf1fba76c7c73a57435`;
target and migration remain absent. No Azure start/change was requested. The
10:58UTC live bound has expired; a new explicit bounded runtime authorization is
needed before starting the existing source/runner. Do not silently extend it.

## Installed-GUI live acceptance — September22 12:16–13:20UTC

After a separately approved new bounded session, the operator prepared Remember
and saved/reused target inputs/LoadJob offline before compute startup12:22:03UTC.
Same-VM resize preserved identity/disk evidence. After a screen-lock interruption,
fresh readiness and migration submission reused the remembered source credential
without another password prompt. One approved counts-report transfer completed
export/import in the same flow. Counts and separately approved full canonical
verification passed for5.6Mrows/18labels/64ranges with the frozen root.
See [B03 execution](other-cloud-n526-live-20260922.md).

Automation limits remain observed, not hidden: the retained-operation watch
returned a running state and an explicit Refresh retained migration was needed
to reconcile terminal completion. The reason has not been diagnosed. An initial
start preflight refused stale readiness after the lock wait, requiring an explicit
fresh readiness check. Full-P1 reconciliation/export/import still uses separate
GUI actions. These are follow-up automation gaps, not migration failures or proof
of unattended completion. No code/installed-pin change was made during this run.
