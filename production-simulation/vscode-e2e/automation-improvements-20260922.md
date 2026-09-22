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
