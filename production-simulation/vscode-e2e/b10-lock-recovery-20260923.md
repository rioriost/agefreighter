# B10 explicit interrupted-lock recovery — September 23

## Scope and current verdict

**Implemented locally; B10 remains partial.** The production store and native
command now support an explicit review of recoverable interrupted local locks.
Recovery performs no Azure request, dispatch, retry, reconnect, or workflow edit.
Unit/panel contracts and an isolated real Extension Host SIGKILL/reopen exercise
pass. The native confirmation dialog has not been exercised by a real operator,
and a signed-in crash during an active Azure inventory/load/verifier remains open.

This change does not alter prior accepted canonical reports. No installation,
credential access, operator-storage mutation, compute start, or live Azure action
is part of this implementation subtask.

## Recovery admission and behavior

New locks contain a versioned workflow ID, unique owner token, local PID, kernel
boot identity and creation timestamp. Metadata is synced before the protected
action starts. macOS uses its kernel boot-session UUID. Linux additionally binds
the PID namespace; Windows and unavailable identities cannot qualify recovery.

Only a lock from the exact current boot/namespace with a definitively absent PID
(`ESRCH`) is eligible. A live or reused PID, permission failure, unexpected probe
failure, empty legacy lock, malformed metadata, linked lock, other boot, or unknown
identity remains blocked. Age alone is never evidence that a lock is stale.
Legacy September 18 empty locks cannot be retrospectively adopted.

`AGEFreighter: Review Interrupted Runner Lock` selects a retained workflow and
shows a native modal with workflow ID, former process ID, lock creation time and
lock/workflow hashes. It explains that the remote operation may still be running.
Selection cancellation and modal cancellation make no recovery write.

The review is one-use, bound to exact lock bytes/device/inode and workflow bytes,
and expires after five minutes. Workspace trust, boot identity, owner absence,
review expiry and file identities are rechecked at recovery time. Changed evidence
or failed archive/directory sync prevents lock removal. Original lock text and the
approval snapshot are retained in a private `*.recovered-lock-*.json` archive and
synced before the original lock is removed. Workflow JSON and reports are unchanged.
The command then tells the operator to review/reconcile the retained operation
separately; it does not invoke that operation itself.

A shared short-lived acquisition/recovery gate excludes participating windows.
Normal owner release waits up to two seconds for transient gate contention and
never removes another owner's gate or replacement lock. An interrupted gate is
ambiguous and remains blocked for investigation. Power-loss behavior, unsupported
filesystem durability, and cross-platform runtime behavior are not qualified here.

## Local verification

- TypeScript validation and host-suite compilation pass.
- **44/44 dedicated tests pass:** 33 store/ownership tests and 11 native-panel
  handler tests. They cover cancellation, no replay, unchanged workflow/report
  bytes, live/uncertain/legacy/foreign-boot admission refusal, identity replacement,
  in-place mutation, review tampering/expiry, archive/fsync failure, shared gate
  contention, trust loss and a real separately exited local Node process.
- Production PID-probe tests verify only `ESRCH` is absent; `EPERM` and other errors
  remain uncertain. A reused live PID is conservatively refused.
- Panel tests use inert VS Code adapters. They verify explicit modal approval and
  cancellation semantics, not actual modal rendering or a human click.

## Real isolated Extension Host continuation

The root task executed the updated crash harness serially using actual VS Code
**1.139.0**. The B09 fixture now supplies a synthetic running owned VM plus separate
historical and fresh same-boot readiness controls with independent GET responses;
no production eligibility checks were relaxed for the harness.

Retained evidence: `/tmp/af-host-crash-ske6Ed`.

| Boundary | Killed / reopened PID | Exit codes | Reopened result |
| --- | --- | --- | --- |
| Durable intent, before synthetic dispatch | 66815 / 66920 | 9 / 0 | PASS |
| Synthetic dispatch accepted, before reply | 67035 / 67159 | 9 / 0 | PASS |

Before-dispatch `recovery.json` SHA-256:
`5806f8fc791648e9d6400be9fd94c96019486f53628b63ae94049945a04b47a9`.

After-dispatch `recovery.json` SHA-256:
`216d6d8426e3a785101fbc85a94a8db0070434e4ad1c84390d8f5d8cee53eb8c`.

Each reopened host first confirms the surviving lock blocks exclusive access,
checks the prior GET-only controller behavior with inert persistence, and verifies
cancellation preserves the lock. It then supplies explicit **scripted approval**
to the production store, verifies durable lock archive content, proves unchanged
workflow/report hashes and `submitted` intent, and confirms exclusive read access
is possible again. Lock recovery itself does not call even the inert ARM adapter.
Both results record `scriptedLocalLockRecovery: true`,
`nativeOperatorConfirmationTested: false`, and `cloudRequests: 0`.

These results extend the September 18 isolated process-crash evidence; they do not
close B10's signed-in active-cloud crash or native operator-confirmation gates.
