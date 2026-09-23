# B10 explicit interrupted-lock recovery — September 23

## Scope and current verdict

**Native local recovery verified at the 09:59UTC checkpoint. B10 subsequently
completed its distinct active-cloud case; see the [finite acceptance ledger](b09-b10-b12-acceptance-20260923.md#b10-the-one-required-active-cloud-crash-case--complete).**
The production store and native
command now support an explicit review of recoverable interrupted local locks.
Recovery performs no Azure request, dispatch, retry, reconnect, or workflow edit.
Unit/panel contracts and an isolated real Extension Host SIGKILL/reopen exercise
pass. The coordinator has now exercised the actual native confirmation dialog
in an isolated unsigned-in profile, as recorded below. A signed-in crash during
an active Azure operation was still open at this local-only checkpoint.

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

## Actual native local cancellation and recovery — 09:57–09:59 UTC

The later native check closes the local confirmation gate, not the active-cloud
gate. The test-only `scripts/native-lock-fixture.cjs` creates a fresh private
profile and harmless synthetic CSV draft/report with production `RunnerStore`.
Its own child78447 acquired the production lock, sealed its marker and exited23
inside the protected action, leaving a real same-boot dead-owner lock. No arbitrary
process was terminated and no cloud operation was represented by the fixture.

Retained fixture: `/private/tmp/af-native-lock-fN6Gq6`, workflow
`578684f4-aff7-41ce-a8d3-67e35d31b428`. The real VS Code1.139.0 development host
loaded unchanged implementation3bc0069, bundle SHA
`3108910e2933f73b1b1d0edf92458e7384e2008261469c56da2f19789366da67`.
Normal installed extension and all91operator-store files remained unchanged.

The coordinator used the actual Command Palette production command, selected
the sole fixture workflow and inspected its native macOS modal. The displayed
workflow/PID/lock/workflow hashes agreed with the independent file baseline.

1. Clicked **Cancel**. Independent verification confirmed the exact same three
   store files, original lock inode and bytes, and no recovery archive.
2. Opened a fresh review through the same native command and clicked **Recover
   local lock**. GUI reported archived recovery with no remote action.
3. Independent verification found exactly one997-byte private archive, original
   lock absent, exact original lock bytes and review hashes in the archive, and
   unchanged workflow/report bytes. A new production exclusive local read
   succeeded without changing those bytes. All three retained files are0600.

Archive SHA `630f6a925056258b7558ec4a899887e11a5295e29f0808d16deb67bb57f6aee2`,
approved09:58:25.667UTC. Cancellation evidence SHA
`a284bd3aff7574470a34859bad81f0a50501bf66a7a8bbef63303540df72768b`;
recovered evidence SHA
`d073bcec17b24124ab9fdccfd7032d8f7e27302e3e1f0b344669bca8c97da65d`.
Native modal and success screenshots/accessibility observations are retained in
the task transcript; these are actual clicks, not inert panel adapters or a
scripted call to approve recovery. The fixture verifier itself cannot prove UI
interaction, so its filesystem evidence and these observations remain distinct.

### Environment caveats and scope

Launching a second instance at the same application path caused the UI tool to
bind the normal instance. The coordinator stopped only that newly launched test
instance and copied the unmodified official application inside the fixture,
then bound its exact separate path. Original/copy executable SHA both
`1b58953da3281bddea360f21198ab80f8a5caa72c77f7cf996bd3c9fa74dfb3c`.
The recovery host PID79913 activated AGEFreighter09:57:42.242UTC.

The first generated workspace placed a user-only update setting at workspace
scope, causing a harmless VS Code settings error. The helper was corrected and
its two local tests rerun. The live fixture was preserved; the coordinator closed
the untrusted test workspace and opened only its readme in a folderless window.
No workspace-trust protection was disabled or trust grant made. The lock command
needs no workspace folder. The isolated test window was closed after verification;
its profile, app copy, logs and evidence remain retained.

No Azure sign-in, resource mutation, credential read, real migration or remote
operation took place in this native test. The already-completed isolated Extension
Host SIGKILL tests and this dead-child/native-modal test are separate evidence;
they do not together constitute an unobserved signed-in active-cloud crash.
See [native evidence](evidence/b10-native-lock-20260923.json) and the
[next live scopes](b09-b10-b12-live-next-20260923.md).
