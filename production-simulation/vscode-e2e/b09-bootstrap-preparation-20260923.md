# B09 terminal bootstrap failure — local preparation

September 23, 2026. **Preparation complete; actual guest observation remains
pending.** No Azure write, executable installation, guest action, source read,
GUI mutation or release change was performed by this preparation. The selected
VM, account, bounds and proposed approval are in
[the supervised session](b09-fault-session-20260923.md).

## Concrete artifact

The new private local directory is
`production-simulation/work/b09-bootstrap-negative-20260923/`. It contains the
normal `manifest.json`, mandatory `negative-fixture-provenance.json`, and
`b09-negative-missing-tools.tar.gz`. The directory is0700 and files0600.

| Evidence | Exact value |
| --- | --- |
| Original source commit | `d40d6ccc9a4ddf6e2ca626392cd7bf83140ed6c7` |
| Original archive SHA-256 | `2321022975f85c73068a54fd21a287e2802d5bc33fa21eb9323c142dd7262ff6` |
| Original bytes | 37,197,546; independently rehashed unchanged |
| Derivative archive SHA-256 | `3c33a179916ec08a83ca8ccb3c19e7682862d0382a63b2c05b5369a3cb124e33` |
| Derivative bytes | 18,378,050 |
| Only archive member | Regular file `agefreighter`, 37,690,414 bytes |
| Retained member SHA-256 | `386c3ede5ff1687a5e0fe9d1948faf775561db7c4caba14dcfddc3de5087e9b3` |
| Intentionally omitted member | `agefreighter-tools` |

The [local builder](prepare-b09-bootstrap-fixture.py) validates the fixed original
pin, size, regular-file member identities and absence of extra paths/links. It
creates a new directory only and deterministic USTAR/gzip metadata. It neither
executes the included Linux binary nor modifies the original. The development
manifest's commit identifies the included binary's source; the derivative is
an intentionally unusable fixture, not an unmodified build or release.

Reproduction, using a **new** output directory:

```sh
python3 -B production-simulation/vscode-e2e/prepare-b09-bootstrap-fixture.py \
  production-simulation/work/vscode-runner-build.HwWiUz/manifest.json \
  production-simulation/work/b09-bootstrap-negative-reproduction
```

## Failure and expected state

Unchanged `bootstrapScript` verifies the archive checksum, extracts the main
binary, and then attempts the missing tools member. `set -e` terminates before
either executable install, version/help checks, archive checksum marker or
`bootstrap.complete`. The empty redirected tools file and retained extracted
main binary identify this stage. This avoids changing connectivity, roles,
published content, accepted runners or guest state after provisioning.

After the separate lost-response reconciliation, ARM deployment/VM success may
leave production `phase: provisioned`; that state does not prove guest readiness.
The normal readiness command waits at most45seconds within its existing60second
ARM bound. For terminal cloud-init failure it exits1 before dispatch. Its managed
command may have successful ARM provisioning but `instanceView.executionState:
Failed` and nonzero exit. Production reconciliation must retain
`guestCommand.phase: failed`, no `guestReady`, no readiness receipt, no assessment
and no migration. Reconciliation is GET-only and does not repair/replay bootstrap.

Expected GUI text: “Guest readiness could not be verified. Retained evidence must
be reviewed.” Local source configuration may still be opened; do not claim its
button is disabled. The source panel's `canStart` is false without `guestReady`,
and backend readiness/dispatch gates prevent source effects. No source start,
credentials or actual assessment request is needed to prove this boundary.

## Independent guest observation, prepared but not submitted

[The observer](observe-b09-bootstrap-failure.py) takes **no arguments**, reads
only bounded fixed guest evidence, and emits one sanitized JSON line. Its exact
source SHA-256 is
`1439353225906bc653a849357f8b70346b59b395d9ea356f73329f36ed82f209`.

The root's private session directory contains:

- `observer-runcommand-body.json`: ready ARM body with location `japaneast`,
  `properties.source.script`, `timeoutInSeconds:180`, `asyncExecution:false`.
  The constant shell wrapper has `set -euo pipefail`, `set +x`, `umask 077`, then
  invokes `python3 -` with an inline base64 literal of the public observer source.
  It verifies the source SHA before executing it in memory. No guest file is
  written by the observer and there are no protected parameters or credentials.
- `observer-request-preparation.json`: exact proposed PUT and GET paths, script
  hash and maximum three independently approved observation names.

The first prepared path is the exact proposed new VM ID plus
`/runCommands/af-b09-observe-01?api-version=2024-07-01`. Reconciliation is GET of
the same ID plus `&$expand=instanceView`. Additional named observations, if the
approved scope permits and earlier output is inconclusive, are
`af-b09-observe-02` and `af-b09-observe-03`. Never replay a submitted unknown
observation blindly. Preserve its ID and reconcile it. This independent control
is separate from the single normal production readiness control and does not
produce a production readiness receipt.

Exit0 means this observer saw all of: terminal cloud-init `error` with nonzero
status, failed cloud-final service, exact archive checksum before tar parsing,
one expected archive member, intact extracted binary, empty missing-tools
destination, expected tar failure diagnostics, absent installed executables and
completion/version/checksum markers, and no processes executing either runner
binary. Exit2 means no credit. Pending, unfamiliar output, another early failure,
any bound exceeded, or inability to observe remains inconclusive. The receipt
does not emit raw status text, logs, process arguments, environments or tokens.
Its output is guarded to below4096 JSON bytes; normal complete output is far
smaller. The external request ledger must bind the receipt to the exact approved
VM/workflow; guest output alone is not ARM identity proof.

The observation command creates a managed Run Command resource even though its
guest work is read-only. Root must include these exact resources in the approval
and cleanup list. The observer does not authorize cloud creation, a running VM,
normal readiness, another bootstrap attempt or cleanup.

## Local verification and limits

- 10/10 Python tests: deterministic/preserved fixture bytes, existing-output
  refusal, digest drift, duplicate/extra/link rejection; observer negative
  controls for pending/unknown status, wrong hash before parser invocation,
  unrelated failure, completed installation and a running worker.
- 20/20 TypeScript tests across the new `runnerBootstrapFailure.test.ts` and
  existing `runnerGuest.test.ts`; actual local shell extraction fails at the
  missing member before installation, readiness exits on terminal status, failed
  command reconciliation stays unready, and assessment/dispatch attempts have
  no effects. The local shell uses inert bytes, relocated paths/download and a
  macOS checksum-option adapter; it is not Linux/cloud-init execution evidence.
- Extension typecheck passed. Python syntax and generated shell-wrapper syntax
  passed. Actual derivative archive/member hashes were separately read back.
- Independent agent review confirmed the failure stage and tightened the
  observer to parse only the already-read hash-verified archive bytes.

No real cloud-init failure, live observer output, GUI refusal or real-service
PASS is asserted here. Follow the supervised session's60minute/USD5 scope and
deallocation deadline if approved; normal package setup can stall and the15minute
download bound is not a bound on all cloud-init. Stop on any different failure
or missing evidence. Deallocation retains chargeable disk/storage; deletion must
cover only explicitly approved new resources, roles, controls and deployments.
