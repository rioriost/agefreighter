# R3 Linux assessment boundary

Status: implemented local foundation and source-form controller, **not
Azure-qualified**. This is not complete R3 upload/report transfer or the R4/R5
migration flow.

## Transport and durable execution

The extension submits a unique managed Run Command on the exact retained runner
VM. Its script is constant; a bounded JSON request is encoded into the protected
`AF_RUNNER_REQUEST` parameter, never ordinary parameters, command-line arguments,
webview messages, or local workflow state. The extension persists the command ID
before PUT. Ambiguous responses remain `unknown`; reconciliation is GET-only.
Do not delete a pending managed Run Command to make retry possible.

The guest accepts only protocol version 1 and UUID workflow/operation IDs through
`agefreighter-tools runner dispatch`. The allowlist is `ready`, `profile`,
`inventory`, `status`, and `report`. There is no shell, load, resume, cleanup,
provisioning, or arbitrary executable action in this protocol.

Before assessment, readiness checks the bootstrap marker, retained archive hash,
matching CLI/tools version, Linux x64 architecture and boot identity. The client
requires a check no older than five minutes, binds the request to that boot ID,
and does not renew validity when re-reading an old ARM result. The guest checks
the boot ID before creating an operation. This readiness is not source
connectivity, schema compatibility, a fresh cost check, or migration approval.

Under `/var/lib/agefreighter/workflows/<workflow>/<operation>`, the guest records
an accepted intent, normalized configuration hash, private credential handoff
and a persistent systemd unit before requesting its start. The workflow has an
exclusive active-operation lease. A permanent worker claim prevents replay even
if the unit is restarted. Units have no boot-install target and `Restart=no`;
a changed boot ID makes an unfinished operation `interrupted`, not resumable.
Crash/uncertain-start leases require explicit operator reconciliation; an
automatic repair or lease-deletion operation is deliberately not provided yet.

## Execution and evidence limits

- `profile`: actual CLI sampled profile, 10,000-row sample. It is not an exact
  inventory and must not automatically determine production capacity.
- `inventory`: existing exact Neo4j count-store inventory only. Other sources
  cannot use the generic bounded profile as a substitute for exact totals.
- Systemd: 30-minute runtime, 4 GiB memory ceiling, no swap, 200% CPU quota,
  private temporary directory and read-only system filesystem with only the
  workflow directory writable. No automatic restart or reboot resume.
- Requests: 1 MiB. Reports: 4 MiB. Stderr: 64 KiB, retained privately and never
  returned through the protocol. Environment variables are an explicit allowlist;
  desktop Azure tokens and unrelated process environment are not inherited.
- Source/target credentials in configuration are approved environment handles,
  not embedded values or arbitrary file references. Temporary credential files
  are removed after a normally finalized terminal operation. A pre-execution
  failure/crash can retain the private handoff for operator reconciliation;
  do not claim that credentials are erased after every failure.
- CSV paths must remain in this workflow's upload directory, including after
  symlink resolution. Upload ownership/hash sealing still needs its own R3 path.
- Terminal reports preserve JSON number precision, redact supplied secret values,
  and retain byte length and SHA-256. A `finished` worker means a valid report was
  produced, **not** that its assessment/verification outcome passed.

The report control supports 1,536-byte chunks with operation ID, offset, total
length and full SHA-256. Client assembly checks all chunks and the full hash.
This is a bounded diagnostic fallback, not a practical bulk transport: a 4 MiB
report would require 2,731 control calls. Implement reviewed private artifact
storage/transfer before GUI P1 qualification; do not use this path for CSV files.
Managed Run Command instance-view output has a 4 KB limit; ARM provisioning
success is separate from script execution success.
[Microsoft documentation](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/run-command-managed).

## Tests and remaining gates

The tests invoke the actual CSV profiling implementation in a child process,
check all report chunks, and exercise duplicate dispatch, stale boot identity,
lost start acknowledgement, unreconciled leases, worker replay, configuration
tampering, CSV symlink escape and report tampering. Transport tests confirm
protected parameters, persist-before-PUT, GET-only reconciliation, freshness
and independently verified artifact assembly.

Linux x64 execution under Apple Container/Rosetta and `systemd-analyze verify`
are separate checks. The service manager itself, Azure VM agent transport,
memory/runtime enforcement, actual guest reboot and private-source connectivity
still require live Azure testing. No mocked/local result counts toward the nine
P1 GUI paths.

The four source forms now create secret-reference-only drafts without requiring
a released artifact or VM creation. Network profiles and Neo4j inventory are
connected to explicit modal approval, native secret prompts and the protected
dispatch/status boundary. Successful operation manifests are preserved in a
bounded history before another approved assessment. Failed/unknown operations
cannot be replaced by a new start. No result is automatically accepted as sizing
or migration success, and the full report is not yet imported into the GUI.

Next gates: immutable development artifact; schema/FK recommendations; private
bulk upload/report transport; exact PG/Cosmos/CSV totals with explicit scan/RU
bounds; reviewed custom CA installation for private sources; then R4 target
deployment/resize and R5 durable migration/verification. CSV execution and target
mutations remain disabled pending these gates.
