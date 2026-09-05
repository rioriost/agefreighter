# R3 Linux assessment boundary

Status: implemented local foundation and source-form controller, **not
Azure-qualified**. Storage, report and CSV transfer are connected locally; this is not complete R3 qualification or the R4/R5
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
`inventory`, `status`, `report`, `export-report` and `import-csv`. There is no shell, load, resume, cleanup,
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
  symlink resolution. Each mapped file needs a matching guest seal; the full file
  is rehashed before profiling, not merely trusted because a marker exists.
- Terminal reports preserve JSON number precision, redact supplied secret values,
  and retain byte length and SHA-256. A `finished` worker means a valid report was
  produced, **not** that its assessment/verification outcome passed.

The report control supports 1,536-byte chunks with operation ID, offset, total
length and full SHA-256. Client assembly checks all chunks and the full hash.
This is a bounded diagnostic fallback, not a practical bulk transport: a 4 MiB
report would require 2,731 control calls. The bulk report protocol below avoids
that fan-out. Reviewed storage provisioning/capability issuance and GUI wiring
are now implemented but still need live qualification; do not use diagnostic chunks for CSV.
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
or migration success. Full reports are imported into a script-disabled GUI viewer
only after complete byte/hash verification and private immutable retention.

Next gates: live validation of the pinned development artifact and authenticated
bulk transport; schema/FK recommendations; exact PG/Cosmos/CSV totals with explicit scan/RU
bounds; reviewed custom CA installation for private sources; then R4 target
deployment/resize and R5 durable migration/verification. CSV sampling requires
every mapped upload seal; target mutations remain disabled.

## Bulk report transfer implementation checkpoint

The guest and controller now implement a complete-report data path (up to 4 MiB),
separate from Run Command's short acknowledgement. This path is now exposed in
the wizard with separate storage/RBAC/network approval and SDK-issued
user-delegation capabilities from the existing VS Code account.
No arbitrary SAS input is exposed. The subsequent installed-GUI storage
deployment succeeded, but enforced organizational policy initially disabled its
public network access. The user then authorized the policy-defined official
development exception on that owned storage account only. An authenticated CSV
upload/readback passed; GUI qualification is tracked in [progress](progress.md).

Provisioning success and transfer readiness are distinct. Reconciliation shows
the actual storage network setting. The current desktop transfer implementation
stops before capability issuance or transfer for `Disabled` or
`SecuredByPerimeter` public access. Supporting an approved private route requires
an explicit connectivity design and qualification. The extension never grants
policy exceptions or enables public access as an automatic repair; an existing
organization-approved exception requires explicit operator authorization outside
this generic workflow. A network failure is not a reason to sign in again.

Storage requests must obtain a Storage-scoped session for the selected existing
account. The installed azureauth subscription credential captures the ARM token
and ignores `getToken(scopes)`; it must not be forwarded to the Blob endpoint.
CSV and archive operations also send UTC `x-ms-date`, as required by
[Get Blob Properties](https://learn.microsoft.com/en-us/rest/api/storageservices/get-blob-properties).

1. The controller binds a terminal assessment's independent byte length/SHA-256
   to the retained workflow/operation. It checks the deterministic storage account
   in the same subscription/RG/region through ARM: workflow ownership tags,
   completed provisioning, HTTPS, TLS 1.2 or higher, disabled anonymous Blob access
   and disabled shared-key access. The exact workflow container must explicitly
   return `publicAccess: None`. These are read-only checks; no policy is changed.
2. The existing-account credential provider supplies two separate, ephemeral
   **user-delegation** capabilities: `c` for guest creation, `r` for desktop reads.
   Both must be HTTPS-only and restricted to exactly
   `af-<workflow>/reports/<operation>.json`, not a container. Expiry is at most
   15 minutes ahead with at most five minutes of start-time skew. Unknown query
   fields, duplicates, alternate hosts/ports, redirects and broader rights fail
   closed. SAS validation is a client policy check; Azure validates the signature.
3. Export intent (destination **without** credentials, operation, bytes, hash) is
   persisted before PUT. The create capability goes only through the protected
   Run Command parameter. The guest rechecks the local report before one bounded
   conditional `Put Blob` (`If-None-Match: *`). It does not overwrite, retry,
   re-profile, change the source operation, or include service errors/URLs in output.
4. A lost response is reconciled by GET, never by repeating the export. The
   controller can import a matching destination even when the acknowledgement
   was lost, but cannot use this to clear an uncertain ARM command. A missing or
   rejected export remains unresolved; automatic retry/repair is not implemented.
5. One bounded authenticated GET reads at most the independently expected bytes,
   checks UTF-8/JSON plus full SHA-256, and atomically retains the original JSON
   bytes in private extension storage. Re-import accepts identical evidence;
   changed existing evidence is never overwritten. Parsing and re-serializing
   would lose some int64 values, so the retained bytes are not reconstructed from
   JavaScript numbers. An imported report is **not** a sizing or migration pass.

Storage here is non-anonymous, authenticated Blob storage; it is not a claim of
private-endpoint network isolation. Endpoint reachability, network restrictions,
least-privilege role grants and storage costs require the separate approved
topology before transfer. Real Azure/RBAC propagation tests remain required.

## CSV and development artifact checkpoint

- Desktop CSV inventory streams SHA-256, then asks before uploading contents.
  The workflow has a 10 GiB aggregate bound and 2 GiB/file bound. Upload uses
  8 MiB blocks and the existing storage-audience credential; no account keys or
  container SAS. Blob names include file UUID and digest. Every explicit retry
  rechecks local bytes, and commits use `If-None-Match: *`. A matching remote HEAD
  is only upload reconciliation, never a substitute for guest content verification.
- `import-csv` carries a ten-minute read-only single-blob delegation capability
  through protected parameters. The same exclusive workflow lease, boot check,
  persistent disabled-at-boot unit and permanent worker claim apply. The guest
  checks projected disk use below 80%, reads at most expected size + 1, compares
  full SHA-256, and publishes without replacing any destination. A private seal
  is required before assessment. Failed partial bytes are retained, while normal
  terminal finalization erases the transient capability. Crashes require manual
  reconciliation; there is no automatic lease repair or import replay.
- Import is currently approved per file. The GUI's status control never silently
  starts the next file. Automatic batch queueing/cancellation/recovery needs more
  implementation before low-interaction P1 qualification.
- For unpublished test binaries only, user-level opt-in enables a reviewed
  local manifest/archive. `build-development-runner.sh` builds a clean committed
  snapshot for Linux amd64. The GUI checks full archive size/hash, uploads the
  content-addressed archive and previews a container-scoped VM Blob Reader role.
  Managed-identity download is bounded; hashes are checked before extracting only
  the two expected binaries. Guest readiness checks version and commit. The
  manifest is not a signed provenance attestation; only a reviewed build is
  eligible. Published production release checks are unchanged.

Reviewed API basis: [Put Blob](https://learn.microsoft.com/en-us/rest/api/storageservices/put-blob),
[conditional headers](https://learn.microsoft.com/en-us/rest/api/storageservices/specifying-conditional-headers-for-blob-service-operations),
and [user delegation SAS](https://learn.microsoft.com/en-us/rest/api/storageservices/create-user-delegation-sas).

Local tests exercise exact bytes, int64/Unicode, strict capability scope/lifetime,
one conditional PUT, redirects, ambiguous responses, changed manifests/files,
symlinks, truncated/oversized responses, atomic no-replace import, private ACLs,
and foreign/public/shared-key-enabled storage rejection. They are not live Azure
RBAC, SAS signature or VM-agent qualification.
