# Privacy and security

## Data handling

The AGEFreighter VS Code extension has no publisher-operated service and emits
no telemetry. It does not upload migration configurations, database data,
credentials, logs, reports, or usage events to the AGEFreighter maintainers.

The advanced existing-LoadJob workflow starts the locally installed AGEFreighter CLI. That CLI connects
only to sources and targets configured by the user's selected LoadJob and uses
the existing environment/file secret-reference mechanism.

The runner-first guided preview does not invoke a desktop CLI. It uses the
operator's existing VS Code Azure account to read ARM resource inventory, query
official Azure retail prices and submit an explicitly approved VM deployment.
ARM receives the selected placement and a generated VM/NIC/NSG template. GitHub
serves the matching public release checksum; the VM downloads and checks that
release. Those services see normal request metadata under their respective terms.
The preview adds no public IP or source firewall rule. Source file selection
does not upload CSV contents. Source settings and mappings are saved locally
before final output-folder selection; this does not deploy resources.

Version-2 runner records (resource IDs, source type/location, template, pinned
artifact checksum, costs, deployment phase, source host/database/username,
reviewed mapping/configuration, selected CSV paths and assessment manifests)
are saved in owner-only files
under extension global storage. No ARM token or source password is saved in
those records. macOS/Linux use owner-only directory/file modes. Windows uses
a current-user-only inheritable ACL on this dedicated workflow directory;
failure to set/verify that ACL blocks access. This uses the operating system's
PowerShell security commands, not a desktop AGEFreighter or Azure CLI.
Azure retains submitted templates and operation records; the VM
retains bootstrap evidence on its managed disk. Closing VS Code does not stop
the VM or its charges. This preview does not automatically delete evidence or
resources. Read-only source assessment is an explicitly approved action. For
Neo4j/PostgreSQL, the native VS Code password prompt supplies an ephemeral
password; it is not sent through the webview or saved in local workflow records.
The generated configuration uses environment references. A protected managed
Run Command parameter carries the configuration and secrets to the runner.
Cosmos uses the runner's managed identity, not an ARM token or desktop credential.

The guest retains the operation and configuration in private directories. It
deletes the transient credential handoff after normal finalization; crashes or
pre-execution failures can retain it for operator review. Bounded stderr remains
private on the guest and is not returned to the UI/model. Reports redact supplied
secret values and retain their hash. No source-form data or remote assessment
payload is passed to an AI tool. CSV/report transfer now require separate
approvals; remote migration and final verification are still not enabled.

The bulk-report GUI supports a
workflow-owned, non-anonymous Azure Blob destination with shared-key access
disabled. It uses separate short-lived create-only/read-only user-delegation
capabilities, never an ARM bearer token on a Blob request. Capability URLs are
not retained in local state, sent to the webview/model, or emitted in diagnostics;
the guest creation capability uses a protected Run Command parameter. Verified
report bytes are retained in owner-only local files without overwriting previous
evidence. Reports may still contain sensitive source metadata or sample values:
secret redaction is not anonymization. A native approval creates workflow-owned
Standard LRS storage and grants the signed-in user Blob Data Contributor on
that new account only. The endpoint is network-public authenticated HTTPS, not
private-endpoint isolation. Azure validates the SDK-issued delegation signature.
RBAC/network errors do not fall back to shared keys or another login.

CSV upload asks before sending complete file contents to that account. It uses
the existing storage-audience credential and content-addressed bounded blocks.
Explicit retries reconcile the same data without overwriting a committed blob.
The guest receives only a short-lived single-file read capability via protected
parameters, verifies all bytes, and publishes a private seal before profiling.
The capability is erased after normal finalization; failed partial files remain
as evidence, and crashes can retain private capability transport until review.

Developer qualification can explicitly opt in at user level to a reviewed
commit/hash-pinned Linux archive. The GUI uploads that archive to workflow
storage after approval. The subsequent VM preview includes a container-scoped
Blob Reader grant for its managed identity. Tokens are fetched inside the guest,
not embedded in cloud-init/command arguments. The developer manifest is an
assertion of provenance, not a signed attestation; production release behavior
is unchanged. This path is not a Marketplace or GitHub publication operation.

## Optional VS Code language model use

AI assistance is optional. When a user invokes the `@agefreighter` participant
or permits the `agefreighter_read` tool, VS Code sends a bounded context to the
chat model selected by the user under that model provider's terms and settings.

Before doing so, the extension:

- uses only JSON emitted by a documented read-only CLI command;
- recursively removes keys that indicate paths, connections, credentials,
  secrets, tokens, URIs, DSNs, or queries;
- truncates long strings and arrays;
- rejects the complete payload if its serialized size exceeds 256 KiB; and
- never includes raw LoadJob content, environment variables, stderr, terminal
  output, query text, or source records.

Every extension-provided language-model tool invocation displays VS Code's user
confirmation. The tool accepts only a discovered LoadJob inside the trusted
workspace and never invokes load, resume, cleanup, verification, or mutation.

## Local logs

The AGEFreighter output channel records operation names, local job paths,
timestamps, and locally redacted errors for the current VS Code session. These
logs remain in VS Code and are not transmitted by the extension. Terminal
output follows VS Code's normal local terminal behavior.

## Reporting vulnerabilities

Do not include credentials, connection strings, or private migration data in a
public issue. Follow the repository's
[security policy](https://github.com/rioriost/agefreighter/security/policy) for
private vulnerability reporting.
