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
The preview adds no public IP or source firewall rule and collects no source
credentials. Source file selection does not upload CSV contents.

Version-2 runner records (resource IDs, source type/location, template, pinned
artifact checksum, costs and deployment phase) are saved in owner-only files
under extension global storage. No ARM token or source password is saved in
those records. Azure retains submitted templates and operation records; the VM
retains bootstrap evidence on its managed disk. Closing VS Code does not stop
the VM or its charges. This preview does not automatically delete evidence or
resources. Remote source assessment, CSV transfer and remote migration are not
yet enabled; they require a separate reviewed credential/transfer implementation.

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
