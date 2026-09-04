# Privacy and security

## Data handling

The AGEFreighter VS Code extension has no publisher-operated service and emits
no telemetry. It does not upload migration configurations, database data,
credentials, logs, reports, or usage events to the AGEFreighter maintainers.

The extension starts the locally installed AGEFreighter CLI. That CLI connects
only to sources and targets configured by the user's selected LoadJob and uses
the existing environment/file secret-reference mechanism.

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
