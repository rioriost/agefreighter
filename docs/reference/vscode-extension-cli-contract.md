# VS Code extension CLI contract

The AGEFreighter VS Code extension invokes the separately installed
`agefreighter` binary. The Go CLI remains the only migration engine and the
only owner of job identity, checkpoints, source and target connections, and
graph mutations.

## Process boundary

- Pass the executable and arguments as an argument array without a shell.
- Resolve the executable from `agefreighter.binaryPath`, then `PATH`.
- Set the working directory to the job file's parent directory.
- Do not read, copy, display, log, or send the process environment to a model.
- Treat stdout as UTF-8 JSON for commands documented below.
- Treat a non-zero exit or invalid/oversized JSON as a failed operation.
- Ignore unknown JSON object members for forward compatibility.
- Cap captured stdout and stderr at 4 MiB each and terminate read-only commands
  after the configured timeout.
- Launch long-running or mutating commands in a user-visible VS Code terminal.

## Read-only JSON commands

| Purpose | Arguments | Top-level contract |
|---|---|---|
| Validate configuration | `validate --format json JOB` | validation result, schema version 1 |
| Static plan | `plan JOB` | `apiVersion`, `job`, `source`, `target`, `limits`, `policies` |
| Source profile | `profile --format json JOB` | report document |
| Target readiness | `doctor --format json --target JOB` | report document |
| Durable status | `status --target JOB JOB_ID` | durable job record |
| Migration report | `report --format json --target JOB JOB_ID` | report document |
| Optimization advice | `optimize --format json --target JOB` | report document |

Some read-only commands connect to a configured source or target and can incur
service load. The extension labels this distinction and asks before expensive
or deep checks.

## Validation result

```json
{
  "schemaVersion": 1,
  "valid": true,
  "apiVersion": "agefreighter.io/v2",
  "kind": "LoadJob",
  "job": "migration-name",
  "source": { "type": "neo4j" },
  "target": { "type": "apache-age", "mode": "create" }
}
```

The result deliberately excludes paths, connection strings, query text,
credentials, and credential-reference names.

## Mutating and long-running commands

The extension requires direct modal confirmation and then launches these in a
terminal so the exact command remains visible:

- `load JOB`
- `resume --job JOB JOB_ID`
- `cleanup --target JOB JOB_ID`
- `optimize --apply-analyze --target JOB`

`verify` and `report --include-counts` do not mutate graph data but can perform
large target scans; they use the same confirmation and terminal path. Model
tools never invoke any command in this section.

## Redaction boundary for AI

Only bounded JSON produced by the read-only commands is eligible for model
context. Before use, the extension recursively removes keys whose names contain
`password`, `secret`, `token`, `credential`, `connection`, `dsn`, `uri`,
`query`, or `path`, truncates long strings and arrays, and enforces a total
serialized size limit. Raw job files, environment variables, stderr, terminal
output, and source records are never provided to a model.

