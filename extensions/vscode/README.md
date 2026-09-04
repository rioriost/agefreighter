# AGEFreighter for Visual Studio Code

Plan, run, recover, and review validated graph migrations without moving
migration logic or credentials into an AI model.

AGEFreighter migrates CSV, PostgreSQL, Neo4j, and Azure Cosmos DB graph data to
Apache AGE or PostgreSQL 19 SQL/PGQ property graphs. This extension is a guided
interface for the separately installed AGEFreighter 2.3.0 CLI; the Go CLI
remains the deterministic migration engine and owner of durable checkpoints.

## Highlights

- Start a guided Neo4j migration by entering the source connection in VS Code;
  no hand-written LoadJob is required for source profiling.
- Reuse the Azure account already signed into VS Code, select a subscription,
  and verify an Azure source's region and logical zone from its ARM resource.
- Keep passwords in VS Code SecretStorage and owner-only extension storage;
  generated jobs contain secret references, not secret values.
- Discover AGEFreighter `LoadJob` YAML and JSON files in the workspace.
- Validate configuration and inspect static plans without connecting to a
  source or target.
- Run bounded source profiles and target readiness diagnostics.
- Start and resume migrations in a visible terminal.
- Review durable status, verification, migration reports, and optimization
  recommendations.
- Ask `@agefreighter` to explain bounded evidence using the chat model selected
  in VS Code.
- Expose one confirmed, read-only tool to VS Code agent mode. AI tools never
  start, resume, clean up, or mutate a migration.

## Prerequisites

Install [AGEFreighter from the Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=rioriost.agefreighter),
then install the AGEFreighter 2.3.0 CLI and ensure `agefreighter` is on `PATH`.
Alternatively, run **AGEFreighter: Select CLI Binary** and select the executable.

On macOS, the Homebrew installation is:

```sh
brew install rioriost/cask/agefreighter
```

Linux and Windows release archives are available from the
[AGEFreighter releases](https://github.com/rioriost/agefreighter/releases).
Windows CLI binaries in 2.3.0 are provided without an Authenticode signature;
verify their checksum and GitHub build-provenance attestation before use.

## Start a migration

For the 2.4.0 guided path, open the AGEFreighter view and select **New guided
migration**. Enter the Neo4j endpoint and discovery identity properties, select
the Azure subscription, and identify whether the source is an Azure resource or
is on-premises. The extension validates the source and presents bounded sizing
evidence without requiring a LoadJob first.

Azure deployment remains gated until the profile has a reviewed total, current
region/zone/SKU/quota checks, a cost estimate, and an ARM what-if review. A
bounded lower limit is never presented as a complete capacity estimate.

The existing LoadJob path remains available:

1. Open a trusted workspace containing an AGEFreighter `LoadJob` file.
2. Open the AGEFreighter activity-bar view.
3. Expand a discovered job and run **Validate**, then **Static plan**.
4. Use **Profile source** and **Diagnose target** when their connection cost is
   acceptable.
5. Select **Start migration**, inspect the confirmation, and continue in the
   visible terminal.
6. Keep the UUID printed by `load`. Status, resume, verify, report, and cleanup
   require that durable job ID.

Long-running commands are intentionally opened as terminal processes. Reloading
the extension does not silently resume or create another migration. Use durable
status and the CLI's reviewed `resume` procedure after a failure.

## AI assistance

The extension works without GitHub Copilot or another chat model. If VS Code
chat is available, use `@agefreighter /help` or one of the read-only slash
commands. The selected model may explain evidence and recommend a next step,
but all execution remains in the CLI and all mutations require a direct modal
confirmation outside chat.

Only bounded, recursively redacted CLI JSON is eligible for model context. Raw
job files, environment variables, connection strings, credentials, credential
reference names, queries, source records, stderr, and terminal logs are never
sent by this extension. See [Privacy and security](PRIVACY.md).

## Settings

| Setting | Default | Meaning |
|---|---:|---|
| `agefreighter.binaryPath` | `agefreighter` | Executable path or command name |
| `agefreighter.readTimeoutSeconds` | `120` | Timeout for captured read-only commands |
| `agefreighter.maxOutputBytes` | `4194304` | Per-stream output capture limit |

## Workspace trust and remote development

Job discovery works in restricted mode. No AGEFreighter process runs until the
workspace is trusted. The extension runs where the workspace extension host
runs, so Remote SSH, Dev Containers, and Codespaces need the AGEFreighter CLI
installed in that remote environment.

Virtual workspaces are not supported because the CLI requires filesystem paths.

## Support

- [AGEFreighter documentation](https://github.com/rioriost/agefreighter/tree/main/docs)
- [Configuration reference](https://github.com/rioriost/agefreighter/blob/main/docs/reference/configuration.md)
- [Operations guide](https://github.com/rioriost/agefreighter/blob/main/docs/reference/operations.md)
- [Report a problem](https://github.com/rioriost/agefreighter/issues)

AGEFreighter is open source under the MIT License.
