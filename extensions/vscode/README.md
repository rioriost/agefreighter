# AGEFreighter for Visual Studio Code

Plan, run, recover, and review validated graph migrations without moving
migration logic or credentials into an AI model.

AGEFreighter migrates CSV, PostgreSQL, Neo4j, and Azure Cosmos DB graph data to
Apache AGE or PostgreSQL 19 SQL/PGQ property graphs. This extension is a guided
interface for the separately installed AGEFreighter CLI; the Go CLI
remains the deterministic migration engine and owner of durable checkpoints.

## Highlights

- Start a guided Neo4j migration by entering the source connection in VS Code;
  no hand-written LoadJob is required for source profiling.
- Reuse the Azure account already signed into VS Code, select a subscription,
  and verify an Azure source's region and logical zone from its ARM resource.
- Check current PostgreSQL 18 capabilities, zonal VM SKU availability, service
  quotas, and bounded USD retail rates before saving an Azure proposal.
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

Use VS Code 1.105 or newer, open a trusted workspace folder, and sign in to
Azure in VS Code. Azure Resources and AGEFreighter have separate account-access
permissions: on first use, open VS Code's **Accounts** menu (profile icon) and
approve the AGEFreighter request to use the existing Azure account, then click
**Refresh Azure access** in the wizard. This does not automatically start another
Azure login. After signing in or changing account/subscription filters, use the
same refresh button; subscription lookup failures are not treated as sign-out.
Select an AGEFreighter 2.4.0 CLI using **AGEFreighter: Select
CLI Binary**, or put it on `PATH`. The guided form uses its new `inventory`
command. A released 2.3.0 CLI supports the existing LoadJob commands only.

This is a 2.4.0 development package. Use the CLI built from the matching
development branch until 2.4.0 is released. Installing the extension does not
install or upgrade the CLI.

For the latest released CLI on macOS, the Homebrew installation is:

```sh
brew install rioriost/cask/agefreighter
```

Linux and Windows release archives are available from the
[AGEFreighter releases](https://github.com/rioriost/agefreighter/releases).
Windows CLI binaries are provided without an Authenticode signature;
verify their checksum and GitHub build-provenance attestation before use.

## Start a new migration in 2.4.0

1. Open the AGEFreighter view and choose **+ / New Guided Migration**, or run
   **AGEFreighter: New Guided Migration** from the Command Palette. You can
   start with an empty workspace; no LoadJob file is needed.
2. Enter the Neo4j host, port, database, username, password, and the properties
   that identify nodes and relationships. Select the target Azure subscription.
3. For an Azure source, provide its ARM resource ID to verify its region and
   logical availability zone. For on-premises or another cloud, enter its
   physical location and confirm or change the proposed Azure region.
4. Select **Connect and profile source**. AGEFreighter validates the generated
   draft, reads exact node and relationship totals, and estimates storage from
   a bounded profile.
5. Review the proposed PostgreSQL 18 Flexible Server and AGEFreighter VM,
   including region, common zone, capacity, quota availability, and the retail
   compute estimate. Resolve any reported blockers before proceeding.

The guided path reads the selected subscription's current region, zone, SKU,
and quota metadata, writes an Azure proposal that expires after 24 hours, and
shows a bounded retail compute estimate when available. Drafts and evidence
are saved under `.agefreighter/guided/`; passwords are stored separately.

**Current development-build limit:** the wizard ends at the Azure proposal.
The planned next steps—review and deploy Azure resources, start the migration
after readiness checks, and verify the completed migration—are not implemented
in this build. The wizard does not create Azure resources or start a migration.
Its generated draft is not yet an executable migration job.

## Existing LoadJob workflow (advanced)

Use this path if you already have a configured source and target with a complete
LoadJob. It remains available for all supported connectors:

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
