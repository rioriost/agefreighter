# AGEFreighter for Visual Studio Code

Plan, run, recover, and review validated graph migrations without moving
migration logic or credentials into an AI model.

AGEFreighter migrates CSV, PostgreSQL, Neo4j, and Azure Cosmos DB graph data to
Apache AGE or PostgreSQL 19 SQL/PGQ property graphs. This extension is a guided
interface to the deterministic Go engine and its durable checkpoints. The new
guided path is being moved to a dedicated Linux Azure VM; advanced existing
LoadJob commands continue to use a separately installed local CLI.

## Highlights

- Select Neo4j, PostgreSQL, Cosmos DB for NoSQL or CSV in the runner-first wizard;
  no desktop CLI or project-folder selection is required to open it.
- Reuse the Azure account already signed into VS Code, select a subscription,
  and verify an Azure source's region and logical zone from its ARM resource.
- Preview a private Linux discovery/migration VM in an existing compute subnet;
  check zonal SKU availability, quotas, pinned release checksums and compute cost.
- Require a modal approval and fresh ARM what-if before runner creation. Retain
  deployment IDs and reconcile unknown status without replaying a create.
- Configure all four source types using fields and vertex/edge mappings, without
  supplying a LoadJob file. Save an owner-only local draft before VM creation.
- Approve a sampled remote profile (Neo4j/PostgreSQL/Cosmos) or Neo4j inventory
  after guest readiness. Passwords use a native private prompt and protected
  dispatch; no password is saved with the form. CSV execution awaits upload.
- Check the provisioned Linux guest's installation and boot identity, and
  reconcile the protected control request without replay after communication loss.
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

Use VS Code 1.105 or newer and sign in to Azure in VS Code.
Azure Resources and AGEFreighter have separate account-access
permissions: on first use, open VS Code's **Accounts** menu (profile icon) and
approve the AGEFreighter request to use the existing Azure account, then click
**Refresh Azure account** in the wizard. This does not automatically start another
Azure login. After signing in or changing account/subscription filters, use the
same refresh button; subscription lookup failures are not treated as sign-out.
The guided path does not invoke a desktop CLI. Runner deployment requires a
published matching 2.4.x Linux release with its checksum; an unpublished release
blocks deployment instead of falling back to an incompatible older binary.
Azure subscription permissions must allow the reviewed VM/NIC/NSG deployment.
Use an existing non-delegated compute subnet with source connectivity, private
DNS and outbound access for Azure VM agent services and release installation.
The wizard adds no public IP, SSH ingress, source firewall rule or peering.
VS Code workspace trust is required for deployment, but opening the wizard does
not require an output folder. The final flow will choose that folder only after
target review. CSV files can be selected earlier without upload.

**Advanced local LoadJob commands only:** select a CLI 2.3.0 or newer with
**AGEFreighter: Select CLI Binary**, or put it on `PATH`. Installing the extension
does not install or upgrade the desktop CLI.

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
2. Select the source type first. Neo4j/PostgreSQL support Azure, on-premises or
   another cloud; Cosmos uses Azure; CSV uses local files. For Azure, select the
   source subscription/RG and discover candidates. A VM is only a candidate,
   not proof of an installed database. No desktop database probe is performed.
3. Review the runner subscription, existing RG/subnet, region, zone and small
   Burstable SKU (or the listed non-Burstable alternatives). Source region/zone
   equality is checked when available; cross-subscription physical-zone mapping
   is not implemented and blocks deployment. Cosmos uses actual data regions,
   not the account metadata location. On-premises region choice is manual.
   **Migration resource group** and **Azure region** are subscription-backed
   dropdowns. The selected migration RG is the common placement for the runner
   and the later Flexible Server target, independent of the source RG. New RGs
   must currently be created in Azure first, then loaded with **Refresh resource
   groups & regions**. RG boundaries do not require VNet peering: a VNet may be
   in another RG in the same subscription. Network connectivity is checked
   separately. Listing a region does not guarantee VM/service capacity there.
4. Optionally select **Configure source & assessment** after entering the runner
   placement fields, before creating any VM. This saves a local draft and opens
   the selected source's form. PostgreSQL table/column mappings and Cosmos
   explicit/Gremlin formats are supported; schema/FK suggestions are still pending.
   CSV maps selected files, stable IDs, endpoints, property types and a null marker.
   Review the generated configuration; it is not the final exported LoadJob.
   Then select **Check prerequisites & preview runner**. Review the immutable resource
   identities, version/checksum, compute cost and additional charges. Approve the
   network prerequisites and costs, then confirm **Approve & deploy discovery VM**.
   The 15-minute preview must still match and a fresh what-if must show only new,
   expected resources. No existing resource is overwritten.
5. **Refresh deployment status** or **Reconnect to a saved workflow** after a
   reload. Closing VS Code does not cancel Azure deployment or stop charges.
   Unknown results are reconciled by ID; they are not resubmitted automatically.
6. Once the VM is provisioned, use **Check Linux guest readiness**, followed by
   **Refresh guest command**. This checks the matching installation and boot
   identity, not source connectivity or migration readiness. Reopen **Configure
   source & assessment**, review the settings and approve sampled source reads
   or exact Neo4j inventory. A native password prompt follows approval where
   needed. **Refresh assessment status** submits/reconciles one bounded status
   check without repeating the source operation. Successful terminal manifests
   remain in the workflow history when a subsequent assessment is approved.

Workflow metadata is held in extension global storage, without source passwords,
before output-folder selection. The VM uses persistent managed OS storage and
has no public IP. Evidence/disks are retained; this preview has no automatic
cleanup, stop or delete action. Operators remain responsible for resource costs.

**Current development-build limit:** source forms, runner provisioning and
protected remote assessment controls are implemented, but not live-Azure
qualified. ARM success is not guest readiness. A finished assessment worker is
not a passing migration. Bulk report retrieval, CSV upload, automatic schema/FK
recommendations, exact non-Neo4j inventories, accepted target sizing/deployment,
same-VM resize, final LoadJob export, remote migration and verification remain
open. Publicly trusted TLS is currently required; custom source CA upload is
not implemented. Do not publish this as a complete guided migration workflow.
The [runner-first plan](https://github.com/rioriost/agefreighter/blob/codex/2.4.0-guided-migration/docs/design/agefreighter-2.4.0-runner-first.md)
tracks the remaining gates.

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
