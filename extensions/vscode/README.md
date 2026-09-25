# AGEFreighter for Visual Studio Code

Plan, run, recover, and review validated graph migrations without moving
migration logic or credentials into an AI model.

AGEFreighter migrates CSV, PostgreSQL, Neo4j, and Azure Cosmos DB graph data to
Apache AGE or PostgreSQL 19 SQL/PGQ property graphs. This extension is a guided
interface to the deterministic Go engine and its durable checkpoints. The
guided path runs on a dedicated Linux Azure VM; advanced existing LoadJob
commands continue to use a separately installed local CLI. Guided target
provisioning creates private PostgreSQL 18/Apache AGE servers. PostgreSQL 19
SQL/PGQ remains a separate advanced LoadJob/CLI path.

The development engine includes the 2.3.1 PostgreSQL native floating-point fix.
Old PostgreSQL checkpoints are not replayed with the changed fingerprint;
retain failed-run evidence and use a fresh job and target for corrective tests.

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
- Approve a sampled remote profile or complete Neo4j/PostgreSQL/Cosmos/CSV inventory
  after guest readiness. Passwords use a native private prompt and protected
  dispatch; no password is saved with the form. CSV requires a full-hash guest seal.
- Prepare workflow-owned transfer storage with explicit account-scoped user
  data permissions. Upload CSV in bounded blocks, import/seal files on the VM,
  and retrieve full hash-verified assessment reports without shared keys.
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

### Fewer interactive waits in guided qualification

- Before starting compute, use **AGEFreighter: Prepare, Replace or Forget Source
  Credential** after reviewing source settings. Choose **Use once**, or opt into
  encrypted VS Code SecretStorage reuse for this workflow, connection, account
  and CA. Reuse expires after eight hours or an earlier target deadline; changed
  connections and retained failures require new entry. The encrypted entry is
  removed on next access after expiry, or immediately with **Forget**. No password
  is stored in the form, workflow JSON, LoadJob, logs or model context.
- Target inputs are saved after each field and survive cancellation, reload and
  preflight failure. Reuse or edit them; no deadline is automatically extended.
  **Save plan only** does not require a running VM. A deployment still requires
  review, fresh readiness and live ownership/network/quota/budget checks after
  input is complete. Saved inputs are not deployment authorization.
- Active source operations are monitored while their panel is open, including
  after reconnect. Migration execution watches the retained job automatically.
  Polling is bounded to 30 minutes/15 status steps; failures stop the watcher.
  Cancel stops watching, not the Linux job. No migration is automatically retried
  or resumed. Existing receipt-capacity and service time limits still apply.
- A single approval for a specific sealed report covers export, bounded polling
  and hash-verified import. Reopen a pending transfer without repeating export or
  its approval. Report import is not full migration qualification.
- The native same-VM resize approval now covers deallocation, resize and restart
  of that exact idle runner, for at most 20 minutes or the earlier target deadline.
  A retained, unchanged grant can continue after reconnect; uncertainty never
  causes replay. Migration, new resources and verifier installation remain
  separately reviewed operations.

MFA and native OS permissions cannot be pre-approved by this extension. Prepare
credentials and review plans while resources are stopped where possible. Installed
GUI/live qualification remains distinct from automated regression tests.

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
For isolated developer qualification only, an explicit user-level
`agefreighter.allowDevelopmentRunnerArtifacts` opt-in enables **Prepare Pinned
Development Runner (Qualification Only)**. It accepts a reviewed local manifest
and Linux archive with matching commit/version/size/SHA-256, uploads to owned
storage, and grants the eventual VM identity Blob Reader only on that workflow
container. It neither publishes a release nor builds mutable source on a VM.
The manifest is a developer assertion, not a signed build attestation.
Azure subscription permissions must allow the reviewed VM/NIC/NSG deployment.
Use an existing non-delegated compute subnet with source connectivity, private
DNS and outbound access for Azure VM agent services and release installation.
The wizard adds no public IP, SSH ingress, source firewall rule or peering.
VS Code workspace trust is required for deployment, but opening the wizard does
not require an output folder. The final flow will choose that folder only after
target review. CSV files can be selected earlier without upload.

**Advanced local LoadJob commands only:** CLI 2.3.1 is recommended; select it with
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
   explicit/Gremlin-shaped NoSQL formats are supported. After guest readiness,
   PostgreSQL catalog discovery can propose mappings and supported FK edges for
   explicit review; other relationships require manual mapping.
   Known Cosmos property types can be declared as `score=score:float64` (or the
   other supported scalar/array types). Undeclared fields retain JSON inference;
   declarations require an updated Linux runner and a fresh job if changed.
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
   or a complete source inventory. A native password prompt follows approval where
   needed. **Refresh assessment status** submits/reconciles one bounded status
   check without repeating the source operation. Successful terminal manifests
   remain in the workflow history when a subsequent assessment is approved.
   Neo4j and PostgreSQL sources using a private CA can select a certificate-only
   PEM bundle. Its bytes are never put in the generated LoadJob or webview; the
   bundle is hash-bound to review, rechecked before each operation and sent to
   Linux only through the protected command channel. Hostname verification stays
   enabled. For Cosmos, separately choose **Grant / verify Cosmos Data Reader**
   after readiness. This creates and GET-verifies one account-scoped built-in
   Data Reader assignment for the owned runner identity; it never grants writes,
   uses account keys, or starts source reads.
7. In the source editor, **Prepare / refresh transfer storage** has its own
   network/cost/RBAC approval. It creates a new Standard LRS account, disables
   anonymous/shared-key access, and grants your signed-in user Storage Blob Data
   Contributor on that account only. Its HTTPS endpoint is network-public, not a
   private endpoint. RBAC propagation and network access can delay readiness.
8. For CSV, **Upload reviewed CSV files** hashes the selected data and asks before
   sending contents to Azure. Maximum 2 GiB/file and 10 GiB/workflow. Retrying an
   interrupted desktop upload reconciles the same content-addressed blocks and
   never overwrites a committed blob. **Import next CSV / refresh import** starts
   one approved VM download or checks its retained status. Import requires fresh
   guest readiness and space below 80%, and seals only a full size/SHA-256 match.
   Review mappings again when all mapped files are verified; only then can the
   sampled CSV assessment start. A matching guest advertising `csv-inventory-v1`
   also enables **Approve complete source inventory**: a full typed scan capped
   at 64 files, 10 GiB, 100 million mapped records and 30 minutes (or a shorter
   configured timeout). Older guests cannot run this operation. Exact mapped
   counts are not proof of unique identities, valid endpoints or a completed
   migration. Capacity estimates still require target/cost review; scaling a
   prefix sample by exact totals does not make it deployable.
   Failed/reboot-interrupted guest imports require
   evidence review; automatic lease repair/restart is not implemented.
9. **Transfer / open verified report** first exports the terminal report, then
   reconciles and imports it on a subsequent click. It retains original JSON
   bytes (including int64 values) in private extension storage and opens a
   script-disabled escaped viewer. A report import is not migration approval.
10. For a complete, imported source inventory, **Review / reconcile private
    migration target** opens native fields for a new PostgreSQL 18/AGE server, non-overlapping
    delegated subnet, target storage, same-VM migration size, authorized deadline
    and cost reserve. It rechecks private placement, service/SKU and both quotas,
    and unique live Linux/PostgreSQL prices. Review the single-server/no-HA trial
    configuration and additional retained-resource costs; then select a folder
    for the secret-reference-only LoadJob and target plan. Save-only performs no
    deployment. Separately approved deployment is create-only and uses generated
    credentials in VS Code SecretStorage/ARM secure parameters. Reopen the control
    to reconcile an uncertain submission; it does not replay it. The existing VNet
    may be in a separate network resource group; only the reviewed new delegated subnet is deployed there, with
    permission checked in both groups. Target creation alone does
    not resize the VM, prepare AGE or start/verify migration.
    Target database/configuration writes are serialized. If the retained
    deployment failed **only** on `shared_preload_libraries` with `ServerIsBusy`,
    reopen target review for a separately approved, one-setting repair. It
    requires all other resource operations to have succeeded, unchanged private
    placement/ownership and the unchanged default preload value. It preserves
    the failed deployment and existing resources; uncertain repair responses
    are read-only reconciled, never automatically retried. Other failures or
    custom settings require operator review. Any required restart stays separate.
11. **Continue / verify Linux migration** requires a matching migration-capable
    guest and complete inventory. Separately approve the AGE preload restart
    and an idle same-VM resize. The resize approval covers deallocation, size
    change and restart within its retained deadline. Unknown responses are
    reconciled without replay; NIC, identity and managed disk bindings are checked.
    An active inventory or migration blocks resize before a resize intent.
12. Separately approve a new create-mode migration. The job UUID is retained
    before writes. The fixed Linux worker prepares AGE over verified TLS, loads,
    then runs complete counts verification. Reconnect with **Refresh retained
    migration**, then **Transfer / open migration verification**. A passing
    counts report is not an independent full-property digest. PostgreSQL uses one
    exported repeatable-read snapshot; Cosmos requires the disclosed source-
    immutability window. Failed runs
    require operator reconciliation; the workflow never automatically resumes.
13. **Inspect same-job recovery (read only; does not resume)** is available for
    failed/interrupted migrations. It requires a pinned
    guest advertising `resume-inspection-v1` and fresh guest readiness. It uses
    the retained guest configuration and the protected target credential, never
    a new source configuration or source password. Target metadata is read in a
    read-only transaction. Job, graph generation, submitted configuration,
    checkpoint and rejects are checked; counters/identities remain lossless
    decimal strings. Re-select the action to reconcile a pending ARM response.
    The result is **review required**, never a successful migration or permission
    to resume. This inspection clears no lease and starts no worker.
14. **Explicitly resume the retained job and counts verification** requires a
    fresh recovery-readiness check, matching checkpoint inspection, separate
    approval and an unchanged `explicit-resume-v1` pinned runner. Only jobs started
    with the new recovery identity binding are eligible. The same source, target,
    VM disk/NIC/identity, configuration, job and graph generation are preserved.
    A new continuation operation retains separate logs, while the old evidence
    and one-use continuation claim remain. The guest checks that the previous
    systemd service is inactive, excludes competing submissions, rechecks target
    identity/checkpoint and replaces only its predecessor's retained lease.
    It calls `resume`, never `load` or AGE preparation, then checks the original
    committed generation and runs complete counts verification. Lost responses
    require reconciliation, not resubmission. New/changed jobs, active workers,
    stale checkpoints, rejected rows and unhealthy guests block admission.
    The defined CSV process/reboot and Neo4j network-interruption recovery cases
    passed installed-GUI migration, counts and complete canonical verification.
    This does not cover every source or interruption timing. A retained desktop
    crash lock requires explicit local review; it is not automatically removed.
    Counts verification is not the separate full P1 canonical qualification.

Workflow metadata is held in extension global storage, without source passwords,
before output-folder selection. The VM uses persistent managed OS storage and
has no public IP. Evidence/disks are retained; the guided workflow has no automatic
cleanup, stop or delete action. Operators remain responsible for resource costs.

## Qualification and current limits

The September 25, 2026 qualification ledger records **9/9 defined P1 base routes
and 12/12 finite extended branches PASS**. The base migrations include complete
canonical verification of 5.6 million mapped records, not counts alone. Extended
branches combine the explicitly recorded live GUI, isolated-host and local
contract evidence; a branch PASS does not make every negative case a live Azure
test. See the [qualification ledger](../../production-simulation/vscode-e2e/remaining-validation.md)
and [progress record](../../production-simulation/vscode-e2e/progress.md).
Final release packaging, compatibility checks and publication are separate M6
gates; these qualification totals alone do not certify a published release.

- The guided target is a new private PostgreSQL 18/AGE Flexible Server with HA
  disabled, using an existing reachable VNet and reviewed delegated subnet.
  Inline resource-group creation, automatic peering/VPN/firewall changes and
  automatic stop/delete are not provided. Cost estimates and deadlines do not
  themselves stop Azure billing.
- Cosmos support is **Cosmos DB for NoSQL**, including supported Gremlin-shaped
  documents accessed through NoSQL. It is not native Cosmos Gremlin API support.
  Guided source access uses the owned runner's system-assigned managed identity
  and a separately reviewed Data Reader grant. The finite access test does not
  establish other credential modes or a propagation-time guarantee.
- PostgreSQL catalog discovery proposes mappings for explicit review, including
  supported foreign-key edges. Nullable/composite or otherwise unsupported
  relationships still require manual mapping; proposals never silently choose
  graph semantics. Neo4j/PostgreSQL private-CA bundles use protected transport
  with hostname verification enabled.
- A retained report import is distinct from migration and full verification.
  PostgreSQL inventory uses a repeatable-read snapshot; Cosmos requires the
  disclosed source-immutability window. P1 results do not guarantee throughput,
  capacity or correctness for every production dataset.
- Managed Run Command capacity remains bounded. The native archive/removal
  action covers eligible successful, unreferenced historical readiness controls;
  it is not general automatic command retirement. Longer qualification sessions
  also required separately reviewed archive-first retirement of other controls.
  Preserve evidence and current/active controls; a full command list blocks new
  dispatch rather than silently deleting history.
- On-premises and other-cloud GUI choices were qualified with Azure-hosted
  endpoint-only simulations, not every third-party network topology.
- B02 initial-runner quota refusal retains local-contract evidence. B12 invalid
  verification covers its stated mixed-layer cases. Historical inconclusive
  probes remain inconclusive. Same-VM resize and active-job refusal do not imply
  continuous worker monitoring or lifetime disk-SKU continuity.

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
workspace is trusted. Advanced LoadJob commands run where the workspace
extension host runs, so Remote SSH, Dev Containers and Codespaces need the CLI
in that environment. The guided Azure runner path does not require a desktop
CLI; it still requires supported local files and trusted execution. Local GUI
qualification does not establish every remote-host topology.

Virtual workspaces are not supported because the CLI requires filesystem paths.

## Support

### Retained runner readiness evidence

The guided runner currently stops new submissions at 25 managed Run Command
records. This is the extension's admission threshold, not a claim about an
Azure service quota. Repeated readiness checks can consume these slots.

New successful readiness reconciliations retain a hash-sealed, field-allowlisted
receipt in the private workflow store before a later command can replace the
current control. Failed, pending, or uncertain results never produce successful
receipts. Protected parameters, raw ARM responses, source configuration and
credentials are not copied into these receipts.

Run **AGEFreighter: Archive runner readiness receipts** from the Command Palette
to publish one retained receipt as a create-only, hash-verified local evidence
file and display it. This works before target creation and without Azure login,
a VM restart, or a desktop CLI. The archive is retained under the extension's
private `runner-v2` storage as `<workflow>.report-<ARM-command-UUID>.json`.
The picker marks receipts still referenced by workflow state. Legacy records
without a sealed receipt are not adopted automatically.

**Archiving alone does not free command slots or authorize deletion.** For a
separate review, use **AGEFreighter: Review / Reconcile Readiness Control Removal**.
It requires an already-running, owned VM, no active/uncertain local operation,
an unreferenced sealed readiness receipt and a separate newer readiness receipt
from the same boot. Both must match current successful ARM evidence. The newer
receipt must show an idle, healthy guest with the pinned installation, observed
within five minutes; re-reading old output does not renew that time limit.
It will not start/stop a VM for cleanup. Pending/Updating ARM evidence blocks it,
including missing historical instance views after a restart. Legacy records are not
adopted, and migration/verification commands cannot be selected for deletion.

After a native confirmation for one exact command, the extension durably saves
and verifies a separate archive, rechecks account/trust/resource state, persists
a single-use removal intent, and submits one DELETE. The ARM control record is
permanently removed; local archives, guest evidence, disks and data remain.
Selecting that record again only checks its existence. Lost replies and retained
intents never trigger another DELETE, even if a crash happened before dispatch.
An HTTP acknowledgement is not completion: absence must be confirmed by GET.
Coordinate exclusive access to the VM/control record during this review; these
checks cannot prevent another Azure client from modifying it between requests.
If durable directory synchronization is unsupported on the extension host,
removal fails closed. That host requires separate qualification; the archive
command remains available. The defined installed-GUI/Azure readiness
archive/removal lifecycle passed in B09; this is not general resource cleanup
or qualification of every extension-host filesystem.

### Recover an interrupted desktop lock

Use **AGEFreighter: Review Interrupted Runner Lock** only after an interrupted
extension operation. Newly created locks identify the local process and OS boot
session. Recovery requires proof that this process no longer exists in the same
boot session, unchanged workflow/lock evidence, and an explicit confirmation
within five minutes. Windows local-lock recovery is unsupported because this
implementation cannot obtain the required Windows boot identity. Read-only
review refuses that unknown identity before recovery; it does not clear the
lock. Durable POSIX directory synchronization is also unavailable on Windows,
so readiness-control removal fails closed before any control DELETE. A report
archive may already have been saved when that synchronization check refuses
removal. Preserve the evidence and investigate manually. Ordinary lock release
and read-only reconciliation are separate operations.

On supported hosts, original lock metadata is durably archived before removal;
workflow records and reports are preserved. Live, uncertain, legacy, malformed
or different-boot locks remain blocked for investigation.

This action is local only. It does not cancel, reconnect, resume or replay an
Azure operation: the remote worker may still be running. After recovery, review
and reconcile the retained operation separately before approving further work.
A crashed lock-acquisition gate also remains blocked; it is not automatically
removed. Actual cloud crash/recovery qualification is separate from the isolated
local Extension Host regression tests.

Hash-valid JSON that fails full P1 canonical validation remains retained as
rejected evidence, not an accepted result. Import errors identify bounded
rejection categories without exposing capability URLs or report contents.
No new PASS tab or successful qualification state is created by a rejected import.

- [AGEFreighter documentation](https://github.com/rioriost/agefreighter/tree/main/docs)
- [Configuration reference](https://github.com/rioriost/agefreighter/blob/main/docs/reference/configuration.md)
- [Operations guide](https://github.com/rioriost/agefreighter/blob/main/docs/reference/operations.md)
- [Report a problem](https://github.com/rioriost/agefreighter/issues)

AGEFreighter is open source under the MIT License.
