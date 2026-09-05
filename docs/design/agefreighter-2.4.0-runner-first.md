# Runner-first guided migrations (2.4.0)

Date: 2026-09-05. This design supersedes the local-CLI reference path and the
optional-runner policy in the earlier guided migration plans.

## Roles and complete workflow

The macOS/Windows extension is the control plane. A dedicated Linux x64 VM is
the execution plane for discovery, profiling, migration, recovery and validation.
The operator never needs to install AGEFreighter or Azure CLI on the desktop
for this guided path. Existing LoadJob commands remain a separate local-CLI path.

1. Open the wizard without selecting a workspace. Choose Neo4j, PostgreSQL,
   Cosmos DB, or CSV. Choose Azure/on-premises/other-cloud for the first two,
   Azure only for Cosmos, and local files only for CSV.
2. Select the source subscription/RG/resource or enter the endpoint. ARM finds
   candidates, not database identities: VM names and open ports are not proof.
   CSV selection opens a local file picker only after selecting CSV; it does
   not imply permission to upload file contents.
3. Choose an existing, source-reachable compute subnet, region and zone for the
   runner. Do not alter the source firewall or create peering/VPN. Private DNS,
   source data-plane access and outbound access to Azure VM agent services and
   the pinned software distribution must be available. No public IP or inbound
   SSH is added. A new isolated VNet cannot solve source reachability.
4. Preview a small Burstable Linux VM and persistent disk configuration. Show
   resource changes, software version/checksum, estimated costs, uncertainty,
   quota/zone restrictions and retained-disk charges. Obtain separate approval
   for this early deployment; do not ask for source passwords before this gate.
5. Create VM/NIC/NSG in a selected existing resource group using a stable workflow
   identifier. Install a pinned release and verify its SHA-256; use the same
   version through migration. Verify guest readiness, not merely ARM success.
6. Collect credentials through protected channels, or use the runner's managed
   identity with explicitly approved source data-plane RBAC. VM identity alone
   grants no Cosmos read access. For CSV, preview upload size, then transfer to
   persistent storage with resume and checksum verification before profiling.
7. Run connector-aware read-only discovery/profile as a durable guest job with
   CPU/memory/time/RU limits. Review PostgreSQL/CSV graph mappings and identities.
   Retain evidence on managed storage; return bounded, redacted summaries only.
8. Propose target Flexible Server and final runner size/disk/IOPS. Do not infer
   loader memory from the source Neo4j heap. P3's 128 GiB source VM and roughly
   2.62 GiB loader peak RSS are different observations. B-series profile speed
   is not a migration-throughput benchmark.
9. After reviewing target settings, select an output folder and save configuration
   and redacted evidence. Cancel preserves the workflow without deployment.
   VM jobs reference Linux paths/secret handles, not desktop password files.
10. Approve resize and target deployment. Seal evidence, ensure no active guest
    job, deallocate the runner, resize the same VM within a compatible x64/SCSI
    size family, restart, and verify actual guest resources/network/identity.
    Check zonal capacity and both regional and VM-family quota at execution time.
    Failure preserves the VM/disks and blocks migration; replacement is a new
    explicit approval, not an automatic fallback. Never resize the source VM.
11. Prepare/verify AGE and target readiness, finalize and validate LoadJob, and
    explicitly start migration on the runner. VS Code closing must not stop it.
12. Reconnect to durable workflow/job IDs for status or explicit recovery; run
    counts/integrity/digest verification. Missing evidence is incomplete, not
    pass. Stop/deallocate and delete are separate approved lifecycle operations.

## Control and durability

### Resource-group and region selection decision

The discovery VM and the later Flexible Server target share the selected
`RunnerInput.subscriptionId` / `RunnerInput.resourceGroup`. R4 must use this
same reviewed group instead of introducing an unrelated target RG. Source
resources are independent and are never moved into that group.

The preview currently offers **existing migration resource groups only**, in a
subscription-backed dropdown. Operators needing a dedicated new group can
create it in Azure and refresh the list. This keeps new-RG permissions, policy,
tags, metadata-location approval and partial-creation recovery out of the current
VM-only deployment boundary. This is an implementation/approval boundary, not
a networking restriction: a VM and its VNet may reside in different RGs. Creating
a new RG does not itself require peering, and selecting an existing RG does not
prove source reachability. Future inline RG creation can reuse an existing VNet
without automatically creating another VNet or peering.

The **Azure region** dropdown is fetched from the selected subscription and shows
display name plus canonical name, e.g. `Japan East (japaneast)`. It does not use
the RG's metadata location as a VM placement default. Candidate source region
is preselected only when it appears in the list; Cosmos account metadata is not
a data-region default. Refresh preserves valid reviewed choices. Changing the
runner subscription clears RG, region and subnet, and stale responses cannot
repopulate another subscription. Empty/failed listings never fall back to a
hard-coded region or unvalidated free text. SKU, zone, source/subnet placement
and quota gates remain authoritative; listing a region is not service-capacity
evidence. On-premises region selection remains an explicit operator decision.

Reference: [Azure virtual network configuration](https://learn.microsoft.com/en-us/azure/virtual-network/manage-virtual-network)
documents independent resource-group placement and same-subscription/region
requirements for resources attached to a VNet.

### Execution and state

ARM deployment and managed Run Command are the control transport, not SSH or a
public runner web server. Managed Run Command launches short allowlisted control
operations; long jobs must live in persistent systemd units on the guest, not
inside an interactive request or a transient unit that disappears on reboot.
Source credentials cannot appear in templates, customData, command text, URLs,
logs or public parameters. Use protected transport or a scoped secret store;
never relay an ARM token as a Cosmos data token. The AI receives no secrets.

Version-2 local workflow metadata lives in extension global storage before any
project selection. It stores source type/location, reviewed deployment, operation
IDs and phase, but no source passwords. Record operation intent before PUT;
after errors/reload query that exact ID instead of submitting a new deployment.
Use a fresh preview hash with expiry and single-flight mutation locks. Ignore
stale UI selection responses. Treat already-existing resource names as collisions.

Do not keep artifacts on ephemeral disks. Keep VM identity/NIC and managed
disks through a size change. Explicitly account for temporary interruption,
dynamic IP changes, egress policy, source allowlists and zone equivalence when
subscriptions differ. A successful control-plane update does not prove that
the guest is running the requested size or that bootstrap succeeded.

## Implementation and review gates

- R1: replace the local-CLI wizard with source selection, late workspace binding,
  resource inventory and persistent runner planning. No local AGEFreighter calls.
- R2: reviewed ARM template, quota/SKU/subnet checks, fresh what-if, bounded cost
  evidence, explicit create approval, durable deployment ID and status refresh.
- R3: guest readiness and protected remote dispatch, connector discovery and
  mapping editors, CSV resumable upload, full retained-artifact retrieval.
- R4: sizing, same-VM resize, target deployment, LoadJob export/finalization.
- R5: durable remote load/resume/verification, reconnect and lifecycle controls.

R1/R2 can be packaged for review while R3-R5 remain visibly disabled. A runner
deployment is not a completed assessment or a migration. Release must wait for
the four source integration paths, Linux guest tests, controlled private-network
deployment and failure-injection evidence. No production-scale rerun is implied.

Review: this removes desktop binary/version drift and reuses private connectivity.
The main risks are private-subnet egress, source credential delegation, Burstable
throttling, zonal resize capacity, and interrupted control requests. Fail closed
on each; do not silently expose databases or claim a provisioned VM is ready.

## Earlier R1/R2 implementation checkpoint — 2026-09-05

R1/R2 control-plane implementation is present on `codex/2.4.0-guided-migration`.
The old local-CLI wizard is no longer registered; advanced existing LoadJob
commands remain unchanged. Source selectors/candidate inventory, late workspace
binding, reviewed private-VM templates, checksum bootstrap, approval, atomic
record storage, cross-window locks and no-replay status refresh are implemented.
R2 is **not live-Azure qualified**. The matching 2.4.0 Linux release is not yet
published, so its release gate intentionally blocks provisioning.

Review findings and dispositions:

- Critical: a desktop CLI in the discovery path defeats private-network access.
  Removed from the active wizard; no temporary public-access workaround added.
- Critical: lost PUT responses and simultaneous extension windows can duplicate
  deployment. Persist intent before PUT, re-read under an exclusive lock, and
  reconcile by exact ID. Crash locks stay retained; status remains readable.
- High: ARM success cannot prove package installation or source reachability.
  `provisioned` is separate from guest readiness; assessment stays disabled.
- High: source and loader sizing were easy to conflate. Preserve separate roles;
  burstable discovery performance must not become a throughput promise.
- High: remote credentials, connector mapping, CSV uploads, resize and durable
  guest jobs need their own integration/failure tests. R3–R5 remain release gates,
  not assumed functionality. No source password or live load is accepted now.
- Medium: manual subnet selection and on-premises region review remain necessary.
  Do not infer a private route or geographic location from a hostname.

Validation: TypeScript/unit tests and packaging; VS Code Extension Host smoke
tests on macOS arm64 at minimum version 1.105.0 and installed version 1.136.1.
The smoke test opens the wizard without a workspace or local CLI chooser.
No live Azure resource was created or changed during this implementation check.
Controlled integration with a published pinned Linux artifact and R3–R5 work
are required before release; this is not an end-to-end completion claim.

## Current R3 implementation checkpoint — 2026-09-05

The source-form controller now extends that earlier preview. A local-only draft
can be reviewed before VM creation or release availability; blank artifact and
template fields cannot pass deployment gates. The selected source and CSV file
identities follow the same draft into a separately approved runner preview.

Neo4j/PostgreSQL/Cosmos/CSV forms produce secret-reference-only configurations
without YAML input. PostgreSQL generates read-only queries from table/column
mappings; Cosmos supports explicit labels and Gremlin document interpretation;
CSV includes typed properties and null markers. Native password prompts occur
only after explicit source-read approval and fresh guest readiness. Protected
start/status operations retain identity, boot/configuration hashes and successful
report manifests across reconnects. No source host resolution through ARM is
performed by the IP/host-only assessment form.

R3 is still not complete: CSV upload, bulk report transfer/acceptance, schema/FK
suggestions, exact non-Neo4j totals and custom-CA delivery remain open. Source
passwords can now be accepted for approved guest assessments, but target mutation
and migration are still disabled. None of these local checks qualifies a P1 Azure
GUI branch. See the [live qualification ledger](../../production-simulation/vscode-e2e/progress.md).

References: [Azure resizing](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/resize-vm),
[managed Linux Run Command](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/run-command-managed),
[Bsv2 CPU credits](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/general-purpose/bsv2-series).
