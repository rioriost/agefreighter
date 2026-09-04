# 2.4.0 source-first discovery and network access revision

Superseded execution policy: [runner-first design](agefreighter-2.4.0-runner-first.md)
makes the reusable Linux runner mandatory, including CSV. Local CLI execution
and the optional-runner ordering below are retained as design history only.

Date: 2026-09-05

Status: design reviewed against the current implementation and official Azure
interfaces; implementation pending. This document supersedes the source,
workspace, and discovery-network assumptions in the original guided plan.
It does not authorize a live deployment or a source firewall change.

## Required experience

Opening **New guided migration** opens the source-type selector immediately.
An open project folder, a LoadJob file, and a successful Azure subscription
lookup are not prerequisites for displaying that selector.

| Source | Location choices | Discovery and operator review |
|---|---|---|
| Neo4j | Azure; on-premises; another cloud | Select an Azure VM candidate or enter an endpoint; authenticated discovery of database, labels, relationship types and identity properties. |
| PostgreSQL | Azure; on-premises; another cloud | Select a Flexible Server/VM candidate or enter an endpoint; discover accessible schema, tables, columns, keys and relationships, then approve graph mappings. |
| Azure Cosmos DB | Azure only | Select account, supported API, database and container/graph; discover supported graph document structures and review identity/partition semantics. |
| CSV | Local files only | Select a local folder/files after choosing CSV; inspect headers, encoding and delimiter, then approve vertex/edge/identity/endpoint mappings. |

No hidden default Neo4j source. Selecting another source clears incompatible
fields, source passwords, previous discovery results and sizing proposals.
PostgreSQL relations and CSV headers are not sufficient evidence to decide
graph semantics automatically. Existing explicit-mapping LoadJobs remain valid.
Cosmos means the APIs/document formats supported by the existing connector,
not all Cosmos APIs; unsupported account types must be explained, not presented
as usable sources.

## Revised sequence and folder timing

1. Choose source type.
2. Choose source location, or select local CSV files. CSV has no cloud-location
   radio buttons. Cosmos has no on-premises/other-cloud options.
3. For Azure: source subscription -> resource group -> candidate resource ->
   source database/container. For other networks: endpoint and credentials.
4. Establish the **discovery execution location** and verify DNS, TCP/TLS and
   authenticated read-only source access. When needed, separately review and
   approve a discovery runner before profiling; see the network policy below.
5. Discover structure, review graph mappings and collect bounded sizing facts.
6. Review target subscription, region/zone, connectivity, capacity and cost.
   Default the target subscription to the chosen source subscription, but allow
   changing it. CSV/non-Azure sources choose a target subscription here.
7. Select a project/output folder and save the reviewed LoadJob configuration
   and non-secret evidence. For CSV, offer the earlier selected project folder
   as the default; never assume the input-data folder must be the output folder.
8. Separately approve target deployment after a fresh what-if/cost review.
   Resolve target references, validate the executable job and run readiness
   checks; explicitly approve migration, then verify its result.

During discovery, any CLI-required draft is an **internal draft**, stored in
protected extension storage, not a user-selected project. It can refer to an
unresolved target secret and must never be presented as ready to run. The saved
configuration at step 7 may also retain unresolved deployment outputs; step 8
must finalize it before load. Cancelling folder selection preserves in-memory
choices and starts neither deployment nor migration.

Opening a folder with `vscode.openFolder` would reload the window and is not
required. Use a folder picker for the output URI and keep the current workbench.
Request trust/consent before local file reads or executable invocation, not
before showing an inert source selector. Without a workspace, only user-level
CLI settings apply; untrusted workspace settings must not select an executable.
On SSH/remote extension hosts, explicitly show where discovery runs. Do not
silently interpret remote files as the user's local CSV files: local CSV must
use the local host path or report that the local flow is unavailable.

Draft migration to the selected output folder must update artifact paths,
retain stable secret references, preserve originals until success, and refuse
to overwrite an existing job without confirmation. Persisted non-secret draft
metadata needs an explicit retention/export/discard policy; source passwords,
raw rows and Azure tokens never enter webview persisted state or AI context.

## Two different meanings of discovery

### A. Resource discovery through Azure Resource Manager

This finds candidate resources even when their database endpoint is private,
provided the operator can read the relevant management resources. It does not
prove database credentials, network reachability or migration compatibility.

- **Flexible Server:** use the resource-group-scoped server list; retrieve FQDN,
  version, primary availability zone, network mode, delegated subnet/private
  endpoint references and relevant provisioning metadata. ARM database listing
  can assist selection; catalog/schema/row discovery still needs SQL access.
- **Cosmos DB:** list resource-group accounts, filter API capabilities, then
  list the matching database/container or Gremlin database/graph resources.
  Read endpoint, region locations, network controls and partition metadata when
  available. Never call `listKeys` just to discover resources. A multi-region
  or zone-redundant account has no single source-VM-style logical zone: show
  the selected read region and an unknown/not-applicable source zone honestly.
- **VMs:** enumerate VM/NIC/private/public-IP references. Image/Marketplace
  metadata and operator-maintained tags can rank candidates. NSG rules can
  supply port hints but are not a list of listening services. A missing hint
  does not exclude a VM; non-default ports and containers are common.
- Keep manual endpoint/resource-ID entry as an escape hatch. A 403, unavailable
  API, unsupported type or truncated list is not equivalent to “no resources”.
  Bound and paginate calls; do not return arbitrary tags/properties to the AI.

The source picker should say **Source subscription**, not **Target subscription**.
Separating these prevents accidentally searching the destination subscription.
Defaulting source and target to the same subscription still provides the simple
flow requested by the user. Account + tenant + subscription must identify a
selection; duplicate subscription IDs across identities must not silently pick
different credentials.

### B. Authenticated source discovery

For a selected VM, suggest 7687 for Neo4j or 5432 for PostgreSQL and permit an
override. These ports alone cannot identify either DB. With operator consent,
connect only to the selected endpoint and port using the actual connector;
verify protocol/TLS and authenticated read-only version/catalog access. No
subscription-wide, subnet-wide or arbitrary-port scan, VM Run Command, SSH or
guest-agent installation occurs as a side effect of listing resources.

Neo4j, PostgreSQL and Cosmos data-plane discovery and query generation remain
CLI responsibilities. PostgreSQL needs a catalog-discovery contract before its
existing explicit query mappings can be generated from UI choices. CSV needs
multi-file mapping controls. AI can explain/offer mappings but never silently
accept identities, relationships or transformations.

## Network policy

Do not require public access. Use this order:

1. **Existing approved connectivity:** profile from the selected execution host
   if its VPN, private routing/DNS or existing allowed public endpoint reaches
   the source. A Mac outside a VNet may already have VPN connectivity; location
   alone does not imply failure. Recheck the final loader-to-source path later.
2. **Private discovery runner:** when the local host cannot connect, recommend
   a small AGEFreighter runner in an existing source-reachable VNet/subnet.
   Require a separate preview and approval showing identity/permissions, subnet,
   DNS/routing, SKU, cost, duration and the stop policy. No public IP or public
   inbound SSH is required by default. Reuse this runner as the loader where
   possible after sizing; otherwise review its replacement/resize explicitly.
3. **Temporary public exception:** an advanced, explicitly approved option only
   where the source's networking mode and organization policy permit it.
   Restrict it to the actual client's egress IP, require TLS and record expiry,
   prior state and ownership. Never allow `0.0.0.0/0` or all Azure services.
   Do not assume VNet-integrated Flexible Server can be made public merely by
   adding a firewall rule. Do not change networking mode or recreate a source.

Resource discovery is possible before a private runner exists; row/schema
profiling is not. Therefore the design must break the original dependency
cycle “profile -> size VM -> create VM -> profile” with a separately approved,
bounded discovery runner. Only the runner's provisional cost is estimated at
that point; unknown source capacity must not trigger target provisioning.

The runner must be in a compute subnet, not the PostgreSQL delegated subnet.
Same region/zone does not establish routing: inspect peering, UDR/NSG, private
DNS links and endpoint approvals. A new isolated VNet is not sufficient.
Do not create VPN/ExpressRoute, peering, role assignments or DNS changes without
their own reviewed scope and approval. If access needs a network administrator,
retain the resource inventory and show a blocked connectivity checklist.

For Cosmos, ARM permission and source data-plane permission are separate. The
current Go connector uses `azidentity.DefaultAzureCredential`; the extension's
VS Code ARM session is **not** automatically inherited by that CLI process.
The runner path should use a scoped managed identity with verified data access.
Local Cosmos profiling needs a separately reviewed VS Code-to-CLI credential
integration that preserves the no-second-login requirement, or an explicitly
selected existing supported identity. Never pass ARM tokens as Cosmos tokens,
put tokens in YAML/command arguments, silently invoke `az login`, or assume a
management-plane role grants source read access. API-specific authentication
support remains a connector acceptance gate, including Gremlin document access.

Expiry/rollback of a public exception cannot rely on VS Code staying open.
Before offering automated exceptions, implement a durable expiry mechanism and
restoration evidence, protect unrelated/concurrent firewall changes, and define
failure alerts. Until then, present the manual/admin path, not an enabled
“temporarily expose” automation button. Runner deallocation, retained evidence,
storage charges and subsequent deletion are distinct, operator-visible choices.

## Placement and sizing boundaries

- Read the primary zone from the typed Flexible Server properties rather than
  assuming every resource has top-level `zones`.
- Logical zone numbers are subscription-relative. After changing the target
  subscription, resolve physical-zone equivalence if authorized metadata permits
  it, otherwise require an explicit fallback. Never copy “zone 1” as proof of
  cross-subscription co-location.
- CSV/non-Azure sources confirm a target geography/region. Do not geolocate a
  file or private IP address; declared location and optional latency are inputs.
- Allocated server/disk size is context, not the size of the selected graph.
  PostgreSQL statistics and sampled rows are estimates; Cosmos sampling has an
  RU budget. Any exact count or full scan has an explicit time/cost limit.
- The current `inventory` CLI is Neo4j-only. Do not call it for other connectors
  or reuse the Neo4j count-store proof for PostgreSQL/Cosmos/CSV.

## Implementation slices and acceptance gates

These augment M1/M2 and precede finishing M3/M4 in the original plan.

| Slice | Work | Acceptance gate |
|---|---|---|
| S1 | Four-way source selector, conditional location controls, late folder selection, protected draft store | Empty window opens selector without folder prompt; CSV prompts only after selection; save cancellation loses no form state and starts no work. |
| S2 | Source subscription/RG/resource selectors and typed management-plane inventory | Managed DB/API classification, pagination, 403/empty/stale results, multiple identities and unknown zones have separate tested outcomes; no guest commands or scans. |
| S3 | CLI catalog discovery and source-specific form-to-LoadJob mappings | All four generated jobs validate and bounded connector tests pass; user reviews PG/CSV semantics; unsupported Cosmos APIs are blocked. |
| S4 | Connectivity gate and reusable private discovery runner | Explicit what-if/cost/network approval; private-only source discovery; no plaintext credentials/logs; failed bootstrap does not provision a target or auto-retry spending. |
| S5 | Save/finalize job, target deployment, load and verify | Correct execution host/file paths and identity; refresh/reload never double-runs; source and target readiness precede load. |

State schema revision must use a discriminated source union, optional workspace
binding, distinct source/target subscriptions, mapping review and connectivity
evidence, and discovery-runner lifecycle. Version-1 Neo4j states require an
explicit migration; do not change the meaning of their existing fields in place.

Tests must also cover source switching, password disposal, transient API errors,
out-of-order picker replies, resource moved/deleted after discovery, user folder
selection outside the opened workspace, local CSV in a remote workbench, private
DNS resolving differently on Mac/runner, public-but-firewalled sources, expired
credentials, Cosmos data-role denial, mapping ambiguity and incomplete sizing.
Use small connector fixtures and isolated approved Azure resources; no new
production-scale simulation is implied by this revision.

## Review findings

1. **Correct the entry flow first:** adding three labels to the Neo4j form is
   insufficient; current input/state/draft generation and `inventory` are all
   Neo4j-specific. Do not advertise end-to-end support before S3 passes.
2. **Separate source and target decisions:** source resource enumeration needs a
   source subscription; final output placement and deployment occur later.
3. **Separate ARM discovery from DB discovery:** private resources can be found
   without local data access; ports/NSG rules are hints, not service identity.
4. **Break the discovery-runner sizing cycle:** a small explicitly approved
   runner can profile first; source exposure is not a prerequisite.
5. **Resolve Cosmos authentication before presenting a working path:** current
   VS Code session reuse does not implement CLI data-plane authentication.
6. **Avoid overbuilding VM detection:** selected candidates plus connector
   handshakes are sufficient; broad port scanning and guest inspection add risk.

Result: design is implementable with these gates. This revision changes no
installed extension, source firewall, Azure resource, or running migration.

## Evidence

Repository baseline inspected: `155d7db` on
`codex/2.4.0-guided-migration`.

- `extensions/vscode/src/guidedMigration.ts`: folder selection before opening
  the panel; Neo4j-only parser and profile flow.
- `extensions/vscode/src/core/guided.ts` and
  `docs/reference/guided-migration-state.schema.json`: Neo4j-only source state.
- `internal/app/inventory.go`: count inventory explicitly rejects other sources.
- `internal/config/types.go` and `internal/config/testdata/valid/`: existing
  CSV/PostgreSQL mappings and Cosmos formats.
- `internal/source/cosmos/client.go`: default Azure credential initialization.

Official references checked 2026-09-05:

- [Flexible Server resource-group listing and properties](https://learn.microsoft.com/en-us/rest/api/postgresql/servers/list-by-resource-group?view=rest-postgresql-2025-08-01)
- [Cosmos account resource-group listing and capabilities](https://learn.microsoft.com/en-us/rest/api/cosmos-db-resource-provider/database-accounts/list-by-resource-group?view=rest-cosmos-db-resource-provider-2026-03-15)
- [Azure Resource Graph resource/VM discovery](https://learn.microsoft.com/en-us/azure/governance/resource-graph/concepts/explore-resources)
- [Flexible Server networking modes](https://learn.microsoft.com/en-us/azure/postgresql/network/how-to-networking)
- [Flexible Server private networking](https://learn.microsoft.com/en-us/azure/postgresql/flexible-server/concepts-networking-private)
- [Cosmos private endpoints and DNS/API considerations](https://learn.microsoft.com/en-us/azure/cosmos-db/how-to-configure-private-endpoints)
