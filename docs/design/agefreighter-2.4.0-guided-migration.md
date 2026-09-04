# AGEFreighter 2.4.0 guided migration workspace

Current architecture: [runner-first redesign](agefreighter-2.4.0-runner-first.md).
The Linux discovery VM becomes the migration VM after sizing. The local-CLI
reference implementation described below is historical, not the new guided path.

Status: reviewed and approved for implementation on `codex/2.4.0-guided-migration`

2026-09-05 revision: the [source-first discovery and networking design](agefreighter-2.4.0-source-discovery-networking.md)
supersedes the initial Neo4j-first scope, early workspace requirement, and
discovery-network sequence below. It adds guided CSV, separate source/target
subscriptions, late folder selection and a separately approved private discovery
runner. That revision is a reviewed design, not completed runtime functionality.

## Outcome

AGEFreighter 2.4.0 replaces the extension's YAML-first happy path with a
guided, resumable workspace. An operator supplies a source connection and a
small number of policy choices; the extension profiles the source, recommends
an Azure topology and size, previews the deployment, writes the deterministic
LoadJob, deploys the approved resources, starts the CLI job, and presents
post-load verification evidence.

Existing LoadJob discovery and commands remain supported. The Go CLI remains
the migration engine and the owner of source discovery, durable job identity,
checkpoints, target mutations, and verification.

## User journey

1. Select **AGEFreighter: New guided migration**.
2. Choose Neo4j, PostgreSQL, or Azure Cosmos DB for NoSQL and enter the source
   endpoint, database, and non-secret identifiers. Password fields are stored
   through VS Code SecretStorage and never echoed, logged, serialized into the
   webview state, or sent to a language model.
3. Test the read-only connection and run a bounded source profile. The UI
   distinguishes exact values, estimates, lower bounds, and unavailable data.
4. Select the source location:
   - for an Azure source, select its ARM resource so location and logical
     availability zone can be read from Azure Resource Manager;
   - for a non-Azure source, supply its physical location and optionally run
     latency probes. A nearest-region result is a recommendation, not a fact.
5. Review an Azure proposal. The default places the loader VM and primary
   Flexible Server in the source region and logical zone when both services and
   the selected SKUs support it. An unsupported or unknown zone is shown as an
   explicit fallback requiring review.
6. Review estimated hourly cost, storage headroom, network path, generated
   LoadJob, and an Azure Resource Manager what-if result.
7. Confirm deployment. The extension submits the deployment with the VS Code
   Azure session and persists the deployment ID and workflow state.
8. After deployment succeeds, it prepares AGE, materializes the target secret
   only in protected extension storage, validates the final LoadJob, and runs
   `doctor`. A failed gate cannot start a load.
9. Confirm migration. The extension opens a visible terminal for `load`; the
   durable job ID is captured by the CLI workflow rather than invented by the
   extension. Reloading VS Code never starts a second load.
10. On completion, run report, counts, integrity, and configured digest checks.
    The result page says pass, fail, or incomplete and links to retained local
    evidence. Cleanup is never automatic.

## Supported scope

### 2.4.0

- Source connectors: Neo4j discovery, PostgreSQL explicit mappings, and Cosmos
  Gremlin discovery. CSV remains available through the existing LoadJob flow.
- Destination: Azure Database for PostgreSQL Flexible Server with Apache AGE.
- Authentication: the user is already signed into Azure in VS Code. The
  extension uses the built-in Microsoft authentication provider through the
  maintained Azure authentication package. It does not invoke an independent
  `az login` flow or collect service-principal credentials.
- Deployment: new resource group or a reviewed existing resource group, private
  VNet, delegated PostgreSQL subnet, private DNS, loader VM with managed
  identity, Flexible Server, AGE allowlist and preload configuration, and a
  protected secret store.
- Resume: the guided workflow records its own phase plus the CLI durable job ID;
  CLI status remains authoritative for migration state.

### Non-goals

- Fully automatic schema or semantic decisions. Ambiguous labels, identities,
  keys, property mappings, and error policies require operator review.
- Inferring a physical Azure availability zone from a hostname or IP address.
  Azure zone numbers are subscription-relative logical zones.
- Geolocating a private/on-premises host from its address, or selecting a region
  solely from public-IP geolocation.
- Creating ExpressRoute, VPN, private peering, DNS forwarders, source firewall
  exceptions, Azure quota, or role assignments beyond the reviewed deployment.
- Automatically accepting cost, deploying, starting, resuming, deleting, or
  cleaning up resources through an AI action.
- Replacing canonical production-scale qualification with a bounded profile.

## Architecture

### Extension host

`GuidedMigrationController` owns one webview and delegates deterministic work
to small services:

- `AzureSession`: enumerates the VS Code user's filtered subscriptions and
  obtains ARM-scoped credentials without exposing tokens to the webview.
- `SourceDraft`: validates form input, stores secrets, and writes a protected
  draft LoadJob for the CLI profile operation.
- `SizingAdvisor`: converts bounded profile facts into transparent storage,
  memory, vCore, and runtime ranges. Every recommendation includes its inputs,
  formula version, headroom, and confidence.
- `TopologyAdvisor`: reconciles source ARM location/zone, user-declared
  on-premises location, region capabilities, SKU restrictions, quota, and
  optional latency measurements.
- `DeploymentPlanner`: renders versioned Bicep parameters and calls ARM
  validate/what-if before allowing deployment.
- `WorkflowStore`: persists a schema-versioned, non-secret state document in
  `.agefreighter/guided/<id>/state.json`; credentials and ARM tokens are never
  part of this document.
- `MigrationOrchestrator`: advances only after explicit confirmations and
  re-reads ARM deployment state plus CLI job state after every reload.

All ARM requests use HTTPS with bearer tokens held only in the extension host.
The webview accepts and emits a versioned allowlisted message protocol. Dynamic
content is escaped, scripts use a per-view nonce, and the content security
policy forbids remote content.

### Files and secrets

The committed/generated workspace may contain:

```text
.agefreighter/guided/<id>/
  state.json                 # non-secret phase, decisions, resource IDs
  source-profile.json        # bounded, redacted CLI evidence
  proposal.json              # formulas, confidence, cost timestamp
  deployment.bicep           # reviewed infrastructure definition
  deployment.parameters.json # no secure parameter values
  deployment-what-if.json    # redacted ARM evidence
jobs/<name>.yaml             # final LoadJob, secret file references only
```

Secret values live under the extension's `globalStorageUri` with owner-only
permissions and are referenced by absolute file paths from the LoadJob. The
same values are also retained in VS Code SecretStorage so a missing materialized
file can be recreated after direct user confirmation. Secret paths are stable
for CLI fingerprint/resume behavior. `.agefreighter/guided` is ignored by
default except for explicitly exported redacted evidence.

### State machine

```text
draft -> source-connected -> profiled -> proposed -> what-if-reviewed
      -> deploying -> deployed -> target-ready -> load-started
      -> load-failed | load-committed -> verifying -> passed|failed|incomplete
```

Transitions use compare-and-set revisions. On activation, the extension reads
the persisted state and queries ARM/CLI before offering the next action. A
phase can be retried, but deployment and load submission use stable identifiers
so a UI retry cannot create duplicate resources or jobs.

## Sizing policy

The first recommendation uses the CLI profile's target-row and graph/identity/
WAL/staging byte ranges. If the profile is bounded, the UI must not extrapolate
from a prefix unless the connector supplies a trustworthy total row count.
Missing totals produce a provisional proposal and block one-click deployment
until the user enters a reviewed upper bound or runs an exact inventory query.

Required headroom:

- Flexible Server storage: high estimate of graph + identity + WAL working set,
  multiplied by 1.30 and rounded up to a supported storage tier;
- loader disk: staging high estimate multiplied by 1.25;
- loader memory: at least 2x configured AGEFreighter RSS limit plus operating
  system headroom;
- Flexible Server compute and loader VM: selected from versioned row/property
  bands, then checked against current regional SKU availability and quota;
- elapsed-time range: shown only when a measured throughput baseline exists.

The recommendation is not a capacity guarantee. The review page exposes each
assumption and allows a larger choice but does not permit storage below the
calculated floor.

## Azure topology policy

- Azure source with a selected ARM resource: use its `location`; use its zone
  only when ARM exposes a logical zone and both target SKUs are unrestricted in
  that zone.
- Azure source without a resource match: require location confirmation and
  label zone as unknown.
- On-premises source: shortlist regions within the selected geography, honoring
  data-residency policy and AGE availability. Rank them by user-declared
  location and optional measured latency. The user confirms the final region.
- Deploy the loader VM and Flexible Server primary in the same logical zone by
  default. Same-zone Flexible Server HA is an explicit production option;
  zone-redundant HA is recommended for the post-migration steady state but may
  add synchronous-write latency during bulk loading.
- Private access is the default. If the source cannot reach the VNet, stop and
  show required networking work; never silently enable public database access.

## Milestones and acceptance gates

### M0 — contracts and reviewed plan

- Save this plan, source/proposal/workflow schemas, webview message contract,
  and threat model.
- Verify current Azure AGE/PostgreSQL compatibility from authoritative service
  metadata at runtime; documentation snapshots are advisory only.

Gate: tests prove secrets and tokens cannot enter persisted state or model
context, and the reviewed plan does not claim hostname-based zone discovery.

### M1 — guided source and profile

- Add the command and webview shell, VS Code Azure session check, source forms,
  secret storage, protected draft generation, connection test, and bounded
  profile presentation.
- Start with Neo4j discovery as the end-to-end reference path; add PostgreSQL
  mappings and Cosmos resource selection behind the same contracts.

Gate: a Neo4j operator can reach a truthful sizing input without authoring YAML;
cancel/reload leaks no secret and performs no mutation.

### M2 — Azure recommendation

- Enumerate subscriptions, regions, source resources, zones, SKUs, quota, AGE
  availability, and USD retail estimates using the authenticated Azure session.
- Implement versioned deterministic sizing and topology recommendations.

Gate: proposal fixtures cover Azure zonal, Azure zone-unknown, on-premises,
SKU-restricted, quota-insufficient, incomplete-profile, and stale-price cases.

Implementation status: complete for the Neo4j reference path. The extension
uses the existing VS Code Azure session to read PostgreSQL 18 capabilities,
PostgreSQL and Compute quota, zonal VM SKU availability, and bounded USD retail
rates. It writes a 24-hour proposal and performs no deployment.

### M3 — infrastructure preview and deployment

- Generalize the production-simulation Bicep into a single-source deployment
  module, add bootstrap/install scripts, ARM validate/what-if, cost summary,
  deployment confirmation, idempotent naming, and status polling.

Gate: what-if contains only expected resource types, secure parameters never
appear in files/output, and an isolated test subscription deployment passes
without public endpoints.

### M4 — LoadJob finalization and migration

- Produce the final LoadJob, prepare AGE, run validate/doctor, and launch load
  in a visible terminal. Persist and reconcile the durable job ID.

Gate: reload/retry cannot double-submit; target readiness failures block load;
the generated job validates with the 2.4.0 CLI.

### M5 — verification and recovery UX

- Present status, failure evidence, explicit resume, report, counts, integrity,
  and digest results as one timeline.

Gate: success requires all selected checks; unavailable evidence yields
`incomplete`, never `pass`; no cleanup is automatic.

### M6 — compatibility, documentation, and release

- Complete unit, Extension Host, mocked ARM, Bicep lint/what-if, Neo4j/PostgreSQL/
  Cosmos integration, packaging, upgrade, and failure-injection tests.
- Update Marketplace documentation and ship 2.4.0 through the existing release
  process.

Gate: existing YAML-first workflows remain compatible and every external
  mutation has an operator-visible confirmation and recoverable evidence.

## Test matrix

| Layer | Required evidence |
|---|---|
| Pure TypeScript | form validation, message protocol, sizing formulas, state transitions, redaction |
| Extension Host | authentication/session selection, SecretStorage, reload, command/view registration |
| ARM mock | pagination, long-running deployment, conflict, stale token, 401 retry, quota and SKU restrictions |
| Bicep | build, lint, validate, what-if allowlist, private networking, zone combinations |
| CLI | generated draft/final jobs, profile, validate, doctor, load/status/resume/verify/report |
| Connectors | Neo4j discovery, PostgreSQL mappings, Cosmos managed identity/default Azure credential |
| Failure injection | VS Code reload, deployment timeout, VM bootstrap failure, source disconnect, target disconnect |
| Security | no credentials in logs/state/YAML/telemetry/model input/VSIX; owner-only secret files |
| Upgrade | 2.3.0 YAML-first workspace continues to work; guided state schema migrates forward |

## Threat model and operational gates

- **Credential disclosure:** password fields travel directly from the webview
  message handler to SecretStorage; sanitized acknowledgements contain only a
  secret handle. Logs and errors use recursive redaction.
- **Token theft:** ARM tokens are never persisted or sent to the webview. The
  extension requests only Azure Resource Manager scopes and refreshes through
  the VS Code authentication session.
- **Injection:** ARM APIs are called through typed HTTPS clients. CLI operations
  use executable/argument arrays; generated names, paths, identifiers, and
  Bicep parameters are allowlist-validated.
- **Unexpected spend:** source profiling is read-only; deployment is disabled
  until a fresh what-if and cost timestamp are reviewed. The confirmation shows
  subscription, region, zone, SKUs, storage, HA, estimated hourly cost, and the
  resource group name.
- **Wrong placement:** unknown source location or zone is never converted into
  a known value. Fallback placement carries its reason into state and evidence.
- **Duplicate mutation:** stable deployment names, state revisions, and CLI
  durable job reconciliation make retries idempotent.
- **AI overreach:** AI may explain the sanitized proposal and verification
  report, but receives no credentials/raw records and cannot deploy, load,
  resume, verify deeply, or clean up.

## Plan review

The requested experience is feasible, but the initial five-step description
left six material risks. This reviewed plan makes the following corrections:

1. **VS Code authentication is the single Azure login.** Depending on the
   deprecated Azure Account extension or starting a second CLI login would
   break MFA, multi-tenant selection, and the user's stated premise.
2. **Location is evidence-based.** A hostname cannot establish a source's
   region or logical availability zone, and an IP address is insufficient for
   an on-premises nearest-region decision. ARM selection or a declared location
   plus optional latency is required.
3. **Profile uncertainty is preserved.** Bounded observations are not silently
   scaled into production capacity. Unknown totals require an upper bound or a
   stronger inventory before deployment.
4. **Deployment has a what-if and cost gate.** Saving YAML and immediately
   provisioning would combine a reversible configuration step with a costly
   external mutation. The user reviews both before ARM deployment.
5. **Network reachability is a first-class gate.** Same region and zone do not
   establish connectivity to an on-premises or separately networked source.
6. **Completion is evidence-based.** A committed job is necessary but not
   sufficient; selected counts, integrity, source immutability where available,
   and digest evidence determine pass/fail/incomplete.

With these corrections, M0 is approved and implementation proceeds with the
Neo4j guided reference path while retaining the connector-neutral contracts.

## Current authoritative references

- [Azure authentication helpers for VS Code](https://www.npmjs.com/package/@microsoft/vscode-azext-azureauth)
- [Azure Account deprecation guidance](https://github.com/microsoft/vscode-azure-account/issues/964)
- [Flexible Server overview and regions](https://learn.microsoft.com/azure/postgresql/overview)
- [Flexible Server high availability](https://learn.microsoft.com/azure/postgresql/flexible-server/concepts-high-availability)
- [Apache AGE on Flexible Server](https://learn.microsoft.com/azure/postgresql/azure-ai/generative-ai-age-overview)
- [Supported AGE versions](https://learn.microsoft.com/azure/postgresql/extensions/concepts-extensions-versions)
- [VM SKU and zone availability](https://learn.microsoft.com/azure/virtual-machines/linux/create-cli-availability-zone)
- [Azure Retail Prices API](https://learn.microsoft.com/rest/api/cost-management/retail-prices/azure-retail-prices)
