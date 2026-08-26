# agefreighter 2.x Cosmos DB Integration Deployment Plan

**Status:** Deployed and Verified

**Approved:** The user approved this plan and authorized Azure deployment.

## Scope and Classification

- **Mode:** MODIFY an existing Go CLI and test infrastructure in place.
- **Classification:** Development/integration-test environment.
- **Scale:** Small, disposable fixture workload.
- **Budget:** Cost-optimized; serverless request billing and minimal retained data.
- **Compliance:** No production or customer data. Keep resources in one confirmed region.
- **Out of scope:** Hosting agefreighter itself in Azure, Change Feed, multi-region
  writes, production availability, and key-based authentication.

## Proposed Azure Context

- **Subscription:** `MCAPS-Hybrid-REQ-51508-2023-rifujita`
  (`67c417f3-5a13-446c-afb9-40cd87f2fdb7`)
- **Tenant:** `16b3c013-d300-468d-ac64-7eda0820b6d3`
- **Location:** `japaneast`
- **Developer principal:** `af888774-9bdc-4d32-a7cf-f9cfaa8c93c7`
- **Developer IPv4:** `59.138.207.107`
- **Basis:** The user confirmed the current Azure CLI account, explicit `azd`
  defaults, subscription, and location.

Subscription policy assignments include inherited MCAPSGov audit, deny, and
deploy/modify initiatives plus Azure Security Baseline. No Cosmos-specific deny
was identified in the initial assignment review; validation and deployment
preview remain authoritative for effective policy. The subscription currently
contains zero `Microsoft.DocumentDB/databaseAccounts` resources, including zero
in Japan East. Cosmos DB does not expose the normal quota API, so the account
count and deployment preview are the capacity checks for this environment.

## Deployment Recipe

- **Recipe:** Azure Developer CLI with Bicep.
- **Rationale:** Azure-only, infrastructure-only test environment with no
  existing IaC. `azd provision` provides explicit environment state and Bicep
  provides reviewable subscription/resource-group deployment.
- **Application hosting:** None. The Go CLI runs locally and connects to Azure
  Cosmos DB and the local Apple Container AGE target.

## Azure Architecture

Create one dedicated `rg-agefreighter-<environment>` resource group containing:

1. One Azure Cosmos DB for NoSQL account.
   - Serverless capacity.
   - Session consistency.
   - One write region in Japan East.
   - Local/key authentication disabled.
   - System-assigned managed identity for Network Security Perimeter support.
   - TLS-only endpoint secured by the perimeter.
2. One database: `agefreighter`.
3. Two fixture containers:
   - `vertices`, partition key `/partitionKey`.
   - `edges`, partition key `/partitionKey`.
4. One Cosmos DB native data-plane role assignment.
   - Built-in Data Contributor.
   - Scope limited to the test account.
   - Principal is the confirmed signed-in developer object ID.
5. One enforced Network Security Perimeter and profile.
   - The Cosmos DB account is associated with the profile.
   - One inbound `/32` rule permits only the confirmed developer IPv4 address.
   - The account uses `SecuredByPerimeter` after the association exists.

No Key Vault is required because the deployment emits only the non-secret
account endpoint and resource names. No account keys or connection strings are
read or emitted.

## Planned Artifacts

- `azure.yaml`: infrastructure-only azd project.
- `infra/main.bicep`: subscription-scope resource group entry point.
- `infra/main.parameters.json`: azd environment substitution.
- `infra/modules/cosmos.bicep`: account, database, containers, and data-plane
  RBAC.
- `infra/modules/network-perimeter.bicep`: enforced perimeter, profile,
  developer ingress rule, and Cosmos association.
- `scripts/azure/README.md`: provision, seed/test, perimeter update, and
  destructive cleanup procedure.
- `internal/source/cosmos/`: source adapter and iterator.
- Cosmos configuration fixture and live integration test.

## Connector Design

### SDK and authentication

- Use the latest compatible stable
  `github.com/Azure/azure-sdk-for-go/sdk/data/azcosmos`.
- Use `github.com/Azure/azure-sdk-for-go/sdk/azidentity` and
  `DefaultAzureCredential`.
- Reuse one Cosmos client and database client per iterator.
- Accept only HTTPS account endpoints and the existing `default-azure`
  credential mode.

### Query and mapping

- Execute configured SQL queries with SDK query parameters; extend the Cosmos
  mapping schema with strictly decoded named JSON parameter values.
- Preserve configuration order and complete all vertex mappings before edges.
- Use the SDK cross-partition query pager with bounded page retention. Do not
  fetch all results or retain prior pages.
- Decode documents with `json.Decoder.UseNumber`, preserve exact signed
  64-bit integers, and fail explicitly on integer overflow or unsupported JSON
  values.
- Resolve configured fields using documented JSON Pointer paths, including
  nested properties.
- Canonically encode mapped properties for the AGE fast path without retaining
  the full source document after emitting a record.

### Resume and consistency

- A versioned opaque resume token binds:
  - complete ordered mapping fingerprint,
  - mapping index and kind,
  - continuation token used to open the current page,
  - record index within that page.
- Resume reopens the page from its starting continuation and skips only records
  already committed. It never advances to the next page after a mid-page
  checkpoint.
- The source remains bounded under replay.
- Cosmos query paging is not a transactional snapshot. The static plan,
  documentation, and job diagnostics state that source mutations can change
  resumed results.

### Reliability and telemetry

- Rely on the Azure SDK retry policy for transient failures and HTTP 429
  responses.
- Bound request concurrency to configured source capacity; preserve
  vertices-before-edges ordering.
- Record cumulative request charge, page count, retry/throttle observations,
  and the latest continuation diagnostic through a source telemetry interface.
- Never log access tokens, authorization headers, account keys, full
  continuation tokens, or source documents.
- Cancellation must interrupt credential acquisition, page reads, and record
  emission promptly.

## Application Integration

- Refactor the app source factory so CSV and Cosmos iterators share the existing
  bounded pipeline and AGE sink.
- Generalize configured label discovery and source rejection/telemetry handling
  without changing CSV behavior.
- Support Cosmos sources for `create` and atomic `replace`.
- Preserve job configuration fingerprints, transactional AGE batches, durable
  checkpoints, and existing failure/resume semantics.

## Test Plan

### Normal CI

- Fake/injected Cosmos page client and credential factory.
- Multi-page and cross-partition result order.
- Mid-page and mapping-boundary resume.
- Nested JSON Pointer mapping and all supported JSON value kinds.
- Exact integer limits and overflow rejection.
- Parameter propagation.
- 429/retry diagnostics, cancellation, and no secret/token logging.
- Bounded-memory scaling with generated pages.
- CSV regression coverage.

### Live Azure

- Seed several logical partitions in both containers using Entra ID.
- Query multiple pages across partitions.
- Load vertices and edges into local Apache AGE.
- Verify counts and graph identities through AGE/Cypher.
- Terminate after a committed batch and resume from the durable Cosmos token.
- Exercise both `create` and `replace`.

The merged statement coverage gate remains at least 90%.

## Security

- Entra ID only; `disableLocalAuth: true`.
- Least-privilege Cosmos native RBAC at account scope.
- Enforced Network Security Perimeter with a single developer IPv4 `/32`
  inbound rule.
- No secrets in files, azd outputs, logs, test failures, or git history.
- Bicep contains no hard-coded subscription, tenant, resource-group, or
  principal IDs; these are azd environment parameters.
- Inspect Azure Policy before generation and adjust tags/network controls if
  required.

## Deployment and Verification Workflow

1. Confirm the proposed subscription and Japan East.
2. Inspect Azure Policy and Cosmos account limits.
3. Resolve current signed-in developer object ID and public IPv4.
4. Implement the connector and normal-CI tests.
5. Generate azd/Bicep artifacts and documentation.
6. Set the plan status to `Ready for Validation`.
7. Run the mandatory `azure-validate` workflow.
8. Run `azd provision --preview`.
9. Deploy through the mandatory `azure-deploy` workflow.
10. Seed fixtures and run live Cosmos-to-AGE integration tests.
11. Run full repository quality gates and independent review.
12. Commit and push two rollback points:
    - `feat: add Cosmos DB source connector`
    - `infra: add Cosmos integration environment`

## Validation Checklist

- [x] All validation checks pass
  - [x] AZD installation
  - [x] `azure.yaml` schema validation
  - [x] azd environment setup
  - [x] Azure authentication
  - [x] approved subscription and location
  - [x] provision preview
  - [x] Go build
  - [x] package validation
  - [x] Azure Policy validation
  - [x] Bicep compilation
  - [x] static Cosmos DB role verification
  - [x] Network Security Perimeter validation

## Initial Validation Proof

- `azd version`: 1.31.2 stable.
- `azure.yaml`: passed the stable Azure Developer CLI schema validator.
- `azd auth login --check-status`: authenticated as the approved developer.
- `azd env get-values`: confirmed `cosmos-dev`, subscription
  `67c417f3-5a13-446c-afb9-40cd87f2fdb7`, `japaneast`, the approved principal,
  and the approved IPv4.
- `az bicep build --file infra/main.bicep`: compiled without errors.
- `go build ./...`: completed without errors.
- `azd package --no-prompt`: completed successfully.
- `azd provision --preview --no-prompt`: completed successfully and proposed
  only creation of `rg-agefreighter-cosmos-dev` and its Cosmos DB account.
- `Microsoft.DocumentDB`: provider is registered and reports Japan East as
  supported for database accounts.
- Azure Policy: the Azure MCP policy operation returned 403 because its
  credential context differed from the authenticated CLI. The required
  fallback `az policy assignment list --disable-scope-strict-match` succeeded
  against the approved subscription. Its three effective assignments cover SQL
  Server, data-protection, and open-source relational database Defender
  provisioning. A management-group Policy hidden from that listing was later
  observed in the account activity log and invalidated the original network
  design.
- Static RBAC review: the approved local developer principal receives Cosmos DB
  Built-in Data Contributor (`00000000-0000-0000-0000-000000000002`) at the
  account scope through a native Cosmos SQL role assignment. This is the
  least-privilege data-plane role needed to seed, query, and delete integration
  fixtures.
- Aspire checks, Docker context checks, and service image packaging are not
  applicable because this is an infrastructure-only azd project with no Aspire
  AppHost or deployable service.

The initial deployment completed, but the inherited
`CosmosDB_PublicNetwork_Modify` Policy changed `publicNetworkAccess` to
`Disabled`. The activity log identifies the management-group assignment
`MCAPSGovDeployPolicies` and the definition display name
`SFI - Disable public network access on Cosmos DB accounts (excluding NSP
configured resources)`. The live test could not seed fixtures. The
policy-compliant Network Security Perimeter design above therefore requires a
fresh validation pass before redeployment.

## Section 7: Validation Proof

Validation completed at `2026-08-26T03:05:37Z`.

- `azd auth login --check-status` and `azd env get-values`: confirmed the
  approved interactive user, subscription, Japan East location, developer
  principal, and developer IPv4.
- `az bicep build --file infra/main.bicep`: compiled successfully. The installed
  Bicep 0.40.2 lacks local type metadata for the stable 2025-07-01 NSP API and
  emits BCP081 warnings; Azure Resource Manager validation below accepted all
  NSP resource schemas and properties.
- `go build ./...`: completed without errors.
- `azd package --no-prompt`: completed successfully.
- `azd provision --preview --no-state --no-prompt`: completed successfully and
  proposed the intended Cosmos identity and `SecuredByPerimeter` changes.
- `az deployment sub what-if`: passed ARM validation and proposed exactly one
  perimeter, one profile, one `/32` inbound rule, one enforced Cosmos
  association, and the expected Cosmos account/data resources. The single
  multiple-deployment diagnostic is intentional: the first account deployment
  creates a Policy-compliant disabled account, and the second switches it to
  `SecuredByPerimeter` only after the association exists.
- `Microsoft.Network`: provider is registered and reports Japan East support
  for Network Security Perimeter.
- Azure Policy: the account activity log identified the inherited
  `MCAPSGovDeployPolicies` modify assignment. The revised infrastructure uses
  the assignment's explicit NSP-configured-resource exclusion rather than
  attempting to bypass or override Policy.
- Static RBAC review: the approved developer retains Cosmos DB Built-in Data
  Contributor at account scope. The account also receives a system-assigned
  identity required for NSP participation; no management-plane role is granted
  to that identity.
- Static NSP review: the association is `Enforced`, its only inbound access
  rule is `59.138.207.107/32`, and the Cosmos account's final public access mode
  is `SecuredByPerimeter`.
- Aspire, Docker context, and application service packaging checks remain not
  applicable to this infrastructure-only azd project.

## Deployment Verification

Deployment and live verification completed at `2026-08-26T03:35:12Z`.

- `azd provision --no-state --no-prompt`: deployed the policy-compliant
  two-phase Cosmos account configuration and Network Security Perimeter.
- `azd deploy --no-prompt`: completed successfully; this infrastructure-only
  project has no hosted application services.
- Azure Portal:
  <https://portal.azure.com/#@/resource/subscriptions/67c417f3-5a13-446c-afb9-40cd87f2fdb7/resourceGroups/rg-agefreighter-cosmos-dev/overview>
- Cosmos endpoint:
  <https://afv7nal73jathdc.documents.azure.com:443/>
- Cosmos account verification:
  - Provisioning state `Succeeded`.
  - Capacity `EnableServerless`.
  - Local authentication disabled.
  - TLS 1.2 minimum.
  - System-assigned identity enabled.
  - Public network access `SecuredByPerimeter`.
- Live role verification:
  - Principal `af888774-9bdc-4d32-a7cf-f9cfaa8c93c7`.
  - Cosmos DB Built-in Data Contributor
    (`00000000-0000-0000-0000-000000000002`).
  - Scope is the exact Cosmos DB account.
- Network Security Perimeter verification:
  - Perimeter, profile, access rule, and association provisioning succeeded.
  - Association access mode is `Enforced`.
  - The associated private-link resource is the exact Cosmos DB account.
  - The only inbound address prefix is `59.138.207.107/32`.
- Apple Container AGE/PostgreSQL/Neo4j smoke checks passed.
- `TestCosmosLiveIntegration` passed against the deployed account and local AGE
  target. It exercised multi-partition fixture seeding, committed-batch resume,
  create verification, atomic replacement, backup cleanup, graph counts, and
  exact fixture deletion.
- Fixture cleanup is registered before writes, covers ambiguous write outcomes,
  treats only 404 as an already-clean result, and surfaces other deletion
  failures. Replacement cleanup covers active, shadow, and backup graph names
  whenever a job ID exists.
- `make check-full` passed after the final cleanup changes: formatting, vet,
  vulnerability scan, unit tests, race tests, and 90.1% aggregate statement
  coverage.
- Independent final review reported no remaining significant issues.
- Azure resources remain deployed. No destructive cleanup command was run.

## Rollback and Cleanup

- Code and infrastructure are isolated in separate commits.
- `azd down`/resource-group deletion is destructive and is not executed without
  separate explicit approval.
- The integration account remains deployed unless cleanup is approved.
- The cleanup procedure targets only the azd environment's exact resource
  group; no broad subscription cleanup is permitted.

## Preparation Results

- Connector implementation and normal-CI tests completed and pushed in
  `f73564b` (`feat: add Cosmos DB source connector`).
- Stable SDK versions are pinned to `azcosmos` v1.5.0 and `azidentity` v1.14.0.
- Full repository vet, vulnerability, unit, race, and 90% coverage gates pass;
  merged statement coverage is 90.1%.
- Independent connector review completed; its CSV quarantine regression finding
  was fixed before the rollback commit.
- `azure.yaml`, subscription-scope Bicep, account/database/container/RBAC
  resources, Network Security Perimeter resources, operations documentation,
  and the live Cosmos-to-AGE integration test are generated.
- azd environment `cosmos-dev` is configured with the approved subscription,
  Japan East location, developer principal, and developer IPv4 address.
- Azure deployment and live Cosmos-to-AGE verification completed successfully.
  The account is secured by the enforced perimeter and the approved `/32`
  inbound rule.

## Official References

- Go SDK:
  <https://learn.microsoft.com/azure/cosmos-db/nosql/sdk-go>
- Query pagination:
  <https://learn.microsoft.com/azure/cosmos-db/nosql/query/pagination>
- Cosmos DB data-plane RBAC:
  <https://learn.microsoft.com/azure/cosmos-db/how-to-connect-role-based-access-control>
- Cosmos DB Network Security Perimeter:
  <https://learn.microsoft.com/azure/cosmos-db/how-to-configure-nsp>
- Network Security Perimeter Bicep:
  <https://learn.microsoft.com/azure/templates/microsoft.network/2025-07-01/networksecurityperimeters>
- Azure SDK `DefaultAzureCredential`:
  <https://learn.microsoft.com/azure/developer/go/sdk/authentication/authentication-overview>
