# Colleague demo rebuild - October 2, 2026 12:42 JST

**Status:** Deployed and Verified

## Request

The October 2 request explicitly asks to repeat the previously approved
screenshot deployment for a colleague demonstration. Reuse that exact
source-only architecture, subscription, region, paid footprint and 24-hour
lifetime. No expanded scope is required. The completed release CI's USD 50
authorization and two-hour lifetime do not apply to this separate deployment.

## Approved repeat scope and Azure context

- Classification: temporary internal demo with synthetic data, not production.
- Subscription: `MCAPS-Hybrid-REQ-51508-2023-rifujita`
  (`67c417f3-5a13-446c-afb9-40cd87f2fdb7`); tenant
  `16b3c013-d300-468d-ac64-7eda0820b6d3`; Japan East, zone 1.
- New group: `rg-agefreighter-demo-20261002-1242-start`.
  Deployment: `af-demo-start-20261002-1242`.
- Source: one private Standard_D2s_v5, 64 GiB Standard SSD, pinned Ubuntu
  `Canonical:ubuntu-24_04-lts:server:24.04.202609040` and Neo4j 5.26.30.
  Exact 100,000 vertices / 250,000 relationships, 18 ID indexes and read-only
  database. This is the smaller screenshot fixture, not full P1 scale.
- Network: isolated VNet, two NSGs, private NIC, source subnet 10.76.1.0/24,
  empty runner subnet 10.76.2.0/24 with Storage service endpoint, NAT gateway
  and outbound-only static IP. Bolt TLS to 10.76.1.4:7687 from runner subnet
  only; no public VM IP, public Browser, SSH ingress, peering or VPN.
- No precreated runner, transfer Storage Account, PostgreSQL target, target
  subnet, Automation, identities or role assignments. Those migration
  resources require the extension's separate approvals.
- Source-only native shutdown: **October 3, 2026 12:42 JST**
  (`2026-10-03T03:42:00Z`), within 24 hours of the request.
  No automatic deletion. Retained disk/NAT/IP charges continue after shutdown.
  Later extension-created resources are not covered by the source schedule.
- New 16-character ASCII-alphanumeric password and fresh TLS/SSH material
  in owner-only Git-ignored local files; never print credentials or use Keychain.
- Same paid footprint as the approved demo. No numerical demo budget ceiling
  was specified. Current compute rate is USD 0.124/hour (USD 2.976/24h);
  disk, NAT, IP and traffic are additional. This is not an all-in quote or cap.
- Recipe: reuse reviewed standalone Bicep, deterministic fixture and bootstrap
  from `production-simulation/work/demo-start-20261001-2145/`; use Azure CLI.
  New artifacts: `production-simulation/work/demo-start-20261002-1242/`.
  Do not modify existing demo records, extension state or unrelated work.

## Read-only preparation evidence and policy constraints

- Current subscription/tenant match the previously approved context and are
  Enabled. No `rg-agefreighter*` groups currently remain.
- D2s_v5 is unrestricted in zones 1/2/3; pinned image is available, x64/V2.
- Quota CLI: regional cores 0/101, DSv5 cores 0/100, VMs 0/25000,
  VNets 0/1000, Standard IPv4 IPs 0/1000, NAT gateways 0/500.
- Current public Linux compute retail was rechecked on October 2.
- Visible subscription policy assignments cover Defender SQL, data protection
  and open-source databases. Inherited controls may additionally apply.
  ARM validation and exact create-only what-if are mandatory.
  No `SecurityControl:Ignore` tag, exemption or policy relaxation is included.

## Execution checklist

- [x] Review prior demo configuration and current Azure state
- [x] Finalize same scope, price estimate and fresh 24-hour lifetime
- [x] Explicit repeat request for previously approved plan; no expanded scope
- [x] Prepare scoped artifacts
- [x] Validate
  - [x] Bicep compilation and lint
  - [x] Actual ARM template validation
  - [x] Exact create-only what-if preview
  - [x] Authentication, SKU/image and quota
  - [x] Policy acceptance without exemptions
  - [x] Static role review: no identities or role assignments needed
- [x] Deploy and verify source data, access and shutdown

## Prepared artifact evidence

- Reused the reviewed templates and bootstrap without adding resources; changed
  only new group identity, deadline and native schedule. Source fixture and
  bootstrap/guest verification code are unchanged from the prior demo.
- Generated the deterministic fixture locally and verified all endpoints and
  properties. Root SHA-256 remains
  `d0a1e58f368eb6462171028766d603556b1d0183b7e70c1f622fda8aea0dc44d`.
- Fresh CA chain and IP SAN 10.76.1.4 verified. CA SHA-256:
  `7f2298bee0f4f03df77c038419da70ed70a9b297b0ebdba672c0dffb9f9a7065`.
- Password is 16 ASCII letters/digits; new credentials and CA differ from the
  previous environment. Local files are 0600 and directory 0700, Git-ignored.
  No credential value printed; no Keychain or Azure resources changed.
- Python parsing and bootstrap/guest shell syntax checks passed.
  Actual ARM validation and exact create-only scope proof also passed.

## Section 7: Validation Proof - October 2 colleague demo

Completed under azure-validate at `2026-10-02T03:46:23.447254Z`.

- `az bicep build --file production-simulation/work/demo-start-20261002-1242/main-sub.bicep`
  passed without errors.
- `az deployment sub validate --subscription 67c417f3-5a13-446c-afb9-40cd87f2fdb7
  --location japaneast --name af-demo-start-20261002-1242 --template-file
  production-simulation/work/demo-start-20261002-1242/main-sub.json --parameters
  @production-simulation/work/demo-start-20261002-1242/parameters.private.json`
  returned `Succeeded`, error null.
- `az deployment sub what-if` with the same input Bicep/parameters and
  `--result-format ResourceIdOnly --no-pretty-print` returned `Succeeded`.
  Exact set comparison proved 10 Creates exclusively in the new group and no
  Modify/Delete; implicit OS disk and inline subnets will be checked live.
- Exact normalized comparisons confirm the only template differences are
  new RG, deadline and native schedule. Fixture/bootstrap/verifier are
  byte-identical to the prior working environment. No policy exemption,
  identity, role grant, transfer Storage Account or target was introduced.
- Current auth, image/SKU and quota passed. ARM accepted the scoped resources
  without policy/tag relaxation.
- `validation-proof.json` retains timestamp plus template/parameter SHA-256;
  `validation-result.json` and `what-if.json` retain actual results.

## Deployment result - October 2 colleague demo

- Submitted `af-demo-start-20261002-1242` once. Subscription and nested
  `neo4j-source-only` deployments succeeded at `2026-10-02T03:50:33Z`.
- Independent authenticated TLS verification completed at
  `2026-10-02T03:52:09.951854Z`: exact 100,000 unique vertices and 250,000
  unique relationships, all nine label/type counts, 18 ONLINE ID indexes,
  read-only Neo4j 5.26.30, digest-pinned container, required Bolt TLS and no
  Browser endpoint. Container running without OOM or restarts.
- Exact ten-resource live inventory includes one private D2s_v5 VM, 64 GiB
  Standard SSD and the expected networking/extension/shutdown resources.
  No VM public IP, VNet peering, identity, new scoped role grant, runner,
  Storage Account, target server/subnet, Automation or policy exemption.
  Runner subnet is empty/nondelegated with NAT and Storage service endpoint.
- Native source shutdown is Enabled at UTC 0342, next
  **October 3, 2026 12:42 JST**. No auto-delete, startup or lifetime extension.
  Later extension resources and retained disk/NAT/IP costs are not stopped
  by this source-only schedule.
- Connection inputs are in
  `production-simulation/work/demo-start-20261002-1242/connection-guide.txt`.
  `credentials.env` contains the new 16-character ASCII-alphanumeric password;
  `source-ca.pem` is the new source CA. Files remain owner-only/Git-ignored.
  Start a new Guided Migration using this RG, not an old workflow record.
- `verification-summary.json`, `source-live-verification.txt`, sanitized ARM
  snapshots and deployment/validation proof retain the evidence. Verification
  ran from the source guest, not from a precreated migration runner.
- Existing groups, prior artifacts, extension installation/state and unrelated
  screenshot-documentation/skills work were not changed.

---

# Release CI rebuild - October 1, 2026 23:28 JST

**Status:** Completed - published and temporary resources removed

## Request and boundaries

The user requested rebuilding the Azure resources needed for release 2.4.1.
The previous CI runner registration is offline; its former resource group and
Cosmos DB account are absent. On October 1 at 23:37 JST the user approved this
plan with an increased USD 50 budget. The two-hour VM limit, same subscription
and Japan East placement, GitHub release, and removal of only the new dedicated
resource group/runner registration after success are approved.

Marketplace publication remains a manual user action. Do not publish there.
Do not modify the current Neo4j demo resources or reuse them as release CI.
Do not weaken the required release gates or inherit expired cost approvals,
policy exemptions, credentials or obsolete endpoint addresses.

## Proposed deployment

- Classification: temporary release-validation environment; synthetic fixtures only.
- Subscription: `MCAPS-Hybrid-REQ-51508-2023-rifujita`
  (`67c417f3-5a13-446c-afb9-40cd87f2fdb7`).
- Region: Japan East, zone 1 for the single VM.
- New dedicated resource group: `rg-agefreighter-release-241-20261001`.
- Recipe: existing Bicep Cosmos modules plus a scoped runner/network wrapper,
  deployed with Azure CLI. Do not reuse stale azd environment IP addresses.
- Runner: one `Standard_D4s_v5`, x64, 4 vCPU / 16 GiB, 128 GiB Standard SSD LRS
  OS disk. Ubuntu image
  `Canonical:ubuntu-24_04-lts:server:24.04.202609040` is available as Gen2 x64.
  Docker hosts the workflow's already-pinned AGE and PostgreSQL test services.
- Network: isolated VNet/subnet, NSG denying unsolicited inbound traffic, one
  NAT gateway/static outbound IP and one NIC. No VM public IP, inbound SSH/RDP,
  peering, VPN, Bastion, or connection to the screenshot demo network.
- Cosmos: one single-region serverless account; database `agefreighter`;
  containers `vertices` and `edges`, both partitioned by `/partitionKey`.
  Synthetic fixture creation/deletion is confined to these new test containers.
- Access: TLS 1.2 minimum, local/key authentication disabled, system identity
  for the Cosmos account, enforced Network Security Perimeter. Permit only the
  new runner's actual static outbound IPv4 /32; no inherited desktop-IP rule.
- Identity: reuse existing enabled `agefreighter-github-cosmos` service
  principal (`f10b1f54-776b-4203-ba62-281905697d36`). Recreate only Cosmos
  Built-in Data Contributor at the new account scope; no subscription-wide
  role or new tenant application/federated credential.
- GitHub: retain the `cosmos-integration` protected environment and its existing
  OIDC subject `repo:rioriost/agefreighter:environment:cosmos-integration`.
  Register a new runner with the required `agefreighter-azure` label, using
  short-lived registration credentials through protected transport. Update
  only the environment's Cosmos endpoint/fixture variables. Do not replace
  unrelated runners or change environment approval/tag policies.
- Use the ordinary required release workflow. No bypass of live Cosmos,
  signing/notarization, package, checksum or provenance gates.

## Cost and lifecycle proposed for confirmation

- Approved incremental release budget: **USD 50**, including compute, temporary
  storage/network, requests and a reserve. This is an operational ceiling, not
  an Azure-enforced billing cap; do not silently increase it.
- Proposed VM runtime: **maximum 2 hours after creation**. Start no new test
  cycle near the deadline. Configure native VM shutdown and an exact-resource
  stop guard; no Azure Automation account. Closing VS Code is not shutdown.
- Public retail prices checked October 1: Linux D4s_v5 USD 0.248/hour
  (USD 0.496 for two hours), E10 128 GiB LRS disk USD 9.60/month plus disk I/O,
  Standard public IPv4 USD 0.005/hour, Cosmos serverless USD 0.285/million RUs.
  Taxes, contract discounts, Cosmos storage, NAT/bandwidth and perimeter charges
  are not included in those component figures. No verified all-in quote yet:
  NAT/NSP retail queries did not return a numerical price; retain a reserve and
  recheck cost admission before deploying.
- Proposed cleanup, requiring explicit approval of this plan: after release
  assets and retained logs are verified, remove only the newly created runner
  registration and `rg-agefreighter-release-241-20261001`. This removes its
  synthetic Cosmos fixtures and VM disk. It never deletes the current demo,
  the old offline registration, existing Entra identity or other groups.
- If validation/release fails, stop the new VM at the deadline, preserve
  diagnostic evidence, and ask before retrying or extending the lifetime.
  Retained disks, Cosmos storage, NAT and public IP continue to incur charges
  until their approved cleanup; VM deallocation alone does not stop them.

## Read-only evidence and policy constraints

- Previous `rg-agefreighter-cosmos-dev` returned `ResourceGroupNotFound`.
  The subscription has no Cosmos accounts and no old release runner VM.
  GitHub runner ID 21 remains offline and idle.
- The existing release service principal is enabled. Its federation still
  matches GitHub's issuer, the protected environment subject and
  `api://AzureADTokenExchange`.
- `az quota list` / `az quota usage list`: DSv5 6/100 vCPU and regional 6/101.
  Adding 4 vCPU fits both. SKU discovery reports no subscription restrictions
  and zones 1, 2 and 3. This is not a capacity reservation.
- The pinned Ubuntu image remains available. Additional network-resource
  limits and all ARM templates must pass validation after approval.
- Current subscription-visible assignments cover Defender SQL, data protection
  and open-source relational databases. The previous deployment also proved an
  inherited Cosmos public-network modify policy that is not shown in this
  listing. Preserve its supported NSP-controlled design, never an exemption.
- Build the account with networking Disabled, associate the enforced perimeter,
  then set SecuredByPerimeter. Verify the final live state; do not infer success
  from template properties or a completed deployment alone.
- Marketplace upload candidate remains local
  `extensions/vscode/dist/agefreighter-2.4.1.vsix` (616808 bytes), SHA-256
  `09b448a43dc1b4c07c519e5f60ffeed2474f153812e071d47302aa534f5ac919`.
  GitHub main is `e0ca70795b715ef645fd5a8ec1c1c979445a8c9a`. Annotated tag
  `v2.4.1` now points to that exact dry-run-qualified commit; its normal release
  workflow is run `36880917495`. GitHub publication completed at
  `2026-10-01T15:08:42Z`.

## Execution checklist

- [x] Read-only discovery and prior-art review
- [x] Proposed architecture, component prices, cost reserve and lifetime plan
- [x] User confirmation (USD 50; October 1, 2026 23:37 JST)
- [x] Infrastructure preparation
- [x] Azure validation
- [x] Deployment and independent readiness verification
- [x] Release publication and artifact verification
- [x] Approved resource cleanup and independent absence verification

### Prepared artifact and absolute deadline

- Operational IaC/bootstrap/parameters are isolated under ignored
  `production-simulation/work/release-241-20261001/`, mode 0700. SSH key and
  parameters are private; no secret is committed or printed.
- The wrapper reuses the existing Cosmos modules and creates only the approved
  new group. Its sole perimeter ingress is derived from the new outbound IP.
- Bicep compilation and shell parsing passed. The installed Bicep lacks NSP
  2025-07-01 local types (BCP081); ARM validation is required, as in the prior
  working NSP deployment. A literal nonsecret administrator-name warning is
  informational, not a password authentication setting.
- Absolute deadline: **2026-10-01T16:38:00Z** (October 2 01:38 JST), conservatively
  set before deployment. Native daily shutdown is 16:38 UTC. The attached local
  exact-VM stop guard will also request deallocation at that deadline.
- Network capacity: 1/1000 VNets, 1/1000 public IPs, 1/500 NAT gateways.
  Microsoft.DocumentDB is registered and lists Japan East.
- Deployment `af-release-241-20261001` succeeded. New runner ID 22,
  `agefreighter-release-241-20261001`, is online. Cosmos account
  `af7q2ct64uxi7dw` uses endpoint
  `https://af7q2ct64uxi7dw.documents.azure.com:443/`; sole perimeter ingress
  is the new NAT address `20.222.89.18/32`.
- Independent live infrastructure verification passed at
  `2026-10-01T14:52:23.311385Z`, including enforced perimeter, scoped SQL role,
  container partition keys, private VM networking and native shutdown.
  Evidence: `infra-verification.json` in the operational artifact directory.
- The first dry-run Azure job failed before integration tests because the
  fresh Ubuntu runner lacked `make`. Installed `build-essential`, `git`,
  `unzip` and `zip` through the exact-VM managed command `af-release-build-tools`;
  verified Make, GCC and Git as the runner user. Updated the retained bootstrap
  and readiness scripts to include these prerequisites; no ARM redeployment.
  Only failed jobs in run `36878496781` were retried. Budget, deadline and
  release gates remain unchanged.
- Retained the original Cosmos environment endpoint for conditional restoration
  after cleanup. Existing runner ID 21, Entra identity and demo remain untouched.
- Full dry-run `36878496781`, attempt 2, succeeded. The Azure job actually ran
  PostgreSQL 19 property-graph regression and live Cosmos integration; neither
  was skipped. Full logs and exact-commit/job assertions are retained locally.
  This does not claim a new P1-scale qualification.

### Section 7: Validation Proof - release CI

- [x] Bicep compilation: `az bicep build` produced `main.json`; no errors.
- [x] Shell build verification: `bash -n bootstrap.sh` passed. The unchanged
  release source already passed PR #30 engine, extension-host and package CI.
- [x] ARM template validation: `az deployment sub validate` returned
  `Succeeded`, error null.
- [x] What-if: `az deployment sub what-if --result-format FullResourcePayloads`
  returned `Succeeded`, exactly 17 Create entries, zero Modify/Delete.
  Programmatic checks bound every identity to the approved new group.
  The implicit OS disk and inline subnet are also declared in the VM/VNet.
- [x] Authentication: current enabled subscription and retained enabled OIDC
  service principal/federation match the approved context.
- [x] Azure Policy validation: ARM accepted the compiled resources. The only
  what-if diagnostic is the expected repeated-account warning for the
  Disabled -> NSP association -> SecuredByPerimeter sequence.
- [x] Static roles: one Cosmos SQL Built-in Data Contributor at the exact new
  account, for the existing CI principal. Required for synthetic fixture
  seed/query/delete. No management-plane elevation or desktop data access.
  The Cosmos system identity is for NSP participation, not cross-service data
  access, so it receives no unrelated role.
- The runner archive version 2.337.0 and SHA-256
  `70920811a4f8ad4328818682bca5c6469c1c942fab52448868071d0063816613`
  match the official GitHub release metadata and are enforced by bootstrap.
- Approved USD 50 reserve is retained; the footprint is one 2-hour VM and one
  serverless synthetic fixture account. No larger SKU, throughput reservation,
  extra VM, automated retry loop or budget extension is permitted.

### Section 8: Publication and scoped cleanup

- Stable GitHub release:
  `https://github.com/rioriost/agefreighter/releases/tag/v2.4.1`
  (release ID `401086249`). All 19 jobs in tag run `36880917495` succeeded.
  Its live Azure job ran on the new runner ID 22.
- All 16 public assets were downloaded and verified. The signed provenance
  certificate is bound to this repository, release workflow, exact source and
  signer commit `e0ca70795b715ef645fd5a8ec1c1c979445a8c9a`, and `refs/tags/v2.4.1`;
  self-hosted attestors were rejected. All 15 signed subject digests match.
  The bundle itself matches the GitHub release asset digest.
- All six archive/VSIX checksums and six SPDX documents passed. The published
  VSIX manifest and JavaScript match the qualified source/local bundle, including
  the retained runner 2.4.0 pin. Public VSIX SHA-256:
  `125c7e8064c4c8b17f545c4c5ebacfd1c3136b85a174061b7ba08e7624c7fcb2`.
  Its ZIP hash differs from the earlier local package; its executable bundle
  is byte-identical. Marketplace was not published by this task.
- The Homebrew tap Formula matches the published Formula byte-for-byte.
  Windows unsigned-binary disclosure remains in the public release body.
- Release logs, asset copies, provenance and verification proof are retained
  under the ignored operational artifact directory.
- Cleanup rechecked the exact new group's tags, all 13 visible resource IDs,
  absence of locks, and idle runner identity. Removed only runner ID 22 and
  restored the original Cosmos environment endpoint after verifying that no one
  had changed our temporary value.
- Submitted one deletion of `rg-agefreighter-release-241-20261001` at
  `2026-10-01T15:11:20.260980Z`; no replay. The VM deletion succeeded at
  `15:11:59Z`. Cosmos/network cleanup completed asynchronously.
- Independent cleanup verification passed at `2026-10-01T15:28:38.597350Z`:
  group absent, no remaining subscription resource IDs within its scope,
  runner 22 absent, old runner 21 retained, and original Cosmos endpoint restored.
  A separate final read again confirmed absence. The stop guard then exited
  with `completed-after-verified-cleanup`. Temporary SSH key files were removed.
  Neither the USD 50 authorization nor the two-hour VM limit was extended.
  These lifecycle records are not a final Azure billing statement.
- Read-only checks confirmed the existing Entra principal remains enabled.
  The earlier screenshot group `rg-agefreighter-demo-20261001-2145-start` is
  now absent: its Azure activity log records a separate deletion starting at
  `2026-10-01T14:37:31.9759756Z`, before this release cleanup. This task did not
  issue a delete or other mutation for that group and did not recreate it.
  The log records timing, not attribution to a person.

---

# Demo source rebuild - October 1, 2026 21:45 JST

**Status:** Deployed and Verified

## Request and approved scope

The user requested another demo Neo4j environment on October 1 at 21:45 JST.
Repeat the previously approved source-only architecture and paid footprint:
100,000 vertices, 250,000 relationships, private D2s_v5 in Japan East zone 1,
fresh 16-character ASCII-alphanumeric password and CA files, and a 24-hour
native source-VM shutdown schedule. No Azure Automation or automatic deletion.
No runner VM, transfer Storage Account or target database is precreated.
Use the previously confirmed subscription; recheck live context and capacity.
No policy exemption or SecurityControl tag is inherited.

## Proposed deployment

- Resource group: `rg-agefreighter-demo-20261001-2145-start`.
- Deployment: `af-demo-start-20261001-2145`.
- Operational artifacts: `production-simulation/work/demo-start-20261001-2145/`.
- Source shutdown: October 2, 2026 21:45 JST (`2026-10-02T12:45:00Z`).
- Recipe: existing reviewed Bicep with Azure CLI, new resource group only.
- Existing resources will be inspected, not changed or deleted.
- Compute baseline: USD 0.124/hour; disk, NAT, public IP and traffic extra.
  No numerical total budget was specified. Shutdown does not remove resources
  or stop retained disk/network charges.

## Execution checklist

- [x] Reconfirm context, existing demo scope, SKU, quota and approved footprint.
- [x] Prepare fresh private artifacts and verify fixture/TLS locally.
- [x] Compile, validate and inspect exact create-only what-if.
  - [x] Bicep compilation.
  - [x] Actual ARM template validation.
  - [x] Exact create-only what-if preview.
  - [x] Azure authentication, current SKU/image and quota.
  - [x] Policy acceptance without exemptions or tag relaxation.
  - [x] Static role review: no identities/role assignments required.
- [x] Submit once; reconcile the named deployment without replay.
- [x] Verify live data, private/source-only scope and shutdown; deliver inputs.

## Preparation decisions and evidence

- Repeat deployment authorized by the current request, using the same previously
  approved source-only plan, subscription and region; no new cost/security scope.
- Subscription `MCAPS-Hybrid-REQ-51508-2023-rifujita`
  (`67c417f3-5a13-446c-afb9-40cd87f2fdb7`), tenant
  `16b3c013-d300-468d-ac64-7eda0820b6d3`, Enabled; Japan East zone 1.
- No `rg-agefreighter-demo-*` resource groups remain. We did not delete them.
- D2s_v5 unrestricted in zones 1/2/3. Pinned Ubuntu image
  `Canonical:ubuntu-24_04-lts:server:24.04.202609040` available, x64/V2.
- CLI quota and usage: regional cores 0/101, DSv5 cores 0/100, VMs 0/25000;
  VNets 0/1000, Standard IPv4 public IPs 0/1000, NAT Gateways 0/500.
- Live Linux pay-as-you-go compute remains USD 0.124/hour (USD 2.976/24h),
  excluding disk/NAT/IP/traffic.
- Reuse reviewed October 1 21:01 templates and deterministic fixture/bootstrap.
  Only group, cutoff metadata, native schedule and fresh protected credentials/
  certificates differ. Existing deployment artifacts are not overwritten.
- Neo4j 5.26.30 image is digest-pinned; database read-only after strict import,
  18 ID indexes, required Bolt TLS and no Browser endpoint. Private IP
  10.76.1.4, source subnet 10.76.1.0/24; empty runner subnet 10.76.2.0/24
  with Storage endpoint and NAT egress. VM has no public IP or inbound SSH.
- Fresh private files prepared successfully; deterministic endpoint/property
  verification matches root
  `d0a1e58f368eb6462171028766d603556b1d0183b7e70c1f622fda8aea0dc44d`.
  Shell/Python syntax and CA chain/IP SAN verified. Fresh CA SHA256:
  `cbf14a7005ecc75a11ed57fab389b134205c46f3662632b32a663be9234b636e`.
  New 16-character ASCII-alphanumeric password; files 0600/directory 0700,
  Git-ignored, no credential value emitted and no Keychain used.

## 7. Validation Proof - October 1, 21:45 request

Completed under azure-validate at 2026-10-01T12:51:06Z.

- `az bicep build --file production-simulation/work/demo-start-20261001-2145/main-sub.bicep`
  passed; only the available-newer-Bicep advisory was emitted.
- `az deployment sub validate --subscription 67c417f3-5a13-446c-afb9-40cd87f2fdb7
  --location japaneast --name af-demo-start-20261001-2145 --template-file
  production-simulation/work/demo-start-20261001-2145/main-sub.bicep --parameters
  @production-simulation/work/demo-start-20261001-2145/parameters.private.json`
  returned Succeeded, error null.
- The same inputs with `az deployment sub what-if --result-format ResourceIdOnly
  --no-pretty-print` returned Succeeded. Exact resource-ID set comparison proved
  ten Creates in the new group, zero Modify/Delete. The implicit managed OS disk
  will additionally be checked in the live inventory.
- Existing policy assignments were inspected. Validation accepted the template
  without a policy exemption, role grant or SecurityControl tag.
- Static review confirms no identities, role assignments, Storage Account,
  PostgreSQL target or Automation. Secure bootstrap enters protectedSettings
  only; it is not printed or included in the readable deployment result.
- Proof retained in `validation-result.json`, `what-if.json` and
  `validation-proof.json` in the current operational artifact folder.

## Deployment result - October 1, 21:45 request

- One submission of `af-demo-start-20261001-2145`; subscription and nested
  `neo4j-source-only` deployments succeeded at 2026-10-01T12:55:42Z.
- Independent authenticated TLS verification completed at
  2026-10-01T12:56:46Z: exact 100,000 vertices / 250,000 relationships,
  matching unique IDs and all label/type counts, 18 ONLINE indexes,
  Neo4j 5.26.30 pinned digest, read-only database, required Bolt TLS and
  disabled Browser. Container running, no OOM or restarts.
- Exact ten-resource live inventory includes the implicit OS disk. One
  private D2s_v5 source VM in zone 1, 64 GiB Standard SSD, Trusted Launch
  and key-only SSH; no public VM IP. Runner subnet remains empty and
  nondelegated with explicit NAT and Storage service endpoint.
- No runner, transfer storage, target database, delegated target subnet,
  Automation, managed identity, RG/child-scoped role grant or policy exception.
- Enabled native source-only shutdown: UTC 1245, next October 2 at 21:45 JST.
  Later extension-created resources are not covered; retained disk/NAT/IP
  charges continue after source shutdown.
- Fresh credentials and CA remain owner-only/Git-ignored in
  `production-simulation/work/demo-start-20261001-2145/`; full input guide is
  `connection-guide.txt`. Do not reuse the old environment's password/CA.
- `verification-summary.json`, `source-live-verification.txt` and sanitized
  ARM snapshots retain the evidence. Verification was from the source guest,
  not a runner-to-source migration test; no runner has been created.
- No existing resource, extension installation or operator workflow was changed.

---

# AGEFreighter screenshot source-only environment: October 1, 21:01 JST

**Status:** Deployed and Verified

The repeat request authorizes rebuilding the previously approved source-only
demo with the same subscription, region, data size, paid resource footprint
and 24-hour lifetime. Preserve every previous environment and workflow.

## Repeat scope and approval

- New isolated RG `rg-agefreighter-demo-20261001-2101-start`.
- Subscription `67c417f3-5a13-446c-afb9-40cd87f2fdb7`, Japan East, zone 1.
- Private Standard_D2s_v5 Neo4j source, 64 GiB Standard SSD, pinned Neo4j
  5.26.30; 100,000 vertices / 250,000 relationships, TLS and read-only.
- Source/runner subnets, VNet, NAT and outbound public IP only.
  No runner VM, transfer storage, target DB, Automation or policy exemptions.
- New 16-character ASCII-alphanumeric password and fresh TLS/SSH material
  in Git-ignored owner-only files. Never print secrets or use Keychain.
- Native source-VM shutdown by October 2, 2026 21:01 JST / 12:01 UTC.
  No deletion; retained disk/NAT/IP charges and later-created resources remain.
- No numerical budget ceiling was specified previously. Keep the same paid
  footprint; report old resources still present without changing them.
- Recipe: reuse reviewed standalone Bicep and deterministic bootstrap.
  Artifacts: `production-simulation/work/demo-start-20261001-2101/`.
- The previous user-applied `SecurityControl:Ignore` tag is not inherited:
  no tag or policy relaxation is authorized for this new group.

## Steps

- [x] Inspect current Azure context, old/new groups, quota, SKU and image.
- [x] Generate fresh artifacts; verify fixture, secret shape and certificates.
- [x] Compile, ARM-validate and prove exact create-only scope with what-if.
  - [x] Bicep compilation.
  - [x] Actual ARM template validation.
  - [x] Exact create-only what-if preview.
  - [x] Azure authentication, current SKU/image and quota.
  - [x] Policy acceptance without exemptions or tag relaxation.
  - [x] Static role review: no identities/role assignments required.
- [x] Submit once; reconcile the named deployment without replay.
- [x] Verify live data, source-only scope, TLS/read-only mode and shutdown.

## Preparation evidence (October 1)

Subscription/tenant match the approved prior context. No previous demo groups
remain and the new group is absent. Japan East regional vCPU quota is 0/101;
DSv5 is 0/100; VM count 0/25000. D2s_v5 is unrestricted in zone 1 and the
pinned Ubuntu 24.04.202609040 image remains available. Network usage is VNet
0/1000, Standard IPv4 public IP 0/1000 and NAT 0/500.
Visible policy assignments were inspected; no policy/tag change is introduced.
Current D2s_v5 Linux compute retail remains USD 0.124/hour (USD 2.976/24h);
disk, NAT, IP and traffic are additional, not a hard spending ceiling.

The existing Bicep/fixture/bootstrap is reused, changing only group/deadline
and generating fresh secrets and certificates. Local fixture generation and
endpoint/property verification preserved root SHA-256
`d0a1e58f368eb6462171028766d603556b1d0183b7e70c1f622fda8aea0dc44d`.
Python AST and bootstrap/verification shell syntax passed. Certificate chain
and IP SAN 10.76.1.4 passed; fresh CA SHA-256
`b4c1c41123e141d9a3825dd78b3cc21903b09bf52eacf934efceb10d29f10400`.
Password shape and owner-only permissions passed without emitting the value.
Artifacts and secrets are Git-ignored, folder 0700 and files 0600.
Use deployment name `af-demo-start-20261001-2101` with the generated
`main-sub.bicep` and `parameters.private.json`. No new external Bicep module,
managed identity, role assignment or additional service is introduced.

## 7. Validation Proof (October 1)

Completed 2026-10-01T12:05:21Z under azure-validate.

- `az bicep build --file production-simulation/work/demo-start-20261001-2101/main-sub.bicep`
  passed, with only a newer-Bicep-version advisory.
- `az deployment sub validate --subscription 67c417f3-5a13-446c-afb9-40cd87f2fdb7
  --location japaneast --name af-demo-start-20261001-2101 --template-file
  production-simulation/work/demo-start-20261001-2101/main-sub.bicep --parameters
  @production-simulation/work/demo-start-20261001-2101/parameters.private.json`
  returned Succeeded, error null.
- Same inputs with `az deployment sub what-if --result-format ResourceIdOnly
  --no-pretty-print` returned Succeeded; exact resource-ID set comparison proved
  ten Creates confined to the new group, no Modify/Delete.
- Generated templates have no managed identity, role assignment,
  SecurityControl tag, storage account, PostgreSQL or Automation.
  Secret payload remains a secure parameter / protected extension setting.
- Safe `validation-result.json` and `what-if.json` persist in the ignored folder.
  App fixture, Python AST, shell and certificate checks passed in preparation.

## Deployment result (October 1)

Subscription deployment `af-demo-start-20261001-2101` and nested
`neo4j-source-only` both succeeded; only one create submission was made.
Final deployment timestamp: 2026-10-01T12:09:26Z.
Independent source verification completed at 2026-10-01T12:10:19Z.

- Exactly 100,000 nodes / 250,000 relationships and matching unique `id`
  counts; all nine label/type counts match the deterministic fixture.
- Neo4j 5.26.30 pinned image, 18 ONLINE ID indexes, read-only database,
  authenticated certificate-verified Bolt TLS; TLS required, Browser disabled.
- Container running, no OOM or restarts, 5 GiB limit; only Bolt port published.
- Source VM D2s_v5, zone 1, 64 GiB Standard SSD, Trusted Launch and key-only
  SSH configuration; private IP 10.76.1.4, no public NIC address.
- Exact source-only ten-resource inventory, including the implicit managed
  OS disk. Empty/nondelegated runner subnet with Storage service endpoint.
  No target subnet, runner VM, transfer storage, target DB or Automation.
- No managed identity, RG/child-scoped role assignment or SecurityControl tag.
  Existing subscription/inherited permissions were not changed.
- Native source shutdown is Enabled, UTC 1201, targeting only `neo4j-source`;
  next shutdown October 2 at 21:01 JST. Disk/NAT/IP remain billable afterward.
- One read-only live verification invocation; its retained result was reused.
  An earlier VM snapshot still showed Updating during bootstrap. A GET refresh
  after completion confirmed Succeeded/running; no bootstrap or command replay.
- Safe snapshots and `verification-summary.json` persist in
  `production-simulation/work/demo-start-20261001-2101/`.
  Fresh credentials/CA and `connection-guide.txt` are available in that folder.
  No installed-extension, retained workflow or product-source changes were made.

---

# Previous: Recreate approved transfer storage: September 29, 20:53 JST

**Status:** Deployed and Verified

User deleted the approved transfer account after public-network access failed,
added a SecurityControl:Ignore resource-group tag themselves, and explicitly
requests recreation of `af719bab06b08645aaaeb7d7`.

- Inspect the exact original approved deployment and current resource group.
- Recreate only the named account and required original private containers /
  account-scoped user role, preserving the extension's ownership contract.
- Public-network HTTPS access is intended, not anonymous blob access.
  Keep anonymous access and shared-key authentication disabled.
- Do not add/change policy exemptions or SecurityControl tags, or mutate
  existing source/runner/target resources and retained extension records.
- Validate the scoped template before deployment, then verify ARM settings,
  account-scoped RBAC and authenticated public-endpoint data-plane access.
- Same existing demo subscription/region; confirm exact scope before creation.
  No restore of deleted blob contents or automatic shutdown is implied.

## Approved recovery scope

Existing RG: `rg-agefreighter-demo-20260929-1955-start`, Japan East,
subscription `67c417f3-5a13-446c-afb9-40cd87f2fdb7`. The user-added group tag
is present; this operation will not change it or create any policy exemption.
The account is absent, globally nameAvailable=true; no residual account roles.

Recipe: ARM JSON through Azure CLI, reusing the exact exported template from
the user's approved deployment `af719bab06b08645aaaeb7d7-transfer`.
`az deployment group show` did not return an inline template; the supported
`az deployment group export` returned the original parameterless template.
Use a distinct recovery deployment `af719bab06b08645aaaeb7d7-recreate-2053`
to preserve the original deployment evidence.

Exactly three original resource definitions:
- Standard_LRS / StorageV2 / Hot account, same name/region/ownership tags,
  HTTPS-only, TLS1_2, publicNetworkAccess Enabled, defaultAction Allow,
  bypass None, anonymous Blob and Shared Key disabled.
- Private container `af-719bab06-b086-45aa-aeb7-d71cd56d21e9`, same workflow
  metadata, publicAccess None.
- Original role ID `2953b573-ea43-4497-9d08-9f29125cabfd`: Storage Blob Data
  Contributor at this account only, original User principal
  `af888774-9bdc-4d32-a7cf-f9cfaa8c93c7`, matching the signed-in operator.

No broader roles, account keys, restored blob contents, automatic deletion,
live workflow editing or existing VM/network mutations. New Standard LRS
storage/transactions retain their normal charges. Artifacts under
`production-simulation/work/demo-start-20260929-1955/storage-recreate-2053/`.

## 7. Validation Proof (storage recreation)

Completed 2026-09-29T11:56:03Z under azure-validate.
- [x] Azure CLI installed/authenticated; account and original user role matched.
- [x] Original ARM JSON parses, has no parameters and exactly three resources.
- [x] `az deployment group validate --subscription 67c417f3-5a13-446c-afb9-40cd87f2fdb7
  --resource-group rg-agefreighter-demo-20260929-1955-start --name
  af719bab06b08645aaaeb7d7-recreate-2053 --template-file
  production-simulation/work/demo-start-20260929-1955/storage-recreate-2053/original-template.json
  --mode Incremental` returned Succeeded, error null.
- [x] Same inputs with `az deployment group what-if --result-format ResourceIdOnly
  --no-pretty-print` returned exactly the original account/container/role IDs as
  Create; every existing resource was Ignore. No Modify/Delete.
- [x] Actual-template policy validation passed without adding exemptions/tags.
- [x] Static role scope and data role verified; original principal equals current user.
- Bicep compilation / Docker build: not applicable, unchanged exported ARM JSON.

## Recovery result

Recovery deployment completed Succeeded, error null, one submission.
Verified at 2026-09-29T11:58:30Z:
- Exact requested StorageV2 / Standard_LRS account exists in the original group
  with the original workflow/ownership tags.
- Public-network access Enabled, firewall default Allow, bypass None,
  HTTPS-only / TLS1_2, anonymous Blob and Shared Key disabled.
- Original private workflow container and account-scoped Blob Data Contributor
  assignment (same role UUID and original signed-in user) recreated.
- `az storage blob list --account-name af719bab06b08645aaaeb7d7 --container-name
  af-719bab06-b086-45aa-aeb7-d71cd56d21e9 --auth-mode login --num-results 1`
  succeeded over the public HTTPS endpoint from this desktop, zero entries.
- Endpoint: https://af719bab06b08645aaaeb7d7.blob.core.windows.net/
- No deleted blob content restored, no write probe or runner command executed,
  no original deployment/retained workflow or group-tag changes.
- Original deployment evidence preserved; verification-summary.json and live
  snapshots saved under storage-recreate-2053. Later policy changes may affect
  access; these are current verified settings, not a policy-exemption guarantee.

---

# Previous: AGEFreighter screenshot source-only environment: September 29, 19:55 JST

**Status:** Deployed and Verified

The user requests another clean screenshot environment. This repeat request
authorizes the same previously approved source-only architecture, subscription,
region, data size, cost-bearing footprint and 24-hour lifetime.

## Repeat scope

- New group `rg-agefreighter-demo-20260929-1955-start`.
- Subscription `67c417f3-5a13-446c-afb9-40cd87f2fdb7`
  (`MCAPS-Hybrid-REQ-51508-2023-rifujita`), Japan East, zone 1.
- One D2s_v5 private source VM, 64 GiB Standard SSD, pinned Neo4j 5.26.30;
  deterministic 100,000 vertices / 250,000 relationships, TLS and read-only.
- Same isolated VNet, NAT, outbound IP and source/runner subnets as before.
  Runner subnet remains empty/nondelegated; target subnet remains uncreated.
- No assessment/migration VM, Storage Account, Flexible Server, Automation,
  managed identity, role assignments or policy exceptions.
- Fresh 16-character ASCII-alphanumeric password, SSH key and TLS CA, stored
  in Git-ignored owner-only files, not Keychain.
- Native source VM shutdown by September 30, 2026 19:55 JST / 10:55 UTC.
  No automatic deletion or startup. Disk/NAT/IP and later extension resources
  are not stopped by the source VM shutdown.
- No numeric budget ceiling was previously specified; retain the same paid
  footprint and report any previous environment still present, without
  deleting or changing it.
- Recipe: standalone Bicep, reusing reviewed September 29 fixture/bootstrap.
  New artifacts: `production-simulation/work/demo-start-20260929-1955/`.

## Steps

- [x] Inspect account, existing groups, quota, SKU and pinned image.
- [x] Prepare fresh isolated artifacts and local fixture/TLS/secret checks.
- [x] Compile, ARM-validate and inspect create-only what-if.
  - [x] Bicep compilation.
  - [x] ARM template validation.
  - [x] Create-only what-if preview.
  - [x] Azure CLI authentication and current region/SKU/image/quota.
  - [x] Azure policy acceptance of actual template.
  - [x] Static role review: no managed identity or role assignment required.
- [x] Deploy once; reconcile the retained deployment rather than resubmit.
- [x] Verify live source data, private scope, shutdown and persistent handoff.

## Preparation evidence (September 29, evening)

Current subscription and tenant match the approved context. No previous demo
groups remain; the new group is absent. Compute quota is regional 0/101 vCPUs
and DSv5 0/100. D2s_v5 has no restrictions and supports Japan East zone 1.
Pinned Ubuntu 24.04.202609040 is available. Network quotas: VNet 0/1000,
Standard IPv4 public IP 0/1000, NAT 0/500. Visible policy assignments reviewed;
no exemption or network-policy relaxation is introduced.
Current Linux D2s_v5 retail price remains USD 0.124/hour, USD 2.976/24h for
VM compute only. Disk, NAT, public IP and traffic are additional.
The complete deterministic fixture passed local vertex/edge/endpoint/property
verification. Fresh CA/server certificate verified for 10.76.1.4.
Fresh password has exactly 16 ASCII alphanumeric characters, mode 0600 in an
ignored mode-0700 directory; its value was not emitted. Bootstrap AST and shell
syntax, including the read-only verification script, passed.

## 7. Validation Proof (September 29, evening)

Completed 2026-09-29T10:58:46Z under azure-validate.

- `az bicep build --file production-simulation/work/demo-start-20260929-1955/main-sub.bicep`
  passed with only a newer-version advisory.
- `az deployment sub validate --subscription 67c417f3-5a13-446c-afb9-40cd87f2fdb7
  --location japaneast --name af-demo-start-20260929-1955 --template-file
  production-simulation/work/demo-start-20260929-1955/main-sub.bicep --parameters
  @production-simulation/work/demo-start-20260929-1955/parameters.private.json`
  returned Succeeded, error null.
- Same parameters with `az deployment sub what-if --result-format ResourceIdOnly
  --no-pretty-print` returned Succeeded. Exact resource-ID set comparison proved
  ten expected Create changes confined to the new group. No Modify/Delete,
  transfer storage, PostgreSQL, Automation or role assignments.
- Safe validation-result.json and what-if.json retained in the ignored folder.
- Auth, current quota/SKU/image, local fixture/certificate/secret shape and
  permissions, template policy acceptance and static role review passed.
- Prior subscription/region/scope approval is reaffirmed by this repeat request.
  No existing resources are modified, removed, or given new access.

## Deployment result (September 29, evening)

Deployment `af-demo-start-20260929-1955` completed Succeeded, error null, after
one submission. Final independent verification: 2026-09-29T11:04:06Z.
The source is Neo4j 5.26.30 with exactly 100,000 vertices / unique vertex IDs
and 250,000 relationships / unique relationship IDs, all nine label/type
counts, 18 ONLINE indexes and read-only access. Authenticated TLS succeeded
with the fresh CA; local/guest CA hashes match. Container running, zero
restarts/OOM; approved fixture root SHA256 unchanged.

Live inventory matches the exact source-only footprint: one running D2s_v5,
zone 1, private IP 10.76.1.4, no public VM IP or managed identity, 64 GiB
StandardSSD_LRS. Runner subnet remains empty/nondelegated with a Storage
endpoint. Only source/runner subnets exist, and only runner-to-Bolt ingress
is allowed. No Storage Account, Flexible Server, runner VM or Automation.
Subscription role listing filtered to the group and descendants returned no
role assignments. No policy exception or access broadening was applied.
There is no runner yet, so source TLS and NSG checks do not claim end-to-end
runner connectivity.

Native shutdown is Enabled at 1055 UTC, next September 30 19:55 JST. It does
not delete resources, stop retained disk/NAT/IP charges, or cover future
extension-created resources.

Persistent handoff is in `production-simulation/work/demo-start-20260929-1955/`:
fresh owner-only credentials.env, source-ca.pem, connection-guide.txt,
protected deployment artifacts, live snapshots and verification-summary.json.
No Keychain, installed-extension change or retained-workflow mutation.

---

# Previous: AGEFreighter screenshot source-only environment: September 29

**Status:** Deployed and Verified

User requested another build after deleting the previous resource group.
Reuse the previously approved source-only scope, subscription, size and
24-hour lifetime. No existing resource deletion or workflow changes.

## Approved repeat scope

- Subscription `MCAPS-Hybrid-REQ-51508-2023-rifujita`
  (`67c417f3-5a13-446c-afb9-40cd87f2fdb7`); Japan East, zone 1.
- New group `rg-agefreighter-demo-20260929-start`.
- One private D2s_v5 VM / 64 GiB Standard SSD, Neo4j 5.26.30,
  deterministic 100,000 vertices / 250,000 relationships.
- Existing recipe: standalone Bicep and protected Custom Script bootstrap,
  reusing the reviewed fixture/bootstrap from September 27.
- VNet 10.76.0.0/16, source 10.76.1.4, empty non-delegated runner subnet
  10.76.2.0/24 with Storage endpoint, reserved uncreated target 10.76.3.0/24.
- No assessment/migration VM, Storage Account, Flexible Server, Automation,
  managed identity, RBAC assignments or policy exceptions.
- Fresh 16-character ASCII-alphanumeric password and fresh CA, saved in
  ignored owner-only files; no Keychain.
- Source native shutdown by September 30, 2026 16:11 JST / 07:11 UTC,
  within 24 hours of the repeat request. No automatic deletion.
- Same cost-bearing footprint as before; no numeric budget ceiling was
  previously specified. Retained disk/NAT/IP and future extension-created
  resources are not stopped by this source-only shutdown.

## Preparation and validation

- [x] Verify current account, previous-group state, quotas, SKU and image.
- [x] Generate isolated artifacts under `production-simulation/work/demo-start-20260929/`.
- [x] Compile and validate Bicep; verify exact create-only what-if.
  - [x] Bicep compilation.
  - [x] ARM template validation.
  - [x] Exact create-only what-if preview.
  - [x] Azure CLI authentication.
  - [x] Azure policy acceptance of the actual template.
  - [x] Static role review: no managed identity or role assignment required.
- [x] Deploy once and verify live TLS, cardinalities, IDs, indexes and read-only mode.
- [x] Verify source-only inventory and shutdown; persist connection guide.

## Preparation evidence (September 29)

`az account show` confirms the previously approved subscription/tenant.
The old September 27 group is still Deleting; leave it untouched. The new group
does not exist. `az quota list` and `az vm list-usage` confirm 0/101 regional
vCPUs and 0/100 DSv5 vCPUs. `az vm list-skus` confirms D2s_v5 in zone 1 with
no restrictions. Pinned Ubuntu 24.04.202609040 remains available.
Network usage: VNet 0/1000, Standard IPv4 public IP 1/1000, NAT 0/500.
Visible policy assignments inspected; no exceptions or access broadening.
Current Linux compute retail is USD 0.124/hour, USD 2.976/24h; disk, NAT, IP
and traffic are extra, as before. No hard spending cap is claimed.
The complete fixture passed local cardinality/endpoint/property verification.
Fresh server certificate verified against the fresh CA for 10.76.1.4.
Password is exactly 16 ASCII alphanumeric characters, mode 0600, Git-ignored;
its value was not emitted. Bootstrap syntax/AST validation passed.
These preparation checks preceded the single deployment submission.

## 7. Validation Proof (September 29)

Completed 2026-09-29T07:15:10Z under azure-validate:

- `az bicep build --file production-simulation/work/demo-start-20260929/main-sub.bicep`
  passed; only a newer-version advisory, no errors.
- `az deployment sub validate --subscription 67c417f3-5a13-446c-afb9-40cd87f2fdb7
  --location japaneast --name af-demo-start-20260929 --template-file
  production-simulation/work/demo-start-20260929/main-sub.bicep --parameters
  @production-simulation/work/demo-start-20260929/parameters.private.json`
  returned Succeeded, error null. Safe result retained as validation-result.json.
- Same arguments with `az deployment sub what-if --result-format ResourceIdOnly
  --no-pretty-print` returned Succeeded and exactly ten Create changes, all
  inside the new group. No Modify/Delete, Storage Account, PostgreSQL,
  Automation or role assignments. Saved as what-if.json.
- Auth, region/SKU/image/quota, local fixture/certificate/password checks
  and template policy validation passed. No policy exception introduced.
- Static role review passed: private standalone Neo4j does not use managed
  identities or Azure data-plane operations; none are provisioned.
- User's repeated same-scope build request authorizes deployment.

## Deployment result (September 29)

Subscription deployment `af-demo-start-20260929` completed Succeeded, error null.
One submission only; no redeployment or previous-resource mutation.
Final verification completed at 2026-09-29T07:21:31Z.

Independent authenticated TLS guest queries confirmed pinned Neo4j 5.26.30,
100,000 vertices / unique vertex IDs, 250,000 relationships / unique edge IDs,
all nine label/type counts, 18 ONLINE indexes and read-only access. The source
container is running with zero restarts and no OOM. Its CA SHA256 matches the
fresh local source-ca.pem. Fixture root SHA256 matches the approved fixture.
This is source TLS verification, not an end-to-end test from a runner that
does not yet exist.

Live resource inventory exactly matches the approved source-only footprint.
VM is running D2s_v5, zone 1, private 10.76.1.4, no public IP or identity,
64 GiB StandardSSD_LRS disk. Runner subnet is empty/non-delegated and has the
Storage endpoint; only source/runner subnets exist. Source ingress allows
Bolt only from runner and denies other ingress. No transfer Storage Account,
Flexible Server, migration/assessment VM or Automation.

Live role verification: `az role assignment list --all` filtered to the new
group and descendants returned no assignments. No identity was provisioned.
An initial CLI attempt combining --all and --scope was rejected locally;
the corrected read-only query succeeded. A local disk-property assertion
was corrected to the actual CLI key diskSizeGB, then all assertions passed.
Neither correction resubmitted or changed Azure resources.

Native shutdown is Enabled, UTC 0711, targeting neo4j-source; next occurrence
September 30 16:11 JST. No startup, deletion or Automation. Disk/NAT/IP charges
continue after shutdown, as do resources later created by the extension.
Old September 27 group was subsequently confirmed absent.

Persistent handoff: `production-simulation/work/demo-start-20260929/` contains
fresh mode-0600 credentials.env and private artifacts, source-ca.pem,
connection-guide.txt, safe deployment/what-if results, live snapshots and
verification-summary.json. Directory mode 0700; secrets Git-ignored.
No Keychain use, extension install, retained-workflow mutation or migration.

---

# Previous: AGEFreighter screenshot source-only environment: September 27

**Status:** Deployed and Verified

Requested September 27, 2026: prepare another Neo4j demo source using the
previous size, budget and 24-hour lifetime. Reuse the last approved source-only
architecture, not the original P1-sized proposal.

## Current request

- 100,000 vertices and 250,000 relationships, pinned Neo4j 5.26.30.
- One Standard_D2s_v5 source VM with 64 GiB Standard SSD in Japan East.
- Dedicated clean resource group; do not delete or mutate previous workflows.
- Private source, empty runner subnet and reserved uncreated target subnet.
- No assessment/migration VM, Storage Account or Flexible Server.
- 16-character ASCII-alphanumeric password in an ignored mode-0600 file;
  no Keychain.
- Native source VM shutdown within 24 hours, no Automation and no deletion.
- Confirm retained budget/context, inspect existing resources and current
  capacity/prices, reuse reviewed Bicep/bootstrap, validate, deploy, then verify.

## Azure context and budget

Confirmed reuse by the September 27 request ("same size, budget and deadline"):
`MCAPS-Hybrid-REQ-51508-2023-rifujita`
(`67c417f3-5a13-446c-afb9-40cd87f2fdb7`), Japan East, zone 1.
The September 26 source-only plan contained no numeric budget ceiling.
Preserve its exact cost-bearing footprint: one D2s_v5, one 64 GiB Standard SSD,
one NAT gateway and one outbound Standard public IP. No paid service upgrades.
An estimate is not a hard spending cap. Retail API returned HTTP 429 for the
disk/NAT price refresh; no total estimate or unverified ceiling is asserted.
Current Japan East Linux D2s_v5 retail price is USD 0.124/hour,
USD 2.976 for 24 hours of VM compute alone. Disk, NAT, IP and traffic are extra.
Deadline: September 28, 2026 11:42 JST / 02:42 UTC, within 24 hours of this request.
Native daily VM shutdown does not delete the disk/NAT/public IP or stop later
extension-created resources. No Azure Automation or automatic deletion.

New group: `rg-agefreighter-demo-20260927-start` (verified absent).
Prior demo groups are absent; no deletion was requested or performed.
Reuse `10.76.0.0/16` without peering, with private source `10.76.1.4`.
Preserve the previously approved demo runner-subnet Storage service endpoint;
do not create a Storage Account or change any account firewall/policy/RBAC.
Generate fresh credentials and CA under ignored
`production-simulation/work/demo-start-20260927/`.
Recipe: standalone Bicep, reusing the reviewed September 26 bootstrap and fixture.

## Validation and deployment

- [x] Review current Azure state and previous budget.
- [x] Prepare scoped artifacts and record shutdown deadline.
- [x] Validate quota, image, IaC and create-only what-if.
- [x] Deploy source-only resources.
- [x] Verify TLS, exact counts, indexes, credential permissions and shutdown.

## Preparation evidence

2026-09-27: subscription confirmed through `az account show`; new group absent.
`az quota list` is available; `az vm list-usage` reports regional 0/101 vCPUs
and DSv5 0/100. D2s_v5 has no restrictions and includes Japan East zone 1.
Pinned Ubuntu `24.04.202609040` is available. Network usage is VNet 0/1000,
Standard IPv4 public IP 0/1000 and NAT Gateway 0/500.
Visible subscription policy assignments were inspected; no exception is used.
Full local fixture generation/verification passed for 100k/250k, including
endpoints and properties. Fresh CA/server certificate verification for IP
10.76.1.4 passed. Password shape and mode 0600 verified without emitting it.
Python AST and generated bootstrap shell syntax passed.
## Validation Proof (September 27)

- Bicep compilation: `az bicep build --file
  production-simulation/work/demo-start-20260927/main-sub.bicep` passed.
- ARM template validation: `az deployment sub validate --subscription
  67c417f3-5a13-446c-afb9-40cd87f2fdb7 --location japaneast --name
  af-demo-start-20260927 --template-file .../main-sub.bicep --parameters
  @.../parameters.private.json` returned Succeeded, error null.
- What-if: the same subscription/name/template/parameters with `az deployment
  sub what-if --result-format ResourceIdOnly --no-pretty-print` returned
  Succeeded and exactly ten Create changes, all within the new group.
  No Modify/Delete, Storage Account, Flexible Server, Automation or role
  assignment. Proof saved in the ignored artifact directory `what-if.json`.
- Authentication, region/SKU, pinned image and current quotas passed.
- Static role review: no role assignments or managed identity in template.
- Policy validation: ARM validation/what-if accepted the actual template;
  visible policy assignments were reviewed; no exemption/bypass tags.
- Bootstrap and fixture were reused only after full local verification.
- Execution is authorized by the user's repeated same-scope deployment request.

## Deployment result (September 27)

Subscription deployment `af-demo-start-20260927` completed Succeeded with no
error. Exactly one VM is running: `neo4j-source`, D2s_v5, zone 1, private IP
10.76.1.4, no public IP or managed identity. No Storage Account, Flexible Server,
Automation or group-scoped role assignment exists.

Independent post-deployment read-only guest queries over authenticated TLS
confirmed Neo4j 5.26.30; 100,000 vertices and unique vertex IDs; 250,000
relationships and unique relationship IDs; all nine label/type counts; 18
ONLINE indexes; database read-only; container running, no OOM and zero restarts.
The local response parser initially expected separate stdout/stderr records;
Azure returned the merged documented markers. The retained response was parsed
and verified without repeating the guest command.
Guest/local CA SHA256:
`4dbae572c42ea01bbc5c7e03c0b6295bba3a047df1cee4191f052664c8be5091`.
Fixture root SHA256 matches the previous deterministic 100k/250k fixture:
`d0a1e58f368eb6462171028766d603556b1d0183b7e70c1f622fda8aea0dc44d`.

Runner subnet remains empty and non-delegated, with Microsoft.Storage endpoint
active. Live source NSG allows only runner CIDR to private Bolt/TLS; the future
10.76.3.0/24 target subnet is not created. No end-to-end runner connection is
claimed before an extension-created runner exists.
Native VM shutdown is enabled at 02:42 UTC daily, next September 28 11:42 JST.
Connection guide, newly generated CA, owner-only 16-character alphanumeric
credentials and non-secret verification evidence are in the new ignored folder.

---

# Previous: AGEFreighter 2.4.0 screenshot source-only environment

**Status:** Deployed and Verified

Approved September 26, 2026: use subscription
`67c417f3-5a13-446c-afb9-40cd87f2fdb7`, Japan East zone 1.
Create a new dedicated group `rg-agefreighter-demo-20260926-start`.
The superseded group is being deleted externally; do not operate on it.

## Current scope

- One private Neo4j 5.26.30 source VM, Standard_D2s_v5, 64 GiB Standard SSD.
- Synthetic supply-chain fixture: 100,000 vertices, 250,000 relationships,
  nine labels and nine relationship types. Stable integer key property: `id`.
- Private VNet `10.76.0.0/16`, source subnet `10.76.1.0/24`, empty
  non-delegated runner subnet `10.76.2.0/24`. Reserve `10.76.3.0/24` for
  the future extension-created target; do not create that subnet.
- Source Bolt/TLS accessible only from runner subnet; no public VM IP or SSH.
  NAT Gateway and outbound-only public IP provide package/runner egress.
- No migration/assessment VM, Storage Account, Flexible Server, Automation,
  managed identity or RBAC assignment.
- Generate a 16-character alphanumeric Neo4j password, save in an ignored
  owner-only local file. Use ARM secure parameters/protected extension settings.
  No Keychain use and no password in logs, repository or chat.
- Native VM auto-shutdown: 09:18 UTC daily; next September 27, 2026,
  18:18 JST. Stopped disks and NAT retain charges. No automatic deletion.
- Formal extension 2.4.0 creates its own runner, storage and target later.
  Its target is create-only and generated target credentials use SecretStorage;
  this task does not change released extension behavior.

## Recipe and artifacts

Standalone Bicep and protected Custom Script bootstrap, stored in ignored
`production-simulation/work/demo-start-20260926/`.
Preserve the historical Cosmos plan below; do not deploy its resources.

## Validation

- [x] Confirm subscription and location with user.
- [x] Verify DSv5 quota: 0/100 used; regional vCPUs: 0/101 used.
- [x] Verify D2s_v5 available in Japan East zone 1 without restrictions.
- [x] Resolve Ubuntu 24.04 image `24.04.202609040`.
- [x] Inspect current policy assignments; no policy exception requested.
- [x] Compile Bicep and validate fixture/bootstrap.
- [x] ARM validation and create-only what-if.
- [x] Deploy and verify authenticated TLS, exact counts, read-only source,
      all 18 indexes, empty runner subnet and absence of excluded resources.

## Validation Proof

2026-09-26 approximately 09:47 UTC: `az account show`, `az quota list`,
`az vm list-usage`, `az vm list-skus`, `az vm image show`, and
`az policy assignment list` succeeded. Actual template validation pending.
Local full 100,000/250,000 fixture verification and Python/shell syntax checks
passed. Password length/character class and mode 0600 verified without emitting
the value. Source CA and server IP SAN verification passed.
New group does not exist. Network capacity: VNet 2/1000, public IP 2/1000,
NAT Gateway 0/500. No role assignments occur in the template.
2026-09-26: `az bicep build --file .../main-sub.bicep` passed;
`az deployment sub validate --name af-demo-start-20260926` returned
Succeeded with no error. `az deployment sub what-if --result-format
ResourceIdOnly` returned exactly ten Create changes, all scoped to the new
approved group. No update/deletion, Storage Account, Flexible Server,
Automation or role assignment. Preview saved to the private artifact folder.

## Current deployment result

Subscription deployment `af-demo-start-20260926` and nested deployment
`neo4j-source-only` succeeded. Independent post-deployment guest queries
confirmed 100,000 vertices, 250,000 relationships and read-only access over
verified TLS. Bootstrap also confirmed unique integer IDs, all nine label/type
counts and 18 ONLINE indexes. Neo4j 5.26.30 is running without OOM/restarts.
Local and guest CA SHA-256 match. VM has no public IP or managed identity.

Live inventory contains exactly one VM and no Storage Account, Flexible Server
or Automation. Runner subnet has zero attached NICs and no delegation.
No group-scoped role assignments were created. The enabled native source
shutdown schedule is 09:18 UTC daily. Connection instructions, public CA,
owner-only `credentials.env` and evidence are in
`production-simulation/work/demo-start-20260926/`.

---

# Historical: agefreighter 2.x Cosmos DB Integration Deployment Plan

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
