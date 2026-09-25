# VS Code runner-first contract (version 2)

The extension is the control plane; a Linux x64 VM is the execution plane.
See the [design and release gates](../design/agefreighter-2.4.0-runner-first.md).
This contract covers the implemented runner control plane. The September 25,
2026 [qualification ledger](../../production-simulation/vscode-e2e/remaining-validation.md)
records 9/9 defined P1 base routes and 12/12 finite branches PASS, retaining
live, isolated-host and local-contract distinctions. M6 compatibility,
packaging and publication remain separate from those qualification results.

## Operations and authority

| Message | Effect |
|---|---|
| ready / accounts | Reuse the VS Code Azure account; no new login session. |
| groups / sources | Read subscription/RG ARM inventory; return candidate names, IDs and placement only. |
| placementOptions | Read existing resource groups and Azure regions for the selected subscription. Populate migration-RG/region dropdowns without deriving VM placement from RG metadata. No RG creation or peering mutation. |
| csv | Show a local file picker after CSV selection; no upload or CLI invocation. |
| preview | Validate typed source/runner input, matching official Linux release checksum, subnet, source placement, SKU/zone, quota, compute price, collisions and create-only ARM what-if. Save a 15-minute immutable preview. |
| deploy | Require workspace trust, matching preview hash, network/cost acknowledgments and modal confirmation. Lock the workflow, re-read state, recheck gates, persist intent, submit exactly one deployment PUT. |
| inventory | Run a complete, bounded source scan on the guest. CSV verifies sealed files; Neo4j uses the transactional count store; PostgreSQL streams all mappings in one exported repeatable-read snapshot; Cosmos streams all mappings during an operator-enforced immutable-source window. |
| migrate-csv / migrate-source | Start one retained create-only load UUID after exact source evidence, private target, same-VM resize and fresh health gates. Network sources are Neo4j, PostgreSQL or Cosmos. PostgreSQL and Neo4j credentials use protected parameters; Cosmos uses the runner identity. |
| target review / deployment | Import complete inventory first; review private PostgreSQL 18/AGE, HA-off sizing, subnet, budget and deadline. Save-only does not deploy. A separate approval creates the exact reviewed target. |
| same-VM resize | Check idle health, persistent disk/NIC/identity and zonal quota; approve deallocation, size change and restart with a retained bounded grant. Active/uncertain work blocks admission. |
| recovery inspection / explicit resume | Inspect the same retained job without writes; separately approve one continuation preserving graph generation/checkpoint/source/target identity. Never automatically resume. |
| restore / refresh | Load retained state and reconcile the exact deployment or operation ID. Status controls may submit a bounded guest status request; they never repeat the source read or load. |
| report transfer / import | Separately approve the sealed report, export it once and reconcile/import full bytes by exact identity, size and hash. Counts and full canonical qualification are distinct. |

The view never supplies templates, shell commands or artifact URLs. These are
constructed by typed extension code. The view receives no ARM token. The source
editor collects connection fields and mappings; passwords use native protected
prompts and optional scoped SecretStorage, not webview messages. AI has no
runner deployment tool. Mutations are direct user actions outside chat.

## Durable state

`<ExtensionContext.globalStorageUri>/runner-v2/<UUID>.json` stores a version-2
`RunnerRecord`: typed source selection, runner inputs, pinned artifact metadata,
generated template, immutable resource/deployment IDs, preview hash/expiry,
compute estimate and phase. Separate atomic files avoid lost cross-window
updates. A workflow-specific exclusive file lock prevents duplicate submission.
Crash locks are retained for operator review; no timed takeover is attempted.
Source credentials, ARM tokens and source rows must never enter these records.
This directory belongs to the extension host (remote host in remote VS Code).

Phases: `previewed → deployment-submitted → provisioned | failed | unknown`.
An ambiguous PUT failure becomes `unknown`. Submission intent is flushed before
PUT. Refresh can reconcile it, but a missing deployment does not authorize
resubmission. `provisioned` means ARM success only, not guest bootstrap, source
assessment, migration or verification success. Each guest phase has
independent durable evidence and an explicit transition gate.

## Current safety limits

- Runner placement uses an existing same-subscription compute subnet and
  resource group, without public IP/inbound SSH, peering/VPN or source firewall
  changes. Source, migration and network resource groups may differ.
- Source-read RBAC, workflow transfer storage and target deployment have separate
  approvals. Cosmos access grants the owned system-assigned identity built-in
  Data Reader on the exact NoSQL account. Storage grants stay account/container
  scoped. Guided target provisioning is private PostgreSQL 18/AGE, HA-off and
  create-only; it is not a general existing-target or SQL/PGQ deployment path.
- Azure source placement is checked where available. VM candidates are not
  database discovery results. PostgreSQL uses its availabilityZone; Cosmos uses
  actual data regions. Cross-subscription physical zone mapping is deferred.
- B2s_v2 is a reviewed starting SKU, not a globally lowest-price guarantee or a
  measured migration capacity. Other listed discovery SKUs require review too.
- Compute retail estimate excludes disks, NAT and network charges, which are
  disclosed and explicitly acknowledged; unknown/ambiguous compute prices block.
- Private connectivity and egress require operator confirmation now; only later
  guest probes can prove them. No ARM provisioning result is used as proof.
- CLI version is pinned to matching 2.4.x and SHA-256 checked before execution.
  The archive must be published; old binaries and mutable branch builds are not
  fallback installers. The explicitly opted-in development-artifact path is
  qualification-only, with reviewed commit/version/size/hash seals; it is not a
  published release or a signed build attestation. Source credentials are not
  included in customData.
- No automatic delete, cleanup, retry or resume exists. Source upload,
  assessment, target deployment, resize and load are explicit, separately
  retained actions. Operator-managed resource costs continue after the window
  closes.

- Native archive/removal covers eligible successful historical readiness
  controls only. General command retirement remains an operator-reviewed task;
  the extension blocks new dispatch at 25 retained managed control records.
- Interrupted desktop-lock recovery requires a verified same-boot identity.
  This implementation obtains it only on macOS/Linux; Windows review fails
  closed before recovery and retains the lock for manual investigation.
  Recovery also requires durable evidence-directory synchronization. Windows
  readiness-control removal fails that POSIX fsync requirement before DELETE;
  saved archive bytes may remain. Ordinary lock release and read-only
  reconciliation do not imply support for destructive recovery.

## Qualification and release boundary

R3–R5 now have the defined four-source migration, mapping, transfer, target,
resize, recovery and verification evidence. The ledger remains authoritative
about finite scope: NoSQL Gremlin-shaped documents are not native Gremlin API;
other-cloud selections used endpoint-only simulations; initial-runner quota and
some negatives retain local or isolated-host evidence. No every-source/timing,
continuous-worker, production-scale or lifetime disk-SKU claim is implied.

Release checks must confirm compatible packaged artifacts, retained advanced
LoadJob behavior, documentation and platform limits before publication. A
successful qualification run is not proof that GitHub or Marketplace has
published the matching binaries/VSIX.
