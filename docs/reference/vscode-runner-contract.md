# VS Code runner-first contract (version 2)

The extension is the control plane; a Linux x64 VM is the execution plane.
See the [design and release gates](../design/agefreighter-2.4.0-runner-first.md).
This contract currently covers R1/R2, not a completed remote migration service.

## Messages and authority

| Message | Effect |
|---|---|
| ready / accounts | Reuse the VS Code Azure account; no new login session. |
| groups / sources | Read subscription/RG ARM inventory; return candidate names, IDs and placement only. |
| placementOptions | Read existing resource groups and Azure regions for the selected subscription. Populate migration-RG/region dropdowns without deriving VM placement from RG metadata. No RG creation or peering mutation. |
| csv | Show a local file picker after CSV selection; no upload or CLI invocation. |
| preview | Validate typed source/runner input, matching official Linux release checksum, subnet, source placement, SKU/zone, quota, compute price, collisions and create-only ARM what-if. Save a 15-minute immutable preview. |
| deploy | Require workspace trust, matching preview hash, network/cost acknowledgments and modal confirmation. Lock the workflow, re-read state, recheck gates, persist intent, submit exactly one deployment PUT. |
| restore / refresh | Load retained state and query the exact deployment ID. Never restart, replay or replace resources. |

The view never supplies templates, shell commands or artifact URLs. These are
constructed by typed extension code. The view receives no ARM token. No source
password or database endpoint is collected in this preview. AI has no runner
deployment tool. Mutations are direct user actions outside chat.

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
assessment, migration or verification success. Each future guest phase needs
independent durable evidence and an explicit transition gate.

## Current safety limits

- Existing same-subscription compute subnet, existing RG; no public IP/SSH,
  peering/VPN, source firewall changes, RBAC assignment or target database.
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
  fallback installers. The guest does not download credentials in customData.
- No delete, cleanup, resize, remote profile, source upload or load action exists
  in this preview. Operator-managed resource costs continue after the window closes.

## Qualification before release

Require unit and extension-host tests plus isolated Azure integration tests for
quota/capacity denial, delegated subnet/egress failure, checksum rejection,
interrupted deployment, cross-window retry, retained evidence and actual guest
readiness. R3–R5 need four source paths, protected credential transport, CSV
transfer, mapping, target provisioning, compatible same-VM resize and durable
load/recovery/verification evidence before claiming end-to-end support.
