# B02 remaining live placement scope — September 24

This audit narrows the remaining live work; it does not mark B02 complete.
No synthetic response substitutes for installed-GUI or actual ARM acceptance.
No cloud write, VM startup, source query, credential access, workflow save or
operator-store mutation was performed by the B02 audit.

## Read-only observations and retention

Dedicated subscription: `67c417f3-5a13-446c-afb9-40cd87f2fdb7`.
Private, Git-ignored evidence directory:
`production-simulation/work/b02-b06-b07-20260924/read-only-b02/`.
Directory mode is 0700; each JSON evidence file is 0600. Per-file hashes are
retained in `SHA256SUMS.json`. Times and exact GET paths are in each receipt.
Azure CLI GET transport was used without exposing tokens or raw failure text.

| Evidence | SHA-256 | Observation |
|---|---|---|
| `result.json` | `0c153b247e1fad986fa1c74b1690ed3f6a532a5bab77619a3d944c4d03189d1f` | One subscription SKU GET, 76,924 rows inspected; 228 rows for the three wizard-offered sizes retained |
| `quota.json` | `f3b741bd3ffb879fa061d638d02ceef1039f82cb317b3eeb49f39bd9d93a0cf6` | Five regional Compute usage GETs, relevant quota rows retained |
| `vm-metadata.json` | `b08551cf87641c32f8208d339651e2b151dbfb63584a4088a3321efdedc0226c` | One subscription VM-list GET; 18 VM identity/region/zone/size summaries, no pagination |
| `proposed-scope.json` | `8afeabb18a29b0d042f4695f122c3ffb774e6a14fd28d08b1bd3aeacf1ab8ce8` | Proposed VNet exact ID returned 404; existing group returned 200; Japan East and Southeast Asia appear in subscription region catalog |

An additional read-only subscription VNet-list inspection found three VNets,
all in Japan East. That console inspection is supplementary; its original raw
response was not retained. None could reach the Southeast Asia SKU refusal
because production preflight checks the VNet region before checking SKU data.

## Unknown-zone VM: no new VM is necessary

Actual existing VM metadata has an absent zone for both:

- `RG-AGEFREIGHTER-VALIDATION-JPE/vm-agefreighter-validation`, Japan East,
  `Standard_D4s_v5`.
- `RG-AGEFREIGHTER-COSMOS-DEV/vm-agefreighter-runner`, Japan East,
  `Standard_B2s`.

Use the validation VM only as a read-only ARM source candidate in an unsaved
normal installed wizard. The UI explicitly says candidates do not establish
database identity. This requires no Neo4j install, source connection, VM start,
or change to either older group's resources.

Required actual GUI observations:

1. Select Neo4j / Azure and the exact trial subscription, then the existing
   validation source group; discover candidates.
2. Keep the approved P1 migration group and P1 Japan East compute subnet as
   runner placement. Set an explicit runner zone before selecting the actual
   unknown-zone candidate, so stale selection clearance can be observed.
3. Select `vm-agefreighter-validation`. Observe inferred Japan East, cleared
   runner zone, unknown-zone review guidance and disabled prerequisite preview.
   A known-zone P1 candidate control is separate; switching source groups
   already clears the zone, so do not mislabel that alone as candidate-specific
   unknown-zone clearance.
4. Explicitly choose runner zone 1, `Standard_B2s_v2` and the existing P1
   non-delegated `runner` subnet. Submit one prerequisite preview. With valid
   current placement, the normal release gate should refuse before pricing,
   what-if or persistence if the matching release remains absent.
5. Capture the exact visible results and independently compare operator-store
   bytes. Close the unsaved wizard. Do not continue to source configuration.

### Supervisor-observed installed GUI result: PASS for unknown-zone VM

The root supervisor completed the above case in the actual normal signed-in
VS Code UI. Its bounded observation window is September 23 **22:58:25 through
23:02:08.211700 UTC** (September 24 JST), from the surrounding cost read through
the final store check. Exact per-click timestamps were not recorded. The earlier
approximate 23:04–23:07 time was corrected before this record was finalized.

Installed JavaScript SHA-256:
`3108910e2933f73b1b1d0edf92458e7384e2008261469c56da2f19789366da67`.

The root first observed actual `af-n44-source` infer Japan East / zone 1.
Changing source group to validation-jpe cleared the zone. Crucially, the root
then **explicitly selected zone 1 after group discovery and before selecting
`vm-agefreighter-validation`**. Selecting that actual unknown-zone candidate
cleared the explicit zone again, displayed unknown-zone review guidance, and
disabled prerequisite preview. Explicit zone 1 plus the existing P1 `runner`
subnet then reached the visible matching AGEFreighter 2.4.0 Linux
release/checksums refusal. The unsaved wizard was closed.

No new workflow, source connection, deployment or cloud mutation followed.
The independently read post-state receipt confirms the same **93 operator files
with exact names, bytes, hashes and modes unchanged**, baseline SHA-256
`af1200be29f21f408ae01c4f76ae15742c6837f2aaddb53681c95b8b0f73ad94`.

The B02 subagent recorded this result from the supervisor's report and read the
post-store receipt; it did not independently observe or control the GUI.
The supervisor viewed UI text and a CUA screenshot, but no screenshot file was
retained. Private receipts:

- `read-only-b02/unknown-zone-gui-supervisor-receipt.json`, SHA-256
  `ac575785195c5170b4a31b1224ecd4ae4d508d48e60d480812f2eb0f3269ee60`.
- `b02-unknown-zone-post-state.json`, SHA-256
  `96d5f6ce2ebbb0a79e3f3046d3d58ee3d74d28e6d97d9dc7fc21339c274b56df`.

A byte-identical, credential-free public copy of the observation receipt is
[`b02-unknown-zone-gui-20260924.json`](b02-unknown-zone-gui-20260924.json), with
the same `ac575785…` hash above. It retains observer/recorder attribution,
bounded timing, installed bundle identity and the post-state receipt hash.

This closes the actual unknown-zone VM transition only. It does not qualify
the unavailable-SKU or initial-runner quota paths.

## Unavailable SKU: precise empty-network proposal

Fresh subscription SKU metadata provides a real refusal candidate:
`Standard_D2s_v5` and `Standard_D4s_v5` in `southeastasia` have a
`NotAvailableForSubscription` restriction on zone 2. Zones 1 and 3 are not
restricted in that response. In Japan East, all three wizard sizes remain
unrestricted in zones 1/2/3. This is control-plane eligibility, not a VM
allocation guarantee.

The unchanged production `parseComputeSkus` parser was also run locally against
the retained real metadata: Southeast Asia `Standard_D2s_v5` yields allowed
zones `[1,3]`; the Japan East control yields `[2,1,3]`. This validates the
proposed case selection only, not an installed-GUI preflight result.

Only the following new resource is proposed for B02:

```
/subscriptions/67c417f3-5a13-446c-afb9-40cd87f2fdb7/resourceGroups/rg-af-vscode-p1-20260905-a/providers/Microsoft.Network/virtualNetworks/af-b02-sku-20260924
```

The exact new VNet ID returned 404. Its reviewed local-only request body is
[`b02-sku-vnet-20260924.json`](b02-sku-vnet-20260924.json): Southeast Asia,
`10.253.240.0/24`, one empty non-delegated `runner` subnet
`10.253.240.0/27`, default outbound access disabled. No VM, NIC, public IP,
private endpoint, NAT, peering, DNS link, delegation, or existing-VNet update
is included. This is a proposed JSON body, not a submitted deployment.
Fresh collision/ownership checks, the current bounded execution authorization
and subsequent cleanup remain the supervisor's responsibility.

After authorized creation and fresh actual ARM readback:

1. New unsaved installed wizard: Neo4j / on-premises, trial subscription,
   existing P1 migration group, Southeast Asia, zone 2, `Standard_D2s_v5`,
   and the new VNet's exact `runner` subnet ID. No endpoint or credential.
2. Click prerequisite preview once. Require the visible refusal
   `The selected discovery SKU is not available in this subscription/zone.`
   A subnet/region/refreshed-catalog error does not satisfy this case.
3. Change only zone to 1 and repeat once. Require placement to reach the
   mandatory release refusal; do not approve deployment. If fresh SKU/quota
   state no longer supports either expected outcome, stop and revise evidence.
4. Capture screenshots and unchanged operator bytes; close the unsaved form.
   Remove only the exact newly owned VNet after verifying no attached resources.

## Quota evidence reconciliation

[`recovery-execution-20260915.md`](recovery-execution-20260915.md#target-preflight-safely-refused-regional-vcpu-exhaustion)
already records a **real installed-GUI private-target/resize preflight quota
refusal**: regional cores 100/101, DSv5 96/100, Bsv2 2/100. The target/migration
metadata stayed absent; no target/subnet/credential/resize/load was submitted.
The original record explicitly credits B02/B09. Preserve that exact evidence
instead of claiming no live quota denial exists, or consuming quota to repeat it.

That historical result does **not** prove an initial-runner wizard quota
refusal. Its initial-runner handler path is covered separately by the thirteen
synthetic placement-to-panel contracts in
[`placement-panel-contract-20260922.md`](placement-panel-contract-20260922.md).
Do not silently widen either claim.

Fresh regional usage is Japan East 82/101, DSv5 72/100, Bsv2 8/100. Southeast
Asia, South Central US, West Europe and Japan West each return 0/100 for all
three relevant quotas. This bounded five-region inspection found no natural
initial-runner quota refusal. No quota changes or exhaustion work are proposed.

For the finite B02 row, retain the original explicit credit for the real
private-target/resize quota denial and the separately stated initial-runner
synthetic-only limitation. Restoring this already-recorded evidence is evidence
reconciliation, not newly qualifying the initial-runner path. The remaining
live B02 acceptance after the actual unknown-zone transition is the
unavailable-SKU rejection and its valid-placement control. No every-screen
quota-denial claim or resource-consuming quota-exhaustion trial is added.
