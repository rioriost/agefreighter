# B02 live placement qualification — September 24

The finite B02 placement branches are qualified at the evidence layers stated
below: actual unknown-zone transition, actual unavailable-SKU refusal with its
valid-zone control, and the previously recorded actual private-target quota
refusal. Initial-runner quota refusal remains synthetic-only. The new empty SKU
fixture VNet has been removed and its absence independently verified; the B02
case is complete. This does not close the combined B06/B07 execution session.
No synthetic response substitutes for installed-GUI or actual ARM acceptance.
The supervisor created the approved empty VNet and operated the actual GUI;
the B02 subagent audited the evidence and, under explicit bounded delegation,
removed only that new VNet after archiving the evidence.

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

This earlier result closes the actual unknown-zone VM transition only. It does
not qualify the unavailable-SKU or initial-runner quota paths. The later actual
SKU result is recorded at the end of this document.

## Unavailable SKU: reviewed empty-network preparation

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
is included. This body was preparation at the time of review; the supervisor
subsequently created the approved fixture and retained the actual readback
below. Fresh collision/ownership checks, bounded execution authorization and
subsequent cleanup remain the supervisor's responsibility.

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
reconciliation, not newly qualifying the initial-runner path. After the actual
unknown-zone transition, the remaining live B02 acceptance was the
unavailable-SKU rejection and its valid-placement control; the later result
below qualifies that pair. No every-screen
quota-denial claim or resource-consuming quota-exhaustion trial is added.

## Fresh checks after the September 24 execution approval

The supervisor recorded the new combined-scope approval at **00:18:14 UTC**.
The B02 subagent then performed four GET-only checks between **00:19:31.186551
and 00:19:39.333141 UTC**, without UI control, source operations, credential
reads or cloud writes. These are execution preconditions, not GUI acceptance.
The exact methods, paths and observation times are retained privately in
`production-simulation/work/b02-b06-b07-20260924/execution-readonly/`;
files are mode 0600 and the directory is mode 0700.

| Receipt | SHA-256 | Fresh observation |
|---|---|---|
| `b02-southeastasia-sku.json` | `835744713933ea282b80afc7a42d76efc2d53019569035df480c1189766f45b5` | Southeast Asia query returned 1,447 rows with no continuation; `Standard_D2s_v5` still restricts zone 2 as `NotAvailableForSubscription`, while zones 1 and 3 have no listed restriction |
| `b02-vnet-absence.json` | `34a50d86dee772b6fb20e27480502ec12de9821241f76590406a95923bdbc027` | Exact `af-b02-sku-20260924` VNet returned `ResourceNotFound` at 00:19:35.821841 UTC |
| `japaneast-quota.json` | `3c3f0bb9a101fb1c4b21ce87bcb71ee106b8c09b817fb02d131671419564aea3` | Regional cores 82/101, DSv5 72/100 and Bsv2 8/100 |
| `target-subnet-availability.json` | `196cc8ba7b7e2e0d8eca64cdbd1340bb9adc2dde708e58d722b9a14c1bdfa935` | `10.246.26.0/24` remains inside the existing VNet address space and overlaps no listed subnet; exact `afpg-0f83d520cd0347928e38` subnet name is absent |

`b02-SHA256SUMS.json` seals these four receipts. No reservation or allocation
is implied by the SKU, quota or address-space reads. Actual creation, ownership,
installed-GUI observations and cleanup remain separate supervisor actions.

## Supervisor-observed installed GUI result: PASS for unavailable SKU

The supervisor created only the approved empty Southeast Asia fixture VNet
for this B02 case. Its private `execution/b02-vnet-created-readback.json`
reports `Succeeded` at **00:25:21.436286 UTC**, the exact approved resource ID,
ownership tags, address space, sole empty `runner` subnet, no delegations,
disabled default outbound access and no peerings. Receipt SHA-256:
`59a91f8a35226ae1d61312014f2299f7d5249f6644010a2f02b23f8d48fc961d`.

The actual normal installed VS Code extension used the approved candidate
bundle SHA-256
`7a2aa5ddf13454ff6dab4d4a8ad2cb9a172842c9c4fbcc7b5d8e94d3f79d1d34`.
An unsaved Neo4j / on-premises wizard selected the trial subscription and P1
resource group, Southeast Asia, `Standard_D2s_v5`, and the fixture's `runner`
subnet. No endpoint or source credential was needed.

1. Zone **2**: prerequisite preview submitted at **00:28:15.866 UTC**;
   the supervisor observed at **00:28:43.923 UTC**:
   `The selected discovery SKU is not available in this subscription/zone.`
2. Changing **only zone to 1**, preview submitted at **00:31:36.200 UTC**;
   the supervisor observed at **00:31:49.949 UTC**:
   `The matching AGEFreighter 2.4.0 Linux release/checksums are not available. No Azure deployment was submitted.`

Both messages were observed through actual CUA accessibility state and viewed
screenshots. No separate screenshot image files were retained. The B02
subagent did not observe or control that GUI; it independently checked the
supervisor's receipt, installed bundle bytes, fixture readback and local store
integrity. The release refusal is the valid-placement control, not evidence of
a successful deployment. No new workflow or source operation followed either
preview.

The supervisor's public credential-free receipt is
[`b02-sku-gui-20260924.json`](b02-sku-gui-20260924.json), byte-identical to
private `execution/b02-sku-gui-root-receipt.json`, SHA-256
`958aa068696e2062d5f5eb2f9df2666acea60f602a7f24efa19ddadd54a7fc7c`.
It records **94 operator files unchanged in names, bytes, hashes and modes**,
against the pre-preview baseline SHA-256
`fb40943295f5d4f3d51c33d7ffd9eda7f3fef2e2e0254c050a4035b126964bbe`.
At **00:33:45.078914 UTC**, the B02 subagent independently re-hashed all 94
normal `runner-v2` files and confirmed the same exact result and current
installed bundle identity. That audit is retained publicly as
[`b02-sku-independent-audit-20260924.json`](b02-sku-independent-audit-20260924.json),
byte-identical to its private execution receipt, SHA-256
`bc51e0ec5027e7fd8116685659f9bec3153ccf9cc72749f1f04098995c1ba49c`.

This completes the finite unavailable-SKU branch and its control. Together
with the actual unknown-zone transition and the historical exact target-quota
refusal above, the finite B02 placement coverage is satisfied. It does not
promote the initial-runner quota contracts to live evidence. The GUI receipt's
cleanup-pending field correctly records its observation time; the subsequent
cleanup completion below supersedes that field without altering the original
receipt.

## Exact new fixture cleanup: complete

After the supervisor left the B02 wizard, it explicitly delegated deletion of
only `af-b02-sku-20260924` to the B02 subagent. Before deletion, a fresh VNet
and subnet GET passed the pinned cleanup helper's ownership checks plus
additional attachment checks. Exact ownership tags, CIDRs, original resource
GUID, sole empty non-delegated `runner` subnet, disabled default outbound
access and absence of peerings or attached resources matched the reviewed
scope. The original `ResourceNotFound`, exact PUT/body seal and readback were
also reconciled.

All **28 B02 evidence files** were archived before deletion in the private
0600, fsynced `execution/b02-evidence-before-cleanup.tar` (**849,920 bytes**),
SHA-256 `0e5305f4ce9711308c4eee4bbb95f4e6949dd75bde7b521dc8b2d564fbedfddd`.
The archive and its file manifest were sealed and reverified before mutation.
The unchanged cleanup helper was pinned to SHA-256
`95ffc085555c3006b1c0cb87742a2d17ec347654d229146137b4e334beb7af6a`.

Its `mutation_once()` sent exactly one application-level DELETE invocation for
the exact new VNet. The common private `safety/cleanup-execution/` journal
retains the create-only durable intent and reply under key
`8c4ffc87c666b98af88d539dfb888889170beacc23414a1aab71b9514f994647`
(SHA-256 of lowercase resource ID plus `:DELETE`). No deletion was replayed.

Independent GETs at **00:45:51 UTC** confirmed the exact VNet absent and the
existing parent resource group still present with `Succeeded`. The subnet GET
also returned `ResourceNotFound`, with the same parent-VNet missing error
digest. It did **not** return a `ParentResourceNotFound` code; the child absence
conclusion is supported by the separately proven absence of its exact parent.
The accepted DELETE reply alone was not used as completion evidence.

[`b02-cleanup-20260924.json`](b02-cleanup-20260924.json) is the public,
credential-free, byte-identical copy of the private cleanup completion receipt.
Its SHA-256 is
`4910075601294c8a58e7900c89a473baa1ce0174438dac5e22235bda53b62392`.
It retains the archive/preflight/journal/verification seals and the exact GET
results. Only the new fixture VNet was mutated by this cleanup; the parent
resource group and all VM, PostgreSQL, source, shared-network and role resources
were outside its mutation scope. **B02 finite qualification and its dedicated
fixture cleanup are complete.** Initial-runner quota live coverage remains
unclaimed, and combined B06/B07 completion remains separate.
