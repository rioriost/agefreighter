# B09 dedicated two-fault session — September 23

This is the retained preparation/setup history. Both live fault gates were
subsequently completed in the [September24 execution](b09-live-faults-20260924.md).
Use that record for current shutdown and cleanup status.

## Historical checkpoint — 14:49 UTC: account-only exception needs confirmation

The final `x8TTEC` companion is active. Its source-free retained seed is unchanged
on activation. The normal native storage approval sent one storage deployment
PUT; ARM reports Succeeded at14:45:58UTC, correlation
`e9c777a8-5934-447d-82e7-2b1ef564febc`. The GUI reconciles the account as
`ready — public network: Disabled (provisioning is not transfer readiness)`.
No archive upload or runner deployment has been attempted; the VM clock has not
started. The exact-VM deadline guard is running, before any runner PUT.

Fresh policy events explicitly identify `StorageAccount_PublicNetwork_Modify`
under `MCAPSGovDeployPolicies` replacing this account's public-network setting.
The current policy definition declares the resource/RG exclusion-tag defaults
`SecurityControl=Ignore`. Earlier approvals for other trial accounts were scoped
to those accounts and do not authorize this new exception.

The prepared additional action is one PATCH to only
`af0d8b4bc9102c4d74827006`: merge `SecurityControl=Ignore` into its existing three
ownership tags and set `publicNetworkAccess=Enabled`. This excludes this account
from policies honoring that tag and permits authenticated public HTTPS access.
No RG/subscription tag or policy assignment changes are proposed. Anonymous Blob
access and shared keys stay false; HTTPS-only and TLS1.2 remain required. Verify
the effective properties and authenticated container access before using the
already-approved immutable negative archive. Remove this account and its scoped
role in the already-approved cleanup; all prior resource/data preservation and
the USD5/60-minute/absolute-expiry bounds remain unchanged.

Private review body `storage-exception-proposed-body.json` has SHA256
`b3fad123c65e44a7fd3af14d36967d1926b0a0cc301cd2caa5b256ec439cecba`.
It has not been submitted. The new policy exception is a material access change
outside the earlier exact approval, so explicit confirmation is required before
that additional action. The two intended B09 live gates remain uncredited.
See the [sanitized checkpoint](evidence/b09-fault-preparation-checkpoint-20260923.json).
Independent local review passes13checks, including the single retained storage
PUT intent, absence of every later effect intent, and the unchanged93-file
normal operator store. A retained intent is not an instrumented wire count.

**Exact live scope and cleanup approved at2026-09-23T14:14:51.122892Z.
The official dedicated host was launched, initially awaiting interactive Azure
sign-in; the user subsequently confirmed sign-in. No cloud resource creation,
deployment, fault execution or new case PASS is established at this checkpoint.**
This continuation is supervised by Astra xhigh with three Astra high
agents for response-loss support, bootstrap-failure support and independent B10
evidence review. The starting commit is `88433be`.

B10 already passes its finite acceptance definition. Its [independent retained
evidence audit](b10-retained-evidence-audit-20260923.md) passes43checks and6negative
controls; another inventory, crash or migration is not required. B09 retains two actual-service gates: deployment response loss and
terminal bootstrap failure. The approval covers the exact dedicated scope below;
preparation, launch and sign-in do not establish either fault result.

## Exact approved scope

- Subscription: `67c417f3-5a13-446c-afb9-40cd87f2fdb7`.
- Existing resource group: `rg-af-vscode-p1-20260905-a`.
- Fresh workflow: `0d8b4bc9-102c-4d74-8270-061ce31ce163`.
- Region / availability zone: `japaneast` / `1`.
- One VM: `af-0d8b4bc9102c4d748270`, `Standard_B2s_v2`.
- Existing VNet / compute subnet: `vnet-af-vscode-p1` / `runner`.
- Source choice: local CSV, without selecting, uploading or reading source data.
- Maximum 60 minutes from the first VM deployment request; begin deallocation
  by minute 55, or earlier after terminal evidence, failure or 15 minutes idle.
- Proposed incremental budget: USD5, within the unchanged cumulative USD800
  ceiling. A time/budget bound is not a promise that delayed billing is current.

Let `B` be
`/subscriptions/67c417f3-5a13-446c-afb9-40cd87f2fdb7/resourceGroups/rg-af-vscode-p1-20260905-a/providers`.
The proposed new resource IDs are:

| Resource | Exact ID relative to `B` |
| --- | --- |
| VM | `Microsoft.Compute/virtualMachines/af-0d8b4bc9102c4d748270` |
| OS disk, 64 GiB Standard SSD LRS | `Microsoft.Compute/disks/af-0d8b4bc9102c4d748270-os` |
| NIC | `Microsoft.Network/networkInterfaces/af-0d8b4bc9102c4d748270` |
| Deny-all-inbound NSG | `Microsoft.Network/networkSecurityGroups/af-0d8b4bc9102c4d748270` |
| Runner deployment | `Microsoft.Resources/deployments/af-0d8b4bc9102c4d748270` |
| Transfer account | `Microsoft.Storage/storageAccounts/af0d8b4bc9102c4d74827006` |
| Private container | `Microsoft.Storage/storageAccounts/af0d8b4bc9102c4d74827006/blobServices/default/containers/af-0d8b4bc9-102c-4d74-8270-061ce31ce163` |
| Transfer deployment | `Microsoft.Resources/deployments/af0d8b4bc9102c4d74827006-transfer` |

The signed-in desktop user receives Storage Blob Data Contributor on only this
new account. Production allocates that assignment UUID during the separately
reviewed native storage approval. The new VM's managed identity receives Storage
Blob Data Reader on only this container; its assignment name is exactly the
workflow UUID. Preserve both assignment IDs for exact cleanup.

The storage HTTPS endpoint is network-public with authenticated access;
anonymous Blob access and shared keys are disabled. It is not a private-endpoint
account. No public VM IP, new VNet/subnet/NAT, source/target restart, existing
network change, accepted-data change or credential change belongs to this scope.

## Exact negative artifact and fault boundary

The locally prepared derivative has SHA-256
`3c33a179916ec08a83ca8ccb3c19e7682862d0382a63b2c05b5369a3cb124e33`,
18,378,050 bytes. It retains only the authentic `agefreighter` member of the
reviewed `d40d6ccc9a4ddf6e2ca626392cd7bf83140ed6c7` build and intentionally omits
`agefreighter-tools`. The binary SHA-256 is
`386c3ede5ff1687a5e0fe9d1948faf775561db7c4caba14dcfddc3de5087e9b3`.
The original 37,197,546-byte archive remains unchanged at SHA-256
`2321022975f85c73068a54fd21a287e2802d5bc33fa21eb9323c142dd7262ff6`.

The derivative is **an intentionally unusable packaging fixture, not an
unmodified build, usable runner or release**. Its production-compatible manifest
identifies the source of its included binary; the separate provenance file must
be reviewed before the native pinned-artifact and VM approvals. It may only be
uploaded to this workflow's new content-addressed Blob. The unchanged bootstrap
must fail at extraction of the absent second member before either executable is
installed or `bootstrap.complete` is created.

The disposable companion retains the unchanged production controller and normal
native approvals, using an instance-bound adapter around the real Azure session.
It must send exactly one approved runner deployment PUT, retain a successful
Azure acceptance receipt, then deliberately withhold that result from the
controller. This is a client/controller boundary fault after an adapter received
the real response. It is not evidence of packet loss, TLS failure or an Azure
service outage. No fault hook is included in the production extension.

## Required live evidence and end state

1. Normal real preflight/what-if, native review and a durable deployment intent.
   Retain the exact runtime preview/template hashes and artifact binding.
2. One actual successful deployment acceptance followed by production `unknown`;
   independent ARM GET/operation evidence for that same deployment; production
   reconciliation without a second deployment PUT or another workflow.
3. Terminal cloud-init failure at the intended missing-member extraction, absent
   bootstrap marker and both installed executables, no worker/source operation.
   A different earlier failure is retained without credit for this intended case.
4. One normal explicit guest-readiness control and GET-only reconciliation of that
   control. It must remain failed/unready with no `guestReady` receipt and no
   assessment dispatch. Allow up to three separately bounded, source-free guest
   observations (`af-b09-observe-01` through `af-b09-observe-03`, only beneath this
   new VM) to obtain terminal evidence; no retries of the deployment. The
   [reviewed observer](b09-bootstrap-preparation-20260923.md) SHA-256 is
   `1439353225906bc653a849357f8b70346b59b395d9ea356f73329f36ed82f209`;
   each180-second request returns one bounded sanitized JSON line and does not
   alter the guest. Its prepared body has not been submitted.
5. Retain local sanitized evidence before deallocating the exact new VM. If
   specifically approved, remove only the enumerated new resources, their exact
   new role assignments, observation controls and two deployment records.
   Preserve every existing VM, disk, database, accepted report and workflow.
   Deallocation alone leaves chargeable disk/storage resources.

The authority for the action-time confirmation was the Computer Use tool's rule
for running unpublished software and creating security-sensitive access, plus the
explicit scope boundary in the preceding live session. Necessary reversible
local preparation is covered by the user's present completion request. The
specific companion, derivative, new cloud/RBAC scope and cleanup were prepared
for review before the user's14:14:51.122892UTC confirmation. That approval does
not extend the exact resource set, runtime limit or source-free boundary.

## Reviewed disposable companion

The original approved preparation used by the first dedicated host was
`/private/tmp/af-deployment-lost-response-au6bmG`.
The runtime bundle SHA-256 is
`df442d480079e66a7d2ce76cfd22c8d5255b2e3df873b7ae93bd7f342b74c6ef`;
the scope-file SHA-256 is
`35e0cd828ffdfc5e40ab790597543dcc18d2ea0ac1b8af24a2f133ca7c922908`.
The loader checks both before activation. All663bundled source-file hashes
were independently matched. The earlier `dW8djI` preparation is superseded and
must not be launched. The final preparation preserves the setup expiry
`2026-09-23T18:08:48.982Z` (September24 03:08:48JST); this is not permission for
four hours of compute. The separate60-minute VM bound and earlier stop conditions
still apply, and the monitor must also stop before setup authorization expires.

Activation is by explicit B09 command, after normal interactive Microsoft/Azure
authentication. Official Azure Resources0.13.0 is prepared in this separate
profile. Authentication caches or credentials are not copied. The companion
seeds only a local production draft and exposes two B09 commands; storage,
artifact and VM confirmation remain actual native production approvals with
the intentional negative-fixture explanation added visibly.

The companion is single-use. An Extension Host restart fails closed; independent
external GET reconciliation is available but a new activation or deployment is
not implied. This trial does not deliberately crash its host. A prepared scope
or helper receipt never asserts live PASS automatically.

Final local validation: **708/708extension unit tests**, typecheck, normal build
and host-test compilation passed. This includes14companion tests:7transport,
6production lifecycle/stage and1compiled-runtime smoke test, all with inert
Azure responses. The23Python fixture/observer tests also passed. Normal production
build inspection found667modules and no test/fault module; its SHA-256 is
`695fba74e45701c07b15f9713bf14c4e96458134c6cd09094685015d7a52f3f5`.
No production source file changed in this continuation. The93normal operator
store files still match their original path/mode/content baseline.

### Native setup failure and receiver compatibility correction

The first `au6bmG` activation at14:21:05.778UTC failed while posting the
production panel's initial busy state: `Proxy.postMessage` raised a JavaScript
private-field receiver error. Its only native event was the initial `ready`
handler entry; its stage ledger was empty and the retained workflow remained a
draft. Thus no Azure stage API, resource creation, intended response withholding
or B09 live qualification occurred in that attempt. The failed root, logs and
private record remain intact; it was neither reset nor rearmed. The private
`initial-native-setup-failure.json` retains this distinction.

The narrowly corrected companion is
`/private/tmp/af-deployment-lost-response-IhNC1t`, runtime SHA-256
`34aa1ae72711020a53d40f156386429b33c85f7972b573f7335af561be85c574`.
Only native Webview/WebviewPanel fallback method and accessor receivers changed:
they bind to their original native instances. Production controllers, transport,
fault boundary, actions and permissions remain unchanged. The scope-file SHA-256
remains `35e0cd828ffdfc5e40ab790597543dcc18d2ea0ac1b8af24a2f133ca7c922908`;
the exact workflow, artifact, resources, data constraints, cost/runtime limits
and18:08:48.982UTC authorization expiry are unchanged.

The new development extension uses the old dedicated `au6bmG/user-data` and
`au6bmG/extensions` directories **in place**. No authentication cache or credential
was copied. It has a fresh companion/store root and retains the failed attempt
separately. The coordinator observed native posting working in the revised host;
that compatibility observation alone supplies no cloud-fault PASS.

At14:30UTC, final post-correction checks passed **709/709extension unit tests**,
TypeScript checking and host-test compilation. The15focused companion tests
include actual JavaScript private-field instance checks and the compiled
production `ready` handler posting both busy states through the facade. These
remain inert local tests. Independent normal-build inspection again found
667modules, no test/fault module, unchanged production sources and unchanged
bundle SHA-256 `695fba74e45701c07b15f9713bf14c4e96458134c6cd09094685015d7a52f3f5`.
Private receipts are `receiver-fix-validation.json`,
`receiver-fix-production-exclusion.json` and
`companion-receiver-fix-preparation.json` in the trial work directory.

### What-if polling compatibility correction before any resource effect

In the corrected `IhNC1t` host, actual native storage approval was followed by
normal ownership checks and the storage what-if POST at14:31:12.108UTC. The
companion then refused Azure's polling URL because its original guard assumed
a regional `Microsoft.Resources/locations/...` route. No storage deployment PUT,
artifact upload, runner PUT or readiness PUT had occurred. The retained record
remained a draft with a previewed storage deployment; its nine stage receipts
and native history remain intact. Root closed that host before continuing.

The official Azure2022-09-01 [resource-group what-if example](https://raw.githubusercontent.com/Azure/azure-rest-api-specs/main/specification/resources/resource-manager/Microsoft.Resources/deployments/stable/2022-09-01/examples/PostDeploymentWhatIfOnResourceGroup.json)
documents a subscription-scoped opaque Location response, so the regional-only
assumption was unsupported. A separate official-CLI what-if observation returned
202 and confirmed the actual same-subscription `operationresults/<opaque>` shape
with `api-version=2022-09-01` and signed query keys `t,c,s,h`. Only its sanitized
shape/hash was retained in `independent-whatif-poll-header.json`; opaque path and
query values were not recorded.

The revised guard accepts only the exact URL learned from an already permitted
what-if response, on HTTPS `management.azure.com` in the same subscription. It
requires the observed bounded path/query structure, rejects duplicate/unknown
query keys and shares a30-request/two-minute budget across a poll chain. Signed
URLs stay in memory. Both pre-validation header receipts and poll GET receipts
retain only hashes, safe shape/key names and the recognized API version. Other
hosts, subscriptions, arbitrary result URLs and mutations remain refused.

The next unlaunched preparation is
`/private/tmp/af-deployment-lost-response-x8TTEC`, runtime SHA-256
`9f5b1f389654eb857c7814798463d99bcee2511cc8990217df31abb1f473630e`.
Its unchanged scope SHA-256 remains
`35e0cd828ffdfc5e40ab790597543dcc18d2ea0ac1b8af24a2f133ca7c922908`.
The loader separately seals the exact prior effect-free draft bytes with SHA-256
`6ad854c0e4ebab4872ba6ee7d0867bb108259540e6ee98253b226a306d86bba7`.
The old root is preserved. The normal storage controller will obtain renewed
native approval and generate its normal new role UUID; this is not a same-role
continuation or a replay of an earlier cloud effect. No controller factory was
replaced and no submitted/ready/source/guest state can enter this seed path.
The same dedicated authentication directories are reused in place, and no cache
or credentials are copied. Artifact, workflow, resource/permission scope,
compute limits and18:08:48.982UTC expiry remain unchanged.

Post-correction local validation passed **727/727unit tests**, including33focused
companion checks, and TypeScript checking. The private preparation and complete
unit log are `companion-poll-fix-preparation.json` and `poll-fix-unit-tests.log`.
Neither compatibility fix nor any local test closes either B09 live-fault gate.

## Preparation observations

Fresh read-only ARM checks in this continuation confirmed the previous B10
runner `af-ae9523105eba42b69fe3` deallocated and its source
`afpg-p1-source-20260907` Stopped. The normal signed-in GUI still shows the same
finished inventory `401490cc-2403-4f63-b53f-c48a66f6d3ec` with its imported
2,947-byte report, SHA-256
`ac52a07c4b43416a954f10c99e528df31455413fdaaac79f810901c2968eb692`.

The selected compute subnet has no delegation and already has the existing NAT
egress and NSG. The SKU is unrestricted in zone1. Regional vCPU use is82/101 and
the Bsv2 family is8/100; the proposed VM requires2vCPUs. A13:52:16UTC group
inventory found no proposed top-level resource ID collision among474 existing
resources. The private normal-store baseline contains93files. These observations
need freshness checks before a later approved deployment.

The transfer account name is available globally and no group-level resource
lock was returned. The official VS Code1.139.0 copy prepared for an unambiguous
test window retains executable SHA-256
`1b58953da3281bddea360f21198ab80f8a5caa72c77f7cf996bd3c9fa74dfb3c`;
its signature verifies. The approved dedicated host has now been launched with
the new companion, and the user confirmed interactive Azure sign-in. Launch and
authentication do not establish a deployment request or start the VM clock.

Current primary retail lookup returned Linux B2s v2 at USD0.109/hour and the64GiB
E6 LRS disk at USD4.80/month, plus separately metered operations and storage.
The13:53:37UTC Cost Management result was USD476.728309915582 for the original
group and USD5.04080123954057 for B01, combined USD481.7691111551226. These are
delayed billing observations, not real-time spend or a final invoice. The first
combined recheck failed; the later successful query is retained separately.

Private preparation and baseline files are under
`production-simulation/work/b09-fault-trial-20260923/`; the negative artifact and
provenance are under `production-simulation/work/b09-bootstrap-negative-20260923/`.
Both are ignored by Git. Live deployment results remain unobserved.

## Approved launch and authentication checkpoint

The user approved the exact companion, negative artifact, new resource/RBAC
scope and cleanup at `2026-09-23T14:14:51.122892Z`. Root launched the official
dedicated host and initially paused at interactive Azure authentication; the
user subsequently confirmed sign-in. This is an executor-reported checkpoint,
not an independent observation of account selection or cloud effects by this
document's updater. No cloud resource or fault-case PASS is recorded here.
The first actual VM deployment request must establish the separate60-minute
clock and minute55 deallocation deadline. Preserve the prior93normal-store
path/mode/content baseline throughout this disposable-profile trial.
