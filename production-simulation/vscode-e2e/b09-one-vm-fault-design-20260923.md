# B09: one disposable VM for two remaining fault observations

Later continuation: the [dedicated session preparation](b09-fault-session-20260923.md)
now names the fresh workflow and negative artifact. The original design-only
checkpoint below is historical; live execution still requires its reviewed scope.

**Design only; no execution, artifact construction, fault implementation or
approval to create resources.** The existing runner/source 60-minute approval
does not cover this plan. Root must obtain a new action-time approval after
reviewing the exact resource IDs, cost, artifact and test-companion mechanism.
No source database needs to start. No target, migration or accepted runner is
part of this scope.

## Feasibility and evidence limit

One new VM can plausibly cover (1) real Azure acceptance followed by deliberate
withholding of the response from the production controller, and (2) a genuine
terminal bootstrap failure on that same VM. These are two separately recorded
observations, not a network-outage claim.

Production has **no supported response-withholding switch**. Deterministic
testing therefore requires a separately reviewed, disposable test companion
using the unchanged production controller/native approval and a narrowly bound
adapter around the real Azure transport. This is not an unchanged normal
installed-command transport test. If that distinction is unacceptable for B09
acceptance, this approach is blocked; do not add a release backdoor or claim an
actual network failure. No mechanism has yet been implemented or execution-
approved.

## Minimum unchanged-production resource scope

Choose one fresh workflow UUID `W` before approval. Let `P` be
`af-` plus its first 20 hyphen-free characters and `S` be `af` plus its first
22 hyphen-free characters. Resolve all below to full IDs in the one approved
subscription/existing resource group, region, zone and existing compute subnet.

| New resource/action | Exact workflow-derived scope |
| --- | --- |
| Storage account | `S`, Standard_LRS; HTTPS/TLS, no shared keys or anonymous Blob access |
| Private Blob container | `S/blobServices/default/containers/af-W` |
| Desktop user's role | Storage Blob Data Contributor on **this new account**, as current production storage setup requires |
| Transfer deployment | `S-transfer` (one storage setup deployment) |
| One immutable negative artifact Blob | `af-W/artifacts/<NEW_SHA256>.tar.gz` |
| Runner NSG and NIC | `P`; NSG retains normal deny-all-inbound rule |
| Runner VM and system-assigned identity | `P`, reviewed supported discovery SKU, preferably B2s_v2 where available |
| Runner OS disk | `P-os`, current template's 64 GiB StandardSSD_LRS disk |
| VM role | Storage Blob Data Reader scoped only to this workflow's container, assignment name `W` |
| Runner deployment | `P`; exactly one runner deployment PUT for the lost-response trial |
| Readiness/observation | One explicit readiness control after terminal cloud-init; any additional bounded read-only observation count must be approved |

**Existing transfer-account reuse is not supported by unchanged production.**
`reportStorageNames` derives the account from `W`; `verifyTransferStorage`
requires the account's exact ID, workflow tag, endpoint and private container.
`developmentArtifact`/`runnerTemplate` bind the artifact URL and reader grant to
that derived account/container. A fresh container inside an old account would
fail these ownership/identity gates. Do not alter tags, reuse an old workflow,
weaken validation or grant broader permissions to reduce this resource list.

No new VNet, subnet, public IP, NAT, peering, route, source firewall change or
target is needed. Existing outbound connectivity must already support the
normal bootstrap. Current storage governance may deny desktop Blob access;
that is a stop condition, not permission to enable public access or change
network policy. The planned account uses the existing production template;
its effective network policy must pass unchanged checks.

## Negative artifact: packaging failure before executable installation

Construct only after its separate preparation approval. Derive a **new** archive
from the reviewed `d40d6ccc9a4d` Linux artifact: retain its authentic
`agefreighter` binary but deliberately omit `agefreighter-tools`. Keep the
accepted original archive and every existing content-addressed Blob unchanged.

Retain explicit derivative provenance: original commit/archive SHA, retained
member SHA, omitted member name, construction method, new exact bytes/SHA and
negative-fixture purpose. The development manifest's commit identifies the
included binary's source; it must not be presented as an unmodified build,
release or usable runner. The reviewer must see the negative-fixture annotation
before the normal pinned-artifact and VM approvals. Local review should verify
the exact tar member list and absence of extra paths/symlinks.

The unchanged `bootstrapScript` first downloads and checks the new archive SHA,
then extracts `agefreighter`, then extracts `agefreighter-tools`, **then** installs
either executable. With `set -e`, the missing second member terminates before
both install commands, version/help checks, checksum marker and
`bootstrap.complete`. No fallback is present. This is an intentional isolated
packaging failure, not a fabricated failure of the accepted Linux build.

Do not corrupt/delete a published Blob, revoke roles, break networking, change
the guest after deployment, or alter cloud-init/production templates to induce
this failure. If a different failure occurs first (governance, download, image,
quota or permissions), preserve it and report the intended packaging case
unexecuted; do not retry/create another VM automatically.

## Narrow lost-response mechanism to review separately

The test companion must run only in a fresh disposable, interactively signed-in
profile/private store. It must never extract credentials/tokens, register a
release fault command, replace global networking, weaken TLS or affect other
extensions. Its allowed writes are explicit stage-specific scopes, not a
generic Azure proxy permission.

1. Use real normal preflight, what-if, freshness, ownership and native approval
   gates. Persist the production `deployment-submitted` intent before sending.
2. For exactly the approved subscription/deployment ID/API version/template
   hash, permit **one** real runner deployment PUT. No second PUT or alternate
   deployment name is allowed. Storage setup PUTs are a separate earlier stage.
3. Await real successful Azure acceptance (for example 201/202), retain a
   create-only **sanitized** receipt outside the workflow record: time, exact
   deployment ID, status and available nonsecret request/correlation identity.
   Never retain raw request/response objects, auth headers or protected data.
4. Deliberately throw before returning that result to the controller. This
   models a client-controller boundary failure **after the adapter saw a real
   response**, not a missing TCP/TLS response or Azure service outage.
5. Require production `unknown` state. Independently read the exact deployment
   and its operations to prove Azure accepted it, then reconcile that same ID
   through production `refreshRunner`, GET-only. Retain a scoped request ledger
   proving one runner deployment PUT and no replay/new workflow. An independent
   ARM observation must not be replaced by the adapter's receipt alone.

`AzureSession.runnerRequest` currently issues one fetch without an application
retry; `submitRunner` persists intent, catches uncertainty and retains
`unknown`; `refreshRunner` performs only a GET and local status persistence.
An adapter failure before Azure acceptance, or a failed receipt write, does not
prove the planned lost-response case. Preserve the actual result and stop.

## Terminal bootstrap evidence and bounded end state

Require independent evidence of actual terminal cloud-init failure on this new
VM, the expected missing-member failure stage, absent `bootstrap.complete`, and
no installed qualification executable/worker. Capture only bounded sanitized
failure fields; never dump guest environments, credentials or arbitrary logs.
Invoke the normal explicit readiness control and reconcile its result by GET.
Production must retain a failed guest command, no `guestReady`/readiness receipt,
and no source assessment, worker or migration. Pending/timeout is not terminal
bootstrap failure; ARM VM success alone is not Linux readiness.

The new approval should name the fresh UUID/full resource list, exact artifact
SHA/provenance, one runner PUT with deliberate response withholding, bounded
read-only observations, maximum runtime/cost and stop conditions. It must also
cover the intended end state: deallocate the new VM and, after preserving
evidence, remove only the enumerated new roles/VM/disk/NIC/NSG/container/account
and deployment records if cleanup is approved. The template detaches the disk
and NIC, so deleting only the VM would leave billable resources. Never delete a
shared group or unrelated resource. If cleanup is not approved, report retained
resources/cost rather than infer deletion authority.

Two-window stale-preview refusal may optionally reuse this private draft before
the one accepted deployment, but needs its separately reviewed refusal harness
and genuine native approvals. It must allow zero runner PUTs during refusal,
preserve records, and finish before any renewal/real submission. It is not
necessary to combine it with these two faults; avoid widening or delaying the
live window merely to combine evidence.

Primary code reviewed: `core/runner.ts` (`bootstrapScript`, `runnerTemplate`),
`core/runnerLifecycle.ts` (`submitRunner`, `refreshRunner`),
`core/runnerReportStorage.ts`, `core/runnerStorageLifecycle.ts`,
`core/runnerDevelopment.ts`, `developmentRunner.ts`, `guided/azure.ts`, and
`core/runnerGuest.ts`. Supervisor design feedback agrees with the conditional
two-case scope and evidence limits; this is not implementation or execution GO.
