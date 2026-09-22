# B03 Neo4j other-cloud selection — approved bounded trial

September 22, 2026. Branch `codex/2.4.0-guided-migration`.
Status: **Approved storage access and artifact upload complete; VM/identity/install approval pending. No compute started.**

## Authorization and boundaries

The user approved proceeding with a maximum two-hour session and the unchanged
cumulative USD800 ceiling, renewing the private fixture TLS certificate and
using a new runner/private target for full migration verification. Start the
two-hour clock at the first resource start/create request that starts compute,
set an absolute shutdown deadline, and reserve stopping latency. Do not silently
extend it. Stop early at failure, completion, or 15 minutes idle awaiting input.

Fresh delayed billing from the preceding safety check is USD390.9644901299309
across the original and B01 trial groups. Retain the USD700 accrued/retention
reserve inside USD800; it is not an additional budget. Recheck billing and
governance before startup. The preceding safety check verified 14 remaining VMs
deallocated and 19 retained Flexible Servers Stopped.

This is an Azure-hosted **simulation of an endpoint-only other-cloud source**,
not a claim of actual AWS/GCP network compatibility. Fixture ARM/guest preparation
is separate from customer source discovery: the guided workflow must use only
the IP/port, database, TLS trust and credentials for the source.

## Reviewed fixture and artifacts

- Existing source only: `af-op-n526-source`, original trial group
  `rg-af-vscode-p1-20260905-a`, approved subscription ending `fdb7`.
- Fresh model: Standard_D8s_v5, Japan East / zone1, retained OS disk
  `af-op-n526-source-os`; tags identify OP-N526 trial ownership. Historical
  expiresAt tag is expired; this new approval is not permission to change tags.
- Fresh NIC: `10.246.5.5`, no public IP, existing `neo4j526-source` subnet.
  No route/peering/NSG/public-exposure change is approved by this preparation.
- Capacity read: regional cores 74/101, DSv5 64/100, Bsv2 8/100. This is quota
  metadata, not proof of actual allocation capacity; refresh before creation.
- Source is the already accepted 1.6M-vertex / 4M-edge P1 fixture. Do not reseed,
  change the native password, delete a container, or touch `af-n526-source`.
- Installed extension remains approved `7338faa`; no new installation planned.
- Linux candidate retained for action-time approval: `d40d6ccc9a4ddf6e2ca626392cd7bf83140ed6c7`,
  archive 37,197,546 bytes, SHA
  `2321022975f85c73068a54fd21a287e2802d5bc33fa21eb9323c142dd7262ff6`.
  Manifest `production-simulation/work/vscode-runner-build.HwWiUz/manifest.json`.

## TLS renewal preparation — not executed

`renew-op-n526-tls-20260922.sh` is fixture-only and passes `bash -n` and
ShellCheck. It checks exact hostname/IP/image, required TLS/read-only settings,
disk/swap/OOM, deadline and original CA hash. It preserves both private keys,
the original public CA and old leaf in evidence, issues renewed public CA/leaf
certificates using the retained keys, updates only the leaf, restarts the same
container, verifies live IP-SAN TLS and unchanged mounts/key files, and seals
sanitized evidence. Keys/passwords never leave the guest. No retry if the
create-only evidence directory exists; diagnose retained state instead.

Static review is not live renewal proof. Set the absolute deadline and fresh
guest gates before dispatch. Export only the renewed public CA and select it in
the new GUI workflow. Never import the expired CA or bypass TLS validation.

## Installed GUI preparation

Normal signed-in VS Code: selected Neo4j / **other-cloud**, the approved
subscription, original trial group, Japan East, zone1, Standard_B2s_v2, and
existing `runner` subnet. Source Azure discovery/ARM fields are absent.
Configure source created workflow **`b775b1b2-81ca-40fc-b669-f136cde904b8`**.
Independent retained-draft inspection confirmed this complete UUID, draft phase,
reviewed source values and create-mode target graph `othercloud_n526_p1_r1`.

Reviewed source fields: name `othercloud-n526-p1-r1`, namespace `p1`, literal
host `10.246.5.5`, port7687, database/user `neo4j`, vertex/edge keys `source_key`.
An input action briefly produced incorrect unsaved key text; corrected and
re-observed both exact key fields before Review source settings. No source read
or credential prompt occurred. New public CA selection remains pending renewal.

Native storage confirmation currently proposes `afb775b1b281ca40fcb669f1`,
Japan East, Standard LRS, account-scoped Storage Blob Data Contributor for the
signed-in user, authenticated network-public HTTPS with shared keys/anonymous
access disabled. Action-time permission was requested; do not click approval
until received. This dialog does not authorize policy-tag/network exceptions.
All compute remains stopped; the two-hour compute window has not begun.

Before cloud submission, the local Linux archive was rehashed and exactly
matches the pinned SHA above. `bash -n` and ShellCheck both passed again for
the renewal script. These are preparation checks, not installed-guest proof.

## Acceptance sequence

1. Exact storage/role approval, verified transfer and fixed Linux artifact approval.
2. Set hard deadline and safety monitoring, start only approved source/runner;
   renew fixture TLS, select its new public CA and verify pinned guest readiness.
3. Private source-password input by user; full inventory 5.6M rows /18labels,
   exact sealed report import with no errors/incomplete coverage.
4. Fresh private PG18/AGE target, separate native approval for credentials/
   delegated subnet; late LoadJob save; same-runner resize preservation.
5. One new durable GUI migration, complete strict counts with zero rejects.
6. Separately approved pinned P1 verifier, all64 ranges and root
   `bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`,
   exact GUI import and independent recomputation.
7. Stop only this session's source/runner/target, preserve all resources/evidence,
   update and push redacted results. No accepted graph or workflow overwrite.

No B03 qualification result is claimed at this preparation checkpoint.

## September 22 06:55–06:59 UTC — storage reconciled, compute still stopped

On continuation, the native confirmation was no longer present and the installed
GUI showed the existing storage deployment as submitted. A refresh reconciled
it to `ready — public network: Disabled (provisioning is not transfer readiness)`;
no duplicate deployment was dispatched. Independent ARM reads confirmed the exact
account Succeeded, anonymous access false, shared-key access false, workflow tags
matching, and only the intended user/account-scoped Blob Data Contributor grant.
The retained deployment records role assignment `83c2202f-2b49-417e-a834-f1be860c22b3`.

Actual public-network access is Disabled despite Enabled in the retained template.
The cause has not been independently attributed. Requested explicit permission
for this exact account's `SecurityControl=Ignore` tag, authenticated HTTPS access,
and upload of the pinned Linux archive to its workflow-only container. None of
those changes or transfers has been performed at this checkpoint. Source VM
`af-op-n526-source` is freshly verified deallocated; no runner/target startup,
TLS renewal, password prompt or assessment. The two-hour compute clock is unstarted.

## September 22 07:04–07:09 UTC — approved transfer complete, VM preview reviewed

The user approved the exact storage tag/access change and fixed archive transfer.
Merged `SecurityControl=Ignore` into this account's existing tags and enabled
public-network access. Fresh independent ARM verification retained HTTPS-only,
TLS1.2 minimum, anonymous access false and shared-key access false. Authenticated
listing of the workflow container succeeded. Installed GUI refresh reconciled
ready/public Enabled; no storage redeployment or source exposure change.

The installed qualification command selected this exact workflow and the pinned
manifest, checked local archive bytes, and uploaded through its normal approved
path. GUI confirmed the archive prepared; retained developmentUpload phase ready.
Independent Blob properties show creation at 07:04:49 UTC, 37,197,546 bytes and
SHA256 metadata `2321022975f85c73068a54fd21a287e2802d5bc33fa21eb9323c142dd7262ff6`.
This is upload/metadata evidence, not a fresh remote-download digest or guest install.

Actual GUI reconnect restored the same workflow and pinned artifact. The UUID
search returned no quick-pick matches because the identifier is in the description;
filtering `neo4j — draft` exposed the exact UUID for selection. No different draft
was selected or edited. Fresh preflight completed 07:08:09.625 UTC: Japan East,
zone1, B2s_v2, USD0.109/hour plus disk/network, preview expires 07:23:09.625 UTC.
Its resources are only the new NSG/NIC/VM `af-b775b1b281ca40fcb669` and the VM's
Blob Reader role on this workflow container. No public IP or source grant.

Requested action-time approval for this VM, scoped identity access and unpublished
software installation/execution. No deployment submitted; source VM freshly
deallocated and new runner absent. Recheck expired preview before any later
submission. Establish the absolute two-hour deadline and safety monitor before
the first compute request; the clock remains unstarted. TLS renewal still pending.
