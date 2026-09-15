# OP-N526 execution and qualification

Status: preparation only; OP-N526 remains unqualified. Overall coverage is 8/9.

## Authorization and reviewed implementation

The user authorizes resource creation and completion of this last route. Preserve
the existing USD 800 ceiling, USD 400 conservative accrued/noncompute reserve and
`2026-09-16T07:14:35.311Z` deadline; do not reset the window. At `04:33Z` on
September 15, approximately 26.7 hours remain. Source D8s_v5 plus runner/target
D4s_v5/D4ds_v5 compute is approximately USD 1.232/hour at previously checked
rates (planning estimate, not billing). A serial route fits the reserve; retain
all prior evidence and stop this route's compute on completion or gate breach.

Read-only gates: original `af-n526-source` is deallocated, 128-GiB Gen2 OS disk
`af-n526-source-os` is Succeeded and retained. No custom/user data or extensions
are present in the original VM model. DSv5 quota is 80/100 and regional quota
82/101; a D8 source and D4 runner fit. Dedicated IP `10.246.5.5` is available;
no RG locks were returned. Recent non-audit activity contains only the preceding
authorized target stop.

Create a cold disk copy `af-op-n526-source-os`, private NIC
`af-op-n526-source-nic`, specialized D8s_v5 VM `af-op-n526-source`, Japan East
zone 1. Reuse the restricted Neo4j 5.26 source subnet. No managed identity,
public IP or new source permissions. Inspect the clone without reading secrets;
retain/stop the copied original container and run the same pinned image/data
with read-only settings and a new IP-SAN TLS certificate. Export only the public
CA. Do not replay original bootstrap or alter native credentials.

Review: cold copy avoids copying a live database and isolates the qualified
source. Copied host keys/logs make this a private fixture harness only, not a
production provisioning recipe. Verify image/version, disk, swap/OOM, TLS chain
and literal-IP identity before assessment. ARM checks belong to the fixture
harness, not the customer workflow's source discovery.

## Required completion sequence

1. Installed GUI: new Neo4j / on-premises workflow, private discovery runner,
   IP `10.246.5.5`, port 7687, database/user `neo4j`, stable vertex/edge identity
   `source_key`, verified TLS with the public CA, private password entry.
2. Complete inventory must report 1.6M vertices and 4M edges without errors or
   incomplete checks. Use the unchanged pinned development loader build.
3. GUI private PG18/AGE target review/deploy in Japan East zone 1, late LoadJob
   save, AGE readiness, and same-runner resize with disk/NIC/identity preserved.
4. New durable migration, strict `--counts --require-complete` verification.
5. GUI full P1 qualification with all 64 ranges / 5.6M typed records and frozen
   root `bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.
   Independently validate the transferred report and recompute roots locally.
6. Stop/deallocate this source, runner and target; preserve resources and all
   evidence. Commit/push redacted results and mark 9/9 only after GUI PASS.

Routine actions are preapproved. Action-time confirmations required by computer
use policy remain distinct, as does private credential entry by the user.

## Prepared source and installed GUI handoff

Cold disk copy, private NIC and specialized D8s_v5 VM creation succeeded.
The original source remained deallocated. Clone inspection at `04:36:44Z`
confirmed Neo4j 5.26.30, pinned image
`neo4j@sha256:037cf5756f0135cbfd66b739b6df7c7c4bb100f9ce11602f6f9538e17e02c74d`,
copied `/var/lib/neo4j526-data`, disk 6%, zero swap and kernel OOM messages.
The reviewed script passed `bash -n` and ShellCheck before guest execution.
The copied original container is stopped with restart disabled and retained.
`af-op-n526-neo4j526` now uses the copied data with 4-GiB heap / 8-GiB page cache,
read-only database default and required Bolt TLS. Native credentials were not
read or changed. Live certificate-chain and literal-IP verification passed;
new container is running with OOM false and zero restarts.

Public CA: 1,521 bytes, SHA-256
`6da7aebf5484f43715daa04330fb6d891833e9d22c4f0c5ed571ab53982cc1ca`.
Only the public CA was exported to
`production-simulation/work/op-n526-20260915/ca.crt` (ignored local work area).
IP-SAN `10.246.5.5`, valid September 15–22. Private keys remain in the guest.

Installed GUI draft `31ce4789-9534-4bd6-bcac-621f460d99cc` selects Neo4j /
on-premises, Japan East zone 1, B2s_v2, existing runner subnet and trial RG.
Source form was reviewed and saved: migration `op-n526-p1-r1`, namespace `p1`,
literal host `10.246.5.5`, port 7687, database/user `neo4j`, vertex and edge
identity `source_key`. GUI-selected CA bytes/hash match the guest export.
No source ARM discovery or password extraction was used.

The final GUI storage dialog is awaiting action-time permission confirmation:
new account `af31ce478995344bd6bcac62`, Storage Blob Data Contributor for the
signed-in user on that account only, HTTPS endpoint network-public with anonymous
access and shared keys disabled. Nothing has been submitted from that dialog.
No new runner, target, inventory or migration exists yet. Source clone remains
running within the existing deadline/budget; all prior resources remain retained.

## Transfer ready; reviewed discovery VM awaiting execution confirmation

Following the user's explicit confirmation, the installed GUI created transfer
account `af31ce478995344bd6bcac62` and user role assignment
`67b08b29-d372-4056-84cf-3c749d80dc67`. ARM and GUI confirm success; independent
role inspection confirms Storage Blob Data Contributor on this account only.
Azure changed networking to Disabled despite Enabled in the submitted template.
The previously authorized trial-storage-only `SecurityControl=Ignore` exception
and original deadline tag were merged into this account only. Authenticated
HTTPS networking is now Enabled, anonymous and shared-key access remain false,
HTTPS-only and TLS 1.2 remain enforced. Anonymous Blob HTTPS was rejected with
409; no RG/source security exception or network relaxation was performed.

The installed GUI selected the unchanged fixed development manifest and uploaded
the reviewed archive through the authenticated transfer path. Independent local
size and SHA-256 checks passed for both loader and P1-verifier archives. Loader
upload is ready: commit `8a23a5109798ec906109532e4cc6c32308b3c824`, 37,079,079
bytes, SHA-256 `52e1d147a13b86a729f5a993e9e72848dd87a89d0ae50a61f26459f5632444f3`.
The verifier is not yet uploaded or executed in this workflow.

After reconnecting to this draft, the installed GUI completed a fresh runner
preview for Japan East zone 1 / B2s_v2, USD 0.109/hour plus additional charges.
Quota after source creation is DSv5 88/100, regional total 90/101. The existing
runner subnet and reviewed placement remain unchanged. The final
`Create reviewed runner` dialog is pending action-time confirmation because it
installs/runs the unpublished fixed build and grants the new VM identity Blob
Reader on this workflow container only. VM `af-31ce478995344bd6bcac` has not
been submitted. No target, authenticated inventory or migration exists yet.
