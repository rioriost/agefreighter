# OP-N526 execution and qualification

Status: installed-GUI migration active; OP-N526 remains unqualified. Overall coverage is 8/9.

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

## 05:02Z onward — discovery VM provisioned

The user explicitly approved the fixed-build installation and container-only
reader grant. The installed GUI submitted the reviewed deployment once; ARM
succeeded at `05:02:54.947Z` and GUI reconciliation accepted it at `05:03:06Z`.
VM `af-31ce478995344bd6bcac` is running, B2s_v2, private IP `10.246.1.15`,
no public IP. Its principal is `f33afa54-ec25-413e-8b65-664c3b192394`;
independent role inspection confirms Storage Blob Data Reader scoped exactly to
container `af-31ce4789-9534-4bd6-bcac-621f460d99cc` in this workflow's account.
No account-wide or source role was granted to this VM.

Storage remains network-Enabled with the trial-only exception/deadline, anonymous
and shared-key access disabled. A later account write was observed; live settings
were checked rather than assuming it disabled the authorized transfer path.
Read-only guest observation at `05:03:46Z` shows cloud-init still running, 4%
rounded root usage, 8-GiB nominal VM memory, zero swap and kernel OOM messages.
No inventory is claimed before pinned-build GUI readiness and authenticated reads.

## 05:06Z — pinned Linux readiness PASS; private password entry required

Cloud-init completed and the guest archive seal matches the pinned build. The
standalone diagnostic's `--version` flag is unsupported by this CLI; the actual
installed GUI protocol check, not that diagnostic, confirmed version/capability
readiness at `2026-09-15T05:06:16.311Z`. Boot ID is
`795c9ef2-c47d-4328-8a1d-885a894a834e`, version `2.4.0-dev.8a23a5109798`,
unchanged archive SHA, required Neo4j inventory/migration capabilities, idle,
3.5082% disk use, swap/OOM zero.

Source recheck at `05:06:35Z`: isolated container running, zero restart/OOM,
TLS 1.3 with chain and literal-IP verification OK, disk 6%, swap zero.
The installed GUI reopened the saved OP-N526 form, reviewed the same IP-only
settings and CA hash, and approved exact Neo4j inventory under the user's route
authorization. It is now at `Read-only source password`, requiring the existing
Neo4j 5.26 `neo4j` credential to be entered privately by the user and confirmed
with Enter. This is the AZ-N526 credential inherited by the cold clone, not the
AZ-N44 password. No secret was extracted, reset, or exported. Inventory has not
been submitted yet, and no target/migration exists. Source and discovery runner
remain running within the unchanged trial budget/deadline.

## 05:14–05:19Z — first inventory rejected by source authentication

The user entered a password privately. Automatic GUI freshness checking passed
at `05:14:43.788Z` (same boot/build, idle, disk 3.5086%, zero swap/OOM). Exact
inventory operation `0598b421-0ed1-4e77-8f16-8f2f15990614` was submitted once
at `05:14:55.114Z` and accepted by the guest, then failed. Configuration SHA-256
is `e257e61da77c7fa2900a951f278d17b2f62c206c8d6864e259ee6019536de610`;
guest configuration SHA-256 is
`a157b16c8d2061de920d07e1df3ec7b1415e125432bccbef9df47fcab5f8d37a`.

Read-only guest classification at `05:18:44Z` identified Neo4j
`Security.Unauthorized`. This is a credential rejection, not successful source
inventory or migration. The private 89-byte stderr remains in the guest;
SHA-256 `05a83f01f9f6a512030e8b3f8daac11c4a7ff44974bf62f1fec9df7d8fdadbe7`.
No raw stderr, passwords or secret request data were exported. The operation,
configuration and claim remain retained. No target deployment, password reset,
source write or automatic retry occurred. Correct existing source credentials
must be entered privately before a new attempt; the failed operation is never
resumed.

Fresh GUI readiness at `05:21:23.142Z` confirms the same boot/build, idle state,
3.5095% disk, no swap/OOM after the failed read. The retry-preparation dialog
requires confirming that the cause has been corrected. Because the correct
credential has not yet been established, that dialog was cancelled; the failed
assessment remains attached, with no new request or source read. Ask the user
to confirm the correct AZ-N526 credential before preparing another attempt.

## 05:30Z — authorized clone-only password recovery

The user explicitly authorized password reset after the September 12 history
confirmed the previous source password was temporary and not persisted. This
supersedes the earlier no-native-credential-change constraint for the OP-N526
clone only. Original `af-n526-source` remains deallocated and unchanged.

Fresh gates confirmed the authorized subscription, no RG locks, only expected
recent activity, source disk 6%, no swap/OOM, and no active source assessment.
Only this source and its discovery runner were running. The existing deadline
and budget are unchanged.

A new random 32-byte password (64 hexadecimal characters) was stored and read
back successfully using the macOS Keychain API before recovery was submitted.
Keychain service/label: `agefreighter-op-n526-neo4j`; account: `neo4j`.
It was passed to Azure as a protected parameter over stdin, not a command-line
argument, console value or local request/password file. Do not rerun the
create-only helper blindly; it refuses an existing Keychain item.

Managed recovery `af-op-n526-password-reset-20260915` succeeded with exit 0.
Guest evidence directory:
`/var/lib/agefreighter/neo4j-password-recovery/20260915T053023Z`.
The system database was backed up before alteration; recovery listened only on
unpublished container loopback. Normal authenticated service was restored and
the new credential returned exactly 1,600,000 vertices and 4,000,000 edges.
The secret-bearing managed recovery command was removed after success. The
Keychain item is retained for future private GUI entry; no secret is in Git.

Completion `2026-09-15T05:31:08Z`; retained system backup SHA-256
`1ceeb54f122df0ea591832ce67355cec5f165157b65950e7c526ae7163675417`;
guest summary SHA-256
`85c81166418513306f7722193e4d267b9b7739d53231f37f827aedeb06e88619`.
Independent post-check confirms the normal container is running, the copied
old container remains stopped, disk 6%, zero swap/OOM and zero restarts.

This corrects source authentication only. The retained failed assessment has
not been replayed and OP-N526 remains unqualified pending a fresh GUI inventory,
migration and full canonical verification.

## 05:37Z — fresh installed-GUI inventory PASS

After fresh same-boot/build idle readiness, the GUI retained failed inventory
`0598b421-0ed1-4e77-8f16-8f2f15990614` in history and prepared a new attempt.
The recovered Keychain credential was supplied to the private password prompt
without printing it. New operation `3f815040-a8ae-4756-9f48-6134ff1c661b` was
submitted at `05:37:05.527Z` with unchanged reviewed connection configuration.
It completed at `05:37:12.416728809Z`: exact Neo4j transactional count-store
totals of 1,600,000 vertices and 4,000,000 edges, outcome pass, no errors,
warnings or incomplete checks. The GUI exported/imported the report through
the approved workflow storage and displayed the hash-verified result.
Independent local validation agrees: 663 bytes, SHA-256
`1fe2e0a44e62abc9906455cb60aaa86e5118e360619892f9049b8d2c9950fe4d`.
Guest configuration SHA-256:
`0ec1d697e4af3d055c95c18f170343fef18e0d8240be96c58262a55896ffc8c7`.

Live transfer storage remains Enabled, anonymous/shared-key access false, with
the existing trial-only exception and unchanged expiry. No new security
exception was applied. ARM subnet inventory confirms `10.246.17.0/24` is unused.
Target planning initially stopped safely because guest readiness expired while
entering the form; no target intent or deployment was created by that attempt.

After fresh GUI readiness at `05:41:49.333Z`, the second target preflight passed.
The native final review dialog now awaits action-time confirmation of creating
the private target and saving its generated credential in VS Code SecretStorage:
`afpg-31ce478995344bd6bcac`, PG18/AGE, Standard_D4ds_v5, 128 GiB, Japan East
zone 1; delegated subnet `10.246.17.0/24`, no public access or peering.
Same-runner migration size Standard_D4s_v5 is a later idle-VM resize.
Reviewed target/runner rate USD 0.736/hour, USD 400 reserve, USD 800 ceiling,
unchanged deadline. Source D8 cost remains additional (prior estimate USD
0.496/hour). No target plan files, credential or deployment have been created
yet; folder selection follows confirmation. Do not treat this stage as migration.

## 05:54Z — approved private target deployment active

The user approved private target creation and SecretStorage retention. The first
approved save created local plan/YAML files, but expired guest readiness blocked
submission before any Azure target deployment. These files are retained, not
overwritten. Fresh GUI readiness at `05:50:06.254Z` allowed a new review with
identical source, sizing, permissions, price, budget and deadline. The GUI saved
the final files under `production-simulation/work/op-n526-20260915/` with mode
0600, using stem
`agefreighter-31ce4789-9534-4bd6-bcac-621f460d99cc-272b8f5590aa`.
YAML SHA-256 `d6753c6448eac2cbf25b22ad77c0c33ce1b850794d8ce64023a5d9898bbb5137`;
target-plan SHA-256 `b67ccec5a6525bbd779c0d1ae5bf3034d16abbab46e2c758458cb13d26419ac3`.
The files contain credential references, not source or target passwords.

ARM accepted deployment `afpg-31ce478995344bd6bcac`, status Running at
`2026-09-15T05:54:02.128028Z`; retained workflow phase is submitted. The dedicated
subnet succeeded and private DNS creation is active. No migration was started.
Reconcile this deployment only; never replay target creation. AGE readiness,
same-runner resize, durable migration and full P1 digest remain required.

## 06:00–06:14Z — target provisioned; resized runner awaits GUI start

ARM deployment succeeded at `06:00:24.161882Z`, including the server, database,
AGE allowlist, preload configuration, subnet and private DNS/link. Installed
GUI reconciliation confirms target provisioned. Live server is PG18, zone 1,
Ready, public network Disabled. The approved AGE restart was submitted once at
`06:02:55.611Z`; subsequent ARM observation shows `pg_stat_statements,age` and
pending restart false. GUI `targetRestart` still needs read-only reconciliation.

Fresh matching runner readiness at `06:03:56.269Z` preceded the approved GUI
resize. Deallocation began at `06:05:26.066Z`; GUI reconciliation then permitted
the size update to Standard_D4s_v5. ARM confirms Succeeded/deallocated with the
new size; GUI retained phase is `ready-to-start`. Disk/NIC/identity seal remained
`8ede536ede5f5147a3c8a5b65c3f0881415a1d7da1324be6d704ad16967ae385`.

The current VS Code execution picker is at `ready-to-start`. Computer-control
actions became unreliable (including `elementHasNoFrame` and
`noWindowsAvailable`), despite a readable screenshot. No lock-state cause is
asserted from that alone. Ask the operator to unlock if needed and bring VS Code
to the foreground before continuing the GUI path. No headless substitution,
runner start, migration or verification has been performed. The source and
private target remain running; the runner is deallocated. Preserve the existing
budget/deadline and all evidence. Next: approve the retained start step, reconcile
resize and AGE restart, establish new-boot readiness, then migrate and qualify.

## 06:24–06:30Z — GUI recovered; new migration submitted

After the user brought VS Code foreground, normal GUI actions worked again.
The approved retained start step ran once, then read-only GUI reconciliation
confirmed resize `finished` and AGE preload `finished`. ARM agrees: D4s_v5,
Succeeded/running, with the retained disk/NIC/identity seal unchanged. New boot
ID is `743ab17a-bf54-4a73-9017-bf99405fe38b`. GUI readiness at `06:27:02.010Z`
confirmed the same pinned CLI/archive, idle state, disk 3.5125%, no swap or OOM.
No RG locks were present. Recent external writes created/attached a target-subnet
NSG with no custom rules; source restrictions and private target access remain
unchanged. The existing USD 800 ceiling, USD 400 reserve and deadline still fit;
no new authorization window was opened.

Installed GUI migration preflight accepted all 5.6M mapped rows and the sealed
inventory. After the routine preapproved confirmation, the existing OP-N526
Keychain credential was supplied to the protected source prompt; no plaintext
was emitted or saved in the repository. Target credentials remain in VS Code
SecretStorage. New durable job/operation
`848306ac-628e-43ac-8af9-31dfdee2a804` was retained at
`2026-09-15T06:29:11.668Z`, before submission of the guest command. Do not replay
this operation. Migration and strict complete counts are active; the separate
full P1 digest is still required before qualification.
