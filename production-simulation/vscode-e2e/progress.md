# Guided migration P1 qualification progress

Updated: 2026-09-23 JST. Overall outcome: **all nine defined P1 base routes qualified (9/9)**.

Latest09:23UTC: **B09/B10/B12 parallel implementation and local validation PASS**.
Three Astra high workers and Astra xhigh review completed the running-readiness
control-removal gate, explicit archived crash-lock recovery and typed rejected
P1 import evidence. One supervisor-found lock-contention race was fixed.
608unit tests,14actual-Go contracts,25real isolated VS Code1.139.0 tests and
two actual isolated SIGKILL/recovery pairs PASS. All91operator files unchanged;
no Azure mutation, normal-profile installation or signed-in cloud fault test.
B09/B10/B12 remain partial, broader ledger6pass/6partial. Next: reviewed candidate
installation and scoped live approval. See
[supervised batch](b09-b10-b12-parallel-20260923.md).

Prior08:58UTC: **B03 PostgreSQL other-cloud GUI migration and full canonical PASS**.
Installed GUI imported the retained23224byte result and displayed full P1 PASS.
Independent comparison matched all5.6M typed records,18labels,64ranges and
recomputed expected/actual canonical root. No migration/verifier replay.
Five explicitly approved historical readiness controls were archived before
manual deletion; exact absence verified, other controls/guest evidence retained.
Only runner restarted for maintenance/export; source and target stayed off.
Source/runner deallocated and target Stopped verified by08:58:29UTC, monitor paused.
Broader defined branch ledger now6pass/6partial; base routes remain9/9.
This is endpoint-only Azure simulation, not real third-party-cloud compatibility.
Native control removal is still unqualified. See
[canonical receipt](evidence/b03-pg-p1-pass-20260923.json).

Prior08:44UTC: **B03 PostgreSQL verifier exit0; result import blocked by control capacity**.
Exact qualification028ec524 finished08:34:52UTC; GUI retained verified phase and
23224byte result seal00938de546f94d1fe50e372044bef9b423e104919855fc8d934a06b4b42f77fc.
Result export was not submitted: extension25managed-command cap reached.
Full report/ranges/root have not been independently imported/validated; no final
B03 PASS. Runner deallocated and target Stopped verified; source remains off,
monitor paused. One historical readiness receipt archived/hash-verified locally.
Native removal preview failed closed because live ARM executionState is Pending
without retained output/times despite sealed historical success; nothing deleted.
Scoped maintenance needs explicit review/approval; no verifier/migration replay.
Existing new-session10:25UTC hardstop remains, no automatic extension.

Prior08:26UTC: **B03 PostgreSQL canonical verification continuation authorized**.
User approved a new two-hour session after Keychain recovery. Existing runner and
private target start requested; source remains off, no migration replay. GUI cost
authorization renewed; fixed stop10:25UTC(begin10:20), cumulative800USD/reserve700.
Scoped monitor active, delayed two-group cost461.0327006541777USD. Canonical not
yet submitted; all prior counts/evidence preserved. Broader ledger remains partial.

Prior04:47UTC: **B03 PostgreSQL counts PASS retained; idle safety stop complete**.
Both exact source/runner VMs deallocated, target Stopped verified04:47:37UTC.
Disks/data/evidence retained; scoped monitor paused. Independent full64range/root
verification still awaits specific approval, not installed or run. No final B03
qualification claim. Flexible Server stop is temporary (automatic start after7days).

Prior04:40UTC: **B03 PostgreSQL GUI migration and counts PASS**.
Imported sealed9619byte report;1.6Mvertices+4Medges/18labels,zero rejects,
24checks pass/no errors or incomplete checks. Independent all-label assertions
and SHA agree. [Counts receipt](evidence/b03-pg-counts-pass-20260923.json).
Full64range canonical check awaits specific verifier approval, not yet installed.
Idle safety stop starts04:45UTC/due04:47UTC if approval still pending; hard
05:45UTC bound unchanged. Broader ledger5pass/7partial remains until full digest.

Prior04:27UTC: **B03 PostgreSQL GUI migration submitted, without credential re-entry**.
Private target provisioned; AGE preload and same-runnerD4s_v5 resize finished.
Source existing container started with valid TLS and health gates; runner newboot
disk3.84%,swap0/OOM0. New jobd707adf8-7cab-4d73-8e38-ec141e72ac18,30minute
service bound, scoped monitor active; counts/full canonical verification pending.
Fixed stop05:45UTC(begin05:40), cumulative800USD/reserve700 unchanged.
Broader ledger5pass/7partial remains: submission is not qualification.

Prior04:00UTC: **B03 PostgreSQL private target deployment submitted via installed GUI**.
User explicitly approved E8ds_v5/128GiB PostgreSQL18+AGE, dedicated private subnet,
same-runnerD4s_v5 resize and5.6Mrow migration/standard verification. Runner started
03:48:08UTC; source remains off during deployment. Fixed stop05:45UTC(begin05:40),
cumulative800USD/reserve700, scoped safety monitor active. ARM deployment Running;
no migration/full canonical verification yet. Broader ledger remains5pass/7partial.
Latest delayed twoRGcost451.2573221409767USD. See
[execution record](other-cloud-pg-live-20260922.md).

Prior02:58UTC: **B03 PostgreSQL source inventory PASS and GUI import verified**.
After explicit user resumption, retained evidence proved60ba3809 finished before
the earlier interruption. No duplicate source scan:1.6Mvertices+4Medges/18labels,
2checks pass,no errors/incomplete checks,2947byte report SHA verified. Both exact
VMs deallocated and scoped monitor paused. Receipt:
[inventory evidence](evidence/b03-pg-inventory-pass-20260923.json).
No target/migration/canonical verification yet; broader ledger5pass/7partial.

Prior02:47UTC: **paused at user's explicit request** after reported Keychain
read trouble. New inventory60ba3809 had been running; no final counts/report
verified. Both exact PGVM source/runner now independently deallocated; monitor
paused and GUI polling stopped. Disks/data/evidence preserved; no target or
migration, no automatic restart/retry. B03 PostgreSQL remains unqualified.

Prior02:36UTC: user entered verified new PGVM credential and Remember; actual
GUI reuse works without re-entry. Approved same-two-VM inventory-only session
started02:36UTC; fixed stop03:34UTC(begin03:29), scoped monitor active. Both VMs
running; health gates/fresh GUI inventory next. No target/migration/new install.
Latest delayed twoRGcost447.08125631327976USD;800USD ceiling unchanged.

Prior02:21UTC: user-approved old PGVM health control archived/pushed then deleted;
reader reset completed and verified-TLS/read-only/18-table access PASS. Dated
PGVM Keychain item is now applied/login-verified, superseding pending state below.
Source and runner deallocated, monitor paused, all guest evidence retained.
No post-reset inventory or migration yet; B03 PostgreSQL remains unqualified.

September23 01:56UTC follow-up: approved PGVM reader reconciliation found login,
expiry,read-only and18table access settings correct. New dated Keychain credential
is **pending, not applied**: reset creation hit source VM's25/25managed command
limit. No password change or inventory retry. Both source/runner deallocated,
monitor paused. Archive/removal of one exact old read-only health control needs
review/approval; no commands deleted yet.

September23 update01:41UTC: approved7976cec extension installed/hash-verified and
actually reloaded. Explicit installed-GUI credential reuse and subsequent
inventory submission needed no re-entry: credential regression PASS alongside
508unit/14CLI contracts/typecheck/build. Approved646f0d4 Linux upgrade completed.
One diagnostic inventory failed immediately with typed category
`postgresql/snapshot-connect/authentication`,exit1,no report. The particular
credential/role condition is not established; no password reset or automatic
retry. Both exact source/runner independently deallocated by01:41UTC, before
02:22UTC hardstop; scoped monitor paused. All evidence retained, no target or
migration. B03 PostgreSQL remains unqualified. Latest delayed twoRG
costUSD444.3963173658658,800USD ceiling unchanged.

Current checkpoint13:27UTC: **B03 Neo4j other-cloud selection now has installed-GUI
migration, counts and full canonical PASS**:5.6Mrows/18labels/64ranges/root match.
Result retained/imported; source and runner deallocated, target Stopped verified
by13:25:54UTC and scoped safety monitor paused. All disks/data/evidence retained.
This supersedes the pending B03 checkpoints below. PostgreSQL other-cloud live
selection remains unqualified, so the broader ledger stays5pass/7partial.
See [canonical evidence](evidence/b03-n526-p1-pass-20260922.json).

Next PostgreSQL other-cloud preparation: existing PGVM fixture remains
deallocated. Retained public leaf expired September20; CA valid untilOctober6.
Installed new-wizard selection verified without cloud mutation. Frozen18mapping
regression and real Go/P1-projection contracts pass:503unit/14CLI contracts,
typecheck/build. Test-only; installed pins unchanged. New bounded runtime/TLS
renewal was subsequently approved with max2hours from first compute start.
New installed-GUI draft cb2ef280 reviewed; all18 persisted mappings exactly match
the fixture. PGVM reader entered; GUI encrypted reuse works without re-prompt.
Approved dedicated storage/scoped user access, account-only tag/HTTPS setting
and pinned archive upload complete; size/SHA match. New private VM preview passes,
and VM/scoped reader/development installation subsequently approved. First
compute request14:14:17UTC; fixed stop16:00UTC (begin15:55), scoped monitor active.
Runner readiness passes; existing PGVM leaf renewed with old evidence retained,
CA/key unchanged, source-local endpoint TLS1.3 verified. GUI submitted inventory
bb184697 using saved credential without re-prompt, but child failed14:19:32UTC,
exit1/no report. Offline exact SHA matching identifies only the generic message
`network inventory initialization failed`, not proof of a network or credential
fault. Local safe typed initialization diagnostics and loopback actual-CLI18mapping
regression pass; fullGo/race,503unit/14contracts pass. Installed pins unchanged.
GUI phasefailed retained, no automatic retry.
Exact source/runner deallocated by14:29UTC; safety monitor paused, all evidence
preserved. No target/migration; next is diagnosis, not another blind attempt. See
[PostgreSQL preparation and gates](other-cloud-pg-live-20260922.md).

Latest B03: user approved the bounded other-cloud-selection trial. Installed GUI
reviewed a new endpoint-only Neo4j draft. Approved storage tag/authenticated HTTPS
access and fixed archive upload are complete; anonymous/shared-key access stays
disabled. Upload size/SHA metadata, actual GUI reconnect and runner preflight pass.
Exact new VM/scoped Blob Reader/unpublished install approved; runner submitted
07:18:49UTC and provisioned, dedicated source startup requested07:19:36UTC.
Hard stop09:15UTC (begin stopping09:10) and scoped safety monitor active.
Guest readiness passed; fixture TLS renewal and retained-evidence reconciliation
passed (mount difference was array order only). Updated CA selected in GUI.
User entered credential; exact GUI inventory23f3ef65 failed with sanitized
unauthorized category/exit1 and no sealed report. No automatic retry/reset;
evidence retained, exact runner/source deallocated verified by07:37UTC and safety
monitor paused. User subsequently explicitly authorized clone-only password reset:
one recovery completed07:49:24UTC, authenticated1.6M/4M counts verified, system
backup/checksums retained and canonical Keychain item updated/readback-verified.
No inventory retry; source and runner independently verified deallocated by
07:53:40UTC, recovery monitor paused. Subsequent user continuation restarted only
these same VMs inside the original09:15UTC window. Fresh GUI readiness/source TLS
pass; failed inventory retained in history, new reviewed inventory waiting for
private updated-password entry. No new inventory submitted, target or migration.
At08:16UTC the prompt was still pending and runner idle: exact source/runner
deallocation requested, unsubmitted input cancelled. Both verified deallocated by
08:18:19UTC, heartbeat paused. Delayed costUSD398.71273; no new qualification result.
User requested continuation08:24UTC: same source/runner restarted, actual GUI
reconnect and fresh08:27:58 readiness/source TLS pass. Private password prompt
visible again. User entry submitted inventory731f5d8f at08:29:47UTC; GUI
reconciled finished at08:36UTC with a sealed663-byte report, SHA4e0efa9b18b6….
User-approved transfer imported and hash-verified08:41UTC: exact1.6M vertices,
4M edges, no errors/incomplete checks. Aggregate-only report does NOT prove
all18 labels; no full inventory/route qualification claimed. Exact two VMs
verified deallocated by08:46UTC, monitor paused; original09:15UTC unchanged.
Cost refresh429; no new target resources or migration.
Continuation target review selected E8ds_v5/128GiB and same-VM D4s_v5 with
candidate10.246.24.0/24. Stopped-guest preflight correctly refused; no target
plan persisted or deployment/credential creation. Both VMs remain deallocated;
next live phase needs a new bounded runtime authorization and fresh readiness.
User subsequently approved restart/max2hours with hardstop10:58UTC, monthlyUSD3750
and dailyUSD100 cost constraints (user-reported all-group estimateUSD1919).
Fresh readiness and target preflight passed; target remained unsubmitted. User
raised test-duration concerns; diagnosis identified repeated source-secret prompts,
readiness expiry during serial input and manual orchestration overhead. Pending
folder/deployment cancelled, exact idleVM stop requested09:07UTC; no new target
or migration. Automation improvements proposed, not yet implemented.
Both exact VMs verified deallocated by09:08UTC; safety monitor paused.
User then requested implementation and live continuation. Local automation
improvements are now implemented: optional scoped encrypted source credential
reuse, retained target inputs/offline saving, post-input readiness refresh,
bounded retained-operation status watches, one-approved-report transfer/import,
and an exact-scope bounded same-VM resize sequence. Typecheck/build and502
unit/adapter tests pass; isolated real Extension Host25/25 on baseline1.105 and
installed1.138 also pass. [Implementation and boundaries](automation-improvements-20260922.md).
User-approved operator-profile installation and actual Reload Window completed
11:51–11:53UTC. Installed bundle matches the approved5f93f3c VSIX; new credential
command opens the retained B03 workflow without Azure or credential mutation.
The existing663-byte inventory remains hash-matched/imported. Live continuation
was not performed under the expired10:58UTC authorization. User then approved a
new maximum two-hour session at11:55UTC and entered the source credential with
Remember. Installed GUI saved the LoadJob/target plan offline and reused all
inputs successfully12:16–12:19UTC. Local target is previewed only; creation
approval is pending, both VMs remain stopped and the runtime clock is unstarted.
Saved deadline14:10UTC; activate scoped safety monitoring before startup.
Latest delayed two-group costUSD420.304221975638. No Azure qualification is
inferred from local installation/tests or offline plan save/reuse.
User approved target creation; scoped monitor activated and first exact two-VM
startup requested12:22:03UTC. Fresh runner health/source TLS pass. Installed GUI
reused the plan and submitted target once; ARM Running12:28UTC. Hard stop14:10UTC,
stop initiation14:05UTC. No migration yet; retained inventory is not replayed.
Target provisioned, AGE preload finished and same-runner D4s_v5 resize finished
by12:43UTC with preservation digest verified. Post-resize readiness succeeded
with a new boot, disk3.62%,swap0/OOM0 and unchanged Linux pin. Mac screen lock
blocks GUI reconciliation/start; no migration yet and stored-secret reuse is
not yet exercised. Unlock requested; conservative idle stop12:59UTC (initiate
12:57UTC) for exact source/runner/target if still waiting, hard14:10UTC unchanged.
User returned/unlocked12:53UTC, ending that wait. Fresh GUI readiness passed;
remembered credential reused without another password prompt. Migration job
99ae1d29-f7b4-43f2-a91a-149464617a22 submitted once12:57:35UTC; ARM acknowledged
accepted and GUI automatic retained-operation watch started. This is not load
completion/counts/full-digest success. Worker30min bound and14:10UTC hard stop
remain unchanged; no inventory replay. Full verifier still requires its own
action-time approval before installation/run on this guest.
Migration finished13:05:07UTC (about7m17s). Installed GUI imported the sealed
9619-byte report/SHAa66aa8c3a183… and displayed Counts PASS. Independent assertions
confirm1.6Mvertices+4Medges/all18labels/zero rejects/24passed checks/no errors or
incomplete checks. Correct Neo4j raw-ID verifier8a23a5109798/SHA60ed56a6773e…
awaits action-time approval; Gremlin-profile e70e02068c68 is not applicable here.
No full-digest/B03qualification claim yet. Pending-approval idle stop13:27UTC
(begin13:25), original14:10UTC hard stop unchanged.
User immediately approved correct raw-ID verifier placement/run/import13:11UTC;
GUI submitted operation6fb20b39 at13:12:26UTC, bound to job99ae1d29. Approval
wait ended; healthy25min/4GiB verification is now active under the unchanged
14:10UTC hard stop. Full canonical result is still pending.
The subsequent canonical PASS is recorded in the current checkpoint above.
Broader branch totals remain 5 pass / 7 partial. See
[B03 preparation and gates](other-cloud-n526-live-20260922.md).

September 22 cost-safety follow-up: eight old retained PostgreSQL targets were
unexpectedly Ready with all trial VMs deallocated. Ownership and recent activity
were reviewed, then stop requests submitted for those exact targets; no deletion,
data/credential/network change or new test window. Fresh delayed two-group cost
is USD390.9644901299309 under the unchanged cumulative USD800 ceiling. The next
proposed B03 other-cloud-selection trial subsequently received bounded runtime
approval (see latest checkpoint above); expired fixture TLS still needs renewal. See
[safety-stop evidence and proposed next trial](idle-cost-safety-20260922.md)
for final stopped-state verification. Qualification totals are unchanged.

Latest local B02 follow-up: 13 added contracts connect the production preview
handler and preflight to synthetic ARM replies. SKU/quota denials stop before
release fetch or persistence; exact quota and unknown-zone reviewed controls
retain the release gate. Typecheck, 476 unit tests and build PASS. Test-only;
no installed extension or Azure changes. This does not close the three remaining
live B02 gaps or change branch totals. See
[isolated placement contracts](placement-panel-contract-20260922.md).

Latest B02 control: post-fix installed GUI accepts read-only placement checks
then refuses missing 2.4.0 release/checksums; no deployment or preview persistence.
Unsaved wizard closed and all 82 operator files unchanged. The preceding four GUI
placement refusals plus this control are complete. Read-only Japan East capacity
shows all three discovery SKUs available in zones 1/2/3 and sufficient quotas, so
unavailable-SKU/quota-denial cases cannot be qualified there without a different
fixture. Unknown-zone VM also remains open. B02 partial, 5 pass / 7 partial; see
[completed control and capacity](placement-arm-readonly-20260922.md).

Latest B02 installed-GUI batch: approved `7338faa` package installed/reloaded,
matching JavaScript hash and corrected VNet guidance verified. Four actual GUI
previews correctly refuse nonexistent subnet, actual DB delegation, VNet-region
mismatch and source-VM zone mismatch. No startup/mutation/source access or saved
preview; all82operator files unchanged. Post-fix valid-placement release control
was interrupted by unreliable native menu input and remains unverified. Unknown
zone/SKU/quota cases still pending; B02partial, totals5pass/7partial. See
[installed results](placement-arm-readonly-20260922.md).

Latest B02 follow-up: signed-in GUI recovered and visibly refused a fresh preview
because matching2.4.0release/checksums are unavailable; unsaved wizard closed,
all82operator files preserved. Moved GET-only placement preflight ahead of release
lookup locally, retaining mandatory artifact/deployment gates and fresh submit
checks. Five handler regressions added:463unit/25isolated-host tests and build
pass. Corrected build installation and actual GUI backend refusals remain pending.
No startup, deployment, source access or release. B02 remains partial; see
[ordering correction](placement-arm-readonly-20260922.md).

Latest B02 batch: **6/6 unchanged production placement checks pass against real
ARM responses**, 19 GETs, no mutations/startup/source data. Actual nonexistent
subnet/group, delegated subnet, VNet-region mismatch and source-zone mismatch
are refused; valid placement passes. This is not installed-GUI rejection proof.
GUI capture failed before a new preview outcome; all82operator files unchanged.
Corrected stale same-group-only VNet guidance in source; typecheck,458unit tests
and build pass, but not installed/visually verified. B02 stays partial, totals
5pass/7partial. See [read-only placement evidence](placement-arm-readonly-20260922.md).

Latest offline B12 batch: **real isolated VS Code1.138.0 host25/25PASS**, including
12new retained canonical-report cases. One valid control opens the actual PASS
tab;11invalid reports are rejected without new PASS presentation, persistence,
redownload or changed evidence. Production controller/store/VS Code API are real;
reports and storage ARM replies are synthetic. No signed-in Azure fault claim.
All457unit tests pass; all82operator files and installed production bundle remain
unchanged. No cloud startup or mutation. B12 remains partial and branch totals
stay5pass/7partial. See [host rejection evidence](p1-retained-host-20260922.md).

Current B01 phase: **full canonical qualification PASS; exact compute stopped**.
Installed GUI imported the23,310-byte sealed result and displayed full P1 PASS.
Independent byte/hash/job checks and both recomputed roots match all64ranges,
1.6Mvertices+4Medges/18labels with typed properties/identities/endpoints.
Exact VM deallocation/target stop requested September22,01:19:46UTC; VM
deallocated/target Stopped/private verified01:21:56UTC. Safety monitor PAUSED;
disks/evidence retained, storage/Cosmos charges continue. Defined branch ledger now
has5pass/7partial; nine base routes remain9/9, not all-branch release qualification.
See [B01 final result](separate-network-live-20260921.md) and
[sealed evidence summary](evidence/b01-p1-pass-20260922.json).

Previous B01 phase: **corrected GUI renewal/readiness PASS; full verifier submitted once**.
Exact existing VM/target starts requested September22,01:06:39UTC. Hard deadline
02:00UTC/11:00JST (less than60minutes), safety stop begins01:55UTC; scoped monitor
and exact VM shutdown configured before startup. Cumulative USD800/reserveUSD700,
combined compute1.448/hour; delayed costUSD370.284238355361. Finished migration,
job/counts report retained. No new migration/source/security changes. Fresh pinned
readiness passed on new boot, disk3.673%, no swap/OOM. Installed GUI submitted
operation8c5c9c6f-be92-4edf-88eb-22ce062e575e at01:12:51.293UTC; reconcile its
25minute/4GiB execution and full64range report, not yet PASS.

Previous B01 phase: **corrected extension installed/reloaded; retained B01 reconnected; compute stopped**.
Approved a4b61d8 installation and actual Reload Window/reconnect completed.
Installed bundle matches reviewed SHA; all81operator files are byte-identical.
No replay/startup/verifier submission. Next: GUI renewal and approved bounded
verification session. B01 remains partial until64ranges/root match.

Previous B01 phase: **post-load time-window renewal corrected locally; installation pending; compute stopped**.
User approved a60minute exact-resource verification session and the pinned
read-only verifier. Before startup, installed GUI refused renewal solely because
the migration already finished. No window saved, startup or verifier dispatch.
Local fix accepts only sealed passing finished counts (or pre-migration), keeps
active/uncertain/failed paths blocked and preserves all non-cost evidence.
Typecheck,457unit tests/no skips and build PASS. Corrected extension installation
and actual GUI renewal remain pending action-time approval; no manual record edit.
Fresh delayed two-group cost USD370.284238355361; cumulative800/reserve700unchanged.
The60minute live window has not started. See [renewal correction](separate-network-live-20260921.md).

Previous B01 phase: **safely stopped; GUI migration/counts PASS; canonical digest pending; B01 partial**.
Idle approval bound triggered exact VM deallocation and target stop. Activity
timestamps show13:42:16UTC, sixteen seconds late; no retroactive extension.
VM deallocated verified13:42:29UTC; target still Stopping then.
Both terminal states verified13:44:37UTC, target Stopped; asynchronous DB stop
completed after the conservative idle deadline, without extending authorization.
Retained report SHA unchanged, disks/data/evidence preserved, no verifier
uploaded/installed/run. Scoped safety heartbeat PAUSED. Storage charges continue;
Flexible Server warns of automatic startup after seven days. Any next live
session requires explicit pinned-verifier approval and bounded exact-resource
restart. See [safety stop](separate-network-live-20260921.md).

Previous B01 phase: **GUI migration and complete counts PASS; full digest awaiting approval; B01 partial**.
Same job completed13:27:18UTC,14m53.21s after submission including preparation,
load and verification. Installed GUI imported sealed9,619-byte report and visibly
displayed Counts verification: PASS. Independent SHA/job/fingerprint assertions
and all18source-label counts agree:1.6Mvertices+4Medges, zero rejects,24checks pass,
no errors/incomplete checks. Exact worker inactive/dead; disk4%, swap0/no OOM.
Delayed two-group costUSD349.336185663541, capUSD800/reserveUSD700unchanged.
Full64range Gremlin verifier e70e02068c68 is not installed/dispatched; action-time
approval requested for this exact VM/job. If approval does not arrive, stop exact
VM/target by13:42UTC conservative idle bound; outer15:00UTC unchanged. Existing
safety monitor active; evidence retained. See [counts acceptance](separate-network-live-20260921.md).

Previous B01 phase: **same-VM sizing complete; GUI migration accepted, Linux worker running; B01 partial**.
GUI advanced the existing VM from B2s_v2 to D4s_v5 while preserving its disk,
NIC/identity and placement. AGE preload applied during target start; pending=false.
Fresh post-resize readiness passed with pinned Linux d40d6ccc9a4d, disk3.65515%,
idle/no swap/OOM. One GUI migration submitted13:12:24.916UTC: job
`37624cff-6aea-449e-9de3-38c6aa5c984d`, accepted against the retained5.6M/18label
inventory. Exact worker independently active/running13:13:32UTC, loader31,912KiB
RSS, disk4%, swap0/no OOM. Counts and canonical verification remain pending.
Worker30minute bound and15:00UTC outer stop unchanged; cumulativeUSD800 cap.
Scoped safety monitor active; no new resource/source/security changes. Full64range
verifier installation remains separately gated. All62targeted regressions pass.
See [live migration evidence](separate-network-live-20260921.md).

Previous B01 phase: **approved bounded migration session starting; B01 partial**.
User explicitly approved existing B01 VM/target restart, AGE readiness, same-VM
D4s_v5 resize and5.6M-row migration/verification. Start requests issued13:03:43UTC;
fixed stop15:00UTC (September22 00:00JST), less than two hours. Scoped monitor and
exact-VM auto-shutdown enabled before startup. Cumulative USD800 cap/USD700reserve
unchanged; latest cost refresh429, previous delayed original-groupUSD341.803553466625
and B01 charges unreported. Installed GUI recorded renewed authorization at the
same USD1.448/hour compute rate. Fresh readiness submitted13:04:12UTC; no resize,
migration or canonical verification claimed yet. No new resources/source/RBAC/
network changes or deletion. See [bounded migration session](separate-network-live-20260921.md).

Previous B01 phase: **Azure target deployment and corrected installed-GUI reconciliation PASS; B01 partial**.
User-approved local extension `07d1e45` installed on Mac VS Code1.138.0;
bundle hash matches. Actual Reload Window/reconnect preserved all80operator
files. GUI read-only review now displays `Private target: provisioned`; only
this workflow's target.phase changed from unknown to provisioned. All source,
plan, inventory and other evidence retained; no replay or migration. VM remains
deallocated and target Stopped. Prior11:45UTC deadline expired, monitor paused.
Runtime AGE readiness, same-VM resize, migration and64range verification need
a new bounded live authorization. See [installed-GUI acceptance](separate-network-live-20260921.md).

Previous B01 phase: **Azure target deployment PASS; installed-GUI reconciliation blocked by a corrected local defect; B01 partial**.
Parent deployment succeeded09:57:46UTC; all reviewed parent/child resources pass.
The installed extension counts successful targetless ARM output-evaluation rows
as extra resources and retains `unknown`. Local fix preserves strict leaf checks;
456unit tests/typecheck/build pass, corrected read-only live ARM reconciliation
returns `provisioned` without changing operator records. Corrected VSIX install
and actual GUI recheck remain pending. Preload configured but requires restart;
runtime AGE readiness, resize, migration and64range verification not performed.
Exact target Stopped and VM deallocated verified by10:02:33UTC; monitor disabled,
disks/evidence retained. See [completion and reconciliation defect](separate-network-live-20260921.md).

Previous B01 phase: **approved separate-network target deployment submitted; B01 partial**.
User approved exact private PostgreSQL18/AGE E8ds_v5/128GiB-storage target,
dedicated subnet/DNS and SecretStorage, existing VM restart, maximum2hours;
no resize or migration. Hard stop11:45UTC/20:45JST, USD800/USD700reserve unchanged.
VM start09:47:40UTC and fresh pinned readiness pass. Installed GUI renewed
plan `395b74a99c91…`, passed separate-scope what-if and submitted one parent;
ARM Running09:50:54UTC. Reconciliation is read-only, no replay. Latest delayed
original-group costUSD341.803553466625; new-group charges not yet reported.
Scoped stop monitor active; stop exact VM/new target at outcome or bound.
See [bounded deployment session](separate-network-live-20260921.md).

Previous B01 phase: **fresh GUI target preflight and save-only plan PASS; B01 partial**.
Approved15minute existing-VM restart began08:55:35UTC. New-boot readiness passed,
then installed GUI reviewed the separate-network-group subnet and private target
with current USD1.448/hour combined compute, USD700 reserve/USD800 cap. Saved
LoadJob and target plan (`20abaf421602…`) only; no credentials, target deployment,
resize or migration. Exact VM deallocated by09:01:22UTC and heartbeat disabled,
before09:09UTC short-session bound. Evidence/disks retained. Actual deployment,
migration and64range canonical verification remain pending. See
[target preview and safe stop](separate-network-live-20260921.md).

Previous B01 phase: **source inventory and sealed GUI import PASS; B01 partial**.
Existing operation completed; report generated08:42:01UTC. Installed GUI exported
and imported exact2,944bytes/SHA `e34bf7857c91…`. Independent assertions match
all18frozen P1label counts,1.6Mvertices+4Medges; no failed/incomplete checks.
Actual report/imported UI verified. Exact VM deallocated by08:44:36UTC and
heartbeat disabled, before09:30UTC. Disks/evidence/approved reader grant retained.
No target/migration; canonical migration verification remains pending. See
[accepted inventory and safe stop](separate-network-live-20260921.md).

Previous B01 phase: **complete source inventory running; not yet accepted**.
User approved exact VM account-wide Cosmos Data Reader, specified-container
reads/report transfer and18:30JST stop extension; USD800 unchanged. GUI grant
verified, existing VM start08:29:25UTC and new-boot readiness passed. Inventory
`68831e91-96e4-4b85-a720-90c7aa7ab14f` started08:30:50UTC with30min/4GiB bound;
08:31:51UTC active/running, disk4%, guest used259MiB, no swap/OOM. Monitor retains
the exact operation and approved report import, then VM deallocation; no source
writes, new DB or migration. B01 remains partial. See
[current bounded inventory session](separate-network-live-20260921.md).

Latest B01 corrected live readiness: user-approved VSIX installed and actual
Reload Window/reconnect passed with all79 operator files unchanged. Existing VM
started08:00:02UTC; one explicit readiness command succeeded08:00:46UTC and was
reconciled in the installed GUI. Pinned Linux version/commit/SHA match; idle,
disk3.51%, memory245MiB, no swap/OOM. Successful receipt retained; prior exit127
failure preserved. No source grant/read, target or migration. This validates the
already-bootstrapped restart path, not initial-boot pending behavior or B01
qualification. VM deallocated verified by08:03:32UTC; scoped heartbeat disabled,
before the unchanged08:35UTC bound. Disks/evidence preserved.
Delayed original-group costUSD337.313050576123, capUSD800. See
[readiness-only live evidence](separate-network-live-20260921.md).

Previous offline B01 correction: readiness waits up to45seconds for bootstrap
within its existing60second command bound. Still-running bootstrap is explicitly
pending, never successful/automatically retried; failed bootstrap remains failed.
Source/migration dispatch and pinned checks remain unchanged. Typecheck,452unit
tests and build pass; actual corrected GUI/Azure qualification is pending.
VM freshly verified deallocated;08:35UTC deadline unchanged. See
[bootstrap race and correction](separate-network-live-20260921.md).

Latest B01 follow-up: user renewed maximum2hours/unchangedUSD800. Created
dedicated migration group `rg-af-vscode-p1-b01-20260921`; actual GUI saved fresh
Cosmos workflow `5cb990c1-2a25-4de5-a10d-09fab2ef0b18`, using the original group's
VNet and reviewed zone1. After exact approval, transfer-account deployment
succeeded06:25:28UTC with account-scoped user Blob access. Following exact user
approval, applied the account-only exception/public HTTPS setting, retaining
anonymous/shared-key disabled. Authenticated access and installed GUI ready
state pass. Pinned37,197,546-byte archive uploaded06:31:38UTC with matching SHA
metadata and retained ready state; GUI reconnect preserved the workflow.
Fresh GUI VM preview passed06:33:04UTC atUSD0.109/hour plus storage/network;
exact VM/container-reader/install approval was subsequently received. GUI
submitted06:36:05UTC; deployment succeeded06:36:46UTC, no public IP and correct
other-group subnet/container grant independently verified. Hard stop08:35UTC.
Initial readiness failed exit127 at06:37:24UTC because tools were not yet
installed; cloud-init finished06:37:26UTC and a later read-only diagnostic found
the tool present, disk4%, memory275MiB, no swap/OOM. Early-readiness/bootstrap
race is supported; failure retained without retry. Requested exact VM stop
under the terminal-failure rule; deallocated verified06:40:15UTC. No assessment,
source grant, target or migration.
Older78operator files unchanged;28target regressions pass. Delayed old-group
costUSD333.523399055421. B01 remains partial. See
[bounded separate-network trial](separate-network-live-20260921.md).

Latest read-only follow-up (September21,06:03–06:06UTC): installed GUI refuses
removal of the historical FK readiness record because current ARM execution is
Pending (independently confirmed; no output/start/end). A referenced catalog
receipt is also refused; workflow-selector cancellation preserves all78files.
All42removal regressions pass. Exact VM remains deallocated; no Azure mutation
or install. Removal confirmation-modal cancellation was not reached and is not
claimed. B09 remains partial; preserve the record without bypassing admission.
See [receipt admission evidence](command-receipt-lifecycle-20260918.md).

Latest offline follow-up: installed-GUI readiness-receipt archive, selector
Cancel and repeated archive pass using an actual FK-trial receipt. Cancel left
all77 operator files unchanged; first archive added one1,434-byte mode0600 file,
all originals unchanged; repeat kept all78files identical. Independent hashes
and24targeted regressions pass. No Azure operation, install or deletion.
See [receipt lifecycle](command-receipt-lifecycle-20260918.md). B09 remains partial:
this is local evidence preservation, not live record-removal qualification.

Current B04 result (September21, 02:17UTC): **defined FK catalog/adoption/inventory
slice PASS**. Installed GUI imported the exact sealed2,087-byte report;
independent hash/count assertions match five vertices plus three edges/all3labels,
two passed checks, no errors/incomplete checks. Explicit properties and actual
Reload Window/reconnect already passed; nullable FK stays manual-review-only.
B04's defined acceptance is now satisfied; broader branch coverage remains partial.
This adds no migration/digest claim. Exact VM deallocated and source Stopped/private
verified by02:20:24UTC, before02:39UTC; monitor disabled. Evidence retained.
No new resources/updates/grants/target/migration. Delayed RG costUSD326.07607295488.

Previous B04 step (September21, 01:58UTC): explicit adoption of the isolated
fixture's two vertices and safe FK edge, reviewed properties and actual Reload
Window/reconnect passed while stopped; sourceDraft SHA unchanged. User approved
existing VM/source restart for at most30minutes. Starts accepted01:39:15UTC;
hard stop02:08UTC (11:08JST). Fresh readiness passed, but no inventory was
dispatched before the fifteen-minute idle input limit01:57:30UTC. Cancelled
the unsubmitted private input and requested exact VM/source stop01:58UTC.
VM deallocated and source Stopped/private verified by02:01:16UTC; monitor disabled.
Mapped five-vertex/three-edge inventory remains pending. Latest delayed RG cost
USD326.07607295488; capUSD800 unchanged. Mappings/evidence preserved.
No new resource, update, grant,
target or migration. See [FK trial](postgres-fk-live-plan-20260920.md).

Previous B04 follow-up (September 20, 09:33 UTC): the approved
[isolated FK trial](postgres-fk-live-plan-20260920.md) produced a complete sealed
two-table catalog. Installed GUI imported its exact bytes/SHA and displayed two
vertex candidates, one safe products-to-suppliers FK edge and the nullable-FK
manual-review warning. All candidates remain unselected. Adoption, reconnect and
the five-vertex/three-edge mapped inventory remain pending; no migration claim.
The monitor only reconciled existing export/import. At09:35UTC, after fifteen
minutes without a worker since export completion, it requested exact VM
deallocation and source stop, preserving schema, disks and evidence. Both stopped
states were verified by09:37:30UTC, before the09:42UTC bound; heartbeat disabled.

Previous offline follow-up: [CSV choice binding audit](csv-choice-bindings-20260920.md)
passes for the accepted P1 route: actual installed-GUI restoration of all 18
file/type/ID/endpoint selections, exact retained configuration/verified-transfer
bindings and unchanged 74 operator files. Added regressions pass (445 unit tests,
13 Go configuration contracts); no production change, install or Azure mutation.
B04's live FK proposal display now has evidence above, but explicit adoption and
mapped inventory remain open. Other partial rows in the branch ledger remain
open as well. Accepted P1 records/data were not rebound or modified.

Previous B04 step (September 20, approximately 08:45 UTC): installed-GUI recovery,
fresh runner readiness and exact sealed inventory export/import **passed**.
All 18 mapped-label counts match the frozen P1 fixture: 1.6M vertices + 4M edges;
report pass, no errors/incomplete checks, independently matched 2,947 bytes/SHA.
VM deallocated and source Stopped/private verified; safety heartbeat disabled.
No source reread, target or migration was started. The catalog-to-mapped-inventory
slice passes; at that checkpoint B04 still lacked live FK recommendations and
exact CSV choice bindings (the latter audited above). This inventory is not new
migration/canonical-digest evidence.

Previous B04 step (September 20, approximately 08:38 UTC): user-approved fix
`3dff349` is installed with matching bundle SHA and actual Reload Window/reconnect.
Installed GUI successfully retained the rejected export after expiry/absence checks;
the same finished inventory and seal are preserved, with no automatic replay.
Fresh runner readiness was invalidated and is required before transfer; catalog
recommendation refresh is gated until then. VM deallocated/source Stopped remain
verified. Next: bounded runner-only readiness and sealed report export/import.
Exact inventory count acceptance is still pending; no target or migration started.

Previous B04 step (September 20, approximately 08:17 UTC onward): user submitted
report-transfer approval after VM shutdown. Azure rejected the export with HTTP
409; exact command and report blob are absent, with unchanged sealed inventory.
Local extension recovery fix preserves the rejected attempt and requires explicit
review, expiry/absence proof, fresh readiness and separate transfer approval.
It also blocks initial exports to stopped VMs before intent creation. Installation
and real GUI recovery/import remain pending. VM/source were not restarted.

Previous shutdown checkpoint (September 20, approximately 06:41 UTC): report transfer remains
unapproved after 15 minutes of idle wait. Source Stopped and exact runner
deallocated verified by approximately 06:42 UTC; safety heartbeat disabled,
with all disks/data/reports preserved. Inventory worker
exit is successful, but sealed import and exact counts remain pending. No retry
or migration was started. Latest delayed cost USD 295.608869711763 / USD 800.

Previous worker checkpoint (September 20, approximately 06:25 UTC): inventory worker
`3a4ffae1-8727-464b-8968-40bed2f42c69` finished successfully. Sealed report is
2,947 bytes / SHA-256 `821638b1e510ff545c424dbfaacf4508d9727c78d9368afc8c34aa98285b9a18`.
Exact transfer approval is pending; complete report and all 18-label counts are
not yet accepted. Source stop requested; runner retained briefly for export.
Fresh delayed RG cost USD 295.608869711763, below USD 800. B04 remains partial.

Prior restart checkpoint (September 20, approximately 06:10 UTC): user approved restarting
the existing runner/source and appropriate time extensions; USD 800 unchanged.
Current session bound is 08:00 UTC / 17:00 JST, with exact-VM auto-shutdown and
dedicated safety heartbeat. Source is Ready/private; refreshed installed-GUI
readiness passes on the new boot (disk 3.71%, idle, zero swap/OOM, pinned build).
After private reader entry, inventory `3a4ffae1-8727-464b-8968-40bed2f42c69`
was submitted once and reconciled accepted, with unchanged reviewed configuration.
Full completion, sealed import and exact 18-label counts are pending. The monitor stops both
resources after 15 minutes of idle user-input wait, terminal outcome or bound.
Latest cost refresh returned 429; USD 284.682696744039 is delayed, not current.

Previous offline B04 checkpoint (September 20, approximately 03:23 UTC): fresh GUI workflow
`ae952310-5eba-42b6-9fe3-9db00e93cdac` preserves the previous failed attempt.
User-approved transfer storage and account-scoped Blob role were created;
The approved exact-account exception tag/authenticated HTTPS access now pass
ARM and authenticated listing checks; anonymous/shared-key access remain off.
After explicit upload approval, the pinned Linux archive is transferred and
reconciled ready. User-approved private runner deployment succeeded, its
container-scoped Blob Reader grant is verified, and post-bootstrap GUI readiness
passes (disk 3.51%, zero swap/OOM, exact pinned build and catalog capability).
The first readiness check preceded bootstrap completion and failed; its evidence
is retained. Metadata-only catalog and sealed import passed: complete 18-table
report, 23,753 bytes, independently matched SHA. Nine intended vertex candidates
were explicitly adopted in the GUI. Offline GUI edits now preserve all nine
vertex and nine edge mappings, matching the frozen PostgreSQL P1 fixture exactly
(ignoring row order), including explicit identity-property projections. The
generated configuration passes `assertP1Projection`. Actual Reload Window and
reconnect restored all 18 mappings and connection fields without replay.
New complete inventory remains open; B04 is partial. VM deallocated and source Stopped verified
by 02:43:55 UTC, before the 02:45 bound. All evidence/disks are preserved.
See [catalog receipt](evidence/pgfs-catalog-r2-20260920.json).
See [fresh draft evidence](postgres-catalog-gui-20260919.md#fresh-catalog-draft--september-20-approximately-0220-utc).

Latest B04 checkpoint (September 20, approximately 02:10 UTC): approved extension fix `d85965c` is installed; actual
GUI preview renewal preserved the same workflow and pinned Linux archive.
The private discovery VM was provisioned once and the GUI verified the exact
guest version/hash and `postgresql-catalog-v1` capability. Disk 3.49%, no swap
or OOM. The user-approved first catalog attempt failed during authentication
(SQLSTATE 28P01), before collecting schema metadata. **B04 remains partial.**
Failed operation/evidence are retained with no automatic retry. The new VM is
verified deallocated; source Stopped is also verified after credential recovery.
The historical file guidance was corrected: the pgfs staging file is the
administrator password, not the reader password. Approved reader rotation now
**succeeded at 02:07:32 UTC**, after the user authorized Keychain access:
committed transaction, separate verified-TLS/read-only login, unchanged
non-password role attributes, empty stderr. The named Keychain item is now
applied; old password files are unchanged. Both temporary reset transports were
removed with guest evidence retained. A separately reviewed fresh catalog
attempt is still required. Latest delayed RG September cost is
USD 284.682696744039 (delayed billing); USD 800 / September 20 16:14 JST outer
limit remains, with an earlier 11:45 JST session shutdown bound. One retained
completed target found Ready was safely stopped and verified; no data deleted.
See [September 20 execution and safety ledger](postgres-catalog-gui-20260919.md#fixed-installed-gui-deployment-and-readiness--september-20-00480059-utc).
The checkpoints below retain historical states, not current installation status.

September 19 installation checkpoint: user-approved `d40d6cc` VSIX is now
installed on MacStudio's VS Code 1.138.0. The real PostgreSQL source editor shows
schema discovery controls and blocks catalog actions for the retained old-runner /
already-assessed workflow; all 70 operator files remain unchanged. The matching
Linux archive is built but not uploaded or installed. Fresh Azure reads show
10 VMs deallocated / 18 Flexible Servers stopped. Cost refresh returned 429 and
external resource writes require review before a new live session. B04 remains
partial, not remote catalog/adoption PASS. See
[installed-GUI checkpoint and next gates](postgres-catalog-gui-20260919.md).

Follow-up: new GUI-created local draft `66a26571-953f-4221-9659-a4b35460ffc4`
is ready for scoped catalog trial preparation; no Azure resources or operations
were created. Both cost-query scopes returned 429; the user subsequently supplied
a currently updated September portal total of **USD 265.69**, clearing that cost
gate without claiming an API refresh. The user-approved new transfer account
and account-scoped role are now created and ARM deployment succeeded. The GUI
correctly distinguished provisioning success from transfer readiness. Following
the user's continuation instruction, the exact-account exception tag and
authenticated public HTTPS access were applied; anonymous/shared-key access
remain disabled. Authenticated listing and the installed-GUI pinned Linux
archive upload now pass. The new private B2s_v2 runner preview is complete;
action-time approval for guest installation and container-scoped Blob Reader
is requested at the native creation dialog. No VM/source start has occurred.
The USD 800 ceiling and September 20 16:14 JST deadline remain.
The later continuation exposed an expired-preview renewal bug: the installed
view lost its retained draft ID and looked for the unpublished release instead
of reusing the pinned guest archive. Both attempts failed before deployment;
the proposed VM is absent. The extension-only fix passes 430 unit tests and
13 isolated host tests, with a new VSIX packaged but not installed. Explicit
renewal retains the workflow/artifact while preserving expiry, placement,
concurrency and fresh-consent guards. B04 remains partial.
The expected P1 source
has primary keys but no declared FKs; explicit manual edge mappings remain
necessary, and a catalog-only result must not be called full P1 qualification.

All routes completed the installed-GUI workflow, complete counts verification,
and full P1 canonical comparison (1.6M vertices, 4M edges, 64 ranges). This is
the defined P1 functional qualification, not production-scale certification,
recovery qualification, or a claim that every possible configuration is covered.
The dated sections below retain historical failures and intermediate states.

September 18 local PostgreSQL catalog follow-up: extension-side retained intent,
capability/fresh-readiness checks, sealed report import and explicit GUI mapping
selection/adoption are now connected to the bounded Linux catalog operation.
Existing manual and unsaved mappings are preserved; adoption requires a fresh
source review and complete inventory before sizing. All **428 extension unit
tests / 13 isolated host smoke tests**, typecheck and build pass. Earlier real
local PostgreSQL 18.6 TLS CLI/metadata evidence remains applicable to the unchanged
guest implementation. This is **not installed or Azure-qualified**: B04 stays
partial, with signed-in GUI/Linux qualification next. Operator records and the
installed extension are unchanged. See
[execution boundaries and evidence](postgres-recommendations-20260918.md#stage-3-extension-controller-and-explicit-gui-adoption).

**Latest Gremlin state (September 18 05:13 UTC): FULL CANONICAL PASS.** Installed
GUI imported the sealed result for `b0530700-ccd4-4f33-84fa-0854c8f4037b` and
displayed **P1 full canonical digest: PASS**. Independent Mac recomputation agrees:
**1.6M vertices + 4M edges, 18 labels, all 64 ranges**, typed properties, composite
identities and endpoints. Root:
`8a048faa36fad90404c263d3ce75073d117e5d96a15f8a614a42347cbd7a0ef4`.
The active verifier survived actual Reload Window/reconnect without replay.
B05 is pass for **Gremlin-shaped NoSQL**, not a native Gremlin API claim; B10
forced Extension Host crash remains open. At **05:13:54 UTC**, exact VM
**deallocated** and target **Stopped** are verified, preserving all data/disks/
evidence; this safety monitor is paused. See
[full result receipt](evidence/gremlin-full-pass-20260918.json).

Historical completed checkpoint: migration
`c257a458-be95-44be-a3ad-54e2318b1856` and complete counts **PASS** in the
installed GUI. Report generated at **03:27:12 UTC**, approximately 13m 34s
after submission: exact **1.6M vertices + 4M edges / 18 labels**, zero rejects,
no failed/incomplete checks. Actual active Reload Window/reconnect preserved
the job, which subsequently finished without replay. Full **64-range Gremlin
canonical digest is not run**; B05 remains partial. At **03:34:36 UTC**, the exact VM was verified deallocated
and the new target Stopped, with all evidence retained; the monitor is paused.
USD 800 ceiling, USD 650 reserve and **06:00 UTC / 15:00 JST
today** session bound remain. See the
[execution record](gremlin-gui-execution-20260918.md#migration-and-complete-counts-pass--september-18-03270332-utc)
and [count receipt](evidence/gremlin-counts-pass-20260918.json).

September 18 JST: a [fresh installed-GUI Gremlin draft](gremlin-gui-execution-20260918.md)
selected the prepared source and private runner placement with the existing Azure
session. The approved new transfer storage's exact-account `SecurityControl=Ignore`
tag and public HTTPS access are now applied; anonymous/shared-key access remain
disabled. Authenticated listing and the installed-GUI pinned runner upload pass.
The approved VM provisioned with no public IP and container-scoped Blob Reader;
its pinned Linux readiness passed with idle worker, disk 3.51%, zero swap/OOM.
Cosmos account-wide Data Reader is approved/verified. The first full inventory
failed during mapping resolution with an opaque error; evidence is retained and
no retry occurred. Independent bounded source reads prove current access and a
separate deterministic discovery-limit problem: 10,100 rows/2 unique labels
exceed the configured 10,000-document cap. Correct catalog discovery before retry;
the exact original error is not proven. The runner is verified deallocated to
avoid idle compute; evidence is retained. No target or migration was dispatched.
B05/B10 remain open.

The subsequent local correction uses bounded exclusion-based exact catalog
enumeration, preserving empty-page continuation and keeping sampled profiling
distinct. Safe inventory diagnostics now emit fixed failure categories only.
The Go suite, focused race tests and all 300 extension tests/build pass; see
the [correction/review record](gremlin-gui-execution-20260918.md#local-discovery-correction-and-review--september-18-jst).
The approved pinned Linux correction was subsequently installed and verified
in the actual GUI (commit `290c6efa3b3e`). Retry preparation then exposed an
extension defect: failure retention required the old boot even after a healthy
restart. No new source operation was submitted. The TypeScript correction and
all **309** extension tests/build pass locally. Following specific approval,
the reviewed VSIX is now installed/reloaded and cross-boot failure retention
passes in the actual GUI. Fresh inventory `c5ce0e77-ac49-472a-a112-b10e5d375b0f`
started September 18 at `00:14:18Z` and finished successfully at **00:25:25Z**
(**11m 6.63s**). The installed GUI imported its hash-verified report: exact
**1.6M vertices + 4M edges**, all **18 label counts** matching the prepared
manifest, no errors/incomplete checks. Actual active assessment Reload
Window/reconnect preserved its operation and original start time without replay.
Disk 4%, swap/OOM zero; latest delayed RG cost USD 218.41. The exact runner is
verified deallocated and the inventory heartbeat is paused. No target or
migration is started and no additional route is qualified. A supplemental local
runner-test rerun hit the Mac's real 81%-used disk capacity gate; Cosmos/app and
309 extension tests pass. See the execution record for exact scope.

The subsequent installed-GUI target review exposed an admission defect: Gremlin
auto-discovered labels were incorrectly compared with an empty manual mapping
list. It stopped before target planning or deployment. The narrow local fix
accepts the sealed complete catalog with configuration/count/bounds checks;
**314** extension tests and compilation pass, including the real retained
5.6M-row report. A new VSIX installation/reload is pending specific approval;
installed-GUI target review and migration remain unqualified. No Azure resource
was started or changed for this correction.

Following specific approval, the pinned `60ea712` VSIX is now installed and
reloaded; its installed-code hash matches. The actual GUI restores the same
sealed report and passes Gremlin label admission. Target preflight now correctly
requires a running/freshly checked guest. The runner remains deallocated after
its old 01:26 UTC bound; a new bounded readiness session is awaiting approval.
No target intent, credentials or deployment were created. Latest billing refresh
returned 429; the previous USD 218.41 total is delayed evidence, not current cost.

The user subsequently approved a bounded same-VM session until September 18
15:00 JST. Actual GUI readiness passes: idle, disk 3.90%, zero swap/OOM, same
pinned Linux artifact. Fresh delayed billing is USD 220.81. D4ds_v5 correctly
failed PostgreSQL family quota (62/64 vCores used); E8ds_v5 passes full target
preflight without deleting resources or raising quota. The GUI saved the
reviewed LoadJob/private-target plan only: E8ds_v5 / 128 GiB storage, later
same-VM D4s_v5, USD 1.448/hour compute, USD 650 reserve, USD 800 ceiling.
Specific new target/subnet/credential approval is pending. No target deployment,
source replay or migration occurred; runner deallocation was verified after
the completed readiness/preview session to avoid idle approval-wait costs.

The [isolated Gremlin source preparation](gremlin-source-execution-20260917.md)
passed at September 17 13:50:12 UTC: all 5,600,000 successful writes and
independently drained remote rows match, and all 18 file counts match the pinned
manifest. The checksummed result and failed first-attempt evidence are retained.
The exact temporary writer was removed and its absence verified; the preparation
VM was deallocated with its OS disk retained. The preparation-only heartbeat is
paused. The other eight VMs and all 17 Flexible Servers were already stopped.
This is a Gremlin-shaped NoSQL fixture, not Gremlin API qualification. The fresh
installed-GUI migration and full target canonical digest remain pending.

The earlier [Gremlin live preparation checkpoint](gremlin-live-preparation-20260917.md)
rechecked stopped compute, private Cosmos access, preserved workflow state and
all 18 source-file hashes, and packaged the source transfer capsule. Following
explicit approval, the pinned candidate was installed/reloaded, existing Azure
sign-in and saved workflow reconnection passed, and the new Gremlin type field
was confirmed without saving changes. That checkpoint made no Azure mutation or live migration
qualification; all 66 saved workflow/report JSON files remain unchanged.

The [Gremlin target/profile follow-up](gremlin-target-preflight-20260917.md)
adds stored composite-ID and physical-endpoint verification. Tiny tests pass
on real AGE 1.7/PostgreSQL 18.1, and the GUI controller rejects profile mixing.
The signed-in Azure Gremlin migration and full target P1 digest are still open;
no new route is qualified by these local verifier tests.

September 17 follow-up: [Gremlin fixture and offline full-P1 verification](gremlin-offline-p1-20260917.md)
now pass for all 5.6M records/64 ranges, retaining partitions, typed properties
and endpoints. This source-decoder-only result does not add a live qualified
route. B05 still needs target composite-identity verification and a fresh
installed-GUI/Azure migration; no existing target or installed candidate changed.

Remaining branch/failure/recovery qualification started on 2026-09-15;
see the [case ledger](remaining-validation.md). Local regression results must
not be read as additional live GUI/Azure passes. In particular, explicit remote
same-job resume requires live qualification before the guided recovery trial passes.
The first recovery implementation increment adds read-only guest/GUI checkpoint
inspection with 204 passing extension unit tests. It does not enable resume or
change the installed qualifying Linux artifact; see the ledger for exact scope.
The subsequent increment implements explicit continuations and the native resume
action with 206 passing extension tests and race-tested guest controls. These
local results are not a live recovery pass; no retained base target was resumed.

CSV recovery r2 is now separately **live-qualified**: actual SIGTERM and
loader-VM reboot, two explicit installed-GUI same-job continuations, complete
counts and all 64 canonical ranges PASS. The frozen root was independently
recomputed locally. [Evidence](evidence/csv-recovery-r2-p1-pass-20260916.json).
Network-source recovery is now also qualified below; the other remaining branch
cases are still open. This is not overall release qualification.

Neo4j network recovery r2 has now completed its actual fault, connectivity
restoration, explicit same-job resume and installed-GUI counts verification
(5.6M records, zero rejects). The installed GUI now also displays full canonical
PASS: all 64 ranges match, with both roots independently recomputed locally.
B11 is complete for the defined CSV process/reboot and Neo4j network faults;
other fault timings and sources are not implied. See the
[network recovery record](network-recovery-20260916.md).

The subsequent signed-in VS Code 1.138.0 [GUI branch audit](branch-gui-audit-20260916.md)
confirmed source/location choices and three CSV picker-cancellation paths,
preserving all 64 saved workflow/report files. Seven added regressions bring
the unit suite to 227/227 PASS; nine real CLI configuration contracts also
pass. The subsequent [CSV transfer trial](csv-interruption-20260916.md) passed
real Cancel, changed-manifest refusal and explicit retry, with full 1.84 GB Blob
readback matching every reviewed hash. Earlier blobs/artifacts are unchanged.
The September 17 [Linux CSV negative trial](csv-guest-negative-20260917.md)
now also passes same-size corruption rejection, no final-file/seal publication,
verified positive control, mapping receipt gating and no implicit retry. Its
isolated VM is deallocated; evidence and all previous qualification files are
preserved. The subsequent [installed-GUI lost-acknowledgement trial](csv-lost-ack-20260917.md)
also passed: a real Azure commit followed by injected response loss, explicit
HEAD-only reconciliation, unchanged ETag and full readback. The temporary test
extension was removed by restoring/reloading the normal candidate (including
the corrected warning), with the saved result visibly preserved. B08's defined
cases now pass; neither transfer/import trial is a migrated P1 graph or overall
release qualification. All 9 retained VMs and 17 Flexible Servers remain stopped.

The next [remaining-branch batch](remaining-next-batch-20260917.md) has started:
20 additional P1 report-controller regressions cover invalid digest imports,
transport failures and corrupted retained-report reopening. B12 remains partial
pending signed-in GUI interaction; this batch does not start cloud compute or
replace the installed candidate.

A subsequent [placement and resize audit](placement-resize-audit-20260917.md)
confirmed same-RG/Japan-East/zone-1 coverage, live identity preservation on three
stopped runners, installed-GUI completed-migration resize refusal and actual
approval cancellation. Eighteen added regressions bring the unit suite to
273/273 PASS. Alternative placement and live active-job/layout rejection are
still open; no infrastructure was started or changed.

A [Cosmos access and approval audit](cosmos-access-approval-audit-20260917.md)
then reproduced and corrected missing workspace-trust, reviewed-grant binding
and VM-principal revalidation guards. The 292-test suite and normal package
build pass. Existing Cosmos scope/security were independently read-only checked;
no roles or cloud resources changed. Following the user's continuation, the
[corrected candidate was installed and reloaded](cosmos-guard-gui-20260917.md).
The actual signed-in GUI displays the corrected Cosmos distinction, refuses
replay of a retained P1 failure without creating PASS, and cancels four more
native execution confirmations without changing any of 66 saved JSON files.
All nine VMs and 17 Flexible Servers remain stopped. B06/B09/B12 still have
unrepresented cases; terminal-state reload is not active-operation B10 coverage.

| Route | Final evidence | Outcome |
| --- | --- | --- |
| CSV-MAC | [Local CSV](evidence/csv-mac-qualified-20260906.json) | PASS |
| AZ-N44 | [Azure Neo4j 4.4](evidence/az-n44-qualified-20260906.json) | PASS |
| AZ-N526 | [Azure Neo4j 5.26](evidence/az-n526-qualified-20260912.json) | PASS |
| AZ-PGVM | [Azure PostgreSQL on VM](evidence/az-pgvm-r3-p1-pass-20260914.json) | PASS |
| OP-PG | [IP-only PostgreSQL](evidence/op-pg-r1-p1-pass-20260914.json) | PASS |
| AZ-PGFS | [Azure Flexible Server source](evidence/az-pgfs-r1-p1-pass-20260914.json) | PASS |
| AZ-COSMOS | [Azure Cosmos DB, corrected r2](az-cosmos-r2-execution-20260915.md) | PASS |
| OP-N44 | [IP-only Neo4j 4.4](evidence/op-n44-r1-p1-pass-20260915.json) | PASS |
| OP-N526 | [IP-only Neo4j 5.26](evidence/op-n526-r1-p1-pass-20260915.json) | PASS |

The next [Gremlin prerequisite batch](gremlin-types-preflight-20260917.md)
reproduced a missing numeric-schema declaration path and corrected the core
adapter plus guided form. Full Go tests, relevant race tests, 295 extension
unit tests and ten real CLI configuration contracts pass. The correction is
local source/build evidence, not an installed GUI or Azure P1 Gremlin pass.
Preparing its partition-preserving canonical fixture/oracle remains necessary.
No saved workflow, accepted target or cloud running state changed.

### OP-N526 — full installed-GUI qualification PASS

GUI displays **P1 full canonical digest: PASS** for job
`848306ac-628e-43ac-8af9-31dfdee2a804`. The result generated at
`2026-09-15T06:49:51.855460904Z` matches all 5.6M records / 64 ranges,
including typed properties, identities and endpoints. The 23,218-byte report
SHA-256 is `3d46ccd84d95252f8c4d6abf647ee9494309a5b57b9def3873a2a97c02d45875`.
Independent local recomputation of both sets of leaves agrees with frozen root
`bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.
No loader, graph, credentials or network settings were changed by verification.
Source and runner post-checks show 6% disk and no swap/OOM; target observed
storage maximum is 13.7756%. By `06:55:54Z`, all 17 trial VMs are deallocated
and all 13 Flexible Servers are Stopped. Data and all prior failure evidence
are retained. Storage/Cosmos charges continue; Flexible Server automatically
restarts after seven days if left stopped. No new budget/window was opened.

#### Qualification sequence (historical)

The user authorized completing the final route. A cold clone of the stopped
qualified Neo4j 5.26 source is running privately, with original source/data/auth
unchanged. Exact image/version, disk 6%, zero swap/OOM, and literal-IP TLS
checks passed. The installed GUI draft `31ce4789-9534-4bd6-bcac-621f460d99cc`
uses on-premises mode, host `10.246.5.5`, port 7687, stable `source_key` IDs,
and the hash-bound public CA. No source ARM discovery was used.
The user confirmed transfer-account/scoped-user-role creation; GUI and ARM agree
on success. The approved trial-account-only security exception restored
authenticated HTTPS while anonymous/shared-key access remain disabled. GUI
upload of the unchanged pinned development archive is ready and its hash is
independently verified. The fresh B2s_v2 / Japan East zone 1 VM preview passes at
USD 0.109/hour plus other charges. Following explicit approval, the installed GUI
created the private B2s_v2 runner; ARM and GUI agree on provisioning success.
Container-only Blob Reader for its new identity is independently verified.
Guest installation completed. Installed-GUI readiness passed at `05:06:16.311Z`:
matching pinned build/capabilities, idle, 3.5082% disk, swap/OOM zero. Fresh source
TLS/IP validation also passes. After private credential entry and automatic
readiness refresh, inventory `0598b421-0ed1-4e77-8f16-8f2f15990614` was submitted
at `05:14:55.114Z` and failed with Neo4j `Security.Unauthorized`. The 89-byte
private guest stderr is retained and hash-sealed; no secret was exported or
changed during that failed attempt. On explicit user authorization, the clone's
password was reset at `05:31:08Z`, following a system-database backup and
unpublished-loopback recovery. New authentication returns exactly 1.6M nodes
and 4M edges. The password is retained in macOS Keychain service
`agefreighter-op-n526-neo4j`, account `neo4j`; original AZ-N526 is unchanged.
The source is healthy, disk 6%, swap/OOM zero. Fresh GUI inventory
`3f815040-a8ae-4756-9f48-6134ff1c661b` passed at `05:37:12Z`, exact 1.6M
vertices / 4M edges, no errors or incomplete checks. GUI transfer and independent
local SHA-256 validation agree on the 663-byte report. Private target planning
passed after fresh readiness. The user approved private target creation and
SecretStorage retention. After another readiness refresh, GUI-saved plan/YAML
and ARM agree: deployment `afpg-31ce478995344bd6bcac` succeeded at `06:00:24Z`.
Private PG18/AGE D4ds_v5, 128 GiB, Japan East zone 1; same runner later resized
to D4s_v5, target/runner USD 0.736/hour. No deployment replay or migration
occurred. AGE preload is applied with no pending restart. GUI-approved runner
resize reached `ready-to-start`; ARM confirmed Standard_D4s_v5/deallocated with
the disk/NIC/identity preserved. After operator foreground assistance, installed
GUI start and read-only reconciliation completed both resize and AGE preload.
Fresh new-boot readiness passed at `06:27:02Z`: idle, disk 3.5125%, swap/OOM zero,
and the same pinned loader. GUI submitted new job
`848306ac-628e-43ac-8af9-31dfdee2a804` at `06:29:11.668Z` for all 5.6M rows.
The existing source credential was supplied from Keychain to the protected
prompt without plaintext output. Guest processing completed at `06:37:42Z`
(8m21s), and GUI report import confirms complete counts PASS: 1.6M vertices,
4M edges, zero rejects, all 24 checks pass, no errors or incomplete checks.
Independent local hashing agrees with the sealed 9,619-byte report.
Source, runner and private target are running. Full P1 canonical verification
remains pending; this is not yet qualification PASS.
Budget USD 800,
reserve USD 400 and deadline `2026-09-16T07:14:35.311Z` are unchanged.
[Reviewed execution and handoff](op-n526-execution-20260915.md).

### OP-N44 — full installed-GUI qualification PASS

The installed GUI displays **P1 full canonical digest: PASS** for job
`6ad5c2d0-9d25-4ceb-a382-9516af0c22cc`. The result generated at
`2026-09-15T03:39:56.226453231Z` agrees for 1.6M vertices / 4M edges / all
64 ranges, including typed properties, identities and endpoints. Independent
local validation agrees with the guest/GUI receipt: 23,217 bytes, SHA-256
`c1ad1305e8cfcf5c8e2fa1f0a0ca9df37c8d1cd3baf1984d6918df1b54ed491c`.
Recomputing both roots from all leaves gives the frozen canonical root
`bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.
No acceptance criteria or committed graph were changed. By `03:47Z`, all 15
trial VMs are deallocated and all 12 Flexible Servers are Stopped. All resources
and previous failed-run evidence are retained. Storage/Cosmos charges continue;
Flexible Server automatically restarts after seven days if left stopped.
[Redacted qualification evidence](evidence/op-n44-r1-p1-pass-20260915.json).

#### Qualification sequence

A cold copy of the stopped qualified 4.4 source now has its own private VM and
IP-SAN certificate. The original source/disk/certificate are unchanged. Live TLS
chain and literal-IP checks pass; disk is 6%, swap/OOM zero. This is preparation,
not authenticated inventory or qualification. Installed GUI workflow
`75e4e508-4f38-467b-b3c1-07ed05607603` uses on-premises mode without source ARM
discovery. The user approved dedicated transfer storage and the account-scoped
user data role; GUI provisioning and independent role verification succeeded.
The selected public CA hash and IP-only form are saved. Public networking was
Disabled despite Enabled in the submitted template; its changing actor is not
yet established. Following explicit approval, the trial-account-only exception
is applied: HTTPS is Enabled with anonymous/shared keys still disabled and TLS
1.2 required. Anonymous access was independently rejected. The GUI selected the
unchanged fixed Linux build; after explicit approval its authenticated 37,079,079-byte
upload completed and the durable artifact phase is ready. Fresh GUI VM preview
passes for B2s_v2 / Japan East zone 1 at USD 0.109/hour plus other charges.
The user approved runner creation and continuing routine actions for this route.
GUI deployment completed; the container-only Blob Reader grant was independently
verified. First guest readiness ran before installation finished (retained failure);
after cloud-init completed, fresh GUI readiness passed at `01:45:59.825Z` with
the pinned version/hash, idle state, 3.508% disk and zero swap/OOM. The separate
source clone is running; its IP-SAN TLS handshake now passes after startup.
The user entered the source password. Fresh Linux readiness passed, then complete
inventory `4864b359-4e9b-477e-9c4e-163da4462f34` failed immediately at
`2026-09-15T01:54:23.971296230Z`. Read-only guest diagnostics identify Neo4j
`Security.Unauthorized`, not a completed inventory. The private stderr is retained
(89 bytes, SHA-256 `05a83f01f9f6a512030e8b3f8daac11c4a7ff44974bf62f1fec9df7d8fdadbe7`);
no password or raw log was exported. The failed attempt remains retained.
The user retrieved the existing credential from the Mac Keychain and entered it
privately. Fresh inventory `0f3d9bcf-355c-41f1-a16f-68f4cf88ad4f` passed at
`2026-09-15T02:09:00.145645164Z`: exact transactional count-store totals of
1.6M vertices and 4M edges; no errors, warnings or incomplete checks. GUI report
import and independent local hashing agree (663 bytes, SHA-256
`6373da2080e0ce7aea06404c4b6831d58e57308b27c6a8ee72409b0dd818753d`).
The GUI reviewed and saved the private PG18/AGE D4ds_v5 / 128-GiB target and
later same-VM D4s_v5 resize, Japan East zone 1, with the original deadline and
USD 800 ceiling / USD 400 reserve. Target deployment, AGE preload restart and
same-VM resize are now complete. Post-resize readiness at `02:29:49.488Z` passes
with the pinned build, idle state, 3.513% disk and zero swap/OOM. After private
credential entry and another fresh healthy readiness check, the installed GUI
submitted job `6ad5c2d0-9d25-4ceb-a382-9516af0c22cc` at `02:34:53.311Z`.
Guest execution began at `02:35:24.720350577Z` and finished at
`02:44:54.736459959Z` with exit 0. The installed GUI displays migration finished
and counts pass. Independent report validation agrees: all 24 checks / 18 labels,
1.6M vertices + 4M edges, zero rejects, errors, warnings or incomplete checks.
The 9,619-byte report SHA-256 is
`f80bac38b0af32a553a2393b434615f5bc526bf589c52ef7a7b2071700747785`.
At `02:41:14Z`, loader RSS was 35,404 KiB, disk 4% rounded, swap/OOM zero.
Fresh GUI guest readiness passed at `02:53:39.521Z`; target storage is at most
13.803% through `02:53Z`. No replay or source credential change occurred.
After explicit execution approval and refreshed idle health at `03:35:42.043Z`
(disk 3.5325%, zero swap/OOM), the installed GUI submitted frozen verifier
operation `7200c95d-3450-4c93-a7a3-6db6b03f766c` at `03:37:23.152Z`.
Guest execution ran from `03:37:46Z` to `03:39:56Z`, succeeded, then the GUI
transferred and verified its report above. The unchanged USD 800 ceiling and
`2026-09-16T07:14:35.311Z` deadline remain in force. OP-N526 was not started.
All previous data/evidence are retained.
[Reviewed preparation and handoff](op-neo4j-preparation-20260915.md).

### AZ-COSMOS r2 — full installed-GUI qualification PASS

The installed GUI displays **P1 full canonical digest: PASS** for fresh job
`d5edef98-bb51-4040-b6ed-0274e252de26`. The final report generated at
`2026-09-15T01:01:19.174440026Z` agrees for all 1.6M vertices / 4M edges /
64 ranges, including typed properties, identities and endpoints. GUI transfer
and independent local validation agree on the 23,220-byte report SHA-256
`a5d14d4b2f18673ac2fe12d59d46e195f14f174ff4f435ef8deba427dda9efe9`.
Recomputing both roots from all leaves gives the unchanged expected root
`bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.
No old graph was patched and no acceptance criterion was weakened.
By `2026-09-15T01:08Z`, all 13 trial VMs are deallocated and all 11 Flexible
Servers Stopped. Old and new evidence/data are retained, not deleted.
Cosmos/storage charges remain; Flexible Server has a seven-day automatic restart.

#### Qualification sequence

Mac unlocked and installed extension reloaded. Fresh workflow
`d138f4e4-bcf3-40fe-a876-ee9ce062e08a` discovers the retained Cosmos source
through Azure subscription/RG selection. All 18 mappings entered in the GUI
are independently identical to the typed P1 fixture. Following explicit user
approval, the GUI created the dedicated transfer
account and account-scoped user role. Policy initially disabled networking;
the user then explicitly approved the trial-account-only exception. Authenticated
HTTPS is now enabled while anonymous access/shared keys remain disabled. GUI
upload of the reviewed fixed Linux archive succeeded. Fresh private B2s_v2 VM
preview passed and the user approved deployment. The new private VM is running;
its pinned fixed binary/capability and Linux readiness pass: idle, disk 3.48%,
swap/OOM zero. Its container-only Blob Reader grant is independently confirmed.
The user approved Cosmos Data Reader on the test source for this new identity;
GUI and independent ARM checks confirm the exact grant. Complete typed inventory
`5aabd41f-38bb-4b24-8623-27ae92a0ccc0` started at `2026-09-14T23:48:31.379Z`
and passed at `2026-09-14T23:57:35.722167732Z`: all 18 mappings / 5.6M records,
zero errors and incomplete checks. GUI report import and independent SHA agree
(`8327409449b181afca0b205a7760bf0f766b4b197ba08dd8c88fe57b8f7bf17e`).
After refreshing stale guest health, the GUI reviewed/saved a private PG18/AGE
D4ds_v5 / 128-GiB target and later same-VM D4s_v5 resize, Japan East zone 1.
Target provisioning, AGE restart and same-VM D4s_v5 resize are complete.
Post-resize health passes with matching pinned binary, idle state, 3.482% disk,
zero swap/OOM. Fresh typed job `d5edef98-bb51-4040-b6ed-0274e252de26`
was submitted once through the installed GUI at `2026-09-15T00:22:21.554Z`.
The `00:36:07.826405692Z` counts report passes all 24 checks / 18 labels,
1.6M vertices + 4M edges, zero rejects/errors/incomplete checks. GUI import and
independent SHA agree:
`f1824243ff13d1cc2f44493c47151a68d30be5d92b353dc87347b5326624c938`.
After explicit user approval and fresh healthy guest checks, the installed GUI
submitted the independent typed 64-range comparison at `2026-09-15T00:58:29.341Z`,
operation `cde9e6fe-ab51-45e6-8427-c978a748321e`. It read only the existing
qualified-counts job and subsequently passed the full canonical comparison above.
The target/runner compute estimate while running was USD 0.736/hour plus
other charges; both are now stopped/deallocated.
Budget and deadline are unchanged.
[Current r2 execution handoff](az-cosmos-r2-execution-20260915.md).

### Earlier AZ-COSMOS r2 preparation — explicit numeric types

Optional Cosmos `propertyTypes` now preserve declared floats, including integral
JSON numbers. The GUI accepts `score=score:float64` and
`distance_km=distance_km:float64`; old Linux runners without the new capability
are rejected before typed assessment/migration. Types bind resume fingerprints;
the unchanged r1 graph and failure evidence must not be replayed or patched.
The frozen 5.6M-record / 64-range offline parity test matches the original
canonical root; it is **not** an Azure qualification. A fresh workflow/job is
required. The Mac is locked, so GUI requalification awaits manual unlock.
All 12 trial VMs and 10 Flexible Servers were independently confirmed stopped
during preparation; no Azure resources were mutated by this fix.
[Implementation review and r2 handoff](cosmos-property-types-review-20260914.md).

### AZ-COSMOS r1 — ordering fixed; numeric type mismatch identified (not qualified)

The installed corrected verifier (`252f14f`) read the same committed job in
operation `c1607b3e-32e6-42c6-a3b6-91be8d95e70f` at `13:38:05.565Z`.
All 5.6M records / 64 ranges were traversed, with identical range bounds and
counts; 63 range hashes differ. The `13:40:40.575490334Z` report is **FAIL**,
not qualification PASS. The original failed operation and diagnostic are retained.
The original inactive marker is retained inside its original evidence directory.
No migration, source, loader, graph or credential was changed.

An independent, offline full-fixture counterfactual changed only integral-valued
floats in `score` and `distance_km` to integers: 40,175 values. Its root exactly
matches the actual target root `33196eb1524a2310b74f5313a6fa64e96ad7704118eafefa895a33f533ae6cb1`.
This isolates the numeric type loss; **it is not an alternative acceptance root**.
The frozen expected root and strict typed verification remain unchanged.
Next: add reviewed explicit Cosmos property-type preservation, retain legacy
inference for undeclared properties, include declarations in fingerprints, and
qualify a separately approved fresh job/graph. Never patch this committed graph
or resume it with a changed mapping. Overall coverage remains **6/9**.
[Requalification and diagnostic evidence](evidence/az-cosmos-r1-ordering-requalification-20260914.json).

At `2026-09-14T13:55:07Z`, all 12 trial VMs are deallocated and all 10 Flexible
Servers are Stopped. No data or resources were deleted. The same USD 800 ceiling,
USD 400 reserve and September 16 deadline apply. Cosmos/storage charges continue;
Flexible Server's seven-day automatic restart remains relevant.

#### Earlier AZ-COSMOS r1 evidence (historical state)

Workflow `7b79f05d-1dc1-40a6-b3dc-6c8129d4e0c1` selects the retained Cosmos
P1 account through the installed GUI's subscription / resource-group Discover
flow. Private access, disabled key authentication and 4,000 RU/s autoscale
maximum were confirmed. No source fixture was rewritten. The user approved
transfer storage; it and the pinned Linux archive are ready. The GUI deployed
the private B2s_v2 runner once, and readiness passed at `12:00:25.914Z`: idle,
3.4785% disk, zero swap/OOM. All 18 explicit mappings were entered and reviewed
in the GUI, then independently matched against the P1 mapping fixture.
The user approved Data Reader on the dedicated Cosmos source only; the GUI
reconciled the grant and started complete inventory
`faa15c58-8f07-4f45-b006-98468ca1b1d4`. Fresh readiness at `12:06:16.851Z`
passed. Inventory completed in about 8m49s: all 18 labels reached EOF,
1.6M vertices / 4M edges, no errors or incomplete checks. The GUI imported
its independently hash-verified report. The private PostgreSQL 18 / AGE target
was submitted once (D4ds_v5, 128 GiB, Japan East zone 1); deployment and
AGE restart completed. The same runner is now D4s_v5, with disk/NIC/identity
preserved. Post-boot health passed (3.5115% disk, no swap/OOM). The GUI started
job `7fa558e4-8027-4335-9a2b-564f70b3df02` at `12:37:48.928Z` using the
read-only Cosmos managed identity. Migration and all 24 complete-count checks
passed in about 10m13s with zero rejects; the GUI imported the hash-verified
report. Full P1 verifier operation `cb0c7805-5d5e-4ccd-bff8-1d39b6015b0f`
failed without producing a comparison report. **This route is not qualified**.
The user approved a separately identified read-only diagnosis, run through the
updated installed extension. Operation `815f2755-4a47-46cb-af1a-3c32e5dbd04f`
returned `target-digest / source-key-order`; the 81-byte receipt hash was verified
independently. The verifier's graph-ID ordering assumption caused the stop.
No data corruption is established, but full integrity is not yet proven either.
Original failure/marker and committed graph remain untouched. All 192 extension
unit tests pass; code is pushed as `9c11095`. Next correct P1 canonical traversal
ordering and requalify the same graph; do not replay migration or weaken checks.
After diagnosis, route compute stop requests were issued again.
At `2026-09-14T13:24:11Z`, all twelve VMs are deallocated and all ten Flexible
Servers are Stopped. Cosmos/storage charges continue; no evidence was deleted.
At `13:04:29Z`, all twelve VMs are deallocated, nine Flexible Servers are
Stopped and this route's server is Stopping. Cosmos/storage charges continue.
Reviewed
target/runner compute is USD 0.736/hour plus the USD 400 additional reserve;
previously stopped compute was not restarted.
Budget and September 16 deadline are unchanged; cost refresh returned 429 and
was not retried. [Current handoff](az-cosmos-r1-execution-20260914.md).

### AZ-PGFS r1 — full installed-GUI qualification PASS

The installed GUI displays **P1 full canonical digest: PASS**. All 1,600,000
vertices and 4,000,000 edges / 64 ranges match, including typed properties,
identities and endpoints. Both canonical roots were independently recomputed:
`bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.
The 23,218-byte result generated at `2026-09-14T10:00:50.732774263Z` was
exported, hash-verified and imported through the GUI; SHA-256
`39f4898473a7c639d322ec0ea129556b352012c2e365a15600de06b0968bdf82`.
Migration and complete counts verification took approximately 5 minutes 18 seconds,
with all 24 checks / 18 labels passing and zero rejects. This is P1 qualification,
not production-scale testing. Remaining routes: **AZ-COSMOS, OP-N44, OP-N526**.
[Redacted AZ-PGFS evidence](evidence/az-pgfs-r1-p1-pass-20260914.json).

Final cost-saving state confirmed at `2026-09-14T10:05:39Z`: all eleven trial
VMs deallocated, all nine Flexible Servers Stopped. No data or resources were
deleted. Retained storage charges and the seven-day automatic database restart
remain relevant; the USD 800 ceiling / September 16 deadline are unchanged.

The following paragraphs retain the setup history preceding this result.

The installed GUI discovered the retained private Flexible Server source from
the approved subscription and resource group. Workflow
`29558917-403e-4a76-aaa0-de07122ea9c6` now holds all 18 mappings, independently
confirmed identical to the frozen P1 mapping fixture. The source was started;
the trial runners and other databases were not restarted. At that initial stage,
no migration, assessment or new target had begun. The user approved dedicated transfer
storage and the account-scoped grant; GUI upload of the pinned Linux archive
passed. Live preview exposed a `Japan East` versus `japaneast` comparison bug.
Commit `dd6b401` fixes it while retaining region/zone checks; 189 tests pass,
and the installed extension is updated. The repaired GUI preflight passed and
submitted the private B2s_v2 runner once (USD 0.109/hour compute). GUI guest
readiness passed at `09:22:35.060Z`: pinned artifact/capabilities match, idle,
3.5079% disk, swap/OOM zero. After private password entry, all 18 mappings
reached EOF in one repeatable-read snapshot: 1.6M vertices / 4M edges, no
errors or incomplete checks. The GUI imported the checksummed inventory.
A private PostgreSQL 18 / AGE target is provisioned (D4ds_v5, 128 GiB,
Japan East zone 1, fresh subnet `10.246.13.0/24`); AGE restart is finished.
The same VM was resized to D4s_v5 preserving its disk/NIC/identity. Post-boot
health passes at `09:46:39.122Z`, idle, 3.5103% disk, swap/OOM zero.
The user subsequently entered the private source password. Migration job
`958d33c4-b7a9-449e-9017-04f7081a23a9` and the full verification above passed.
[AZ-PGFS execution record](az-pgfs-r1-execution-20260914.md).

### OP-PG r1 — full installed-GUI qualification PASS

The actual VS Code now displays **P1 full canonical digest: PASS**.
All 1,600,000 vertices and 4,000,000 edges / 64 ranges match, including typed
properties, identities and endpoints. Expected and actual canonical roots are
`bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.
The 23,216-byte report generated at `2026-09-14T08:29:37.022758513Z` was
exported, hash-verified, imported and independently revalidated locally;
SHA-256 `22b727306457f412b6fd589bc16de47a81c209a3a6edf5dd9b2d59ca317046f7`.
This qualifies the IP/port-only on-premises simulation separately from AZ-PGVM.
It is P1 scope, not production-scale qualification or every PostgreSQL schema.

At OP-PG completion, remaining routes were **AZ-PGFS, AZ-COSMOS, OP-N44, OP-N526**.
At OP-PG completion, all ten trial VMs were deallocated and all eight Flexible Servers were Stopped;
no resources/data/evidence were deleted. Retained storage charges continue.
Latest returned trial cost: USD 36.7436452084183 (delayed, not final billing).
[Redacted OP-PG evidence](evidence/op-pg-r1-p1-pass-20260914.json).

The following paragraphs preserve the steps leading to this result.

Private credential entry was completed. The whole-source repeatable-read
inventory passed: 1.6M vertices / 4M edges across all 18 mappings, no errors or
incomplete checks. Its 2,947-byte report was hash-verified and imported through
the GUI (SHA-256 `e217d947f121501c24ae833e593c5c66475a0718461f2d2a5dfc2126b78fafda`).
A fresh private PostgreSQL 18 / AGE target exists: D4ds_v5,
128 GiB, Japan East zone 1, dedicated subnet `10.246.12.0/24`. Its original
deployment failed only on the AGE preload child with `ServerIsBusy`.
The installed GUI repaired that one setting without replaying deployment;
the original failure is retained. The separate target restart finished.
Future target child writes are serialized; 188 tests and typecheck pass.
The same runner was resized from B2s_v2 to D4s_v5, preserving its disk, NIC
and identity. Post-boot readiness passed with the unchanged Linux artifact,
3.512% disk usage and zero swap/OOM. Budget and deadline are
unchanged. New durable job `bef7834e-3c7f-4d7a-8021-2c99c70cef66`
was submitted at `2026-09-14T08:13:10.085Z` after private credential entry.
The report generated at `08:18:31.411226266Z` passes all 24 checks and 18
exact label counts with zero rejects. The installed GUI imported and
hash-verified it. At that stage full canonical verification was pending (4/9);
the independent final comparison above now qualifies this fifth route.
[Execution evidence and next gates](op-pg-r1-execution-20260914.md).

The paragraphs below retain the earlier setup and password-handoff history.

Workflow `53625ae3-b155-4821-bfc3-910cc8cad6df` was created in the installed
GUI with PostgreSQL / **on-premises**, IP and port, the retained read-only
database user and verified custom CA. No source ARM ID or Azure source
candidate was supplied. All 18 GUI mappings exactly match the P1 fixture;
configuration SHA-256 is
`284397bc7549f79520b426ed66bf182dba52e59e1ffb52ca457ae5691f63ada5`.
Source CA SHA-256 remains
`0b196d1e310a5a68732879c7bf0aa8c9f1fc67bbfe436fe560a31de7afa05b68`.

The private runner placement is the existing trial group / runner subnet,
Japan East zone 1, initially B2s_v2. Published 2.4.0 is unavailable, so the
release prerequisite correctly submitted no VM deployment before the reviewed
development artifact was selected. The user explicitly approved the new
account-only Storage Blob Data Contributor grant. Installed-GUI transfer
deployment succeeded at `2026-09-14T06:50:00.598619Z`. Account
`af53625ae3b1554821bfc391` received the previously authorized storage-only
`SecurityControl=Ignore` exception; authenticated public HTTPS is enabled,
anonymous access and shared keys are disabled, and TLS 1.2 remains required.
The exact 37,056,164-byte archive uploaded successfully, SHA-256
`10a27dd02b53f070ad2529b21c40c5d382a6e67cb59d9175f98cc29531ff8997`.
The GUI reconnected to the draft, rechecked placement, and submitted private
runner `af-53625ae3b1554821bfc3` once at USD 0.109/hour compute plus other
charges. Its identity receives only this workflow container's Blob Reader
role. Provisioning is not readiness or qualification.

Deployment succeeded at `2026-09-14T06:54:44.021368Z`. The first guest check
was too early: retained command `af-7082911a-b627-4f0e-b9e3-d9d2229b62cb`
failed with exit 127 because `agefreighter-tools` was not yet installed.
Read-only boot diagnostics subsequently confirmed cloud-init done, no fatal
errors, and the installed tool. Recoverable Azure IMDS reprovision-data 404
warnings were retained; they are not asserted to be a migration failure.
After that evidence review, a new GUI readiness operation
`7f1999cd-56a7-4f94-b45c-30dff7942ade` passed at 06:57:57 UTC. Version,
commit and archive hash match; `postgresql-native-floats-v1` is present.
Boot ID `2163fb28-cd1e-41e5-8f67-964c96395078`; idle=true, disk 3.508%,
swap=0, OOM=0. The GUI imported this verified readiness.

Only the retained PostgreSQL fixture VM was started for laboratory setup.
That infrastructure operation is separate from the OP-PG discovery under
test, which contains no source ARM reference and uses only IP/port/credentials.
No source database reads, migration job or target exist for this route yet.
Source guest checks confirmed its existing TLS chain and IP verification,
certificate expiry September 20 (more than 96 hours remaining), disk 9%,
swap=0 and OOM=0. The intentionally restart-disabled, retained PostgreSQL
container was explicitly started without changing its data or credentials;
its running/non-OOM state was confirmed. No source ARM discovery is used by
the OP-PG workflow.

The GUI re-reviewed the unchanged 18 mappings and accepted the complete
inventory read approval (30 minutes / 4 GiB / no swap). It now displays
**Read-only source password** for private operator entry and Enter. No
inventory intent exists before that entry; no password is recorded here.
Only the new B2s_v2 runner and the D8s_v5 fixture VM are running; old runners
and all targets remain stopped. The next step is inventory and report import,
then fresh target/resize, migration, exact counts and all 64 canonical ranges.

Fresh baseline at 06:36–06:39 UTC confirms all nine VMs deallocated and all
seven Flexible Servers Stopped. Recent external network writes were inspected:
the runner subnet retains its NSG, no route table, and no added inbound allow
rule. No security control was changed. Cost Management returned USD
35.32679129750154 across September 12–14; billing is lagged, not a final total.
The USD 400 accrued/non-compute reserve, USD 800 ceiling and
`2026-09-16T07:14:35.311Z` deadline remain unchanged.

### AZ-PGVM r3 — complete GUI qualification PASS

The new PostgreSQL 18 VM → private Flexible Server / AGE migration used the
released native-float fix in the pinned development runner. Fresh job
`b4e66d41-cfdc-4bf3-bc84-2181a7ff5a37` passed all 24 counts checks and the
independent full 64-range canonical comparison. The real VS Code UI displays
**P1 full canonical digest: PASS**. All 1.6M vertices / 4M edges, typed
properties, identities and endpoints match canonical root
`bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.
The verifier is unchanged from the rejected r2 trial; old failed graphs and
evidence are preserved. [Pass evidence](evidence/az-pgvm-r3-p1-pass-20260914.json).
At the earlier AZ-PGVM completion, remaining routes were OP-PG, AZ-PGFS,
AZ-COSMOS, OP-N44 and OP-N526. OP-PG has since passed its separate IP/port-only
GUI workflow and evidence, as recorded above; no other route is promoted.
After final idle/no-swap/no-OOM checks, all nine trial VMs are deallocated and
all seven Flexible Servers are Stopped (06:08 UTC). Data and evidence remain;
storage charges continue and Flexible Servers can auto-start after seven days.
Earlier sections below are historical phase observations, not current status.

### Development resumed with the released PostgreSQL fix

Released v2.3.1 (`952b6b4`) is merged into this development tree (`43490f6`).
It preserves native SQL float properties across COPY, cursor and keyset modes,
including integral-valued floats and float arrays/domains, and rejects old
PostgreSQL checkpoint fingerprints. The Extension remains 2.4.0 with its
runner-first GUI and Azure Resources authentication integration intact.
New PostgreSQL assessments/migrations now require the explicit
`postgresql-native-floats-v1` Linux capability; old evidence controls still work.
AZ-PGVM r1/r2 and their failed targets are retained without replay or repair.
The next qualification uses a newly pinned fixed development runner and a
fresh workflow, target and job. This repair is not yet a fourth qualified route.
The merged tree passes all Go package tests, PostgreSQL/runner race tests and
180 Extension unit tests. Live local PostgreSQL 18 / AGE tests preserve exact
float serialization in COPY/cursor/keyset and both pre-encoding paths; native
float arrays/domains and legacy-checkpoint refusal pass as well.
The fixed VSIX is installed/reloaded on the Mac. New GUI workflow
`c275d043-de93-4b0a-b2b0-59cddd13c84f` (`az-pgvm-p1-r3`) has all 18
reviewed mappings and the existing validated CA. Its newly pinned Linux
artifact is uploaded; private discovery VM deployment and guest readiness pass.
The guest advertises native-float preservation, with disk 3.48%, no swap/OOM.
Source container health and current TLS validation pass. The installed GUI
completed the full inventory: 18 mappings, 1.6M vertices / 4M edges, with no
errors or incomplete checks. Its 2,947-byte SHA-256-verified report is imported.
Fresh private target deployment succeeded at 05:23 UTC. AGE preload restart
and same-VM resize to Standard_D4s_v5 are complete. Post-boot guest readiness
passes (idle, disk 3.48%, no swap/OOM). New migration preflight and approval
passed initially. After credential entry, a fresh readiness command succeeded
but the following ARM VM-state gate refused admission; no migration job was
created. A bounded GET-only wait for the matching running VM's Updating state
is now installed (183 tests pass), with all final gates retained. Re-admission
and private source-password entry subsequently succeeded. Fresh job
`b4e66d41-cfdc-4bf3-bc84-2181a7ff5a37` started at 05:46 UTC and its complete
counts report passes at 05:51 UTC (5.6M rows, 24 checks, zero rejects/errors).
Full 64-range P1 verification was submitted at 05:58 UTC and remains pending.
No old failed job was replayed.
See [r3 execution sheet](az-pgvm-r3-execution-20260914.md). This source inventory
pass is not a completed migration or P1 property-digest qualification.

### AZ-PGVM corrective attempt — counts PASS; full digest FAILED (numeric-type investigation)

The user approved the pinned full verifier. The first approval outlasted the
five-minute health gate without submitting work; fresh health and the same
approval submitted operation `4ed0d7dc-c319-4981-ac3b-4acfee6a0f92` at
03:45:23Z. It compared all 5.6M records and all 64 ranges, then failed at
03:47:39Z: 63 range hashes differ, while row counts and range boundaries agree.
The installed GUI reconciled the failure without replay. Result/checksums and
the failed graph are retained. PostgreSQL source readback confirms integral
`double precision` values serialize without a decimal marker; the current
connector consequently interprets them as integers. This is a typed-property
preservation defect requiring repair, not permission to weaken the digest.
See [redacted r2 evidence](evidence/az-pgvm-r2-p1-failed-20260914.json).
The offline read-only diagnostic over the frozen P1 fixture reproduced the
exact failed target root by collapsing 40,175 integral-valued `score` and
`distance_km` floats to integers. It covers all 5.6M records / 64 ranges, so
the type conversion explains the complete observed mismatch, not just samples.
This diagnostic PASS is not migration qualification. All eight trial VMs are
now deallocated and all six Flexible Servers are Stopped; data and evidence
remain, with storage charges continuing. The connector fix and a new qualified
migration remain required; no production conversion behavior was changed here.

After manual unlock, the installed GUI created a separate draft
`22f11b89-e943-4d56-9675-7331a78b6de7`, named `az-pgvm-p1-r2`. All 18
corrected mappings were entered through the GUI and the existing custom CA was
selected with TLS validation retained. Readback exactly matches the reviewed
mapping fixture and passes frozen-P1 projection admission. The user approved
dedicated storage and the scoped user role. Their deployment succeeded at
00:43:53Z; the GUI reconciled it as ready, but actual public network access is
Disabled. Policy-modify events occurred during creation. HTTPS-only, TLS 1.2,
anonymous access disabled and shared keys disabled are confirmed. No upload,
source read, new VM, target or migration had been submitted at that checkpoint.
The user subsequently approved the exception tag and authenticated public HTTPS
on this new account only. Effective settings now confirm Enabled networking,
HTTPS-only, TLS 1.2, anonymous access false and shared keys false. Authenticated
listing and the installed GUI's 37,040,125-byte frozen runner upload succeeded.
The new VM preview passed: Japan East / zone 1 / B2s_v2, USD 0.109/hour compute
plus disk/network. The user approved the pinned test build and managed-identity
Blob Reader on this workflow container only. The new private VM provisioned at
01:26:26Z; installed-GUI readiness confirms the frozen build, idle guest,
3.48% storage use and no swap/OOM. The retained source VM was started for a
current health check, which passed: container running, TLS chain/hostname and
96-hour certificate validity confirmed, disk 9%, no swap/OOM. The GUI accepted
the already-covered complete inventory read confirmation and now awaits the
read-only source password in its secure input. No inventory, target or migration
has been started for r2. Previously
covered approvals need not be requested again; new kinds of authority remain
separate decisions.

The first password entry outlasted the five-minute readiness gate and no
assessment was submitted. The extension now refreshes stale idle health after
credential entry without replaying source operations or weakening health/boot
checks. All 177 tests passed; the corrected VSIX is installed and reloaded in
VS Code 1.137.0. The same corrected r2 draft is reopened at secure password
entry. On the next input, automatic readiness refresh succeeded and the GUI
submitted operation `b78461db-10aa-482c-a75f-ae22d551fe0d` at 02:05:29Z with
the corrected projection. It failed during initialization at 02:05:37Z; a
password-authentication rejection is present in the source log for that same
interval. Failed evidence is retained; no replay, credential reset, target
deployment or migration occurred. Correct existing read-only credentials are
required before a fresh attempt.

The saved credential was subsequently confirmed to be a 64-character random
hexadecimal password, not a hash. After the user entered it, fresh operation
`5132c60a-fddb-47d3-89f0-4e189718f6cf` passed at 02:21:06Z with all 1.6M
vertices / 4M edges and 18 exact mappings, no errors or incomplete checks.
The installed GUI imported/displayed the hash-verified 2,944-byte report
(`33c83a3021d9691333a0220f678e3bbf93d92533fc968b38fb5eb5992dfc5c0e`).
The previous failure remains retained. The installed GUI saved a new LoadJob
and submitted the separate private target at 02:31:21Z: PG18/AGE D4ds_v5,
128 GiB, Japan East/zone 1, new `10.246.10.0/24` delegated subnet, no public
access or peering. The same runner is planned for D4s_v5. Combined compute is
USD 0.736/hour plus USD 400 accrued/non-compute reserve under the unchanged
USD 800 / September 16 deadline. Target readiness, resize, migration and full
canonical qualification were pending at that checkpoint; coverage remains 3/9.

The target deployment and AGE preload restart are now finished. Effective
networking remains private and the target is Ready. GUI same-VM resize began
at 02:51:22Z; deallocation completed and the D4s_v5 update was submitted with
disk, NIC, identity and placement retained. Migration has not started. The
credential-wait readiness repair also covers migration entry now, with 178
passing tests and the updated extension installed/reloaded. Post-boot readiness,
migration, strict counts and the full canonical digest remain required.
The same-VM resize subsequently finished with preservation checks passed.
Post-boot readiness at 02:58:03Z is idle, disk 3.50%, no swap/OOM, pinned
runtime unchanged. GUI migration preflight passed and its covered confirmation
was accepted; the secure source password is required before the first dispatch.
No new migration is submitted or qualified at this checkpoint.

After credential entry the GUI submitted job
`f16b1aac-2b1b-41ee-888c-df7f2177575f` at 03:26:57Z. It finished and its
strict counts report passed at 03:32:12Z: all 5.6M records, 18 exact label
counts and 24 checks, zero rejects/errors/incomplete checks. The installed GUI
imported and displayed the hash-verified 9,619-byte report. All previous graphs
remain intact. The independent 64-range canonical-property comparison is still
pending, so AZ-PGVM is not yet fully qualified and coverage remains 3/9.

Read-only reconciliation of the original verifier did not restart it. The
deallocated VM's current ARM instance view reports Pending/exit 0 without the
old result, so the GUI correctly remains submitted; the prior terminal failure
and guest logs are still retained. This is not success or permission to replay.
The other six existing trial VMs and five Flexible Servers remain stopped. Cost
refresh was again throttled (429); the USD 800 ceiling and September 16 deadline
are unchanged. See the [corrective execution sheet](az-pgvm-r2-execution-20260914.md).

### AZ-PGVM — counts PASS; full verification FAILED, projection repair required

The installed GUI selected the PostgreSQL VM through Azure discovery and saved
the TLS-verified source configuration with all 18 P1 mappings. The approved
dedicated storage and pinned development artifact approvals were applied. The
source and discovery runner are running; the other five VMs and all four old
Flexible Servers are stopped/deallocated. CA persistence defects were fixed,
tested and installed before the GUI started complete source inventory. That
attempt failed during initialization: the source container was stopped and its
certificate had expired. Both are repaired without changing data or disabling
TLS checks. The explicit fresh attempt passed all 1.6M vertices, 4M edges and
18 label counts; its hash-verified report is imported. The reviewed private
PG18 / AGE target began deployment at 13:36:58Z, retaining the USD 800 ceiling
and September 16 deadline. The target is now Ready, AGE preload is applied,
and the GUI completed the same-VM resize to D4s_v5 with disk/NIC/identity
preserved. Post-boot readiness passed; the GUI submitted new migration
`12a2462e-e5a3-4368-a356-54e292650051` at 13:59:55Z. See
the [execution sheet](az-pgvm-execution-20260913.md).
Migration and strict counts passed: all 5.6M records, zero rejects, all 24 checks,
no incomplete checks. The GUI imported the verified report. The independent
full P1 verifier was approved and submitted at 2026-09-13T22:54:07Z as operation
`c0efdd7b-fd18-49db-a872-bbd29d1736c2`, on the same runner and retained graph.
It failed with exit 1 at 22:55:33Z without generating a canonical result.
The retained configuration omits `source_key` and the corresponding identity
properties from all 18 projections; identity fields are not automatically graph
properties. Counts do not prove preservation of omitted source properties.
GUI guidance, frozen-P1 projection admission and plain-text failure receipt
handling are fixed, with all 172 tests passing and the updated extension
installed. A corrected mapping fixture is prepared but not applied. The Mac
locked before final read-only GUI failure reconciliation; the local phase still
says submitted, not success. Azure's command outcome is definitively Failed.
All seven trial VMs are now deallocated and all five Flexible Servers Stopped.
Original data/evidence remain retained. A new corrective migration requires a
reviewed projection and fresh job/graph; do not patch or replay the original.
See [failure evidence](evidence/az-pgvm-p1-failed-20260914.json).
Coverage remains 3/9. Earlier paragraphs describe historical execution states.

### AZ-N526 qualification — PASS at 13:16Z

The installed VS Code 1.136.1 GUI completed the Azure Neo4j 5.26.30 branch from
source selection and exact inventory through private target deployment, AGE
preload, same-VM discovery-to-migration resize, load, counts verification and
independent full P1 verification. Job
`313f6dca-680b-4379-9eac-a7539cb95792` committed all 1,600,000 vertices and
4,000,000 edges with zero rejects. Its 9,619-byte migration report has SHA-256
`4816412aece55c3c70170779503c6ba0477b2ac903c6b4ea154e1a68afd678dc`.

The isolated read-only verifier regenerated the frozen P1 fixture and compared
all 5,600,000 typed records in 64 canonical ranges. Properties, identities and
relationship endpoints matched, and expected and actual canonical roots were
`bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.
The 23,215-byte verifier report has SHA-256
`fe3f6e23e19230c8f54de8615bcbe4bf15932bacbc516ed878515c0cb2d90bff`.

The final GUI health check reported the runner idle, 5.18% disk use, zero swap
and zero boot OOM events. Target storage was 12.79%; no failed Azure activity
event was present from migration start through shutdown. Both VMs are
deallocated and the Flexible Server is Stopped. Data and raw evidence remain
retained. See the [execution sheet](az-n526-completion-20260912.md),
[redacted qualification evidence](evidence/az-n526-qualified-20260912.json) and
[credential-recovery evidence](evidence/az-n526-password-recovery-20260912.json).

### Headless source preparation — active

The operator is travelling and cannot use the installed GUI. GUI state and
SecretStorage-dependent AZ-N526 are preserved without credential reset or
substitute execution. GUI-independent source preparation and product work is
tracked in the [headless checkpoint](headless-source-preparation-20260907.md).
PostgreSQL Flexible Server, PostgreSQL-on-VM and Cosmos P1 preparation passed.
The actual AGEFreighter Linux artifact also completed all-record inventories
for AZ-PGVM, AZ-PGFS, AZ-COSMOS and the IP/port-only OP-PG simulation. Each
returned 1.6 million vertices, 4 million edges and all 18 exact label counts.
Cosmos retains only Data Reader and is back at 4,000 RU/s. The source VM is
deallocated and the Flexible Server is stopped. These are source-fixture and
source-read results, not additional guided-path migration qualifications.

### Headless checkpoint — AZ-N526 retained; four network sources read completely

AZ-N526 has completed the private target deployment, AGE preload restart and
same-VM resize from the discovery SKU to `Standard_D4s_v5`. The Neo4j 5.26.30
source and runner remain unchanged and the retained workflow is ready to start
its create-only migration. macOS is locked while the operator is travelling, so
VS Code SecretStorage cannot release the existing target credential to a
headless process. The credential was not reset or copied, no substitute job was
created, and no GUI qualification is claimed. The runner and source are
deallocated and the PostgreSQL target is Stopped; exact resume state and guest
evidence are retained.

While GUI work is unavailable, the PostgreSQL and Cosmos guided paths were
extended to support complete mapped-record inventories on the Linux runner,
per-label and capacity evidence admission, source-specific runner capabilities,
and protected PostgreSQL/Cosmos create-only migration dispatch. PostgreSQL uses
one exported repeatable-read snapshot. Cosmos requires the source-immutability
window because there is no cross-container transactional snapshot. The actual
commit-pinned Linux binary then read and decoded all 5,600,000 records for
AZ-PGVM, AZ-PGFS, AZ-COSMOS and the IP/port-only OP-PG simulation. All four
inventories passed every exact label count. These source-read results do not
qualify target deployment, migration or canonical verification; those four P1
paths remain open until their complete guided runs pass.

### AZ-N44 qualification — PASS at 12:59Z

The installed VS Code 1.136.1 GUI completed the Azure Neo4j 4.4.48 path from
resource selection and complete inventory through private target deployment,
same-VM discovery-to-migration resize, load and independent verification. Job
`45e2d8bb-641b-424d-9074-d55e14b6ac2a` committed all 1,600,000 vertices and
4,000,000 edges with zero rejects. The read-only verifier compared all
5,600,000 typed records in 64 ranges; expected and actual canonical root are
`bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.
See the [execution sheet](az-n44-completion-20260906.md) and
[redacted evidence](evidence/az-n44-qualified-20260906.json). Both VMs are
deallocated and the Flexible Server is Stopped; data and evidence are retained.
This passes AZ-N44 only; seven other remote-source branches remain open.

### CSV-MAC qualification — PASS at 07:47Z

The installed VS Code 1.136.1 GUI completed the full local-CSV path. Job
`99f22ed1-d5b4-4432-b761-063e923b726c` committed all 1,600,000 vertices and
4,000,000 edges in 1,120 batches, with zero failed batches and zero rejects.
The hash-verified counts report passed. An isolated read-only verifier then
regenerated the frozen P1 fixture and compared typed properties, identities and
endpoints in all 64 canonical ranges. All 5,600,000 records matched; expected
and actual root are
`bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.
See the [redacted qualification evidence](evidence/csv-mac-qualified-20260906.json).
This passes CSV-MAC only; it does not qualify the eight remote-source branches.
The runner VM is deallocated and the Flexible Server is Stopped; all resources,
CSV data, graph data, private reports and guest evidence remain retained.

### Earlier checkpoint — 07:20Z

The Mac is unlocked. The actual installed GUI has reconciled the private PG18
target, applied AGE preload, and completed the same-VM resize to D4s_v5.
VM and target are running within the unchanged USD 800 / Sep 9 08:55Z limits.
The first migration failed before graph/metadata creation; a GUI read-only
diagnostic proved the empty target, and the failed job is retained in history.
The repaired guest now reuses the established hosted AGE session initialization.

Pinned guest: `2.4.0-dev.72f45536b052`. Its new full inventory
`bcb9a509-7558-412a-879e-c80e4ae45a89` has been imported and displayed in VS Code:
all 1.6M vertices / 4M edges / 18 mappings, no errors, unchanged input hashes.
Report SHA-256 `a66e2231baf6b785917538ad68a7143deab7ecf9fec1c0bcd7c33b1cc7871af0`.
Next: approve a new create-only migration, import complete counts verification,
then run the isolated full P1 canonical verifier through the development GUI.
Extension tests: 153 pass; relevant Go tests pass. CSV-MAC and the other eight
scenario branches remain unqualified until their actual end-to-end evidence passes.
See the [execution sheet](csv-completion-20260906.md) for retained failures,
diagnostics, artifact pins and archived ARM receipts. Entries below are historical.

## Earlier inventory / target checkpoints

CSV-MAC completion is the active scope. **The actual installed VS Code GUI
completed the Linux whole-source CSV inventory on 2026-09-06**: 1.6 million
vertices + 4 million edges across all 18 mappings, no rejects, and matching
before/after file hashes. The worker ran from 03:43:29Z to 03:43:52Z (22.97 s).
See [live evidence](evidence/csv-inventory-linux-20260906.json) and the
[execution sheet](csv-completion-20260906.md). This is source inventory, **not
a migration pass** or a uniqueness/endpoint/canonical-graph proof.

After the Mac was unlocked, the GUI imported and displayed the exact 3,220-byte
whole-source report. No source scan was replayed and no private workflow file
was manually patched. The current `c880a67` guest and all old evidence are retained.

The actual GUI then reviewed source counts, live prices/capabilities/quotas,
selected a save folder and submitted a private PostgreSQL 18 target. ARM reports
**Succeeded**, server initially **Ready**, D4ds_v5 / 128 GiB / Japan East zone 1, with public
access disabled and a dedicated delegated subnet in the runner VNet. Shared
preload `pg_stat_statements,age` is pending a restart. No migration has started.
Compute quote: USD 0.736/hour for the planned D4s_v5 runner plus target; USD 100
accrued/non-compute reserve under the unchanged USD 800 / Sep 9 08:55Z deadline.
This is a conservative exposure gate, not a finalized bill.

The same-VM resize state machine and fixed Linux CSV prepare/load/counts-verify
sequence are now implemented for qualification, with a retained UUID before
target writes, explicit approvals, no automatic resume/replay, and independent
report-hash validation. They require a new pinned guest artifact and a fresh
complete inventory before use. Local extension tests pass (148); live R5 and
the full P1 property digest remain unqualified. The Mac locked again before
GUI target reconciliation. Latest guest health at 05:23:29Z was idle, disk 6%,
swap 0, kernel OOM 0. Fresh ARM reads confirm VM **deallocated** and Flexible
Server **Stopped**. All data/resources/evidence are retained; disk, storage and
network charges continue. Flexible Server can automatically start after seven
days; the authorized September 9 deadline is earlier and remains binding.
Unlock is required for the next actual GUI steps; no metadata was patched to
claim a completed deployment reconciliation or migration.

## Earlier retained transport and sample checkpoints

Latest checkpoint: **the bounded Linux CSV trial passed its transport and
assessment controls**, not a P1 migration. After user reauthentication, the
installed VS Code 1.136.1 GUI verified all **18 CSV imports / 1,168,576,671 bytes**.
Independent Linux full-byte readback matched every desktop manifest. All 18
typed mappings matched the P1 reference. A GUI-approved Linux sample profile
completed and its **26,514-byte** report was hash-verified, retained privately
and displayed in VS Code. See [Linux trial evidence](linux-runner-20260905.md)
and [full-file readback](evidence/csv-guest-readback-20260906.json).

The profile itself reports **incomplete**, as designed: it observed only a
bounded prefix of 10,000 vertices / 0 edges. Mapping validation and read-only
checks passed; whole-source counts, capacity acceptance and migration success
remain unproven. No target or migration has started. VM deallocation was
requested at 02:18Z and independently confirmed; disk, CSVs, reports and failure
evidence are retained.

Corrected the earlier blocker attribution: old MFA errors did not prove the
cause of every later PUT failure. Fresh diagnostics exposed HTTP 400 with
exactly Azure's maximum 25 Managed Run Command resources. Safe capacity
preflight and bounded HTTP diagnostics were added; 127 unit tests, package and
CI `34004772287` passed. Completed command receipts were archived/pushed and
freshly hash-matched before removing only historical successful ARM command
resources. Latest and failed receipts and all guest data were retained. This
was operator maintenance, not automatic housekeeping or a source replay.

## Authorization / live resources

- Subscription: `MCAPS-Hybrid-REQ-51508-2023-rifujita` (selected account verified).
- Approved ceiling: 800 USD; live window: 96 hours from first creation.
- Conservative first creation: **2026-09-05T08:55:00Z**; deadline:
  **2026-09-09T08:55:00Z**. No old P3 authorization or result is substituted.
- Created dedicated `rg-af-vscode-p1-20260905-a` in Japan East and its tagged
  `vnet-af-vscode-p1` / `runner` compute subnet. Both provisioning states succeeded.
  The first one-VM Linux trial was submitted through the installed GUI at
  **2026-09-05T12:48:08Z**, after the user approved the pinned development build.
  At that initial checkpoint no Flexible Server or Cosmos resources existed;
  the CSV Flexible Server described above was subsequently created. Private run metadata
  retains exact IDs; ARM completion is not a guest or migration pass.
  Initial reserve was **25 USD**, superseded by the **100 USD** target-plan reserve above. B2s_v2 Linux compute is 0.109 USD/hour;
  Standard NAT is 0.045 USD/hour plus 0.045 USD/GB, and its Standard public IP
  is 0.005 USD/hour. At most 92 remaining hours plus a 10 USD disk/data reserve
  fits the initial ceiling. The global 800 USD / 96-hour limits are unchanged.
  A bounded pre-creation setup watcher installs a 16:00 UTC Azure daily shutdown
  on the exact owned VM once it exists, or deallocates it if setup fails.
- Corrected an initial preflight interpretation: `defaultOutboundAccess=false`
  **disables** implicit egress ([Azure documentation](https://learn.microsoft.com/en-us/azure/virtual-network/ip-services/default-outbound-access)).
  Created zone-1 Standard NAT `nat-af-vscode-p1-egress` and its public IP on the
  task's runner subnet; no VM public IP, ingress rule, peering or source firewall
  change was made. No exception tag was applied to either network resource.
  NAT/IP incur charges even with the VM stopped; retain them within the same
  deadline/cost gate until later qualification or separately authorized cleanup.
- Actual installed VS Code 1.136.1 GUI selected CSV/local, enumerated the existing
  Azure Resources account, and selected the authorized subscription and dedicated
  resource group. This is account/placement evidence, not an Azure migration pass.
- After action-time approval, the installed GUI created Standard LRS account
  `af83c6b829acdc4405aa2dfb`, its workflow container and operator Storage Blob
  Data Contributor scoped to that account only. Deployment succeeded at
  **2026-09-05T09:24:00Z**. Anonymous and shared-key access remain disabled.
- **Initial network blocker (resolved for this account):** the enforced `StorageAccount_PublicNetwork_Modify`
  policy in `MCAPSGovDeployPolicies` modified `publicNetworkAccess` to `Disabled`.
  The retained activity observation identifies the modification at 09:23:33Z;
  a read-only authenticated Blob request failed the storage network rules.
  This was not an Azure sign-in failure. The policy definition was subsequently
  read: its explicit exclusion defaults are `SecurityControl=Ignore` on a resource
  or RG. The user identified this as the official development exception and
  explicitly authorized it **on this storage account only**. Assignment parameter
  reads were denied, so effective behavior was verified rather than assumed.
- Merged the approved tag on the owned account, preserving ownership tags, then
  enabled its authenticated public endpoint. The setting remained `Enabled`;
  anonymous access and shared keys remained disabled. The RG has no exception tag.
  An actual `Carrier.csv` upload at 10:00:18Z and authenticated download passed:
  235,855 bytes, SHA-256
  `0a2c6e8ecf1fdfe2540c4058202527e458f66e1d7611d75c03d51144089fe88b`.
  This single-file CLI probe is **not** a GUI/P1 migration pass.
- The installed GUI selected all 18 CSV files (1,168,576,671 bytes). Its first
  upload failed before committing a blob. Inspection found that azureauth's
  `AzureSubscription.credential.getToken` returns a captured ARM token regardless
  of requested scope. File/report/archive transfers now request a Storage-scoped
  session for the same VS Code account, without ARM/shared-key/account fallback.
  Regression tests cover scopes, refresh and missing/foreign sessions. The GUI
  retry uploaded **all 18 files / 1,168,576,671 bytes**, and independent full-byte
  authenticated readback matched every size and SHA-256 at **10:17:36Z**.
  See [retained transfer evidence](csv-transfer-20260905.json). Anonymous read was separately rejected
  (HTTP 409), while the signed-in user's readback succeeded.
- Source servers were not prepared; preparing dedicated sources is authorized.

## Completed local foundation

1. Saved and reviewed the [implementation and qualification plan](plan.md).
2. Added opt-in typed CSV properties for scalar/array integers, floats, booleans
   and strings. Legacy mappings remain strings; type changes invalidate resume.
3. Added `verify --require-complete`, preserving report output while failing
   incomplete verification. Existing default CLI behavior remains compatible.
4. Added a controller-side counts-verification decision module with wrong-job,
   stale evidence, digest mismatch, reject, missing coverage and count mismatch
   tests. **It is not yet connected to an enabled guided migration operation.**
5. Regenerated full P1 on MacStudio: 1,600,000 vertices + 4,000,000 edges,
   64 shards, seed 20260829. All 1,170 fixture files match the earlier fixture root:
   `f74220f6c58f0c1a62f80a567520ffcde43a2499ba48100667ee7b78ff4e2e2f`.
6. Exported 18 headered CSV files, 18 typed Cosmos-ready JSONL files, mapping
   metadata and checksums. All 5,600,000 JSONL documents are now present in the
   dedicated private Cosmos account; the exact remote-count seal is tracked in
   the headless checkpoint above.
7. Read **all 5,600,000 converted CSV records** through AGEFreighter's actual CSV
   connector and compared all 64 canonical ranges with the original fixture.
   Both roots are:
   `bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.
8. Added the [R3 Linux assessment boundary](runner-assessment-protocol.md):
   protected control transport, boot-bound readiness, durable read-only workers,
   no automatic replay/reboot resume, and hash-verified bounded report retrieval.
   Actual CSV profiling runs through the child-process boundary in tests. The
   GUI can request guest readiness independently of deployment state.
9. Added field-based source forms for Neo4j, PostgreSQL, Cosmos explicit/Gremlin
   documents and CSV. They generate configurations without YAML or SQL input;
   CSV includes property types and an explicit null marker. Local drafts can be
   reviewed before a release is available or any VM is created.
10. Wired approved network-source profiles and complete Neo4j, PostgreSQL and
    Cosmos inventories to native secret prompts, protected dispatch and retained
    status checks. Successful operation manifests remain in history. Finished
    workers are not accepted as capacity or migration passes. PostgreSQL/Cosmos
    whole-source inventory and migration dispatch are locally tested but not
    Azure-qualified.
11. Implemented guest/controller bulk-report export/import with a single bounded
    data transfer, independent full hash/size verification, private immutable
    local retention and loss-of-acknowledgement reconciliation. Exact storage
    ownership and non-anonymous/shared-key-disabled policies are checked before
    transfer. The GUI now connects reviewed Standard LRS storage creation,
    account-scoped user Blob Data Contributor, SDK user-delegation capability
    issuance through the existing Azure login, and a script-disabled report viewer.
    The designed endpoint is network-public/authenticated HTTPS, not
    private-endpoint network isolation; this is disclosed before approval. Live
    provisioning subsequently exposed the policy blocker recorded above.
12. Connected streamed CSV review/upload (8 MiB blocks, conditional commit,
    content-addressed destinations) and protected asynchronous guest import.
    Guest full size/hash seals, 80% disk gate, 10-minute download limit and
    non-replaying workers are required before CSV profiling. Limits are 2 GiB per
    file / 10 GiB per workflow. Failed/interrupted guest import repair and bulk
    automatic import orchestration are still open; imports are approved per file.
13. Added explicit user-level opt-in for reviewed commit/hash-pinned Linux test
    artifacts in workflow storage. Production still requires an official release.
    VM identity gets container-scoped Blob Reader only; cloud-init fetches through
    managed identity with no SAS/token in the template. Guest readiness also
    checks the pinned commit. The build helper requires a clean committed tree;
    no mutable-branch guest builds or unapproved release publication are used.
14. Actual installed-GUI preparation exposed and corrected the ARM what-if
    response envelope and existing-resource `Ignore` handling. Only independently
    enumerated unchanged resources may be ignored; unexpected mutations still
    fail review. Restored drafts now restore source/placement fields. Added
    shallow, bounded CSV-folder selection and explicit storage-network status
    with a pre-transfer stop for disabled/perimeter-restricted public access.
15. The Linux trial GUI exposed a real Retail Prices ambiguity: Bsv2 Cloud
    Services shares `serviceName=Virtual Machines` and the ARM SKU with ordinary
    Linux compute. Excluded that distinct product without selecting arbitrarily
    among ambiguous Linux meters. All **117 unit tests**, typecheck, packaging,
    and CI **33966950563** passed on `050ee1f`; installed bundle hash is
    `31d154d7bb2d00b0ee87b6d26f4b1db2078e54a66123c30774cf374f84296b72`.
    The GUI then displayed the unique 0.109 USD/hour price and passed what-if.
    The guest build remains the separately approved `6ff072c0db71` archive,
    SHA-256 `5fee11ec3fde605ef7e18afaa56131862113348de8ddcb8bc37b52aba1c3620b`;
    no release/Marketplace publication was made.

Local retained data (ignored by Git):

- `production-simulation/work/vscode-p1-20260905/manifest.json`
- `production-simulation/work/vscode-p1-portable-20260905/portable-manifest.json`
- `production-simulation/work/vscode-p1-portable-20260905/csv-source.json`
- `production-simulation/work/vscode-p1-portable-20260905/canonical-verification.json`

## Required execution stages still open

Validation of this foundation: `go test ./...` passed; extension typechecking and
all 110 unit tests passed at the transfer/CSV/development-artifact checkpoint. Targeted guest/tools
tests also passed with Go's race detector. All five GUI-generated source formats passed the actual
Go CLI validator; the contract test is now part of extension CI. These commands
do not run the Azure P1 GUI scenarios.
At the preceding checkpoint, nine Linux guest tests passed under Apple Container/Rosetta,
including validation of the generated unit by `systemd-analyze verify`. This
does not exercise a running service manager or an Azure VM agent. The installed
VS Code 1.136.1 executable passed three isolated Extension Host smoke tests,
including opening all four source editors without a release, workspace, CLI or
ARM calls. The actual webview scripts also have source-branch and edit/approval
gating tests. This is not a live connected GUI migration.

The first branch CI run exposed an existing Windows-specific permission test:
POSIX group/other bits cannot verify a Windows ACL. Runner storage now explicitly
sets a current-user-only inheritable Windows ACL and its Windows test checks the
actual file ACL. A failed ACL setup blocks reads/writes; macOS/Linux retain
owner-only modes. CI also uses the extension's actual VSIX version and supported
minimum VS Code 1.105.0 instead of the stale 2.3.0/1.100.0 constants.
The follow-up [CI run 33952182561](https://github.com/rioriost/agefreighter/actions/runs/33952182561)
on commit `75f7885d2d4d31bd778870be3866191d027730c6` passed all six jobs:
90 unit tests on each of Linux, Windows and macOS, the five actual-CLI source
contracts, the Extension Host suite, and VSIX packaging. In particular, the
Windows private-directory and inherited-file ACL checks passed on Windows.

At the bulk-report checkpoint, [CI run 33954366675](https://github.com/rioriost/agefreighter/actions/runs/33954366675)
on commit `67ff793b8e00efd2096e290b1541a37461d81f9d` passed all six jobs.
Each of Linux, Windows and macOS passed 99 extension tests. The Linux job also
ran guest/tools tests with the race detector and validated all five source
configuration formats through the real CLI. Extension Host and packaging passed.
MacStudio's actual VS Code 1.136.1 separately passed all three isolated host
smoke tests after this change, and the matching VSIX was installed. None of
these tests exercised Azure storage provisioning, real SAS/RBAC, or P1 migration.

The following is the historical two-route snapshot, retained for audit.
Use the report's opening summary for current qualification and resource state.

| Stage | Historical status at two-route qualification |
|---|---|
| Dedicated Azure fixture topology / ownership and cost watchdog | RG/VNet/subnet, explicit NAT, transfer storage/RBAC and one private runner VM tested; account-only approved exception; exact-VM 16:00 UTC shutdown enabled; whole-suite cost automation remains open |
| Source preparation: Neo4j 4.4 / 5.26, PG VM / FS, Cosmos | All dedicated source fixtures are retained. PGVM r7, PGFS r3 and Cosmos r3 passed exact preparation checks; all six VMs are deallocated and all four Flexible Servers are stopped |
| P1 local CSV | Prepared; complete local canonical comparison passed; installed-GUI storage upload, Linux import/sealing and independent full-byte readback all passed for 18 files |
| R3 remote source configuration, mapping, assessment, upload | CSV-MAC and Azure Neo4j 4.4 passed in the installed GUI. Commit-pinned headless complete inventories also passed for AZ-PGVM, AZ-PGFS, AZ-COSMOS and OP-PG; this does not promote them to full guided-path qualifications |
| R4 target deployment and same-VM resize | CSV-MAC and AZ-N44 actual Azure paths passed; remaining source branches open |
| R5 durable migration / resume / verification controller | CSV-MAC and AZ-N44 clean migrations, counts and canonical verification passed; recovery and remaining source branches open |
| Installed VS Code 1.136.1 full GUI branches | CSV-MAC and AZ-N44 passed; seven branches not run |
| Nine P1 base paths and additional branch/failure ledger | 2 / 9 complete |

At that point the installed preview had two end-to-end GUI/Azure qualifications
(CSV-MAC and AZ-N44); the other seven were not yet qualified.
The preview VSIX is installed into MacStudio's VS Code 1.136.1. Installation
and bundle identity are rechecked with each packaged update; these do not imply
that the live GUI branches passed.
After the Storage-scoped session fix, typechecking, all 116 unit tests and packaging
passed. The installed and built JavaScript SHA-256 is
`92d50336ae387cb1be19c81fcc60551f948ede95085943bc393c32840622c7a7`.
An already-open extension host needs a window reload to pick up this build.
No Marketplace publication was performed here.

At that transfer checkpoint, [CI run 33956445191](https://github.com/rioriost/agefreighter/actions/runs/33956445191)
passed all six jobs: 110 unit tests on Linux/macOS/Windows, five real-CLI source
contracts, Extension Host, and packaging. The exact Linux development archive is
retained locally with its build manifest; it has not been uploaded or executed.

The subsequent [CI run 33957782167](https://github.com/rioriost/agefreighter/actions/runs/33957782167)
for `4c12b32756a7e405a64f0d186a9dbf7fdbee45b7` (what-if and restored fields)
also completed successfully. This predates the final network guard/folder tests.

Storage-audience fix [CI run 33960028041](https://github.com/rioriost/agefreighter/actions/runs/33960028041)
on `c9d477b5003189082a575ab7afd7a67b1780cdbb` passed all six jobs: Linux/macOS/
Windows unit tests, source contracts, Extension Host and packaging.

## Review notes for the next implementation stage

- Matching release/bootstrap is still mandatory in production. The released 2.4
  artifact remains a production gate. The approved test-only commit/hash-pinned
  artifact and managed-identity bootstrap passed the actual Azure Linux CSV trial.
  Do not install a mutable branch on guests or publish an unapproved release.
- Managed Run Command capacity now fails before submission at 25 resources.
  Evidence-preserving lifecycle management is still manual. Plan bounded command
  retention before long GUI sessions; do not delete active/uncertain commands or
  guest data to recover capacity.
- The live CSV profile sampled 10,000 vertices and no edges. Do not feed its
  lower-bound estimates into automatic VM/target sizing. Whole-source inventory
  or explicitly representative assessment remains an R4 acceptance gate.
  Complete CSV inventory has now passed local P1 testing, and prefix-scaled
  estimates no longer pass the capacity gate even with exact total counts.
- The form requires explicit reviewed mappings. PostgreSQL schema/FK catalog,
  guest execution/sealing, extension-side operation/import and explicit GUI
  recommendation adoption are locally tested; installed-GUI/Linux qualification remains pending
  (see [implementation evidence](postgres-recommendations-20260918.md)). Current table/column/graph identifiers
  are limited to ASCII letters/digits/underscores. Cosmos explicit mappings use
  a top-level label field; only the public Azure NoSQL endpoint is supported.
- TLS validation remains mandatory. A custom Neo4j/PostgreSQL source CA can now
  be selected locally, digest-bound to the reviewed source, rechecked at each
  approved operation, transported only as a protected parameter, validated as
  CA-only PEM on Linux, and removed after use. Live private-IP qualification is
  still required; hostname verification is never disabled.
- Neo4j inventory uses count-store totals; generic profile `exact` is still
  bounded to 1,000,000 rows and must not masquerade as a full inventory.
  PostgreSQL, Cosmos and CSV now have connector-specific, explicitly approved
  complete streams with row/time bounds (and Cosmos RU disclosure). Their local
  implementation does not replace path-specific Azure migration qualification.
- Managed Run Command instance-view output is limited to 4 KB. Use it only for
  bounded control/acknowledgements, not full reports or CSV transfer. Source
  secrets require protected parameters; raw command output is not safe UI data.
  [Azure managed Run Command](https://learn.microsoft.com/ja-jp/azure/virtual-machines/linux/run-command-managed).
- Keep exact-count verification and full canonical property/endpoint equality
  separate. A count pass alone cannot qualify the P1 scenario.
