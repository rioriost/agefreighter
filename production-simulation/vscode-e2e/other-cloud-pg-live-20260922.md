# B03 PostgreSQL other-cloud selection — offline preparation

September23,2026. Status: **post-reset source inventory PASS/imported; private
target provisioned; AGE preload and same-VM resize completed; installed GUI
migration/counts PASS imported. Full canonical verification awaits specific
verifier approval. Both VMs deallocated and target Stopped verified04:47:37UTC;
scoped monitor paused. All data, disks and evidence retained**.
Latest outcome supersedes the historical preparation checkpoints below.
This follows the completed Neo4j other-cloud simulation; it is a separate
PostgreSQL selection and must not reuse the Neo4j credential or qualification.

## Existing fixture and blocking checks

Read-only ARM inspection confirms existing source VM `af-pgvm-source` is
deallocated, Standard_D8s_v5, private IP10.246.1.20 and trial-owned. Existing
Flexible Server source `afpg-p1-source-20260907` is also Stopped; it is not the
selected fixture and must not be started for this trial.

Use the previously accepted OP-PG fixture: endpoint10.246.1.20:5432,
databasep1source, read-only accountagefreighter_reader, verified TLS and frozen
`p1` schema. Laboratory ARM power/health checks remain separate from discovery:
the extension receives only endpoint/database/credentials/CA and mappings.
Prior route workflow53625ae3-b155-4821-bfc3-910cc8cad6df and all its evidence
remain unchanged; a new other-cloud workflow/target is required.

The retained September13 renewed public leaf expired2026-09-20T12:59:55Z.
This is a retained-certificate observation, not a fresh live handshake. The
retained CA is valid until2026-10-06T13:12:43Z, file SHA-256
`0b196d1e310a5a68732879c7bf0aa8c9f1fc67bbfe436fe560a31de7afa05b68`.
Do not weaken TLS or submit inventory against the known expired certificate.
After a new bounded authorization, inspect the exact guest certificate first.
If unchanged, renew only its public leaf with existing CA/key, preserve previous
certificates/evidence, reload the source and verify chain/hostname/IP-SAN/live TLS.
If existing signing material is unavailable or identities differ, stop and report;
do not invent a new CA, reset the reader password or extract private keys.

## Stopped-compute preparation completed

- Opened a genuinely new installed-GUI wizard, preserving accepted workflows.
  Selected PostgreSQL then other-cloud; source Azure subscription/resource
  group/candidate/ARM fields disappear and private-connectivity guidance remains.
  No project-folder request, source access or cloud deployment was triggered.
  The selection is currently unsaved; no workflow identity is claimed.
- Frozen18mapping file SHA-256
  `ac02ab254bb85f929abe033407c4d3358c2addca91c20864c4dce8be4d072e8f`
  matches the reviewed OP-PG properties and endpoints.
- Added an exact-fixture regression: all9vertex/9edge generated mappings preserve
  every property,identity,endpoint and quoted read query; other-cloud/on-premises
  configurations agree and contain only a source DSN environment handle, no ARM
  metadata or embedded endpoint/password.
- Added an actual-Go-validator contract using these18mappings plus the full-P1
  projection admission check. Unit503/503, CLI contracts14/14, typecheck and
  bundle build PASS. These are local tests, not source/migration qualification.
- Runtime production code, installed extension5f93f3c and Linuxd40d6ccc9a4d are
  unchanged. No new development installation is needed for these test-only edits.

## Proposed bounded live sequence — not yet started

1. Obtain a new maximum2hour compute window for this PostgreSQL subcase; the
   completed Neo4j session's14:10UTC deadline is not silently extended or reused.
   Keep cumulativeUSD800, monthlyUSD3750/dailyUSD100 constraints and conservative
   accrued/retention reserve. Refresh billing/governance before cloud mutations.
2. Review a new workflow, dedicated transfer container/storage and private
   B2s_v2 runner in the existing approved trial group/VNet/region. No source
   public access, SSH ingress or peering change. Resolve exact resource names and
   required scoped access/unpublished-artifact approvals before dispatch.
3. Prepare all18mappings and the correct PostgreSQL-on-VM reader credential while
   compute is stopped, with explicit Remember only if the user chooses it.
   PGFS reader credentials are a different source; never infer equivalence.
   Do not start a timed compute session merely to wait for a password.
4. At first compute start set a fixed absolute deadline and scoped monitor.
   Verify source TLS/current fixture and runner disk<80%,memory<=4GiB,swap0/OOM0.
   Run one reviewed full inventory and import its sealed1.6Mvertex/4Medge,
   all18label report. Any failure retains evidence and never automatically retries.
5. Review/save target inputs before deployment; fresh private PostgreSQL18/AGE,
   same-runner resize after assessment, then one explicitly approved GUI migration.
   Preserve prior accepted graphs and jobs. A fresh target/subnet/secret has its
   own exact-scope approval; previous Neo4j target is not repurposed.
6. Import complete counts with zero rejects, then action-time-approved raw-ID
   verifier8a23a5109798 / archiveSHA60ed56a6773e6cbb64f7a0c03bc407f8aea135c7f1a75d7b8494db17cf09f79d.
   Require all64ranges and canonical root
   `bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.
7. Verify exact source/runner deallocated and target Stopped at completion/failure,
   prolonged idle input wait or deadline, preserving resources/disks/evidence.

This is another Azure-hosted endpoint-only simulation, not AWS/GCP certification.
B03 remains partial until this actual PostgreSQL GUI route passes. No new
cloud cost or runtime approval is inferred from local test success.

## September22 13:54UTC — approved continuation, stopped-compute preparation

The user's continuation after the explicit certificate/max2hour question approves
the existing PGVM leaf renewal and a maximum two-hour compute session. The clock
has **not started**: no VM start/create or server start was submitted. The prior
Neo4j deadline is not reused. Exact new resource/access/artifact gates remain.

Installed GUI created workflow `cb2ef280-a891-4edf-b45b-de75e33eb7b8`,
`othercloud-pg-p1-r1`, PostgreSQL/other-cloud, existing trial RG, Japan East,
zone1, B2s_v2 and existing runner subnet. Source is endpoint-only10.246.1.20:5432,
p1source/agefreighter_reader; no source ARM identifier. Selected the retained
public CA, whose file SHA matches the pinned value above. All18 mappings were
entered through the GUI and Review source settings persisted them. Independent
deep equality against the frozen mapping fixture passes. SourceDraft JSON SHA-256:
`9ce861d71b8a638203b7a5b1b602e7dbe0f6652a08f4c687a19684d4b9fc8e29`.

Opened Prepare/reuse source credential for this exact new workflow while compute
is stopped. The private prompt identifies the PGVM endpoint and reader role;
user input is pending. Flexible Server and Neo4j credentials are not substitutes.
No password was retrieved or written to a file; Remember is the user's choice.

Prepared a new **local public leaf only**, signed with the existing CA/key and
retained CSR. Chain, DNS and IP SAN validation pass; public key matches the
September13 retained leaf. Validity September22 13:54:11UTC to September29
13:54:11UTC; public certificate SHA-256
`4b10b4f74d827f22967eab716bac117ec25c19ca56070f6131281e25449746f3`.
The CA/private keys were not changed or exported. No guest certificate update or
live TLS validation has occurred; those require fresh exact guest identity and
health checks after the bounded compute session starts. Prior leaf retained.

Next: prepare the correct source credential, then exact storage/access/artifact
review; set a fixed absolute stop time and safety monitor immediately before the
first compute start. Do not spend the compute window waiting for credentials.

## September22 14:10UTC — credential reuse and artifact transfer ready

User entered the PGVM reader credential. Invoking Prepare/reuse again for this
exact workflow returned without another password or Remember prompt; encrypted
reuse is working. This proves preparation/reuse, not successful authentication
against the stopped source. No secret was inspected or exported.

User explicitly approved dedicated storage and account-scoped Blob Contributor.
Installed GUI submitted deployment `afcb2ef280a8914edfb45bde-transfer` once;
ARM creation succeeded and GUI reconciled ready. Account public-network access
was initially Disabled. User separately approved this account's
SecurityControl=Ignore tag, authenticated HTTPS enablement and exact pinned
archive upload. Merged the tag without replacing ownership tags; fresh ARM
confirmed Enabled with anonymous/shared-key access still false. No source
exposure, VM start or target creation occurred.

Actual installed-GUI development preparation uploaded Linuxd40d6ccc9a4d to
`afcb2ef280a8914edfb45bde` / `af-cb2ef280-a891-4edf-b45b-de75e33eb7b8`.
GUI reports prepared; independent Blob properties confirm37,197,546bytes and
metadataSHA2321022975f85c73068a54fd21a287e2802d5bc33fa21eb9323c142dd7262ff6.
GUI reconnected to this draft and prerequisites/preview passed for exact VM
`af-cb2ef280a8914edfb45b`, B2s_v2, Japan East/zone1, USD0.109/hour compute.
New VM/container-scoped reader and unpublished installation approval is pending.
No compute request has occurred; maximum2hour runtime clock remains unstarted.

Fresh billing returned originalRGUSD423.333012629158 plus B01RGUSD3.69074183488681,
totalUSD427.02375446404481 (delayed billing, not real-time); cumulative800USD,
reserve700USD, monthly3750USD/daily100USD constraints unchanged. RG identity and
recent control-plane activity inspected; stale expiry/budget tags were not
rewritten as new authorization. The old Neo4j safety monitor remains paused.

Prepared `renew-pgvm-fixture-certificate-20260922.sh` locally; bash syntax and
shellcheck pass. It pins old/new leaf and CA hashes, preserves old evidence,
requires a bounded stop epoch, verifies fixture counts/health and matching key,
and changes only the public leaf. It has **not run** on the guest. A live TLS
check is still required after approved startup and certificate installation.

## September22 14:12UTC — compute approval and fixed safety bound

User explicitly approved the exact new runner, workflow-container Blob Reader
and pinned development execution. Existing source start/leaf renewal remains
approved. Fixed hard stop **2026-09-22T16:00:00Z** (September23 01:00JST), begin
deallocating15:55UTC; this is conservatively less than two hours from the first
forthcoming compute request. Exact-scope safety monitor updated and ACTIVE before
startup. Fresh ARM confirms sourceVMaf-pgvm-source deallocated/trial-owned and
runner subnet10.246.1.0/24 without delegation. No target is approved or created.
Only af-pgvm-source and af-cb2ef280a8914edfb45b belong to this stop scope.

Installed GUI submitted exact runner deployment once at14:14:17.980UTC;
existing source startup requested14:14:19UTC. The maximum2hour clock is now
running; fixed16:00UTC stop remains. No inventory or target request was submitted.

## September22 14:21UTC — live TLS and complete inventory submitted

Runner ARM provisioned, installed-GUI readiness14:15:27UTC succeeded:
boot8b207a68-b8fc-4bb2-b4a5-10c5ff5348aa, exactd40d6ccc9a4d pin,
disk3.481685%,idle,swap0/OOM0. Exact runner principal's Blob Data Reader grant
was independently confirmed at this workflow container only.

Source bootb68acdb3-17f0-4b75-ba86-ac46163b9649, disk10%,swap0/OOM0.
Fresh guest old-leaf/CA hashes matched pins; reviewed certificate renewal
succeeded and old/new certificates were retained under
`/var/lib/agefreighter-source/evidence/tls-renewal-20260922`.
Existing PostgreSQL18.1 pinned container was stopped and was started by the
approved renewal. Post-renewal health/evidence checks pass: retained18tables,
5.6Mcounts, no password environment/secret mounts. Actual endpoint TLS1.3
handshake verifies the CA, hostname and IP-SAN. Retained counts alone are not
new inventory proof.

Actual GUI source-read approval submitted inventory
`bb184697-3121-41d1-be23-fe0c8ff78593` at14:19:26.974UTC on the same runner boot,
configSHA`a2d3038228711c8d455c0b06ee0a0aa69eae8ef8e1400cbfed2e127a994f8498`.
GUI entered automatic retained-operation watch without a password re-prompt.
Managed dispatch completion is not inventory success; no replay was performed.
Worker limits30minutes/4GiB/no swap; target still absent. Reconcile and require
the sealed full18label/1.6Mvertex+4Medge result before further target review.

## September22 14:29UTC — inventory failure retained; safe stop verified

The durable guest state proves the child inventory started14:19:32.548274885UTC
and failed14:19:32.653073659UTC, exit1, without report bytes/hash. Installed GUI
reconciled `failed`; the earlier accepted display was a stale dispatch receipt,
not a healthy active worker. Systemd unit success/inactive is **not** child CLI
success: the worker supervisor retained the child failure in state.json.

Retained51-byte stderr SHA-256:
`d1d694a1717436e62073c3f131686fba9f5f11455eb128dd409544982872d94b`.
An in-guest fixed-category scan matched connectivity; it did not match explicit
auth/TLS/permission/validation keywords. This is only coarse classification,
**not** proof of a network fault or a correct/incorrect password. No raw secret,
DSN or SDK error was exposed. Exact cause remains unresolved. Source-local live
TLS success does not prove the runner's authenticated database connection.

Read-only inspection retained state, unit/journal metadata and artifact names.
An initial diagnostic used jq, absent from this runner; repeated only the
read-only inspection with available tools. No worker retry/restart was performed.
Guest evidence includes job.json, state.json, worker.claim and stderr.log; no
report.json or transient secrets.json remained. Latest runner disk4%,swap0/OOM0.

Requested source deallocation after terminal failure, then runner deallocation
after collecting bounded sanitized evidence. ARM independently confirms **both
PowerState/deallocated by14:29UTC**. Scoped heartbeat PAUSED after verification.
All disks/data/certificates/failed evidence retained; no target exists. One final
read-only GUI status command `af-d661e725-c371-49d7-8000-7a051ec5180e` was still
locally submitted at shutdown; it is not an inventory replay and must not justify
restarting the VM. It can be reconciled via ARM without guest startup.

Next is diagnosis of the connection-stage failure, not another inventory attempt
or migration. Do not reset the reader credential or weaken TLS based on this
coarse error alone. The original16:00UTC bound is not extended by early stopping.
B03 PostgreSQL remains unqualified; base9/9 and broader5pass/7partial unchanged.

## September23 01:46UTC — approved exact-reader authentication reconciliation

User approved checking the PGVM reader authentication settings and resetting
its password if necessary. Started only existing`af-pgvm-source`; runner remains
deallocated. Separate conservative stop bound02:10UTC (begin02:08UTC), inside
the preceding02:22UTC limit; no inventory/target/migration is authorized by this
credential-only step. Scoped safety monitor activated before start. Resource
ownership/size/stopped baseline rechecked; existing expired fixture tag noted,
not rewritten. Existing800USD cap and latest delayed444.3963173658658USD remain.

Read-only catalog check excludes password verifier values and raw authentication
logs. Its first transaction aborted on an incorrect pg_authid.rolconfig
reference; corrected to pg_db_role_setting, no role/password change occurred.
Prepared a create-only, deadline-bound reset path for the existing reader only:
new random value retained in a distinctly named Keychain item, protected ARM
transport, psql password encryption, post-change TLS/read-only/18-table access
checks and before/after role-attribute equality. Never use a PGFS/Neo4j item or
administrator password for the reader. No secret values go to chat, argv or Git.

## September22 14:45UTC — offline error identification and diagnostic correction

Without restarting either VM, the fixed CLI message
`inventory: network inventory initialization failed` plus its trailing newline
reproduced both the retained51byte length and exact SHA-256
`d1d694a1717436e62073c3f131686fba9f5f11455eb128dd409544982872d94b`.
Thus the earlier keyword match was the generic word **network**, not independent
evidence of a network fault. The old binary deliberately discarded the connector
initialization cause. Authentication, TLS, connection configuration and snapshot
initialization remain possible; no cause can be recovered from this fixed string.

Implemented a local-only diagnostic correction. PostgreSQL parse/connect/begin/
export failures now retain private typed, fixed stage/category labels. Known
authentication, permission, database, TLS verification, DNS, network and timeout
categories are selected by typed causes/allowlisted SQLSTATE values, never raw
error text. Raw errors, connection strings, server messages and certificate names
are discarded; no original error chain is exposed. Inventory preserves those
safe labels while other connector errors remain opaque. No read query, TLS mode,
credential, source fingerprint or migration semantics changed.

Local verification PASS: full `go test ./...`; PostgreSQL/app race tests;
extension typecheck503unit tests and14actual-CLI contracts. The18mapping P1
contract now reaches an actual Go iterator against a loopback synthetic PG
server, proves its authentication category, and excludes a private server-message
canary from output. Typed TLS/permission/unknown-error redaction cases also pass.
External-DSN integration tests remain opt-in; this is not Azure retry evidence.

Fresh ARM checks at14:44UTC confirm both exact VMs still deallocated. Installed
extension5f93f3c/Linuxd40d6ccc9a4d remain unchanged, no credentials read/reset,
no inventory replay and no target created. The diagnostic correction cannot
retroactively identify the first failure. Prepare a pinned local build; live
installation and a new explicitly reviewed attempt require their exact-scope
approval, inside the original16:00UTC hard bound or a newly authorized window.

Pinned local Linux candidate built from committed/pushed
`646f0d4b361aec55fcc3f44cb689e06fd915d6e1`:
`agefreighter-2.4.0-dev.646f0d4b361a-linux-amd64.tar.gz`,37,215,444bytes,
SHA-256`07ba1744b5daf7e226915c50d46fa5eda704949fc7d175e2b39a150813f41692`.
Independent archive checksum and two expected ELFamd64 member inspection pass;
this build has not run on a Linux guest. Local manifest:
`production-simulation/work/vscode-runner-build.qCO0dA/manifest.json` (ignored).
Pending exact approval covers existing workflow-storage transfer, existing runner
upgrade, startup of only the same source/runner and one new GUI inventory,
max30minutes/4GiB/no automatic retry within16:00UTC. No new target, network,
credential, RBAC or source-data changes. A successful inventory is not migration
qualification. Keep both VMs stopped while awaiting approval.

## September23 01:12UTC — approved new window; stopped-compute credential defect

User approved a new maximum60minute window from the first compute start for the
same source/runner, pinned646f0d4 diagnostic installation and one inventory.
The September22 deadline is expired, not silently extended. This new clock has
**not started**; both VMs remain deallocated. No target or migration is approved.
Fresh delayed billing: originalRGUSD440.138794871022, B01RGUSD4.2575224948438,
totalUSD444.3963173658658. Existing800USD ceiling/reserve700USD and monthly/daily
constraints unchanged. ARM confirms exact runner identity/scoped placement;
storage retains authenticated public HTTPS, anonymous/shared-key disabled and
existing authorized tag. Overnight policy/Defender/EventGrid actions are visible;
these controls were not changed or removed. The previous read-only status command
remains Pending, not an inventory retry; reconcile it after a legitimate start.

User entered the PGVM credential through the native GUI. The subsequent explicit
reuse check prompted again. Code inspection reproduced a separate extension bug:
`savedSourceCredential` unconditionally removed even freshly prepared credentials
whenever the current retained operation was failed/interrupted. Watch callbacks
also unconditionally deleted credentials on repeated failure observation. The
user's input was not evidence of an authentication failure. Cancelled the duplicate
prompt without reading its value; compute stayed stopped.

Corrected failure binding to include the unique sorted current/history failure
IDs. A new failure still invalidates an older credential; explicit post-failure
entry survives repeat observation and archival of that same failure. Watchers
now validate the latest record instead of unconditionally deleting its session.
No plaintext workflow storage, automatic credential retrieval, extended8hour
expiry, changed-connection reuse or automatic source retry was introduced.

Local typecheck/build,508unit tests and14actual-CLI contracts PASS. Regressions
cover assessment/catalog/migration failure invalidation, explicit preparation,
archive/reload/history reordering, repeated failure callbacks, a second failure,
expiry/connection changes and no credential in workflow JSON. Installed GUI
validation remains pending; the old extension is still installed. A separately
pinned updated VSIX must be approved before installation/reload, then credential
preparation can be verified before spending the new compute window. This fix
does not identify the prior PostgreSQL initialization failure or qualify B03.

Extension fix committed/pushed as `7976cec`; packaged local candidate
`production-simulation/work/vscode-credential-build.lKeVSH/agefreighter-2.4.0-7976cec.vsix`
(ignored), VSIXSHA-256
`e5865e69c956df4e4e07ab7595dd52094f78af192c2c85404c32b353fcd660d6`,
bundled extensionSHA-256
`cb5f1023974fda24435686dbf46c84a681c35161532873dd0b82b6c16a2e6fcf`.
Archive member checksum matches the locally tested bundle. Requested specific
unpublished extension installation/reload approval; not installed yet.
Linux diagnostic candidate remains646f0d4/SHA07ba1744b5da unchanged.
No compute start, source read, artifact upload or scope expansion occurred.

## September23 01:23UTC — installed credential fix; bounded diagnostic session started

User approved7976cec VSIX installation. Official VS Code installation succeeded;
installed bundleSHA matches`cb5f1023974fda24435686dbf46c84a681c35161532873dd0b82b6c16a2e6fcf`.
Actual Developer: Reload Window completed. User entered the PGVM reader credential
and chose Remember. Explicit Prepare/reuse on the same retained failed workflow
returned without a password prompt: installed-GUI credential regression PASS.
No credential value was inspected. Existing runner container-scoped Blob Data
Reader assignment was independently confirmed; no grants were changed.

Both exact VMs were freshly confirmed deallocated before starting. At01:22:40UTC
the safety clock was fixed; start requests for only`af-pgvm-source` and
`af-cb2ef280a8914edfb45b` followed. Conservative hard stop is02:22:00UTC
(11:22JST), begin stopping02:20UTC; less than60minutes from first request.
Scoped heartbeat activated before start. Approved next actions remain pinned
646f0d4 Linux upgrade and one inventory/report import, not target or migration.
Old failed operation and pending read-only status command remain retained.

## September23 01:37UTC — diagnostic runner installed; one inventory submitted

Source VM startup did not automatically start the retained PostgreSQL container.
After disk10%, no swap/OOM and pinned container identity checks, started only the
existing container. Public leaf/CA/hostname checks pass; a fresh source-local
TLS1.3 handshake verifies chain and IP-SAN10.246.1.20. No certificate, credential,
grant, network or source-data change. Retained fixture manifests still describe
18tables/5.6Mrows; these are not a new inventory result.

Installed GUI uploaded approved646f0d4 archive to the existing workflow container
and submitted upgrade`16b27aeb-4ce3-4ace-b92c-2eac37c9fcc1` at01:31:16.412UTC.
Reconciliation returned finished with exact approved version/SHA, preserving old
installation. Fresh readiness01:34:09.596UTC: same boot
`0a11bbe3-8d37-4644-8c13-d0dee31a529a`, idle, disk3.79923%, swap0/OOM0.
GUI archived failed`bb184697-3121-41d1-be23-fe0c8ff78593` without deletion/replay.

Reviewed source and submitted one complete inventory through the installed GUI
at01:36:46.590UTC, without another credential prompt:
operation`3f2b3c12-84d9-4921-b6c7-7e11a175b01c`, command
`af-2ba232bb-4c61-47da-aeec-d296ef5c7007`. Reviewed configurationSHA remains
`a2d3038228711c8d455c0b06ee0a0aa69eae8ef8e1400cbfed2e127a994f8498`.
Submission is not success. Await exact retained outcome; 30minute/4GiB worker
bound and02:22UTC session hardstop remain unchanged. No target or migration.

## September23 01:41UTC — authentication rejection identified; resources stopped

Exact guest state independently confirms new inventory started
01:36:53.621890453UTC and failed01:36:53.694531693UTC,exit1,no report.
Sanitized diagnostic is`[postgresql/snapshot-connect/authentication]`.
This category comes from typed PostgreSQL SQLSTATE28P01/28000, not keyword
matching; connection was rejected at authentication, before snapshot creation or
mapped row inventory. It does not prove whether the supplied password, role
configuration or another authentication condition is responsible. Do not blame
user entry or reset credentials without explicit direction. The old coarse
error cannot retrospectively establish the same cause for September22.

Retained stderr96bytes SHA-256
`e66d29627bb360a68db463fda03c479c37f4b547630915734fb6708055c67a46`;
job.json10298bytes,state.json387bytes,worker.claim36bytes. Guest configSHA
`a89ed9cdc6262b11fd044548618685a6e66a0a62fe5d1aa6f862b0b22be006f5`.
No credential or CA file remains in the operation-directory listing. The
supervisor unit's Result=success is not child inventory success: child state is
failed. Unit inactive/MainPID0,runner disk4%,swap0/OOM0. Initial read-only guest
diagnostic could not use absent jq; repeated only the bounded evidence read with
standard tools, without installing utilities or replaying the inventory.

Installed GUI refreshed to phasefailed; latest status command
`af-477c52ca-9143-4756-aba6-7ecc0e206ea2` is finished. September22 failure remains
in assessmentHistory. Requested deallocation of only both exact VMs at01:40UTC;
independent ARM confirms both PowerState/deallocated by01:41UTC. All disks,
source data, former binaries, upgrade receipts and failure evidence retained.
No automatic retry, password change, target/migration or security changes.
Scoped monitor paused after verified stop. Next is explicit credential/role
reconciliation, not another paid retry with unverified credentials.
B03 PostgreSQL remains unqualified; base9/9 and broader5pass/7partial unchanged.

## September23 01:56UTC — role checks PASS; reset blocked before execution

User-approved read-only catalog reconciliation: reader login enabled, no expiry,
SCRAM verifier present (never displayed), default read-only on, no elevated role
flags or inherited memberships. CONNECT/schema USAGE true and18/18tables SELECT
permitted;7HBA rules,no parse errors,one SCRAM host rule. Disk10%,swap0/OOM0.
These checks do not authenticate the previously entered password.

Fresh random value saved only in Keychain service
`agefreighter-af-pgvm-source-agefreighter_reader-20260923`, account
`agefreighter_reader`: **PENDING / NOT APPLIED; do not use for login yet.**
Protected reset command`af-pgvm-reader-reset-20260923` creation failed.
ARM ResourceNotFound proved absence; one reconciled request reused the exact
pending value,without regeneration,and again failed. Activity Log confirms
BadRequest: managed Run command limit25,current25. No guest reset/SQL password
change occurred. No secret value or raw request diagnostic was emitted.

Stopped source on capacity blocker; both source and runner independently
PowerState/deallocated by01:56UTC. Monitor paused. Original02:10UTC reset bound
not extended. All disks/data/failed jobs/25control resources preserved; no
inventory,target,migration,password/grant/network change.

Candidate for separately approved archive/removal is only
`af-pgvm-r2-health-20260914`. Inspected script is the existing read-only fixture
health check,no parameters/protected parameters,180second timeout. Historical
September14 execution log records Succeeded/exit0; current ARM says Pending
with no timestamps/output,not fresh terminal proof. Source is deallocated.
Before any approved control removal preserve current metadata/script and
historical evidence; never delete guest evidence or preparation jobs for space.
No removal performed; pending Keychain value remains unchanged.

## September23 02:18UTC — approved control archived/removed; reset accepted

User specifically approved archive/removal of the old health control and reset
continuation. Archived full non-secret definition/current view plus historical
evidence reference in`evidence/pgvm-health-control-archive-20260923.json`, SHA
`9e0ed0d05f9e38da433389cc4928aed6ac845fa3d45ea83e26a72ab76d1d5d97`,
committed/pushed`dfc9ce3` before deletion. Azure rejected deletion while VM was
deallocated (OperationNotAllowed), so started only the source with scoped monitor
and fixed02:22UTC shutdown,begin02:21UTC. No runner/target start.

Exact control entered Deleting after the approved request; by02:18:06UTC it was
absent and managed control count24. No other old control or guest evidence was
removed. Same pending Keychain value was reused; the protected reset command
was accepted once after capacity recovery. Reset script starts only the existing
restart-disabled PostgreSQL container if required, changes only reader password,
checks identical role attributes and verified-TLS read-only/18-table access.
Dispatch deadline02:19UTC,guest admission deadline02:20UTC,180second command bound;
no automatic retry. Accepted/Pending is not applied or login-verified yet.

## September23 02:21UTC — reset and login verified; source deallocated

Exact reset command executed02:18:36–02:18:41UTC, Succeeded/exit0 and fixed
success marker. The same previously pending Keychain value is now **APPLIED AND
LOGIN VERIFIED**: service`agefreighter-af-pgvm-source-agefreighter_reader-20260923`,
account`agefreighter_reader`, endpoint10.246.1.20:5432/databasep1source. Its metadata
was updated to distinguish verified application from pending creation. No secret
value was printed, copied through chat, read into model context or saved in Git.

`psql` encrypted the password change; login used verify-full with the existing
CA and IP-SAN. Three assertions pass: exact user/database with read-only default,
TLS in pg_stat_ssl, and SELECT access on all18tables. Before/after pg_roles JSON
SHA is identical`bddecc207ed57b59b4e802f24709183cf19850b00ff53a17d3cfed03ec902bcf`.
Reader-check SHA`bd7c10b69530b76942ffe06edc52707c82b3d2a9a0ae91a33ac51c826c0a9e41`.
Sanitized receipt retained in`evidence/pgvm-reader-rotation-20260923.json`; guest
evidence remains at`/var/lib/agefreighter-source/evidence/reader-reset-20260923`.
Only the explicitly approved old health control was deleted; its full definition
is archived. New protected reset control and guest evidence remain retained.

Source deallocation requested immediately after success; independently verified
PowerState/deallocated by02:21:04UTC, before02:22UTC bound. Runner remained
deallocated throughout credential recovery. Scoped monitor paused after states
confirmed. No post-reset inventory or migration has run; source-local login is
not runner-to-source or GUI migration qualification. Next retry must use the
new canonical PGVM item, never the PGFS or Neo4j credentials, within a reviewed
bounded session. Broader qualification ledger remains5pass/7partial.

## September23 02:36UTC — post-reset GUI retry window opened

User entered the applied/login-verified dated PGVM reader credential and chose
Remember, then requested continuation after the proposed maximum60minute window
for the same two VMs, inventory and report import only. Installed-GUI
Prepare / reuse credential returned without another input prompt while both VMs
were still deallocated. No credential value was inspected by the agent.
Exact ownership/power states and recent Activity Log reviewed before start;
recent entries were the prior source deallocation and Resource Health update.
Scoped safety monitor enabled before first start requests02:36:06UTC. Only
source`af-pgvm-source` and runner`af-cb2ef280a8914edfb45b` started; both running
confirmed. Conservative fixed stop03:34UTC, begin03:29UTC, less than60minutes
from first start. No new target, migration, upgrade, grants/network changes or
credential reset. Existing failed attempts remain retained; next is one fresh
GUI inventory after source/runner health checks. Refreshed delayed Cost API:
original RG442.711240190565USD+B01RG4.37001612271476USD=
447.08125631327976USD; cumulative800USD/reserve700 unchanged.

## September23 02:43UTC — fresh post-reset inventory submitted

Source startup correctly left restart-disabled PostgreSQL container stopped.
After exact pinned image/disk10%/swap0/OOM0 checks, started only that existing
container. Existing CA/leaf verification and IP-SAN10.246.1.20 pass; leaf valid
untilSeptember29. Runner installed-GUI readiness succeeded, boot
`08d8a5eb-e188-4606-988c-fcb871407870`, fixed646f0d4b361a, disk3.82055%,idle,
swap0/OOM0. Preserved both prior failed inventories in assessmentHistory.
Reviewed unchanged18 mappings; configurationSHA remains
`a2d3038228711c8d455c0b06ee0a0aa69eae8ef8e1400cbfed2e127a994f8498`.

Installed GUI source-read approval accepted; new inventory
`60ba3809-4c8c-4d2f-bc34-46c5686e62dc` submitted02:43:11.016UTC via control
`af-4c315a62-6c06-4a46-a5f7-bb6045d03856`, no password re-entry. Existing
30minute/4GiB/no-swap worker bounds; accepted is not successful inventory.
Fresh ARM transfer account check: public HTTPS Enabled, anonymous/shared key
false, existing approved policy tag unchanged. No new settings or role changes.

## September23 02:47UTC — interrupted at user's request; both VMs stopped

User reported a Keychain read problem and explicitly interrupted the work.
No further Keychain access, credential diagnostics/reset, source reads, retries
or qualification work were initiated. At02:46:45UTC retained GUI phase was
running for inventory`60ba3809-4c8c-4d2f-bc34-46c5686e62dc`. Earlier independent
guest evidence02:44:12UTC had confirmed actual worker start02:43:25.414189071UTC,
same boot, RSS27196KiB,disk4%,swap0/OOM0. No final report or complete counts were
verified; this is an interrupted/unqualified trial, not an authentication
failure or a successful inventory. The reported Keychain issue was not diagnosed.

Fresh exact VM ownership/power states checked and deallocation requested for
only runner`af-cb2ef280a8914edfb45b` and source`af-pgvm-source` immediately after
the request. Both independently confirmed PowerState/deallocated by02:47UTC.
Scoped heartbeat PAUSED and active GUI assessment tab closed to stop automatic
polling. Source data, OS disks, all operation evidence and prior failure history
preserved. No target created or migration performed. No resource deletion.
Do not restart to read evidence or automatically resume/retry; await explicit
user continuation and a new bounded session. Broader ledger5pass/7partial.

## September23 02:53UTC — explicit continuation after Keychain recovery

User reported recovery after re-login and explicitly requested resumption.
Installed-GUI Prepare / reuse credential returned without another input while
both VMs were stopped. Re-login also opened VS Code1.139.0 release notes; no
extension or Linux artifact upgrade was performed. Ownership and recent Activity
Log reviewed: prior deallocations succeeded; an automatic status request at
02:47:12 had failed409 after deallocation and was not a source read replay.
Scoped safety monitor re-enabled before starts02:53UTC. Only the same source
and runner started. New conservative stop03:50UTC(begin03:45), within60minutes.
Same inventory/report-only scope, no target/migration/install/security changes.
Interrupted60ba3809 must be reconciled before deciding any fresh attempt.
Unknown status controlaf-14792620 confirmed absent by ARM and reconciled as
failed in installed GUI; it was not resubmitted. Latest cost447.08125631327976USD.

## September23 02:57UTC — retained completion discovered; no repeated source scan

Independent guest evidence shows60ba3809 actually finished02:45:52.376327734UTC,
before the user interruption/deallocation, exit0, empty stderr. This supersedes
the earlier unconfirmed/running observation; the job was not automatically
resumed on restart. Report2947bytes SHA
`d67f999ba62d427fa776b05b4f9a99f24418ea1d0343870e25000377d9edb381`
retained; disk4%,swap0/OOM0,idle/no loader process. Installed GUI reconciled the
same operation to finished. No new inventory dispatched and no password prompt.
Approved report-only transfer submitted02:57:03.814UTC via control
`af-1153ebf8-6f46-4a5a-862b-d20e9d57e72e` through existing exact workflow
container; hash/count acceptance pending import. Source health/start command
completed with valid CA/IP-SAN and disk10%,swap0; no repeat scan was required.
Source deallocation requested immediately once retained completion established;
only runner remains needed for report export. No target/migration.

## September23 02:58UTC — installed-GUI inventory PASS/imported; both VMs off

GUI transferred/imported the exact2947byte report and displayed the hash-verified
source report. Independent local SHA matches retained manifest. Assertions PASS:
outcome pass,2passed checks,errors/incompleteChecks empty; all18label counts sum
to1,600,000vertices+4,000,000edges=5,600,000rows. Count method is a complete
PostgreSQL mapping stream in one repeatable-read snapshot. No source re-scan,
credential prompt, Keychain read/reset, or new operation was needed after resume.
Report warning correctly says counts do not verify unique identity/endpoints
or migration. Redacted receipt `evidence/b03-pg-inventory-pass-20260923.json`.

Both source and runner independently PowerState/deallocated by02:58:21UTC;
scoped monitor paused. All source data/disks/current and former evidence retained.
No target/migration/full canonical digest yet. Next is offline target sizing
review and specific target/deployment approval; broader ledger remains5pass/
7partial. Base9/9 qualification is unchanged and is not this B03 variant.

## September23 03:43UTC — stopped-compute target inputs prepared, approval pending

User requested continuation. Both source/runner independently confirmed
deallocated; no cloud resource mutation performed. Installed-GUI target review
read the accepted inventory and persisted only partial targetDraft inputs:
new server`afpg-cb2ef280a8914edfb45b`, existing migration RG, JapanEast/zone1,
PostgreSQL18/AGE Standard_E8ds_v5 with128GiB storage, same runner laterD4s_v5.
Proposed delegated subnet10.246.25.0/24 does not overlap the freshly listed
existing10.246.0.0/16 VNet subnets(.1 through.24). New private DNS/subnet only;
no public access/peering or source firewall changes are proposed.
Inventory storage high estimate9,900,772,000bytes plus25% fits128GiB; compute
selection maintains the prior P1 test configuration, not a throughput guarantee.

Current Azure Retail Prices API has a unique applicable consumption meter each:
PostgreSQL E8ds_v5=1.200USD/hour, runnerD4s_v5=0.248, existing sourceD8s_v5=0.496.
Target+runner1.448/hour; allthree1.944/hour, plus retained disks/storage/network.
Proposed next window is at most2hours from first compute start, cumulative800USD
ceiling/reserve700 retained. This longer window and exact target/credentials/
subnet creation have NOT yet been approved. Stopped the GUI at its explicit
deadline prompt without inventing an approved deadline. Draft inputs retained;
no final LoadJob/target plan saved, no credentials generated, no target deployed,
no VM restart/resize/migration. Await one bundled action-time approval, then
repeat live placement/quota/pricing/readiness gates before submission.

## September23 03:48UTC — target/migration window explicitly approved

User approved the bundled new private E8ds_v5/128GiB target, dedicated subnet/
credentials, same-runnerD4s_v5 resize and5.6Mrow migration/standard verification,
maximum2hours from first compute start. Saved credential reuse reconfirmed through
installed GUI without another password prompt. Exact ownership/deallocated states
checked; recent Activity Log contains Resource Health notifications and prior
deallocation, no new in-scope security changes observed. Refreshed delayed cost:
446.774812390391USD originalRG+4.48250975058573USD B01=
451.2573221409767USD, cumulative800USD/reserve700 unchanged.
Scoped monitor activated before first runner start03:48:08UTC. Fixed conservative
hardstop05:45UTC(begin05:40), less than2hours. Source stays off until needed.
New target creation is now authorized but not yet submitted. No dedicated
canonical verifier installation is included in this action-time approval.

## September23 04:00UTC — private target submitted through installed GUI

Readiness refreshed the runner boot to97569248-10c4-4f01-85e0-c2bc1d3b4915:
fixed Linux646f0d4b361a, idle, disk3.82691163%, swap0/OOM0. The first review
correctly failed closed on the changed boot before saving final files or creating
a target. Operator reviewed the new health and reused unchanged saved inputs;
no deployment or source-read replay occurred. The final secret-reference-only
LoadJob and target plan were saved under the ignored dedicated local folder.

Installed GUI submitted target plan hash
`76fc93db4304e4d4c66835b2968de6ca616cbf2dc91e039bdaf8ece64d6689ad`.
Independent ARM deployment `afpg-cb2ef280a8914edfb45b` is Running,
timestamp04:00:19.891143UTC with no current top-level error code. This is not
deployment success or migration qualification. Existing source VM remains off.
Continue read-only deployment reconciliation, then approved AGE readiness and
same-VM resize/migration only within05:45UTC hard bound. No verifier installed.

## September23 04:19UTC — target provisioned, AGE preload applied

ARM deployment succeeded04:06:09UTC and installed GUI reconciled provisioned.
Target ownership/workflow, PostgreSQL18/E8ds_v5/128GiB and public access Disabled
confirmed. External governance deployed advanced threat protection04:06UTC;
preserved without override. GUI submitted approved target-only restart04:11:38UTC.
Fresh ARM now Ready and shared_preload_libraries=pg_stat_statements,age with
pending restart false. Source VM remains deallocated.

Same-VM resize authorization saved04:14:37UTC with its20minute bound. First
fresh-readiness cycle passed guest health but VM running/provisioning gate stopped
before any resize submission. Subsequent independent ARM confirms running and
Succeeded/B2s_v2. Continued the same still-valid authorization, not an uncertain
mutation replay; a fresh readiness command is pending. No migration or verifier.

## September23 04:27UTC — installed-GUI migration submitted, no re-entry

Same-VM resize completedD4s_v5 with retained disk/NIC/identity preservation SHA
`ee5c3b4b075c92bd441ea050d0bf23bd8c19ecf89f5561310707ec7726cda01f`.
GUI reconciled AGE preload finished without replaying its restart. Existing
sourceVM started04:22UTC; exact existingpostgres:18.1 container started after
CA/IP-SAN/expiry checks passed, disk10%,swap0/OOM0. No fixture changes.
New runner bootfc19bb34-c428-43ca-b959-139cb72c16b0, fixed646f0d4b361a,
disk3.84237553%,idle/swap0/OOM0 accepted through installed GUI.

GUI submitted operation/job`d707adf8-7cab-4d73-8e38-ec141e72ac18` at
04:27:10.720UTC using reviewed inventory and saved source credential without
another password prompt. Control`af-2fb1a1a3-42e7-48b1-9242-d1df4e346f89`.
The create-only new graph load and complete counts verification use the same
fixed runner; no replacement, retry or resume.30minute service bound remains.
Scoped heartbeat updated to reconcile only this exact operation and retain the
05:45UTC hard stop(begin05:40). Submitted is not migration success. Dedicated
canonical verifier has not been installed or action-time approved for this guest.

## September23 04:40UTC — migration/counts PASS imported; canonical approval pending

GUI reconciled exact migration finished exit0 and imported the9619byte report
through the existing workflow container, SHA
`5bdf8e8c6c3f9bda62c504fdf9da98fe0a8d6ed81455f66bb20d16277ccbe4dc`.
Actual installed GUI displays Counts verification: PASS. Independent assertions
confirm18mapped labels,1,600,000vertices+4,000,000edges; accepted,committed,live
physical and live identity counts all match inventory; rejects0 including the
separate unclassified.rejects summary field;24checks pass,errors/incomplete empty.
Report generated04:32:01.345623733UTC. Receipt
`evidence/b03-pg-counts-pass-20260923.json`. No migration replay or source rescan.

Read-only guest health04:36:54UTC: worker inactive, disk4%,swap0/OOM0. Its optional
state summarizer could not run because guest jq is absent; no software installed.
The sealed terminal result was independently reconciled by the installed GUI.
Recent scoped Activity Log contained policy audit/auditIfNotExists only outside
our run-command operations; target Ready/public access Disabled unchanged.

Full64range/root verification is NOT performed. Correct raw-ID verifier
8a23a5109798/archiveSHA60ed56a6773e6cbb64f7a0c03bc407f8aea135c7f1a75d7b8494db17cf09f79d
was locally rehashed; requested specific upload/install/run/private-import
approval, max25min/4GiB. No verifier installed/run. While waiting, begin stopping
exact source/runner/target04:45UTC, verify by04:47UTC where Azure latency permits,
conservatively15minutes after report generation. A fresh approved active worker
supersedes this idle condition, never the original05:45UTC hardstop. All evidence
retained; broader B03 qualification remains partial.

## September23 04:45–04:47UTC — idle safety stop verified

No new verifier approval or retained qualification operation was present at
04:45:15UTC; migration/report transfer both remained finished, counts PASS.
Fresh exact ownership checks matched runner workflow and source fixture tags;
target Ready/private and no intervening scoped governance change observed.
At04:45UTC submitted deallocation for only runneraf-cb2ef280a8914edfb45b and
sourceaf-pgvm-source, plus stop for only targetafpg-cb2ef280a8914edfb45b.
Both VMs PowerState/deallocated verified04:46:36UTC. Target was then Stopping;
did not replay stop, and Stopped verified04:47:37UTC after Azure service latency.

No resource, disk, graph, credential or evidence was deleted. No verifier was
installed/run. Scoped heartbeat paused after all three stopped states verified.
Complete migration/counts evidence remains PASS, independent64range/root
qualification remains pending. Azure warns that a stopped Flexible Server
automatically starts after7days; retained storage continues to incur charges.
Do not restart merely to inspect evidence or silently extend the original window.

## September23 08:26UTC — new user-approved two-hour canonical session

User requested resumption after Keychain recovery and explicitly authorized a new
two-hour window. No migration replay: only the existing runner and private target
are needed for the previously described raw-ID verifier and private report import.
The source VM remains off. Installed GUI recorded renewed cost authorization with
hardstop10:25UTC(begin10:20), cumulative800USD/reserve700 unchanged; the previous
authorization remains in history. First compute start request08:26:35UTC, so the
fixed bound is conservatively less than two hours. Scoped safety monitor was
activated before start. Qualification has not yet been submitted.

Normal Azure login recovered management access; VS Code Azure Resources shows
the signed-in account and both expected tenants. No secret extraction or reset.
Fresh delayed cost461.0327006541777USD across both test groups. Exact ownership,
private target and unchanged D4s_v5/disk/NIC/identity checked. External application
updated platform patch settings08:05UTC; preserved without override. Latest guest
health must be rechecked after startup before executing the verifier. Archive
60ed56a6773e6cbb64f7a0c03bc407f8aea135c7f1a75d7b8494db17cf09f79d rehashed locally.

## September23 08:33UTC — installed-GUI canonical qualification submitted

Runner and target are Running/Ready; source independently remains deallocated.
GUI reconnected the exact workflow without source re-entry or migration replay.
Readiness completed exit0: boote44e4208-b324-4826-b108-57123dc481c4,
fixed646f0d4b361a, idle, disk3.860608995%,swap0/OOM0. Native verifier review
selected8a23a5109798 with the previously disclosed archive hash and accepted the
read-only25minute/4GiB action. Existing target SecretStorage was reused without
prompt; no credential extraction/reset. GUI retained qualification operation
`028ec524-990f-4ed3-8265-6eb024d5cf57`, submitted08:32:46.205UTC, for the same
jobd707adf8-7cab-4d73-8e38-ec141e72ac18. No automatic retries. Scoped monitor
tracks this exact operation; hardstop10:25UTC unchanged. Submitted is not PASS.

## September23 08:44UTC — verifier finished; capacity blocker and safe stop

Exact qualification028ec524-990f-4ed3-8265-6eb024d5cf57 finished exit0 at
08:34:52UTC. Installed GUI reconciled phase verified,23224byte result SHA
`00938de546f94d1fe50e372044bef9b423e104919855fc8d934a06b4b42f77fc`.
The next GUI action failed before export submission at the extension's25managed
Run Command cap. No qualification/migration replay, no export control created.
The sealed result remains on the runner disk. Full local range/root assertions
and installed-GUI final PASS are still pending; exit0 is not final qualification.

Guest health after completion08:35:09UTC showed no active verifier process,
disk6%,swap0/OOM0. The completed transient unit was gone; no peak RSS measurement
is claimed. Exact ownership/private-target checks passed before safety stop.
Recent governance operations on other targets were not altered. Requested only
runner deallocation and target stop at08:41UTC; runner deallocation Succeeded
08:41:27.8589076UTC, both deallocated/Stopped independently verified by08:44UTC.
Source remains off, disks/data/evidence retained. Scoped monitor paused.

Installed GUI archived one unreferenced historical readiness receipt for
`af-618186b2-cee7-4a82-95b8-962dfff30de3` (September22 14:15:27UTC).
Local archive1433bytes SHA
`aee3895eda7371f6b1e12905af8578eacfa489722784106dcd99f790b7a21b08`,
receipt SHA08486937aec4c7719a3bd96508336d1b994c4a14df79473008e4e07a61dc632a.
Native removal preview correctly refused: current ARM provisioningState is
Succeeded but instanceView executionState is Pending,exitCode0, with output and
start/end times absent. Two other historical readiness controls show the same
condition; the cause of missing live execution evidence is not established.
The archive is historical proof, not a claim that current ARM proves success.
No removal intent or DELETE was submitted; current readiness, migration, counts
and qualification controls remain preserved. A separately approved exact-scope
maintenance deletion is needed to free capacity, or a reviewed non-deleting
export path. Do not weaken the existing removal guard or edit workflow metadata
to bypass it. Any resumed result retrieval stays inside10:25UTC hardstop and
must not rerun the verifier, migration or source scan.
