# B06/B07 continuation after the overnight stop

The user unlocked the Mac on September 25 JST. The additional authorization
remains the same: **96 hours and USD 400**, observed September 24 at 12:45:06 UTC
and ending September 28 at 12:45:06 UTC. No new approval is inferred from the
unlock itself. This is the second bounded session within that existing scope.

Final status: **B02/B06/B07 complete for their defined finite scopes; all 12
defined branches PASS; dedicated trial cleanup independently verified**.
This remains separate from release qualification. The sections below retain
the sequence and its earlier incomplete checkpoints.

The first session's guard recorded `compute-stopped` at September 24
17:12:58.853522 UTC, with the VM stopped and target absent. Its stop journal
indicates no stop POST was submitted by the guard; this proves the observed
state, not the cause of the earlier stop. A fresh exact Azure GET at September
25 00:00:58 UTC confirmed the trial VM deallocated, its original system identity,
expected B2s_v2 shape and no additional disks. The original session's clocks,
receipts and journals remain unchanged.

## Initial preparation: same resources, separate session clock

Resume only workflow `8e9f853a-7e9b-4570-8045-952341217524`, VM
`af-8e9f853a7e9b45708045`, in `rg-af-vscode-p1-20260905-a`. Its existing trial
storage and pinned software are retained. The pending private target remains
`afpg-8e9f853a7e9b45708045`, PostgreSQL 18/E8ds_v5/128 GiB/zone 1/HA off,
with subnet `10.246.26.0/24`. All other resources and data remain preserved.

The new session guard uses a separate private root
`sessions/02-20260925/safety`, an actual **before-resume** marker, the previous
stopped receipt and the original first-VM intent as provenance. Resume must not
be described as a new VM creation or overwrite the old first-intent timestamp.
Stop begins 230 minutes after the actual resume intent; the maximum is 240
minutes, also bounded by the original outer deadline. No resume has been
submitted at this preparation checkpoint.

Reserve another **USD 30** for this session. Preserve the first session's
USD 30 reservation. The target wizard therefore uses a cumulative workflow
ceiling of **USD 60** and an additional-cost reserve of **USD 45** (previous
session 30 plus current noncompute 15), leaving USD 15 for current compute.
The new guard's session limit stays USD 30. The normal fresh pricing check must
pass before target deployment.

One fresh delayed CostManagement read at September 24 23:59 UTC returned
USD 638.99647374916802 for the two retained groups. Keeping the prior USD 20
allowance and both USD 30 reservations gives USD 718.99647374916802, below the
conservative USD 800 ceiling. Conservatively assigning the whole observed
increase since the earlier snapshot to the new authorization, plus both
reservations, gives USD 127.56697196627880 against USD 400. This is delayed
billing and reserved headroom, not a final invoice or attribution to this VM.

## Historical preparation checkpoint

The strict B06 access transition already passed independent review: actual
same-principal/action denial, normal GUI Reader grant, then the same constant
query succeeded. No additional B06 observer requests are permitted or needed.

The original inventory `0bb518bd-f782-48eb-8234-7128a276443f` is finished, with
a 2,940-byte report sealed as
`5cbe10f11102ff5897ba20098b163d84922ecb8a290e4aa782f5842e84b09e31`.
Its original assessment boot identity remains intact. Installed-code review
confirms terminal report export/import survives an ordinary restart and checks
the original operation, bytes and hash. After restart and fresh operational
readiness, use the normal GUI transfer/import and inspect the full report.
No replacement inventory is justified solely by the boot change.

Then genuinely review and hold the unchanged source panel before target
creation, provision the reviewed private target, complete a normal same-VM
resize while idle, and run the second originally authorized inventory. Bracket
the native active-inventory resize refusal with exactly two process observers
bound to the actual second operation and the new resume clock. The complete
lifetime command limit of 60 and retained limit of 25 span both sessions; they
are not reset. Reconcile and import that same second inventory, stop compute,
archive evidence and clean up only the owned trial additions.

The installed bundle remains SHA-256
`7a2aa5ddf13454ff6dab4d4a8ad2cb9a172842c9c4fbcc7b5d8e94d3f79d1d34`;
all 96 pre-trial operator files remain unchanged at this checkpoint. B06 full
acceptance, B07 active-case acceptance and final cleanup remain pending.

## Resume submitted

The independent guard review passed 12 offline cases and its actual local
pin check. Root launched the new guard, then retained a fresh exact owned,
deallocated VM observation and an exclusive resume intent at
**September 25 00:10:24.679169 UTC**. One ordinary Azure VM start request returned
HTTP 202; no repeat was sent. This operational restart is separate from the
installed extension's qualification actions. The new guard fixes stop at
**04:00:24.679169 UTC** and maximum at **04:10:24.679169 UTC** (13:00 and
13:10 JST). Both earlier creation and stop receipts remain byte-identical.

## B06 completion

B06 now passes its defined fixed-managed-identity branch. At approximately
00:20 UTC, root observed the original inventory report as imported in the
normal GUI and opened its hash-verified viewer. Independent review compared
all 2,940 report bytes and the normal-store copy against the original seal,
verified the imported native record, and confirmed 18 configured mappings:
1,600,000 vertex records plus 4,000,000 edge records, with no errors or
incomplete checks. The original operation and assessment boot remained
unchanged across restart. Together with the new strict same-principal/action
denial, normal Reader grant, matching constant-read success and explicitly
retained prior trust/principal refusals, this completes the finite B06 cases.
The [scoped evidence receipt](evidence/b06-strict-access-and-import-20260925.json)
separates root's native GUI observation from independent byte verification.
Earlier inconclusive results remain unchanged; propagation latency,
open-modal trust loss, alternative credential modes and migration correctness
are not claimed. B07 and final stop/archive/cleanup/preservation remain open.

## Private target submitted

The normal target wizard retained the reviewed settings. An initial readiness
refresh was followed by a provisioning preflight refusal before any target
intent; a fresh observation then found the same VM in `Succeeded` state.
A separate price lookup returned HTTP 429. Both outcomes are retained as
pre-submission results, not failed deployments. Reusing the unchanged saved
inputs subsequently produced a `submitted` target record with a plan generated
at 00:30:33.333 UTC. No deployment intent was replayed.

The plan specifies private PostgreSQL 18, Japan East zone 1, E8ds_v5, 128 GiB,
HA disabled, a dedicated delegated subnet and private DNS, followed by a
separate same-VM D4s_v5 resize. Its deadline is 04:00:24.679 UTC, cumulative
ceiling USD 60 and reserve USD 45. The reviewed compute rate is USD 1.448/hour;
the plan-time remaining compute plus reserve is USD 50.06. Independent review
confirmed the seven-resource template, unchanged source settings and original
Reader identity. Fourteen command definitions are retained across both sessions.
At this checkpoint, provisioning, resize and B07's active-worker bracket remain
pending. GUI events are root-attributed; the independent review verifies the
retained submission and configuration, without claiming a separately retained
HTTP 429 response.

## First B07 observation pair remains inconclusive

Normal target reconciliation reached `provisioned`, and the ordinary bounded
same-VM sequence completed B2s_v2 to D4s_v5. A separate retained ARM observation
confirmed the unchanged disk/NIC/identity seal. After fresh guest readiness,
the held, previously reviewed source panel submitted inventory2 at 00:45:36 UTC.

The first observer proved a running inventory process. Root then observed the
normal read-only resize action refuse with the expected message; the native
record remained byte-identical. The second observer failed with exit 2 and
`activeInventoryProcessProven:false`. Therefore same-worker continuity across
the GUI refusal is **not proven**, and this pair is **inconclusive**. Both
observation slots are consumed; neither operation is replayed. The generic
failure does not establish a cause or authorize another observation.

The same inventory subsequently reconciled to finished and was imported through
a second ordinary source panel, preserving the original reviewed panel. Its
complete 2,939-byte report contains all 18 mappings and 5.6 million records with
no errors or incomplete checks. This successful report does not close B07's
missing process proof. See the [inconclusive receipt](evidence/b07-active-refusal-inconclusive-20260925.json).
The original 96-hour / USD 400 user authorization covers completion of the
requested cases. The two-observation limit was root's narrower operational
scope, not a separate user-imposed attempt ceiling. A new inventory and one
additional pair therefore require a separately sealed, independently reviewed
root amendment under that existing authorization. It preserves the consumed
pair and common journal, fixes four observations in total, and retains the
current stop deadline and budget. No new user approval is claimed. Preparation
alone does not enable another submission.

At 01:18:09.183261 UTC, root sealed this bounded amendment after independent
review of all 53 offline cases. It permits one fresh complete inventory and
one additional before/after pair, with four B07 observations in total. The
original pair remains inconclusive and consumed. The actual amendment receipt
is SHA-256 `ea5e2bb60f958427e0426692b8a8f3eceef9146ca6d2b13d26f332df82f1f617`;
the independently reviewed dispatcher is
`5e05683df2e78710a0601735dd62e950da1637c67c8054336e86aa13dfc23a9b`.
This records root's operational review under the existing authorization,
not a new user approval. The 04:00 stop, 04:10 maximum and USD 60/45 cumulative
budget/reserve stay unchanged. The next inventory and pair have not yet run
at this checkpoint.

## Inventory3 imported; no process observers sent

The next normal inventory, `a2437775-0b18-4c23-b664-b6ad23338329`, completed.
Root observed its ordinary GUI import and hash-verified report viewer at
**September 25 01:54:41.591 UTC**. Independent review verified the complete
2,940-byte report, its identical normal-store copy and the native imported
transfer. All 18 mappings completed: 1,600,000 vertex records and 4,000,000 edge
records, with no errors or incomplete checks. Its report SHA-256 is
`1e84830e30a4eef21fd5e6037868b7a85fe7a0fadc42761d3cd28286024d85f5`;
the retained imported native record is
`28b75ce484648a2b1cff103e4b017e335b9dc359e85963a5c305c5d7e25f4085`.

The normal watcher submitted a status control before the intended launch
snapshot was frozen. Preparation refused before creating an observer binding
or sending either observation. No historical normal record was reconstructed.
Inventory3 therefore used **zero B07 observer calls**; independent local review
confirmed no corresponding PUT intents and only the original consumed pair
in the common journal. The full report is valid inventory evidence, but it does
not establish the missing active-worker bracket or change the earlier
inconclusive pair.

## Historical checkpoint: final bounded pair in progress

At 01:56:44.307196 UTC, root sealed amendment SHA-256
`70c3503796841fdbe14947456efca6eea998fbed734ae1c088fd14c99ce628b8`.
Under the same existing 96-hour / USD 400 user authorization, it permits one
final fresh inventory and the remaining two process observations. It preserves
the earlier amendment and both consumed observations, excludes observations
for inventory3, and keeps **four B07 observations in total**. This is a root
operational amendment, not renewed user approval. The same VM, 04:00 stop,
04:10 maximum, USD 60 workflow ceiling and USD 45 reserve remain unchanged.

Inventory4, `62ef7018-2374-46bc-8ffd-23811f27e4ea`, has now started through the
normal GUI. Its launch submission, actual ARM launch response and later native
state are retained separately. The first observer reported an active inventory
process; root then observed the expected native read-only resize refusal at
02:15:03.718696 UTC. At this checkpoint the second observation and independent
same-worker correlation are pending, so **B07 is not yet PASS**. Final report
completion/import, compute stop, full archive, owned-resource cleanup and
preservation checks also remain pending. Successful inventory counts do not
claim migration correctness.

## Final active case and inventory4 import completed

The final before/after observers both finished successfully and identified the
same running inventory worker across the native resize refusal at
**02:15:03.718696 UTC**. All four process snapshots agree on the service
invocation, main/child process identities and start ticks. The host-side
before-result retrieval precedes the GUI observation, which precedes the
after-observation submission; no host/guest clock equality is assumed.
Independent review verified both complete ARM outputs, sealed results and
pinned request bodies. **The B07 active-inventory refusal case is PASS.**
The common journal contains exactly four observation intents: the original
inconclusive pair and this successful pair. Inventory3 has none. The earlier
inconclusive outcome remains unchanged.

The same inventory4 operation then completed. Root observed its normal GUI
import and verified report viewer at **02:22:29.938203 UTC**. Independent review
checked the entire 2,940-byte report, its identical private normal-store copy,
and the native finished/imported binding. All 18 mappings completed with
1,600,000 vertex records and 4,000,000 edge records; the report outcome is pass,
with no errors or incomplete checks. The report SHA-256 is
`5db2d92a8e04b2d4551549f9fbcdd718329f361279f69b3d9101221bb3e55452`;
the imported native record SHA-256 is
`719f23f8099898c5dfcde938cf9ff186303088de8ea7458559e38e82d8599331`.
GUI observation is attributed to root; the independent audit establishes
retained-byte and normal-store consistency. These are mapped-record counts,
not proof of unique identities, endpoint existence or migration correctness.
See the [active-case and import receipt](evidence/b07-active-inventory-refusal-pass-20260925.json).

Compute stop, complete archive, owned-resource cleanup and final preservation
checks remain separate pending work at this checkpoint. This result does not
claim release qualification or completed cleanup.

## Final stop, cleanup and preservation

After the verified import, root requested an early stop through the existing
guard at 02:22:58 UTC. The guard recorded the dedicated VM and PostgreSQL
stopped at **02:24:32.186090 UTC**, before the fixed 04:00:24 UTC cutoff.
Fresh exact GETs confirmed the VM deallocated and PostgreSQL Stopped. Neither
was restarted for cleanup. The stopped database rejected child enumeration;
the explicitly reviewed cleanup therefore deleted only the exact new parent,
without claiming an empty or individually inspected database/configuration list.

Initial cleanup preparation stopped before deletion because its disk-type
constant expected Standard SSD. Fresh disk metadata and the actual VM responses
before and after this final resize instead show the same original 64 GiB
Standard HDD (`Standard_LRS`) disk. One comparison was corrected to that exact
type, with 72 independent offline checks passing and all identity, ownership,
attachment, archive and no-replay checks retained. The earlier actual identity
projection reported SSD; the earlier transition time/cause remains unresolved.
The final resize itself preserved the complete observed disk configuration.

The sealed 12,974,080-byte predelete archive has SHA-256
`670adaeb9ef824f895e980d930b633c6c282d56b3566e7cf6712e24b18b2dcbb`.
Root's cleanup completed at **02:49:04.860165 UTC**. PostgreSQL deletion took
several minutes after acknowledgement; only GET reconciliation was used, and
its DELETE was never replayed. Independent reads at 02:49:32–02:50:02 verified
all **42 final manifest identities plus 10 previously retired controls absent**.
The journal contains **26 unique DELETE intents**, including the earlier ten;
52 absent identities do not imply 52 individual deletions. Parent cascades and
automatic account-derivative removal are counted as absence observations.

The original 96 operator files retain their names, bytes, permissions and
hashes. The current workflow and all four imported reports are also unchanged.
The shared VNet and original 25 subnets, original 26 topic/source associations,
resource group and source privacy projection are preserved. Cleanup verification
made no source-data or storage-content queries. There is no full-configuration
baseline for every older storage account; topic associations are narrower proof.
See the [final cleanup receipt](evidence/b06-b07-final-trial-cleanup-20260925.json).

The final private archive is 66,938,880 bytes, SHA-256
`dc507362a3ad1401ddf0f46fd612af5946d2e75fff1d2f8a6e2ffd35e423396c`.
Independent verification checked all 715 regular members, including 711 trial
evidence files, the unchanged predelete archive, the pinned Linux runner,
installed extension and member manifest. Closing receipts are retained separately.

The delayed billing snapshot plus all retained reserves is **USD 722.10**
against the conservative USD 800 total ceiling. The conservatively attributed
additional-window increase plus its two reserves is **USD 130.67** against
USD 400. These are bounded estimates from two resource groups, not a final
invoice or an isolated charge for this trial. No software release was made.
