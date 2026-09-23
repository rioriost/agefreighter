# B09 two-fault execution — September 24 JST

**B09 and B10 complete for their defined scopes; all new trial resources removed.** This continues
the [approved dedicated trial](b09-fault-session-20260923.md) after commit
`1fa5ac8`. B10 remains independently revalidated and complete; no new inventory
or source-data operation is needed.

## Resumed authorization and preserved state

The user's `continue` directly answers the preceding question to apply
`SecurityControl=Ignore` to the exact new transfer account and continue the
trial. The old setup window ended with the guard result
`EXPIRED_WITHOUT_DEPLOYMENT_INTENT`, `cloudMutation=false`. No trial VM or archive
upload existed. The old companion, its one storage deployment intent, the ready
storage record and both earlier failed setup attempts remain preserved.

The explicit resumed setup window ends at **2026-09-23 23:15 UTC
(September24 08:15 JST)**. It does not extend compute or spending limits: maximum
60minutes from the first actual VM deployment request, start deallocation by
minute55 or earlier after evidence/failure/15minutes idle, incremental USD5 total
for this dedicated trial within the existing USD800 ceiling. The new guard also
leaves five minutes before setup expiry, with additional ownership-read headroom.

All resource IDs, workflow `0d8b4bc9-102c-4d74-8270-061ce31ce163`, source-free
scope, negative artifact SHA `3c33a179916ec08a83ca8ccb3c19e7682862d0382a63b2c05b5369a3cb124e33`
and observer limits remain unchanged. The old host was closed through its native
UI before preparing a sealed continuation. Authentication stays in the existing
dedicated profile; no credentials/cache are copied. The storage deployment must
not be repeated.

## Actual account-only exception

One exact PATCH at21:18UTC merged the approved exception tag into the three
ownership tags and enabled authenticated public HTTPS access on
`af0d8b4bc9102c4d74827006`. At21:19UTC, independent ARM readback confirms
Enabled, anonymous Blob access false, shared-key access false, HTTPS-only true
and TLS1.2 minimum. An authenticated list of the exact workflow container
succeeded and was empty. No resource-group/subscription exception was added.

Fresh read-only inspection also confirms the previous B10 VM deallocated and
its source DB Stopped. Independent local audit passes11checks: all93normal
operator files are unchanged; only one storage PUT intent exists; no artifact,
runner, readiness or source operation exists. These checks are preparation
evidence, not credit for the two intended faults.

Private resumed receipts are under
`production-simulation/work/b09-fault-resume-20260924/`, with guard preparation
under `production-simulation/work/b09-fault-trial-20260923/resume-20260924/`.

## Actual native execution and independent verification

The renewed companion runtime SHA is
`d4f95188b7169b611dd2e100fce2be1ad3e620dbac7be4cd2caad935934c1fd7`,
scope SHA `b47e7bee1caa3e2a5db287d10154ee2325fd5867ffe66ac181fa37c20bea4cc7`.
The supervisor and an independent agent checked the sealed continuation, exact
ready-storage seed and inherited create-only storage PUT claim. The supervisor
also verified all667bundled source hashes. Only the setup expiry changed;
the earlier storage creation cannot replay. The new helper passes730unit tests,
typecheck and host compilation. Production source and the release bundle were
not changed.

The normal native source panel refreshed the account to Enabled. The operator
selected the exact negative manifest in the real file chooser and accepted its
visible negative-fixture notice. Independent authenticated Blob properties
confirm18,378,050bytes and the approved SHA metadata. No source files were selected.
The normal runner preview rechecked placement, network, quota and ARM what-if;
the compute estimate was USD0.109/hour, with storage/network extra.

| UTC event | Observed result |
| --- | --- |
| 21:32:02.852 native Create reviewed runner | Approved exact VM and container-only Reader grant; negative fixture clearly identified. |
| 21:32:14.983 one runner PUT intent | Exact reviewed template SHA `13f8c7196e45aa7b8bb200dd1040565c35097ff0d030bac795bce76a4ca01be0`. |
| 21:32:16.724 HTTP201 | Adapter retained actual Azure acceptance and deliberately withheld it from the unchanged controller. |
| 21:32:16.752 controller return | Phase unknown; native UI says refresh, do not resubmit, with Deploy disabled. |
| Independent ARM observation | Same deployment Succeeded; six operations contain only approved VM/NIC/NSG/container Reader and output evaluation. |
| 21:33:24.453 native Refresh return | Same deployment reconciled to provisioned using one GET; no second runner PUT. |
| 21:35:43 guest observation | Actual terminal cloud-init/cloud-final error; correct archive and retained binary; missing tools extraction; no installation markers or runner process. |
| 21:36:42.571 normal readiness submission | Exactly one unchanged production readiness command. |
| 21:37:09.388 native status reconciliation | Failed, exit1, `Linux bootstrap did not complete successfully.`; no guestReady or readiness receipts. |
| 21:37:36 source panel opened | No assessment or report; no source credentials, source dispatch, target or migration. |

The [response-loss receipt](evidence/b09-response-loss-20260924.json) passes
23independent predicates. This proves deliberate loss at the transport/controller
boundary after real acceptance, not TCP packet loss or an Azure outage.
The [bootstrap receipt](evidence/b09-bootstrap-failure-20260924.json) independently
matches both returned ARM scripts to their reviewed source and checks every
terminal-failure predicate. Only observer01 was needed, within the maximum3.
The normal UI says guest readiness could not be verified; source-panel
`canStart=false` is additionally derived from the unchanged production code and
retained record. No source operation was attempted to establish that gate.

## Evidence retention and shutdown

Before cleanup,102companion files were copied to the private durable
`retained-companion/` directory, preserving native receipts, stages and workflow.
Its retention manifest SHA is
`50372540b77e1f01327cdceaa7186d82311c2aef9eab2eadb55b2c87ff37730c`.
Authentication caches were not copied. Independent ARM responses, observer
output, failed-readiness response and guard records remain private alongside it.

The exact-owner guard initiated deallocation at21:38:12.648UTC. Azure records
success at21:38:25.550UTC and the guard retained verified DEALLOCATED by
21:38:35.160UTC, about6minutes20seconds after the first VM PUT, within the
60minute maximum. No restart is authorized or needed. The delayed month-to-date
Cost Management read was USD511.610086 across the two qualification groups;
it is not a final invoice for this short trial.

The frozen cleanup manifest lists only the approved new resources and two
retained terminal commands, one exact artifact Blob and the two scoped roles.
All93normal operator files still match the original names, modes and hashes.
At21:39:21.653UTC Azure definitively rejected the first command-resource DELETE
with HTTP409 OperationNotAllowed because the VM was already stopped. The
create-only intent and rejection evidence are preserved; no DELETE replay or
VM restart occurred. A separately reviewed continuation removes the owned
VM together with its two retained child controls, then resumes the original
exact dependency cleanup and execution journal.

The first continuation check also stopped before mutation: after deallocation,
both ARM command instance views became exactly `Pending`/exit0 with no output
or timestamps, although all other command properties still matched their pinned
originals. This is not new terminal evidence. The revised condition requires
freshly verified VM ownership and deallocated state, exact unchanged command
properties, and the independently retained pre-stop Succeeded0/Failed1 results;
only the observed two-field unavailable view is accepted. Eight mocked checks
and an independent review pass; changed bodies, extra status fields and a
running VM remain refused. The live read-only preflight then passed.
The continuation SHA is
`6c5ca24c462ac71f48ac10b8d90b87615d1400d8a05d3d14b9c4d3a44d0ae4db`.
The owned VM and both command resources were independently absent at21:54:02UTC.
The original cleanup then removed the disk, NIC, NSG, both scoped roles,
container, account and two deployment records. Unavailable observations after
disk/NIC/NSG deletion caused conservative stops; fresh GETs established absence
before continuing the same journal, without replaying a prior DELETE.

At21:59:02UTC, separate final GETs established absence for all14exact resource
identities. Each Azure CLI response rendered `Not Found` with a structured ARM
missing-resource code; numeric HTTP status lines were not separately retained.
The container, VM Reader role and Event Grid child returned
`ParentResourceNotFound`; their exact parent account/topic were independently
absent. The frozen cleanup's final reporting rejects that code, so its exit
status is not the completion proof. The separate final verification is.

The new account's automatically created Event Grid topic and its sole scanning
subscription disappeared with the account; no Event Grid DELETE was sent.
No shared scanner, security policy, resource-group tag or existing resource was
modified by cleanup. Ten explicit resource DELETE requests were accepted, plus
the one preserved rejected command DELETE; there was no duplicate DELETE or
VM restart. The account-only exception disappeared with its deleted account.

See the [sanitized cleanup receipt](evidence/b09-trial-cleanup-20260924.json).
All93normal operator files retain their original names, modes and hashes. Fresh
independent reads at21:46:33UTC also verify the previous B10 VM deallocated and
source database Stopped; neither was included in any new-resource mutation.
B10's retained5.6M-record evidence remains unchanged. The dedicated VS Code was
closed through normal native Quit after retention; its Extension Host was then
confirmed absent. No credential cache was copied or deleted. Base routes remain9/9;
extended branches are9pass/3partial, with B02/B06/B07 outside this completed
batch. This is not release qualification.
