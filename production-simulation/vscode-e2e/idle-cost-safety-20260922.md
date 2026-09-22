# Retained trial compute safety check — September 22

Scope: the existing original and B01 P1 trial resource groups, subscription
ending `fdb7`. This preserves the prior completed-session stopped state; it is
not a new paid test window or qualification result.

## Fresh observations

At approximately 05:37–05:40 UTC, read-only checks found all 13 remaining VMs
in the original group deallocated, plus the B01 VM deallocated. The source
Flexible Server and B01 target were Stopped, but eight old migration targets
were Ready. Each target's application/purpose/workflow tags and exact ID matched
its retained operator workflow. No loader VM was running. Historical local
`submitted` flags are retained as historical uncertainty, not silently rewritten
as success or treated as a running guest on an absent/deallocated VM.

The bounded preceding 24-hour activity-log query returned 247 events, including
failed TLS-setting writes and resource-health events, but no PostgreSQL
start/restart action. It does not prove who/what started the servers. Azure CLI
warns that stopping Flexible Server permits automatic restart after seven days;
this is a plausible explanation, not a confirmed per-server cause.

Fresh MonthToDate ActualCost/PreTaxCost query:

| Scope | USD |
|---|---:|
| `rg-af-vscode-p1-20260905-a` | 388.915409197442 |
| `rg-af-vscode-p1-b01-20260921` | 2.0490809324889 |
| Combined | 390.9644901299309 |

Billing is delayed, not real-time spend. The cumulative USD800 ceiling is not
renewed; the retained USD700 accrued/non-compute planning reserve is not an
additional allowance. Stopped databases/disks/storage still incur retention
charges. The old live-session deadline is not extended by this safety check.

## Exact safety-stop targets

After ownership, VM state and recent activity review, submitted stop (not delete)
for these eight existing targets only, all in the original trial group:

- `afpg-2595fb2ddf9d4582b8b8`
- `afpg-7b79f05d1dc140a6b3dc`
- `afpg-29558917403e4a76aaa0`
- `afpg-c275d043de934b0ab2b0`
- `afpg-83c6b829acdc4405aa2d`
- `afpg-22f11b89e9434d569675`
- `afpg-53625ae3b1554821bfc3`
- `afpg-1f480fe1490d4789bc18`

All eight stop requests were accepted and then observed Stopping at
05:39:53 UTC. By 05:42:57 UTC the original-group inventory showed 17/18
servers Stopped; the sole remaining target `afpg-29558917403e4a76aaa0` was
independently confirmed Stopped at 05:44:09 UTC. B01's target was separately
confirmed Stopped at 05:42:02 UTC. Thus all eight safety-stop targets are verified
Stopped, and all 19 retained servers across the two groups have stopped-state
observations in this batch. All 14 remaining VMs were observed deallocated.
No source/graph/disk deletion, credential, role, network, policy-tag or TLS-setting
change was made. The installed GUI was only observed; no test operation was
submitted. At 05:41:09 UTC all 82 operator files still retained their prior aggregate SHA
`210699618b47cae309efc601398bfb171ed37248675d1436af2363c04735eb4a`.

## Remaining test blockers and proposed next live batch

B09's B01 historical readiness command `af-6a99dd57-eb7d-43a7-bc1c-7fab7cd90203`
is locally unreferenced but currently ARM provisioning Succeeded / execution
Pending, with no start/end times. Exit code 0 alone cannot qualify it. Do not
weaken admission to reach a removal confirmation or delete the command.

For a next B03 endpoint-only **other-cloud selection** trial, the accepted
Neo4j 5.26 IP-only clone `af-op-n526-source` is the candidate source; never call
this actual third-party-cloud infrastructure. Its retained public CA has expired:
notAfter `2026-09-22T04:38:18Z`, independently read from the local public certificate.
Do not bypass TLS validation or start with that expired trust chain.

Proposed, **not yet authorized or started**: a new maximum two-hour live window,
same cumulative USD800 ceiling, renew only this private fixture's TLS material
without changing the database password or network exposure; use a fresh GUI
workflow/runner/private target, full 5.6M-row migration and all 64 canonical ranges,
then stop the source/runner/target. Before any start, refresh ownership, cost,
capacity and exact certificate/source state, pin the artifacts and set an absolute
shutdown time. New resource/RBAC/unpublished-artifact action-time approvals and
private source-password entry remain separate gates. Never overwrite accepted
graphs, replay accepted jobs or use source ARM discovery in the customer flow.

This proposal does not close B03 or authorize new resources. All branch statuses
remain 5 pass / 7 partial; nine base routes remain qualified.
