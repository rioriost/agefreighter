# Installed candidate: Cosmos wording, cancellation and retained-failure refusal

September 17, 2026, approximately 06:01–06:10 UTC. Follow-up to the
[Cosmos approval audit](cosmos-access-approval-audit-20260917.md).
The user's continuation authorized the previously presented candidate
installation/reload and display/refusal/Cancel-only checks. No Azure permission,
network, compute-start, guest-execution or migration action was approved or
performed by this batch. Budget and deadline are unchanged.

## Installed artifact

Source baseline: `b559150881701f415019415e6c1abffeab98d90e`.
The normal, unpublished 2.4.0 VSIX was installed through the VS Code CLI and
the actual desktop was reloaded with Developer: Reload Window.

- VSIX SHA-256: `ceb7cc74ca1f60723fb693b659b0ead1d4e2b25631979375252cc3a7db1a5165`.
- Installed bundle SHA-256: `34dc9fc71284d7a356aa52f40b03772a06197b40887dfd0ade79b4216707d91e`.
- Prior installed bundle: `3061aea43c315113f47df3ad1da31dc68ad73c27b47a18571ddfb66bf742e929`.

The installed bundle matched the packaged candidate. The existing Azure
session loaded without a second sign-in. This reload occurred with terminal
workflows and stopped compute; it is **not** an active-operation reload/crash
qualification for B10.

## Actual signed-in GUI observations

Reconnected accepted Cosmos workflow `d138f4e4-bcf3-40fe-a876-ee9ce062e08a`.
Configure source & assessment displayed the corrected explanation: the runner
uses its managed identity, account keys are not a guided-flow option, ARM
`ready` does not establish data-plane propagation, and only successful
assessment proves source reads. The saved ready grant and nine vertex/nine
edge mappings remained present. No grant, form edit or assessment was submitted.
This confirms the installed wording, **not** live changed-principal, trust-loss
or data-plane denial/propagation behavior; those gaps remain in B06.

The following four real native dialogs were opened from the reconnected
workflow's Continue / verify Linux migration action and cancelled:

| Action | Native confirmation | Result |
|---|---|---|
| Inspect same-job recovery | Inspect the retained checkpoint without resuming? | Cancel; returned to retained workflow |
| Diagnose retained target | Read-only diagnosis of the retained target? | Cancel; returned to retained workflow |
| Archive empty-target preparation failure | Archive this preparation failure without deleting or resuming anything? | Cancel; no archival |
| Transfer / open migration verification | Transfer and verify the retained migration report? | Cancel; no transfer |

The transfer dialog bound job `d5edef98-bb51-4040-b6ed-0274e252de26`, 9,619
bytes and SHA-256
`f1824243ff13d1cc2f44493c47151a68d30be5d92b353dc87347b5326624c938`.
No source-password prompt or approval submission followed any cancellation.
These are cancellation-branch observations, not a claim that an accepted
terminal workflow qualifies for approved recovery, diagnosis or archival.
Together with the prior resize Cancel, five execution approval surfaces now
have installed-GUI cancellation evidence; B09's other surfaces/faults remain.

Separately, the command-palette Continue / Verify Linux Migration action
selected failed Cosmos workflow `7b79f05d-1dc1-40a6-b3dc-6c8129d4e0c1`, then
Qualify / reconcile full P1 digest (development only). The actual error was:

> Review retained qualification failure; no automatic retry.

No success panel, retry or new PASS state appeared. Its retained P1 phase
remained `failed`, bound to job `7fa558e4-8027-4335-9a2b-564f70b3df02` and
operation `c1607b3e-32e6-42c6-a3b6-91be8d95e70f`. The historical counts pass
did not override the independent full-digest failure. No synthetic report or
workflow was inserted. This supplies B12's installed retained-failure refusal,
not live coverage of every corrupted/forged report or failed transfer case.

## Preservation checks and limits

All 66 root workflow/report JSON files were byte-identical before and after
these observations. At `06:01:32.332Z`, `06:06:24.579Z` and `06:10:04.025Z`,
the aggregate SHA-256 was:

`819cdcd7c1bae30b57cb669fcd0f7a2d45172c912072e05219f9197b33c28e9f`

Reproduction: include every root `.json` in the installed extension's
`runner-v2` storage directory, sort filenames with JavaScript `localeCompare`,
then hash the concatenation of `filename + NUL + SHA256(file bytes) + LF`.
No JSON files were excluded. Live read-only resource checks around 06:01–06:03
UTC confirmed all nine retained VMs deallocated and all 17 Flexible Servers
Stopped. No infrastructure was started or changed.

The no-write evidence consists of actual Cancel/refusal observations, unchanged
saved artifacts and reviewed production early-return paths; this was not a
packet-capture experiment. No new guest-health, billing or source-read claim is
made. Prior 292/292 unit results apply to this unchanged installed candidate;
they were not rerun as new live tests. Overall release qualification remains open.
