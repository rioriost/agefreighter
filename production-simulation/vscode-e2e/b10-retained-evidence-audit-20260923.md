# B10 retained evidence audit — September 23, 2026

**PASS for the frozen B10 scope. No additional inventory, host interruption,
or graph migration is needed.** This audit re-read retained local evidence; it
did not contact Azure, use the GUI, inspect credentials, or change private files.
It supplements the [finite acceptance definition](b09-b10-b12-acceptance-20260923.md#b10-the-one-required-active-cloud-crash-case--complete)
and [original qualification receipt](evidence/b10-live-active-crash-20260923.json).

The [machine-readable audit](evidence/b10-retained-evidence-audit-20260923.json)
passes all 43 checks. All eight sanitized executor receipts under the private
`/private/tmp/af-b09-b10-live-CM2Opa` directory still match their committed SHA-256
values. The exact normal-store workflow, its original report and its new report
were read without printing the private record. Both 2,947-byte report seals,
the SourceDraft seal, previous assessment history, finished assessment and
imported transfer agree with the committed evidence.

The supervisor also retained a create-only durable private copy of those eight
sanitized receipts under
`production-simulation/work/b10-retained-evidence-20260923/receipts/` (0700
directory,0600files, all original hashes verified). The original temporary
receipts and accepted store are unchanged. Running the same43checks and6negative
controls against that durable copy also passes. This backup and its manifest are
ignored by Git; the historical receipt still identifies its original sources.

A separate read-only hash check also matched all six retained native-lock seals:
Cancel/recovered verification receipts, original lock snapshot, current workflow,
current report and recovery archive. The original lock remains absent. This
supports continued retention of the earlier isolated native result; the new
43-check script is scoped to the active-cloud case.

The observer receipts bind the same boot, service invocation, main process and
inventory child before and after the recorded host exit. Both preserve active
process and bounded health proof; the post-crash child was not inferred merely
from persisted `running` state. Inventory, status and export receipts retain
the same operation/configuration. The current report contains complete source
checks, all 18 mapped labels, 1.6 million vertices and 4 million edges. The
19-control final list includes the five named additions and all Succeeded
states. The retained shutdown receipt confirms the two exact resource names
and stopped states before the hard bound.

Six negative controls on in-memory copies reject altered receipt bytes,
altered current/prior report bytes, removed history, missing imported transfer
and a changed guest-child start identity even with a newly calculated receipt
hash and unchanged output length. The 13 existing offline observer tests also
pass. These are evidence-integrity checks, not six new live fault trials.

Reproduce the audit with the explicit private receipt and normal-store paths:

```sh
python3 production-simulation/vscode-e2e/scripts/b10-validate-retained-evidence.py \
  --ledger production-simulation/vscode-e2e/evidence/b10-live-active-crash-20260923.json \
  --receipts production-simulation/work/b10-retained-evidence-20260923/receipts \
  --store '/Users/rifujita/Library/Application Support/Code/User/globalStorage/rioriost.agefreighter/runner-v2' \
  --check-rejections
```

The script only reads the specified retained files and prints check names and
booleans. Missing private evidence fails closed; a checkout alone cannot
reproduce the complete audit. The saved result contains no private record data.

The original limitations remain: native action/window binding, baseline control
count and live ARM observations are executor-attributed; timestamps do not
independently measure cross-machine clock offset; the control timeline is not
an instrumented HTTP count. No surviving lock existed in this live run, so
native stale-lock recovery retains its distinct earlier evidence. Inventory
continuity does not establish migrated-property/identity/endpoint correctness.
Stopped-state evidence is historical at its recorded timestamp, not a fresh
cloud observation by this auditor.

Stale actionable B10 instructions appeared in the bottom remaining
sequence of `remaining-validation.md`: its closed set omitted B10 and its trial
list included B10 despite the latest status and acceptance table saying PASS.
That sequence now preserves B10's credit and marks the prior bounded session
complete. The local lock-recovery document's top verdict now identifies its
09:59UTC checkpoint, and the active-crash preparation document links the later
completed result. Earlier chronological checkpoints and immutable receipts
remain historical evidence; their old pending states do not reopen the gate.
