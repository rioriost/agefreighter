# New-wizard workflow isolation — September 16

During preparation for the independent network-source recovery trial, the
installed signed-in wizard reproduced a stale-selection defect. Closing the
accepted CSV wizard and opening **New guided migration** created empty Neo4j
fields while retaining the old host-side current workflow and CSV guest-ready
timestamp. Source/placement edits invalidated the visible preview, but several
host actions still used that retained workflow without a displayed-ID binding.
No cloud or guest mutation was made while reproducing this condition.

## Correction and review

Code commit: `e551c31` on `codex/2.4.0-guided-migration`.

- Current workflow, CSV selection and busy state now belong to each panel.
- An async operation from a closed panel can retain its own evidence, but its
  output cannot populate a new panel. Closed listeners reject new requests.
- Shared commands persist records without implicitly changing a wizard's
  current selection.
- Deployment, guest controls, target review and execution require the exact
  workflow displayed by the requesting panel. Missing/stale IDs fail closed.
- Changing source/placement clears readiness, disables retained-job controls,
  and cannot silently use the old host-side selection.
- Reconnecting remains explicit; account refresh restores the saved fields
  before displaying that workflow's saved readiness.

This does not reset/delete durable jobs, modify credentials, alter Linux
artifacts, loosen resume gates, or qualify desktop crash recovery. Accepted
CSV and base-route graphs remain untouched.

## Tests and installation

- TypeScript typecheck PASS; **218/218 unit tests PASS**.
- Seven added cases cover workflow binding, actual production message-handler
  lifecycle with inert adapters, late async completion, missing-ID actions,
  changed input and webview invalidation. They do not constitute Azure tests.
- Current VS Code 1.137.0, isolated Extension Host: **13/13 PASS**, exit 0.
- Packaging reran typecheck, all unit tests and compilation successfully.
- Installed in the normal Mac profile and reloaded through the VS Code GUI.

| Artifact | SHA-256 |
|---|---|
| Qualification VSIX | `c07091e4a0bed80de5ee2811fce8bacaa25218a6e085f4d79b288b8d0aa11844` |
| Bundled and installed JavaScript | `323e6a9e0ecf1906b82358ba16a3a4a1f8c574017c1cd3cc18d370e197c06bee` |

Signed-in GUI retest: explicitly reconnected to accepted CSV workflow
`54da6ddd-27d2-45e0-bb68-cf5f352801db`, saw its saved readiness, closed it and
opened a new wizard. The new source is Neo4j, guest readiness is unchecked and
migration execution is disabled. The saved workflow was not automatically
selected. No cloud operation was submitted.

Before and after this retest, all **14** persisted workflow JSON files have the
same aggregate SHA-256 (sorted filename followed by exact bytes):
`6d60c1807976792b12fba13f29eba75048d211de7e4c86b4165ee93b15cec8c8`.
This covers those workflow files, not a claim that all VS Code application
storage is immutable. Fresh Azure reads at approximately 05:02 UTC confirmed
all six retained VMs deallocated and all fifteen Flexible Servers stopped.

## Remaining

Network-source recovery remains unqualified and has not been started. Prepare
a fresh source/runner/target binding, then review a narrowly scoped reversible
connectivity fault with automatic restoration before injecting anything.
Retain the USD 800 ceiling and September 20 07:14:35.311 UTC outer deadline;
do not disable the existing daily shutdown implicitly. B09/B10/B11/B12 remain
partial/open according to their individual acceptance requirements.

## Next network-source draft

The installed GUI subsequently created a new local draft, separate from all
accepted jobs:

- Workflow: `8a9ae99e-c621-4a94-afd1-a30ff210a201`.
- Name: `neo4j526-network-recovery-p1-r1`; namespace: `n526_recovery`.
- Azure candidate: existing `af-n526-source`, selected from the trial group's
  actual resource list. The GUI defaulted placement to Japan East / zone 1.
- Existing runner subnet; initial proposed size B2s_v2. No deployment preview,
  price approval or runner creation has occurred yet.
- Stable vertex/edge key: `source_key`; TLS CA selected through the file dialog,
  1,513 bytes, SHA-256
  `0b196d1e310a5a68732879c7bf0aa8c9f1fc67bbfe436fe560a31de7afa05b68`.
- Source settings reviewed locally; no password entered, discovery run, target,
  migration or fault injection. A local `canAssess` draft flag is not proof of
  guest readiness or source reachability.

The native final storage confirmation is awaiting action-time approval for
`af8a9ae99ec6214a94afd1a3` and signed-in-user Blob Data Contributor on this
new account only. The endpoint is HTTPS network-public, authenticated, with
anonymous access/shared keys disabled; no source exposure is added. No storage
or role request has been submitted. The GUI permissions boundary was presented
to the user; it must not be bypassed through another control path.

An independent metadata read confirms the new record is only `draft`, with no
storage intent, guest command, target or migration. Excluding this new record,
the original 14 workflow records still have the exact aggregate hash above.
RG locks were empty and the CLI subscription ID matches the authorized trial.
Recheck health, deadline, budget and external governance before later cloud
mutations. Existing compute remains stopped; this is preparation, not recovery
qualification.
