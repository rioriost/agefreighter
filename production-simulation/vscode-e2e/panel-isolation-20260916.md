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
