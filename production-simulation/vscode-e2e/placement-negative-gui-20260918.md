# Placement negative GUI checks and implementation gaps

Checkpoint: September 18, 2026, 09:19 UTC. Installed extension source candidate
`f9e456e`, bundle SHA-256
`e3a8fb0aa518b8d8c339aef8aa1d3d861ccfdc82baa7fec32fc2afeef9964ac1`.
This batch uses the actual signed-in VS Code wizard, not a synthetic webview.
It is configuration/early-rejection evidence, not cloud deployment qualification.

## B01: independent migration group selection

In a new unsaved wizard, selected the approved trial subscription and source
group `rg-af-vscode-p1-20260905-a`. The migration group initially defaulted to
that group. Changed only the migration-group dropdown to the existing catalog
entry `rg-agefreighter-validation-jpe`; the source group remained unchanged.
Cleared and reselected the source group. The explicitly selected migration
group remained `rg-agefreighter-validation-jpe` instead of being overwritten.

No candidate discovery or deployment was directed at the alternate group.
It was a local dropdown selection only. Returned the migration group to the
dedicated trial group before the negative cases. This closes the observed
selection/default-preservation case, not separate-group provisioning.

## B02: installed input rejection

Selected actual `af-n44-source`, which proposed Japan East and zone 1.
Executed three deliberately invalid prerequisite-preview submissions:

| Input | Visible rejection |
|---|---|
| Subnet `not-an-arm-subnet` | `Invalid subnetId.` |
| Structurally valid synthetic subnet under a different subscription UUID | `The runner subnet must belong to the runner subscription.` |
| Structurally valid synthetic subnet under the selected subscription, but source ID `not-an-arm-source` | `Invalid resourceId.` |

The synthetic subnet names/UUID were rejection fixtures, not real resources or
deployment targets. For the third case, changing source identity cleared the
candidate and zone and disabled preview; zone 1 was then explicitly selected
before submitting. The parser still rejected the invalid source identity.

`runnerMigration.ts` calls `parseRunnerInput()` before catalog fetch, release
download, preflight, what-if or record persistence. `core/runner.ts` rejects
these exact cases there. The no-dispatch conclusion is based on observed UI,
matching installed code and unchanged retained files, not a network capture.
This does not qualify live nonexistent-subnet, delegation, quota, region/SKU
capacity rejection or arbitrary cross-subscription source placement.

## Corrected interpretation of remaining work

Two requirements in `plan.md` are implementation gaps, not just missing test
receipts. They must not be closed by replaying the accepted same-group/manual
mapping routes:

1. **B01 independent network resource group:** `targetPreview()` in
   `core/runnerTarget.ts` explicitly rejects a runner VNet outside the migration
   group. Its ARM template declares the delegated subnet in the deployment's
   own group. Accepting such a subnet on the initial runner form is therefore
   not proof the complete target workflow supports it. The existing guard must
   stay until correctly scoped network deployment/reconciliation is implemented.
2. **B04 PostgreSQL key/FK recommendations:** `core/runnerSourceView.ts` explicitly
   says schema/FK auto-discovery is not connected, and offers manually entered
   table, stable-ID, endpoint and property mappings. Retained full-P1 migrations
   qualify those reviewed manual mappings, not recommendation adoption/editing.

Next implementation work should address these before scheduling redundant live
trials. For independent network groups, bind the exact VNet/group to the reviewed
plan, validate both scopes and permissions, preserve source/network isolation,
and retain separate operation identities for any cross-group deployment. Do not
simply remove the current guard or treat an RG boundary as a peering requirement.
For PostgreSQL recommendations, add a bounded read-only catalog operation on the
Linux runner, retain its hash-bound result, and expose explicit adoption/editing
without overwriting existing reviewed mappings. Composite/nullable/ambiguous
keys and relationships need explicit unsupported/review outcomes, not guessed
identities. New live operations still require fresh budget, time and health gates.

## Preservation

Closed the unsaved wizard without saving a draft. All 70 retained operator files
remain byte-identical to the starting baseline, aggregate filename/content SHA:
`fcc85c6021b1191d9207a9fe93eb661cd1665edebbf83e5d33a4602d87f248ef`.
No VM/server start, cloud mutation, source read, password/RBAC change, workflow
save, migration or new verification result occurred. Azure operations were only
subscription/catalog listing and VM candidate discovery in the dedicated group.
No new cloud power/cost observation or Marketplace publication is claimed.

No code changed in this batch; previous 391 unit / 13 isolated host results are
historical validation of this installed candidate, not rerun results today.
B01/B02/B04 remain partial with the more precise distinctions above.
