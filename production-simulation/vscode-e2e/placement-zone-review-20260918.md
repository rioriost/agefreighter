# Placement selection follow-up — September 18

## Installed GUI observation

The user requested continuation. This batch opened the installed signed-in
wizard without a workspace or desktop CLI and used only read-only catalog and
candidate discovery in the approved subscription and dedicated trial group.

- Source RG selection populated the migration RG with the same group.
- Actual candidate discovery listed the retained VMs, including `af-n44-source`.
  Candidate selection filled its exact ARM ID and selected Japan East.
- Availability zone displayed `1`. Because the old view already defaults to
  `1`, this observation alone does **not** prove zone inference from source ARM.
- Japan West is present in the region dropdown, but attempts to select it were
  repeatedly interrupted by the UI tool reporting a changed application state.
  Region override acceptance was not established. No prerequisite preview,
  deployment, source configuration save, assessment or worker was submitted.
- Separate-RG, independent-network-RG and invalid-choice live GUI acceptance
  remain open. The interrupted dropdown is not a successful test.

## Reproduced defect and correction

The actual generated webview script was executed using the existing inert DOM
test adapter. Three new regressions failed before the correction:

1. Selecting a source with known zone `2`, then a source with an unknown zone,
   left `2` selected without indicating that the new source zone was unknown.
2. Changing source type/location or placement could retain the previous zone.
3. A source's logical zone was copied into another subscription, even though
   that does not establish physical-zone equivalence. The backend already
   rejects unproven cross-subscription placement; the UI was misleading.

The view now starts with an explicit zone-selection placeholder. Source type,
location, group, manual identity, candidate, runner subscription and manual
region changes clear stale zone selection. Only a supported source zone in the
same selected subscription and region is automatically proposed. Cosmos account
metadata is not treated as VM-zone evidence. Unknown or unsupported source zones
require the operator to select a runner zone explicitly, with explanatory text.
Preview and new-draft configuration remain disabled until a zone is selected.

Restoring a retained workflow preserves its exact saved zone and labels it as
saved, not newly inferred. Backend source/subnet/SKU/zone/quota checks and all
deployment approvals remain unchanged. Explicit selection is not proof of
source co-location and does not override backend rejection.

## Validation and preservation

- **389/389 unit tests PASS**, TypeScript/build PASS. Nine added tests cover
  unknown/unsupported zones, selection changes, subscription mismatch, Cosmos,
  unavailable source regions, deselection and the initial empty selection.
- Existing **13/13 isolated native Extension Host tests PASS** on VS Code
  1.138.0 arm64. These are not the signed-in interaction retest of this fix.
- All **70 operator files / 19 workflows** are byte-identical; aggregate
  filename/content-hash SHA-256 remains
  `fcc85c6021b1191d9207a9fe93eb661cd1665edebbf83e5d33a4602d87f248ef`.
- Installed bundle remains the previous approved candidate, SHA-256
  `965a4c4422ff2816668611d937123f9f05377b59bb2cd848f72a29222e74f6a3`.
  No Azure mutation, credential/RBAC change, new workflow, or accepted migration
  modification occurred. No current cloud power/billing refresh is claimed.
- B01/B02 stay partial. Next step is approval/install of the pinned correction,
  then actual known/unknown zone and override interaction checks without cloud
  mutation. Live provisioning/active-worker and removal acceptance are separate.

## Pinned candidate for installation approval

- Source commit: `1b929bd`.
- VSIX: `production-simulation/work/vscode-zone-review.U303jI/agefreighter-1b929bd-zone-review.vsix`.
- VSIX SHA-256: `ccf2d695c1bc34e18b77de423c9357f867bc56f5df4510b50adfe91240b4e49f`.
- Bundle SHA-256: `141b1e1bf2181bb76fe61194ba609d7edd2c7a34b3143ab4d0288f648d2cd763`,
  matching the extracted package member and local build.
- Packaging reran typechecking, all 389 unit tests and build successfully.
  No Marketplace publication or installation of this candidate has occurred.
