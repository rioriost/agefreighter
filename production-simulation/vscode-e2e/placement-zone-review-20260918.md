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

## Approved installed-GUI checkpoint — September 18, 08:09 UTC

After explicit user approval, the exact VSIX above was installed through the
native **Extensions: Install from VSIX** picker. VS Code reported completed
installation; **Developer: Reload Window** was executed. Installed bundle hash
now matches `141b1e1bf2181bb76fe61194ba609d7edd2c7a34b3143ab4d0288f648d2cd763`.
This supersedes the earlier not-installed checkpoint, not its historical facts.

Before replacement, private backups were retained under
`production-simulation/work/zone-install-backup.aOJOCs` (not committed):

- Store archive SHA-256: `89ba27ba15c87c48a876bb8a24a80d283603873a3c7e7f069689ec7f2fb70ac0`.
- Extension archive SHA-256: `3cdbc8d4967151eba60cff6f5e18c16818437d8fc97bd85b3921c87a1590a83e`.

Actual signed-in GUI checks:

1. New wizard has **Select a zone after reviewing placement**, not zone 1,
   with explicit unknown-placement guidance and disabled preview.
2. Explicitly select zone 2; change Neo4j location from Azure to on-premises.
   Zone returns to the placeholder, with review-required wording.
3. Select zone 3; change source type to CSV. Location becomes local and zone
   returns to the placeholder. No CSV picker, upload or draft save is invoked.
4. Reconnect to retained Gremlin workflow
   `4043e008-b86e-47b8-8722-1efe637ae12a`. Japan East and saved zone 1 are restored,
   with wording distinguishing a saved zone from newly inferred source placement.
   Reconnection uses the retained record only; no refresh/worker action is clicked.

The 70 retained files / 19 workflows remain byte-identical, with the same
aggregate SHA-256 recorded above. No preview, deployment, worker, cloud control
mutation, credential or permission change was performed. These UI cases pass;
known/unknown Azure candidate transitions, region overrides, separate RG/network
and remaining live branches are not closed by this checkpoint. No new full P1
verification was run or claimed. Marketplace publication is unchanged.

## Candidate and override checkpoint — September 18, 08:27 UTC

Continued in the installed `1b929bd` candidate, using a fresh unsaved wizard
and the signed-in approved subscription. Read-only catalogs and candidates were
loaded from the dedicated trial group. Native menu actions initially lost their
accessibility handles; fresh observation and keyboard selection recovered the
interaction. Only the following visibly confirmed outcomes are accepted:

1. Selecting the trial source RG also proposes that same migration RG.
2. Selecting actual `af-n44-source` fills its ARM identity, Japan East and
   zone 1, with **Known source logical zone selected in the same subscription
   and region**. This time the wizard started with an empty zone, so the result
   establishes candidate-derived proposal rather than the previous default.
3. Manually select Japan West. The zone clears and preview becomes disabled.
   This is UI override/invalidation evidence, not backend permission to deploy
   outside the source region or proof of regional capacity.
4. Explicitly select zone 2 and click prerequisite preview with the subnet
   deliberately left empty. **Invalid subnetId.** is displayed. Inspection of
   the installed matching source confirms input parsing rejects this before
   catalog/release/preflight/what-if requests or a persisted preview record.
5. Switch to Cosmos NoSQL, discover and select actual `afcosmosp120260907`.
   The zone stays empty; account metadata is explicitly not used to infer the
   data region. The previously selected Japan West remains an unvalidated
   manual choice. No source assessment, preview or target operation is started.

The unsaved wizard was closed. All 70 operator files remain byte-identical
(same aggregate SHA-256 above). Installed bundle remains
`141b1e1bf2181bb76fe61194ba609d7edd2c7a34b3143ab4d0288f648d2cd763`.
No cloud mutation, credential entry, worker or new workflow occurred. Cloud
power/billing were not refreshed in this GUI-only batch.

### Follow-up display correction, not yet installed

The region override exposed another misleading explanation: **Source region
selected** remained visible after switching to Japan West and after changing
source type. Zone clearing/backend guards worked, but the explanation was stale.
Two added actual-webview-script regression tests failed before correction.
The local fix now replaces inferred-region wording on manual region, source
identity/type/location/group, subscription and candidate deselection changes.
A catalog refresh preserves the explicit-choice explanation. Saved workflow
restoration still uses saved-placement wording.

Typecheck/build, **391/391 unit tests**, and **13/13 isolated native host tests**
pass. No deployment, placement validation or authorization logic changed.
The display correction is not installed or live-GUI-qualified yet; installation
requires approval of its pinned VSIX. B01/B02 remain partial: unknown-zone VM
candidate transitions, cross-subscription/invalid placement preflight, separate
RG/network and live provisioning branches are not closed by these checks.

### Pinned wording-fix candidate

- Source commit: `f9e456e`.
- VSIX: `production-simulation/work/region-guidance.l66gfm/agefreighter-f9e456e-region-guidance.vsix`.
- VSIX SHA-256: `c6b2ec5cc2d5a54b96a506625fe055f03d07c0f1d110d60956a334bae60e8b04`.
- Bundle SHA-256: `e3a8fb0aa518b8d8c339aef8aa1d3d861ccfdc82baa7fec32fc2afeef9964ac1`,
  identical in the package and local build.
- Packaging reran typecheck, all 391 unit tests and build successfully.
  Not installed or published; next action is approval to install/reload and
  repeat the same read-only GUI checks, without Azure starts or mutations.
