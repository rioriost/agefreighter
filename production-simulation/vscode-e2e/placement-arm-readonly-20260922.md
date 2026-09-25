# B02 real ARM read-only placement checks

Checkpoint: 2026-09-22 02:32:12 UTC. **Six checks pass; B02 stays partial.**
This batch runs unchanged production `preflightRunner` with fresh Azure ARM
responses supplied by a GET-only Azure CLI adapter. It is not installed-GUI
placement qualification, deployment, guest execution or source-data access.

## Scope and method

- Branch: `codex/2.4.0-guided-migration`, baseline `d0879fe`.
- Subscription: approved dedicated trial subscription ending `fdb7`.
- Existing migration group: `rg-af-vscode-p1-b01-20260921`.
- Existing source/network group: `rg-af-vscode-p1-20260905-a`.
- Existing VNet/compute subnet: `vnet-af-vscode-p1` / `runner`.
- Actual source metadata: `af-n44-source`, Japan East, zone 1.
- Discovery choice: `Standard_B2s_v2`; only metadata/quota availability is read.
- No startup, deployment, what-if, source query, password, grant, network,
  policy-tag, operator-record or accepted-data mutation.

The opt-in harness `extensions/vscode/src/test/runnerPlacement.live.ts` is not
part of default tests. It requires `AF_B02_LIVE_READ_ONLY=1`, permits only GETs
to exact reviewed management paths, bounds pagination and command time, refuses
persistence/polling and stops on unexpected outcomes. It does not print tokens,
raw Azure failures or database data. Future execution still requires checking
that the retained trial resources remain in scope; this is not a startup permit.
Running without the opt-in was separately checked: exit 1 at the initial guard,
before creating an evidence directory or making any Azure request.

## Observed results

| Case | Production decision using actual ARM response | GETs |
|---|---|---:|
| Valid existing private placement | Accepted read-only preflight | 6 |
| Nonexistent compute subnet | Actual 404; selected subnet does not exist | 1 |
| Actual PostgreSQL-delegated subnet | Refused; use a non-delegated compute subnet | 1 |
| Japan West runner with actual Japan East VNet | Refused; runner region must match VNet | 2 |
| Nonexistent migration resource group | Actual 404; select an existing resource group | 3 |
| Zone 2 runner with actual zone 1 source VM | Refused; select source availability zone | 6 |

All six expected outcomes match, with 19 GETs and zero mutations. Synthetic
missing names are lookup-only fixtures, not created resources. Successful quota
and SKU reads do not qualify a quota-exhaustion or unavailable-SKU rejection.

Local private result: temporary directory `af-b02-arm-readonly-1zWj7E`,
`result.json`, mode 0600. SHA-256:
`95d44163233bd4c0b7dbc839d9561d9982371425b978b5ed5f608ecc8395402a`.
The sealed report retains per-case outcomes and GET path/status traces.

## Installed GUI boundary and small wording correction

The installed extension remains the previously approved `a4b61d8` runtime;
bundle SHA-256 `8f0061a16bf2d42abe1a41cd6f805697316f70286a905c20e95d56b80fdd2a8e`.
Opened a fresh unsaved wizard in VS Code 1.138.0. The account/catalog loaded,
and the source-location menu was observed. Subsequent screen capture failed
with ScreenCaptureKit code -3811, including a second observation attempt.
No new preview outcome or successful wizard-close action is claimed. No source
configuration or deployment action was submitted from this new form.

Code inspection also shows that a fresh production preview resolves matching
Linux release/checksums before backend preflight. This batch did not bypass
that release gate, rebind a retained development workflow or install software;
it does not claim to have newly observed the release error in the GUI.

The visible guidance still incorrectly required the VNet in the migration
resource group, despite completed B01 cross-group qualification. Corrected only
that wording to permit a reviewed other-group VNet while retaining assessment,
two-scope review and approval. Added a rendering regression. Typecheck, all
458 unit tests and build pass. This source change is **not installed** and its
updated appearance has not been visually verified.

All 82 pre-existing operator files remain byte-identical (sorted filename,
NUL delimiter and file-byte SHA-256):
`210699618b47cae309efc601398bfb171ed37248675d1436af2363c04735eb4a`.
No fresh VM power-state or cost claim is made by this read-only metadata batch.

## Remaining B02 acceptance

All 13 current trial VMs inspected have zone 1; none supplies the unknown-zone
VM transition case. Do not remove a zone or create a new VM just to manufacture
this evidence without a separately scoped plan/authorization.

Still required: installed-GUI backend refusal paths, unknown-zone VM transition,
unavailable SKU and quota denial. Existing synthetic/unit evidence stays distinct
from the real-ARM checks above. Defined branch totals remain **5 pass / 7 partial**;
the nine base migration routes remain qualified, not all-branch release readiness.

## Follow-up: GUI recovered; release gate observed; ordering correction

September 22, approximately 03:00 UTC: normal signed-in VS Code 1.138.0 became
readable again. In the same fresh unsaved wizard, selected Neo4j/on-premises,
the approved subscription, existing B01 migration group, Japan East, zone 1 and
the existing `runner` subnet. No endpoint, source credentials or source-data
operation was used. Independent GitHub release lookup returned `release not found`.
One prerequisite-preview click visibly returned:

> The matching AGEFreighter 2.4.0 Linux release/checksums are not available. No Azure deployment was submitted.

This is actual installed-GUI release-refusal evidence, not backend placement
refusal. Closed the unsaved wizard successfully; all 82 operator files retained
the aggregate SHA above. No what-if, draft save, deployment or resource startup
occurred. The installed build and its mandatory release gate were unchanged.

Corrected the production message-handler order: after input/catalog/draft/version
validation, run the same GET-only `preflightRunner` before release lookup. Invalid
placement can now be explained without being hidden by a missing release. Release
version/checksum and development-artifact opt-in/binding guards still gate pricing,
what-if, record persistence and approved deployment. Submission still performs its
own fresh preflight; no protection was removed and no failed operation is replayed.

Added five production-handler tests with inert UI/Azure/storage/fetch adapters:
placement rejection prevents release fetch/effects; invalid catalog prevents both;
missing and malformed checksums prevent pricing/what-if/persistence; only matching
checksum after valid placement reaches pricing. These test event ordering, not
live Azure metadata or GUI interaction. Initial invalid-catalog fixture used a
parser-invalid region and was corrected to a parser-valid unknown catalog value.
Final typecheck, 463/463 unit tests and build pass. The existing isolated real
VS Code host suite also passes 25/25, without using the operator account/store.

The source fix includes the previous cross-group explanatory text correction.
Installation and real signed-in GUI backend refusal tests remain pending; B02
and overall branch counts are unchanged. No Marketplace publication is implied.

Packaged candidate: source commit `7338faa`, VSIX SHA-256
`3814b4f11ce6f121c2fe8843cea7e960cad72095a6710b3656b43a2f3f361245`,
compiled JavaScript SHA-256
`3f4d147925709b74f861fd65cf2a4a883c81e15c2b46dd62b73a338bfd8a0dcf`.
`npm run package` passes, including its fresh 463-test/build prepublish checks.
Requested action-time permission to install this local, unpublished VSIX and
reload the normal VS Code window, followed only by read-only negative tests.
No installation has been performed at this checkpoint.

## Follow-up: approved installation and four installed-GUI refusals

September 22, approximately 04:58–05:07 UTC: after explicit user approval,
installed the exact `7338faa` VSIX above and executed Developer: Reload Window.
Installed JavaScript SHA matches `3f4d14792570…` in full. The normal signed-in
VS Code wizard visibly shows the corrected two-scope VNet guidance, including
assessment, review and approval; this closes its previous visual-check gap.

Submitted four independent prerequisite-preview checks through actual GUI fields:

| Input | Observed installed-GUI result |
|---|---|
| Neo4j/on-premises, Japan East/zone 1, lookup-only nonexistent subnet `af-b02-nonexistent-20260922` | `The selected subnet does not exist.` |
| Same placement, actual delegated subnet `afpg-5cb990c12a254de5a10d` | `Use a non-delegated compute subnet, not the Flexible Server delegated subnet.` |
| Actual `runner` subnet in Japan East, runner region Japan West/zone 1 | `The runner region must match the existing VNet.` |
| Azure Neo4j source ARM ID `af-n44-source` (actual zone 1), runner Japan East/zone 2 | `Select the source availability zone for the runner.` |

The last case used the visible source ARM-ID field; no candidate discovery or
database connection was needed. It verifies server-side source-zone admission,
not unknown-zone VM proposal behavior. All actual preview submissions used the
approved trial subscription, B01 migration group and original network group.
Each displayed a specific placement refusal instead of the missing-release error.
No preview record, what-if, deployment, source configuration, VM/DB startup or
credential/RBAC/network change occurred. This conclusion combines observed GUI
outcomes, matching installed code ordering and unchanged operator files, not a
network capture. A native menu initially selected another subscription and
triggered automatic group/region listing; that selection was corrected before
any preview or resource action, and was not used for a qualification case.

The user interrupted the turn after these four results. On continuation around
05:10 UTC, the old wizard was no longer open. Opened a fresh unsaved wizard for
the valid-placement/missing-release control, but native menu actions repeatedly
reported external UI changes and did not reliably select the target subscription.
Stopped sending further actions. No success, cancellation or close is claimed for
that unfinished menu interaction. The post-fix valid-placement release refusal
remains pending in the real GUI; it is covered by handler tests, not by these
four negative cases. GitHub's fresh release lookup still returned release not found.

At 05:14:46 UTC, all 82 operator files still match the original aggregate SHA
`210699618b47cae309efc601398bfb171ed37248675d1436af2363c04735eb4a`.
No source code changed in this installation/test batch; the candidate's existing
463-unit/25-isolated-host/build/package results are retained, not newly rerun.

B02 remains **partial**, now with four real installed-GUI backend refusals rather
than only direct preflight evidence. Outstanding items include unknown-zone VM,
unavailable SKU, quota denial and the post-fix valid-placement release control.
No quota exhaustion, unavailable capacity or unknown source metadata was induced
by mutating live resources. Branch totals stay 5 pass / 7 partial.

## Follow-up: post-fix valid-placement control completed

September 22, approximately 05:17–05:19 UTC: resumed the same unsaved form,
confirmed the approved subscription, B01 migration group, Japan East, zone 1,
`Standard_B2s_v2` and actual `runner` subnet. The source choice was
Neo4j/on-premises with no endpoint or credentials entered. Submitted one new
prerequisite preview through the installed GUI. Its final visible result was:

> The matching AGEFreighter 2.4.0 Linux release/checksums are not available. No Azure deployment was submitted.

With the verified installed `7338faa` handler ordering, this proves the real
placement preflight returned successfully before the mandatory release refusal.
GitHub independently still reports no v2.4.0 release. No pricing/what-if,
preview persistence, deployment, guest action or source connection followed.
Closed the unsaved wizard and observed the remaining empty Untitled editor.
All 82 operator files retain aggregate SHA
`210699618b47cae309efc601398bfb171ed37248675d1436af2363c04735eb4a`.

Read-only Azure capacity inspection explains why the remaining denial cases
cannot be honestly claimed in this existing placement:

| Current Japan East choice/quota | Observed state |
|---|---|
| `Standard_B2s_v2` | Zones 1/2/3, no subscription restrictions |
| `Standard_D2s_v5` | Zones 1/2/3, no subscription restrictions |
| `Standard_D4s_v5` | Zones 1/2/3, no subscription restrictions |
| Total regional vCPUs | 74 / 101 (27 available) |
| `standardBsv2Family` | 8 / 100 (92 available) |
| `standardDSv5Family` | 64 / 100 (36 available) |

These are control-plane availability/usage observations, not a guarantee of a
successful VM allocation. All three visible discovery choices fit the observed
quota; none supplies an unavailable-SKU denial. No resources were created to
exhaust quota, no quota was lowered, and no responses were substituted in GUI
tests. Direct unit-test rejection cases remain a separate evidence level.

The installed-GUI batch is now **four placement refusals plus one valid-placement
release-refusal control complete**. Remaining B02 live gaps are unknown-zone VM,
unavailable SKU and quota denial. Those require a suitable scoped fixture or
explicitly labelled isolated fault tests, not repeating successful previews in
the current environment. B02 stays partial and totals stay 5 pass / 7 partial.
