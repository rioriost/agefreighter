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
