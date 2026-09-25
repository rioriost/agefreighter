# B02 placement-to-panel isolated contracts

Date: 2026-09-22. Baseline: `8446c18` on `codex/2.4.0-guided-migration`.
Outcome: **13 added isolated contracts PASS; B02 remains partial.**

## Method and scope

Extended `runnerPanelLifecycle.test.ts` to connect the actual preview message
handler to unchanged production `preflightRunner`, input validation and SKU/quota
parsers. ARM replies are synthetic; UI, storage and release fetch are inert test
adapters. Exact expected read paths, subscription, GET method and absent request
body are asserted. Unexpected requests fail the fixture. This is not an actual
VS Code Extension Host, signed-in GUI, live ARM refusal or deployment test.

No production code, installed extension, operator records or cloud resources
were changed. No credentials, source data, Azure calls, what-if, VM allocation,
quota changes or denial-inducing infrastructure were used. The installed build
and the preceding real-GUI evidence remain unchanged.

## Cases

| Case | Expected and observed result |
|---|---|
| SKU missing from response | Refused before quota lookup or release fetch |
| SKU restricted in selected region | Same refusal |
| Selected zone restricted | Same refusal |
| Regional quota one core short | Refused before release fetch |
| VM-family quota one core short | Same refusal |
| Regional quota absent | Same refusal |
| VM-family quota absent | Same refusal |
| Malformed quota value | Same refusal |
| Both quotas exactly sufficient | Placement passes; mandatory missing-release gate stops preview |
| Restriction applies only to another zone | Same valid-placement release refusal |
| Restriction applies only to another region | Same valid-placement release refusal |
| Source VM zone unknown; explicit runner zone supplied | Same valid-placement release refusal after source metadata check |
| Source VM zone unknown; runner zone omitted | Input rejected before any ARM or release call |

All denial cases assert the panel's expected error and cleared busy state.
SKU denials use four synthetic reads; quota denials use five. Positive controls
use five reads, or six with source VM metadata. All cases assert no persistence
or downstream effects; no preview record is posted after preflight/release refusal.
The fixture initially omitted the VM `properties` object and correctly failed
structured-response validation; correcting that synthetic response resolved the
test without changing production behavior.

## Validation and remaining work

`npm run check`: typecheck PASS, **476/476 unit tests PASS**, build PASS.
No package installation or isolated Extension Host rerun was needed for this
test-only change. Previous 25-host-test results are historical, not a new run.

These contracts strengthen coverage across the production handler boundary;
they do not replace remaining installed-GUI unknown-zone VM, unavailable-SKU or
quota-denial evidence. The preceding real environment has no suitable denial
fixture. Do not deplete quotas or modify qualified VMs to manufacture one.
Branch totals remain **5 pass / 7 partial**, base migration routes **9/9**.
