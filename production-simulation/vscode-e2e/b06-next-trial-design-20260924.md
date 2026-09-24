# B06 next trial: bounded access observation with useful failure evidence

Status: local preparation only. Root created the normal GUI draft
`b425c32a-3ba3-44bc-8415-a48114993c5c` at 05:19:31.788 UTC, with planned VM
`af-b425c32a3ba344bc8415` in Japan East zone 1. No actual VM principal, approval
time or execution deadline has been allocated for these preparations. No cloud
operation or probe was run by this preparation task. The previous trial's deadline has passed. Its one inconclusive
pre-grant attempt remains consumed; subsequent permission to continue cannot
extend that expired trial implicitly. Bind a new normal GUI draft and an explicit
new scope after the previous trial's cleanup is proven.

## Established issue and remaining uncertainty

The first observer required the managed-identity access token to be a three-part
JWT and required several decoded claims. The wrapper also replaced its safe
failure reason with a generic error. The observed receipt establishes neither
the actual failure cause nor whether the Cosmos request was attempted. Local
success with a fabricated JWT does not establish the live token's format.

Microsoft instructs client applications to treat access tokens as opaque and
leave their validation to the resource API. Parsing claims is therefore an
unnecessary client-side compatibility assumption. This is a reason to improve
the next probe, **not proof that a token-format difference caused the failure**.
[Microsoft access-token guidance](https://learn.microsoft.com/en-us/entra/identity-platform/access-tokens)

The proposed probe keeps its token in memory and checks only the documented
IMDS response envelope: a bounded nonempty token, Bearer token type, requested
Cosmos resource, and a numeric future expiry. It does not decode, hash, retain
or output the token, claims or token-response body. The IMDS request selects the
expected managed identity using the documented `object_id` parameter.
[Microsoft VM managed-identity endpoint contract](https://learn.microsoft.com/en-us/entra/identity/managed-identities-azure-resources/how-to-use-vm-token)

## Identity evidence without inspecting token claims

Before each command, root retains a fresh exact VM ARM GET proving the normal
draft's VM ID, ownership tags and sole `SystemAssigned` identity; its principal
must equal the approved principal and no user-assigned identities may be present.
Root checks that no identity mutation, other worker or competing command is in
progress, and records the nonsecret receipt hash. This is an actual prerequisite,
not a caller-supplied hash treated as independent proof.

Inside the VM, one fixed IMDS resource-ID GET must match the approved VM before
the token GET. The token GET requests only the Cosmos resource and the approved
principal's `object_id`. There is no automatic fallback to an unselected identity
if this request fails. Only then may the single Cosmos query be attempted.
Cosmos validates the access token. The receipt labels its identity evidence as
`fresh-arm-system-identity + guest-imds-resource-id + selected-imds-object-id`;
it must not claim local JWT signature or claim validation.

For genuine RBAC denial, retain the existing strict service-response classifier:
403 alone is insufficient. The service must explicitly identify required RBAC
permissions, the same approved principal and a DocumentDB action. Generic 403,
401, network errors and metadata errors remain inconclusive. Successful access
requires the fixed query's exact constant result, `Documents: [1]`.

## Finite observation and diagnostic policy

Each approved invocation allows at most **two IMDS GETs and one Cosmos POST**:
one VM-resource-ID lookup, one selected-identity token request, and one fixed
read-only query `SELECT TOP 1 VALUE 1 FROM c`. This extra metadata lookup must be
included in the fresh scope. No retries, continuation, arbitrary URL, raw source
output, credential fallback or other query is permitted. Disable proxies and
redirects; require normal HTTPS certificate validation for Cosmos. Bound each
network call, the response sizes, total guest runtime and receipt output. Keep
the existing maximum 60-second command runtime with enough outer-timeout margin
for a final safe receipt.

The observer itself owns the exception boundary and receipt serialization; a
second wrapper must not reimplement its error handling. The wrapper verifies
script/config seals, imports the reviewed code, and calls its single bounded
entry point. This entry point must cover initialization as well as network calls.
No `__file__` dependence is allowed in the in-memory execution path.

Retain a compact flushed stage event before each potentially blocking operation
and one final receipt. Use only these enumerated stages:

| Stage | Safe evidence | If it fails |
|---|---|---|
| `config` | script/config hashes and `configValidated` | Refuse before any HTTP attempt. |
| `vm-metadata` | `attempted`, `responseReceived`, `vmMatches`; optional numeric HTTP status | No token request or query. |
| `token` | `attempted`, `responseReceived`, `envelopeValid`; optional numeric HTTP status | No query; no token/body/hash/claim details. |
| `query` | `attempted`, `responseReceived`; allowlisted Cosmos status/activity ID | No retry; classify only the retained single response. |
| `complete` | recognized classification or `inconclusive`, safe reason, counters | Never infer RBAC denial from an earlier-stage failure. |

Set `attempted` immediately before emitting the pre-dispatch event and
`responseReceived` only after obtaining a complete bounded response. Attempts
count conservative pre-dispatch intents, **not function calls, packets sent or
server receipt**: interruption while emitting the event may prevent even the
transport invocation. A timeout after an attempted query leaves
service delivery unknown. If only an intermediate event survives, the last stage
is known but completion remains unproven. No output at all proves nothing about
which operation ran. Use `queryAttempts` and `queryResponsesReceived`, never an
unqualified `dataPlaneRequests: 1` on a failed or unknown execution.

Reasons are fixed enums such as `config-invalid`, `deadline-expired`,
`vm-metadata-http`, `vm-binding-mismatch`, `token-http`, `token-envelope-invalid`,
`transport-timeout`, `redirect-refused`, `response-limit`, `query-unclassified`,
`runtime-unavailable` and `internal-error`. Raw exceptions, HTTP error bodies,
tokens, claims, token hashes, arbitrary header values and source values never
enter a stage event or receipt. HTTP status and booleans are diagnostic evidence,
not permission to broaden the request or repeat it.

## Binding and new-trial acceptance

The next implementation must accept a sealed schema rather than contain the old
trial's identifiers or dates. Populate it only after the new GUI draft and actual
ARM identity observation exist: workflow/VM/source IDs, principal/tenant,
database/container, phase, actual approval time, actual first VM intent, the
earlier of that intent plus the approved compute allowance and absolute expiry,
and the retained ARM identity receipt hash. Validate the production-derived VM
name against the actual workflow. No placeholder config may execute.

Propose at most two pre-grant attempts (initial plus one manually reviewed
diagnostic follow-up) and three manually reviewed post-grant attempts in the next
exact scope; root must finalize those counts with the new resources and user
approval. Stop pre-grant after a recognized denial. Each command gets a unique approved resource ID and
a create-only durable intent. Unknown results consume the attempt and allow only
GET reconciliation. The second pre-grant diagnostic needs explicit inclusion in
the new approved scope and a manual review of the first failure. It is never an
automatic retry, even if the first attempt appears to have failed before the query.

Before native grant, require an actual recognized same-principal RBAC denial.
If the pre-grant attempt is inconclusive, stop this branch for diagnosis without
granting merely to proceed. After the genuine native grant, preserve matching
VM/principal/source/query bindings and record each manually bounded observation.
Stop at the first success. If it succeeds immediately, report no propagation
delay observed; do not manufacture one. Then require a genuine normal GUI
inventory before claiming application readiness. Standalone query observations
are never relabeled GUI inventory evidence.

The finite live trust and principal-change cases retain their separate evidence
requirements. The prior genuine Restricted Mode refusal may be referenced as
retained evidence for that exact installed candidate, with its provenance and
scope stated. Unit tests of trust loss across awaited boundaries remain unit
evidence. A fresh trial's actual principal rotation/refusal still needs the held
reviewed source panel, real native action and preserved causal sequence.

## Offline proof required before approving final bodies

Execute the exact final generated payload under blocked real network/process/file
access and inert transport. Cover an opaque non-JWT token, a JWT without the old
optional claims, malformed/missing envelope fields, wrong IMDS VM, selected token
endpoint failure, expired approval at every boundary, timeout before and after
query invocation, same-principal RBAC denial, unrelated 403, constant success,
unexpected sensitive error and wrapper/module initialization failure. Verify
stage flags, maximum two GETs/one POST, no replay, bounded output and absence of
all injected secrets. Check the corrected entry point under the actual guest's
Python version compatibility constraints without reading guest credentials.

The new standalone observer and NOEXEC builder are implemented separately as
[`b06-observe-access-v2.py`](scripts/b06-observe-access-v2.py) and
[`prepare-b06-observer-v2.py`](scripts/prepare-b06-observer-v2.py). Their strict
schema is the implementation's `FIELDS` set, including phase/attempt; the builder
takes all other fields from an actual approved binding and matches the nonsecret
ARM identity receipt before writing any request artifacts. It prepares at most
two before and three after commands, named `af-b06-v2-before-01/02` and
`af-b06-v2-after-01/02/03`, all on the bound VM. No launcher is included.

The observer uses 10-second transport timeouts and a 55-second process alarm
inside a 60-second command. It rechecks the authorization window before and after
each bounded response, so expiry during a call retains the attempt/response
evidence but produces an inconclusive result. The exact payload's initialization
fallback emits fixed text with unknown request counts, never a fabricated zero.

Thirteen offline tests pass in
[`test_b06_access_v2.py`](scripts/test_b06_access_v2.py), including execution of
the exact generated before/after payloads with opaque tokens, query timeout and
module-initialization failure. Real HTTP, process launch and guest-file access are
blocked in those payload tests. Additional cases cover malformed envelopes,
identity mismatch, scope/attempt validation, expiry before and after each
response, strict same-principal read denial, nonqualifying errors, constant result
checking and builder ownership/no-overwrite. JSON Lines output remains at most
4096 bytes, including all stage events and the final result.

The new observer, builder, config, request bodies and dispatcher require fresh
immutable hashes and independent review. This document does not authorize
execution. The installed extension and the prior trial's sealed scripts, configs,
receipts and journals remain unchanged.
