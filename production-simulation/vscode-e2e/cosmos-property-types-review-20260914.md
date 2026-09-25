# Cosmos explicit types: implementation review and r2 handoff

Date: 2026-09-14 UTC. Scope: guided migration development branch, not a public
release. AZ-COSMOS r1 remains failed; overall GUI qualification remains **6/9**.

## Contract and review

The retained r1 graph differs from the frozen fixture by numeric representation:
the previously recorded full-fixture counterfactual reproduces its root after
40,175 integral-valued floats lose their float type. The fix does not change
the verifier or its acceptance root. Optional `propertyTypes` now flow through
the config/schema, compiled Cosmos mappings, iterator, source fingerprint,
Linux capability advertisement, and VS Code mapping fields/capability gate.

Review findings and disposition:

1. Compatibility: undeclared fields and empty declarations preserve legacy
   inference/fingerprints. A changed type rejects an old resume token.
2. Safety: no string-to-number coercion, fraction truncation, integer overflow,
   unsafe integer-to-float conversion, recursive flattening or source-value
   disclosure. Exact arithmetic has bounded input length/exponent.
3. Deployment: older runners fail closed before typed assessment/migration;
   read-only access to retained failure evidence remains available.
4. Test scope: actual Cosmos iterator plus JSON reserialization and shuffled
   bounded pages is an offline simulation, not Azure or GUI qualification.
   Both default tiny and full frozen P1 tests compare against unchanged roots.
5. New GUI fixture overrides only `score` and `distance_km` as `float64`; other
   fields retain inference. The original r1 fixture remains unchanged. The
   offline full test uses these same two overrides.
6. Residual limits: explicit declarations require a known source schema and
   cannot restore precision lost upstream. Cosmos has no cross-document
   snapshot here; the disclosed immutable-source window still applies.

## Validation

- Full core regression: `go test ./...` **PASS** (including default tiny parity).
- Extension typecheck, 195 unit tests, and VSIX packaging **PASS**.
- Offline full frozen P1 with the exact two GUI overrides: **PASS**, 83.69s,
  5,600,000 records, 64 ranges; expected and actual root
  `bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.
- Full-data invocation uses `AF_P1_COSMOS_FIXTURE` pointing to the retained
  `work/vscode-p1-20260905/manifest.json` and `AF_P1_COSMOS_PORTABLE` to
  `work/vscode-p1-portable-20260905`, then runs
  `go test ./production-simulation/internal/rangedigest -run TestCosmosTypedPortableCanonicalParity -v`.

## Pinned artifacts and desktop installation

Implementation commit pushed: `8a23a5109798ec906109532e4cc6c32308b3c824`.
Built from its clean archived tree; no public release or Marketplace publication.

| Artifact | SHA-256 | Bytes |
| --- | --- | ---: |
| Linux runner `2.4.0-dev.8a23a5109798` | `52e1d147a13b86a729f5a993e9e72848dd87a89d0ae50a61f26459f5632444f3` | 37079079 |
| Read-only P1 verifier | `60ed56a6773e6cbb64f7a0c03bc407f8aea135c7f1a75d7b8494db17cf09f79d` | 7248083 |
| VSIX `agefreighter-2.4.0.vsix` | `ecff799b71e2a3a34bce6c15fde9ac93be622952ca178a0123f756857be965e4` | — |

Local manifests are retained under `work/vscode-runner-build.GXKmVv/` and
`work/vscode-p1-verifier.YxRRzz/`. VS Code CLI confirmed installation; the
installed `dist/extension.js` and packaged build both hash to
`9cb64155b59c2fa72d2481df86cdd6ea280a85b608f3b1958fcbd519331eff9a`.
Window reload and actual updated UI are **not yet verified**, because the Mac
remains locked. No Azure resource writes were performed during this preparation.

## Next live steps (not executed by this fix)

1. Unlock this Mac manually, reload the installed extension, and confirm its
   current UI. Computer Use reported the Mac locked; no unlock bypass is used.
2. Recheck the unchanged USD 800 ceiling, USD 400 reserve, deadline
   `2026-09-16T07:14:35.311Z`, live health and governance before any Azure write.
   A read-only check during this fix confirmed all 12 VMs deallocated and all
   10 Flexible Servers Stopped. Retained Cosmos/storage charges continue.
3. Create a fresh AZ-COSMOS workflow/runner/target/job using the reviewed pinned
   fixed Linux artifact and `fixtures/cosmos-p1-typed-mappings.json`. The old
   runner has reached its 25 managed Run Command limit; do not delete receipts
   or reuse its failed job/graph to make room.
4. Use the existing source account and frozen data without rewriting them.
   Obtain action-time confirmation if a new runner identity needs a new
   security-sensitive access grant. Repeat GUI inventory, target/runner review,
   migration, complete counts, and full canonical verification/import.
5. Accept only all 5.6M records / 64 ranges and the unchanged canonical root,
   displayed as PASS by the installed GUI. Preserve r1 failures and r2 evidence.
   Stop route compute after evidence capture; do not publish or clean up data.

Remaining routes after a future AZ-COSMOS pass: OP-N44 and OP-N526. No live pass
is inferred from offline parity or package installation.
