# Gremlin target verification and profile binding

September 17, 2026, approximately 06:50–07:12 UTC. **Local implementation and
storage-contract checks PASS; live Gremlin migration remains not-run.**
Builds on the [full source-side P1 oracle](gremlin-offline-p1-20260917.md).

## Design and review

The new `P1GremlinTargetManifest` is separate from the existing raw-ID verifier.
The original P1/P3 interpretation and already accepted results are unchanged.

- Resolve the committed job's graph/generation, then read a single PostgreSQL
  **read-only, repeatable-read transaction**. Bound individual statements to
  20 minutes; the guest retains its existing 25-minute / 4 GiB / no-swap limits.
- Read actual composite IDs from retained vertex/edge identities; validate
  their shape and agreement with visible ID properties. Do not reconstruct
  observed IDs from fixture ordinals or source keys.
- Read physical edge `start_id`/`end_id`, require equality with metadata, and
  resolve both through the actual physical-vertex index. Check endpoint labels,
  namespaces and source partition. A consistent but wrong physical+metadata
  endpoint edit is still caught by the independent canonical oracle.
- Start reads from physical tables with a left join to retained identities,
  not an inner join that could hide an orphan. Check per-label and overall
  metadata counts, physical counts, exact catalogs/generations, and empty
  unlabeled root tables. Missing/extra records or labels fail closed.
- Bound the endpoint index to 1.6M records / 256 MiB string payload, and one
  mapping's sorting buffer to 512 MiB. Only the frozen P1 is exposed by the
  public verifier; tiny is admitted solely by the internal test entry point.

The isolated runner accepts an explicit `gremlin-partition64` profile, with
the frozen root `8a048faa36fad90404c263d3ce75073d117e5d96a15f8a614a42347cbd7a0ef4`.
Omitting it retains the existing `raw-id` behavior. Unknown profiles are rejected
before private input is read. Failure receipts remain redacted and create-only.

The extension now binds source configuration, selected artifact manifest,
persisted operation, runner arguments and imported report to the same profile.
It independently recomputes every leaf/root. Source/profile changes fail before
transfer/submission; an offline-source receipt or the old raw-ID root cannot
qualify a Gremlin target. Existing raw-ID receipts remain readable. Frozen
Gremlin admission requires `partitionKey` and explicit float64 declarations for
`score`/`distance_km`. The legacy raw-ID failure diagnostic/requalification path
refuses Gremlin operations pending a profile-specific evidence review; it never
silently falls back or retries.

## Executed checks

- Full `go test ./...` and simulation `go vet` pass.
- Portable/rangedigest/isolated-runner race checks pass, including a local
  PostgreSQL SQL-contract test with 560 records and injected corruption.
- **Real AGE storage contract PASS**, using Apple Container, PostgreSQL
  **18.1** / AGE **1.7.0**, Linux amd64 under Rosetta, 2 CPUs / 1 GiB.
  Image pinned to the existing compatibility matrix digest:
  `apache/age@sha256:e7de1717e487dac7c1be93a1cd5360a2cf07ff4170342c2af2ac4713c21baf00`.
  A fresh verifier connection does not borrow setup `LOAD` or search-path state.
  Query tracing confirms the read-only snapshot and only read/session statements.
- All 560 records match the tiny partition-preserving root:
  `1657c8a13b6add49113b06fc522ee495a3f66fd647e76d86f2cacb2eb34d874e`.
  Orphan identity, physical/metadata endpoint divergence, wrong namespace,
  changed partition, missing metadata, physical orphan, unlabeled data,
  uncommitted job, extra label, changed property, and a consistent wrong endpoint
  are rejected. The final AGE race test completed in 1.23 seconds (2.567 seconds
  for the Go package invocation).
- Extension typecheck, **300 unit tests**, bundling and **ten actual CLI source
  contracts** pass. Source contracts initially refused a missing test-binary
  environment variable; all ten passed after building/selecting the current CLI.
  Controller tests use inert GUI/ARM adapters, not a signed-in live GUI.

The AGE test populates physical AGE types/catalogs with tiny fixture rows and
test metadata using SQL. It validates the verifier, **not** a full CLI migration,
the production metadata DDL, a 5.6M-record target, or Azure/GUI qualification.
The full 5.6M-record result from the preceding batch remains source-side only.

Initial AGE harness attempts exposed two test-setup constraints: the single-letter
graph name was invalid, and `drop_label(..., true)` is unsupported. The harness
uses `gremlin_contract` and the supported false option for its empty test label.
Failed databases were retained; no accepted target was touched. Final retained
AGE database: `af_gremlin_test_1789629042262901000`, in local container
`af-gremlin-age-20260917`. Both this container and the plain PostgreSQL contract
container `af-gremlin-oracle-20260917` were stopped after testing, not deleted.

## Reproduction and next gate

For the local test, point `AF_GREMLIN_SQL_TEST_ADMIN_DSN` at an **isolated local**
PostgreSQL administrator connection. Every invocation creates and retains a
new database. Set `AF_GREMLIN_SQL_TEST_AGE=1` for the real AGE mode; omit it for
the text-table contract. Run:

```sh
go test -race ./production-simulation/internal/rangedigest \
  -run '^TestGremlinTargetSQLContract$' -v -count=1 -timeout=3m
```

After committing the reviewed tree, build the separately identified artifact:

```sh
production-simulation/vscode-e2e/build-p1-verifier.sh gremlin-partition64
```

No installed extension, guest, Azure permission/network, accepted graph or
cloud compute was changed. Fresh ARM reads around 07:10 UTC show nine VMs
deallocated and 17 Flexible Servers Stopped. At `07:11:07.982Z`, all 66 saved
workflow/report JSON files retain aggregate SHA-256
`819cdcd7c1bae30b57cb669fcd0f7a2d45172c912072e05219f9197b33c28e9f`.
Budget remains USD 800 / deadline `2026-09-20T07:14:35.311Z`; actual costs and
guest health still require fresh checks before cloud mutation.

Next: pin/package the corresponding runner, verifier and extension, review the
new Gremlin source/container placement, refresh all authorization/cost/security
gates, then run the isolated installed-GUI/Azure path and full target digest.
Include active-operation reload/no-replay testing for B10. Neither B05 nor B10
is promoted to PASS by this local work.
