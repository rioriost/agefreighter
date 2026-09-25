# Gremlin P1 fixture and offline canonical verification

September 17, 2026, approximately 06:31–06:47 UTC. **Offline preparation PASS;
no new live GUI/Azure migration qualification.** Implements and reviews the
[fixture/oracle plan](gremlin-fixture-plan.md), following the
[explicit numeric-type prerequisite](gremlin-types-preflight-20260917.md).

## What was verified

- Exported the frozen P1 into a new create-only local directory, preserving
  all old fixtures and accepted graphs. Nine vertex labels and nine edge
  types contain 1,600,000 vertices and 4,000,000 edges.
- Every Gremlin element ID retains its partition. Edges retain their source
  partition and destination `_sinkPartition`; original typed properties,
  arrays, nulls and Unicode strings are unchanged.
- The exporter derives partitions from external-ID ordinals. The independent
  expected oracle reads the original verified CSV and derives partitions from
  source keys and label first-key ranges. It does not call the exporter helper.
- Actual records pass through the production Cosmos Gremlin decoder, using
  100-document local pages reversed in order. The final read-only rerun also
  removes all-zero fractional spellings without converting through float64
  (for example `120.00` becomes `120`); explicit floating types preserve values.
- All **64 ranges and 5,600,000 records match**. Normal CSV export separately
  matches the unchanged raw-ID canonical oracle.

| Evidence | SHA-256 |
|---|---|
| Frozen fixture root | `f74220f6c58f0c1a62f80a567520ffcde43a2499ba48100667ee7b78ff4e2e2f` |
| Unchanged raw-ID canonical root | `bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70` |
| New partition-preserving canonical root | `8a048faa36fad90404c263d3ce75073d117e5d96a15f8a614a42347cbd7a0ef4` |
| Export manifest | `7cb9642cd3c3a85fb6503b727697a3178ed43269156086590eea1ca211936860` |

Canonical version: `agefreighter-production-simulation-gremlin-partition64-v1`.
The observed role is `cosmos-gremlin-offline`, never `apache-age`. The normal
target comparator rejects this role/version; the offline comparator cannot
qualify a target. All expected/actual leaves and exported file hashes are in
the [redacted evidence](evidence/gremlin-offline-p1-20260917.json).

Export plus initial CSV/Gremlin verification took **222.43 seconds**, maximum
RSS **986,972,160 bytes**. The tightened numeric-transport read-only full rerun
took **122.94 seconds**, maximum RSS **989,937,664 bytes** (about 0.92 GiB),
zero swaps. These are local macOS preparation timings, not Azure migration
throughput. The checker buffers at most one bounded P1 mapping for ordering;
it is restricted to tiny/P1 and does not claim arbitrary production-size support.

## Regression and review

`go test ./...`, simulation `go vet`, and race tests for the portable/rangedigest
packages pass. Tiny tests reject wrong partition, wrong destination partition,
property modification, missing/duplicate records, checksum changes, wrong table
metadata, wrong labels and trailing JSON. Tests cover partition wrap at ordinals
64/65, malformed IDs, create-only output and lossless number normalization.
The reader verifies document hashes before/after decoding and validates the
canonical tiny/P1 plan and table identities/counts. No cloud client is created.

Review kept original canonical behavior unchanged, separated source-only from
target qualification, retained all partial output, and corrected the simulated
numeric transport to cover `.0000`/`.00` as well as `.0`. The full rerun confirms
the same canonical root after that correction. A strict single-JSON-value
guard was added afterward and is covered by the tiny/race regression suite.

## Reproduction

From the repository root, select an output directory that does not yet exist:

```sh
go run ./production-simulation/cmd/portablefixture \
  -manifest production-simulation/work/vscode-p1-20260905/manifest.json \
  -output production-simulation/work/NEW-GREMLIN-DIRECTORY \
  -cosmos-format gremlin
```

The retained local output for this run is
`production-simulation/work/vscode-p1-gremlin-20260917/`. Do not overwrite it.
The optional `TestGremlinFrozenP1` reads an existing export without changing it;
set `AF_P1_GREMLIN_FIXTURE` and `AF_P1_GREMLIN_PORTABLE` to absolute paths.

## Preservation and remaining gates

No Azure, GUI, installed-extension or guest mutation was performed. Fresh
read-only ARM checks around 06:42 UTC confirm all nine VMs deallocated and
all 17 Flexible Servers Stopped. Stopped compute does not eliminate ongoing
storage/Cosmos charges. No fresh billing or guest-health conclusion is claimed;
the existing USD 800 ceiling and September 20 deadline are unchanged.
At `2026-09-17T06:46:54.113Z`, all 66 saved root workflow/report JSON files
still match aggregate SHA-256
`819cdcd7c1bae30b57cb669fcd0f7a2d45172c912072e05219f9197b33c28e9f`,
using the [previously documented aggregation method](cosmos-guard-gui-20260917.md).

Before B05 can pass: extend the read-only target verifier to check retained
composite identities and actual endpoint bindings, update the development GUI
verification envelope, pin Linux/extension builds, refresh all cloud gates,
and execute a new isolated installed-GUI/Azure migration with full target
verification. Existing raw-ID target verification is insufficient: reconstructing
expected endpoint IDs would not prove retained partitions. B10 active-operation
reload/no-replay verification also remains open. No new approval is requested
by this local preparation step.
