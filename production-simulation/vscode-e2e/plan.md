# VS Code guided migration P1 qualification

Status: implementation in progress; dedicated Azure network preparation started,
but no migration qualification has run. The 96-hour window expires at
2026-09-09T08:55:00Z. See [progress](progress.md) for current evidence and gates.
Branch: `codex/2.4.0-guided-migration`. Date: 2026-09-05.

## Authorization and stop conditions

The user authorized preparation and creation/update/deletion of this test's
resources in subscription `MCAPS-Hybrid-REQ-51508-2023-rifujita`, with an **800 USD**
ceiling and a **96-hour live window**. The window starts with the first resource
creation, not with local implementation. Record its UTC start/deadline and the
resolved subscription ID in private local run metadata before provisioning.
Use dedicated `rg-af-vscode-p1-<date>-<suffix>` resources and ownership tags;
never mutate unrelated subscription resources. No previous P1/P3 authorization
or retained evidence is reused as permission or as this qualification's result.

Before each cloud mutation check elapsed time, conservative accrued/projected
cost (including pending billing), capacity/quotas, active jobs and ownership.
Stop starting trials when projected completion exceeds either ceiling. Preserve
logs/checkpoints and stop/deallocate owned compute at a gate breach. Export
redacted evidence before separately tracked deletion of owned test resources.
VM/disk/target storage must stay below 80%; checkpoints must stay current;
OOM, swap or source mutation invalidate a timed qualification. No unrestricted
source public access is required by the on-premises simulation.

## Required customer path

The operator is already signed into VS Code through Azure Resources. No local
AGEFreighter or Azure CLI is required by the extension. GUI source selection,
runner deployment, protected source configuration, source assessment/mapping,
target review, late LoadJob save, same-runner resize, target readiness, durable
load and verification must all execute. Success is a verification result, not
ARM success, process exit zero, or load completion. Existing local LoadJob
commands are not a substitute for testing the guided GUI path.

## Fixture and source preparation

Reuse P1: **1,600,000 vertices + 4,000,000 edges**, seed `20260829`, all nine
vertex labels and nine edge types, with original identities, endpoint mappings,
Unicode, nulls, variable-width values, integer/float/boolean and array types.
Generate a canonical fixture manifest and independent full-range digest once.
The old P1 source-manifest and canonical-data roots are distinct; do not confuse
them or accept old results as evidence for newly generated data.

Prepare Neo4j 4.4.48 and 5.26.30, PostgreSQL on a Linux VM, PostgreSQL Flexible
Server, Cosmos NoSQL graph documents (including supported Gremlin document
format), and ordinary headered UTF-8 CSV on this MacStudio. Each representation
must round-trip the same typed graph. CSV needs explicit type mappings (legacy
CSV remains string-valued by default). PostgreSQL and Cosmos fixture adapters
must retain source_key and stable edge IDs; no lossy numeric/string conversion.
Verify each source before making it available to the GUI. Keep sources read-only
through migration and verification. Store all secrets outside Git and reports.

## Base GUI paths — every row requires a full P1 migration

| Case | GUI source | Discovery mode |
|---|---|---|
| AZ-N44 | Neo4j 4.4.48 VM | Azure subscription/RG/resource candidate, then authenticated discovery |
| AZ-N526 | Neo4j 5.26.30 VM | Azure candidate, then authenticated discovery |
| AZ-PGVM | PostgreSQL 18 on VM | Azure VM candidate, then schema/mapping review |
| AZ-PGFS | PostgreSQL 18 Flexible Server | Azure managed resource, then schema/mapping review |
| AZ-COSMOS | Cosmos DB | Azure account/container and supported document/mapping selection |
| OP-N44 | Neo4j 4.4.48 | IP + port + credentials only |
| OP-N526 | Neo4j 5.26.30 | IP + port + credentials only |
| OP-PG | PostgreSQL 18 on VM | IP + port + credentials only |
| CSV-MAC | Local CSV | Mac file/folder selection, reviewed mapping and approved upload |

On-premises cases may use Azure-hosted fixtures, but the customer workflow must
never resolve the source through ARM, NIC metadata, Azure DNS/resource discovery
or inferred subscription IDs. Target/runner ARM operations remain allowed.
The fixture preparation harness is separate from this source-discovery audit.
Use IP-reachable private routing where available; a public IP is not required.
TLS must validate the configured identity (test certificates with IP SANs if
using literal IPs); do not silently disable certificate validation.

## Branch coverage, beyond the base paths

Inventory actual selectable UI branches and bind each to a test case. Each
source-specific branch needs integration evidence; small mocked tests do not
replace the full P1 base paths. Independent generic choices can be distributed
across those paths, with a branch-to-case ledger, rather than blindly multiplying
every SKU and region into an unaffordable Cartesian product.

- Azure source default RG versus separate migration RG; independent network RG.
- Source-region/zone default versus reviewed permitted override/unknown zone;
  region/SKU/subnet/quota mismatch must block without mutation.
- Private Azure discovery and IP-only on-premises discovery; no source exposure.
- Neo4j 4/5 discovery, PostgreSQL key/FK recommendations and reviewed mappings,
  Cosmos explicit/Gremlin mapping paths, CSV vertex/edge/type/null mappings.
- Managed identity and other supported Cosmos credential modes; no ARM token
  used as a Cosmos token. RBAC changes need recorded approval and propagation checks.
- Burstable discovery to appropriately sized same-VM migration; reject active-job
  resize and incompatible SKU/architecture; preserve identity/NIC/durable disks.
- CSV multi-file selection, resumable upload, changed file, checksum mismatch,
  transfer cancellation and local folder cancellation; never start load on partial data.
- Cancel every approval; expired preview; duplicate click and multiple windows;
  lost deployment response; guest bootstrap failure; missing artifact; quota denial.
- Close/reload VS Code during assessment/load/verification; reconnect without replay.
- Interrupted loader/network, explicit resume of the same job/generation/fingerprint.
- Counts/digest mismatch, rejected records, missing/truncated/stale/wrong-job
  evidence and `incomplete` verification must not become a completed migration.

## Implementation sequence and review gates

1. **Engine/fixture parity:** typed CSV and fixture adapters; strict verification
   signaling; configuration/schema/fingerprint/regression tests. Tiny local
   round-trips precede P1 generation. Preserve old LoadJob semantics.
2. **Remote runner service:** install a pinned verified Linux artifact; protected
   credential dispatch, bounded read-only assessment, durable guest operation
   IDs, allowlisted operations, protected retained files and no automatic replay.
   Long tasks must survive control-request and VS Code closure.
3. **GUI mappings/data path:** all source forms, PostgreSQL/CSV type/identity
   review, Cosmos container/mapping review, resumable verified local CSV upload.
4. **Target/resize orchestration:** same migration RG, existing reachable VNet,
   separate PostgreSQL delegated subnet/private DNS, approved cost/what-if,
   resize the runner only, verify guest resources, install/verify AGE, doctor,
   finalize and save secret-reference-only LoadJob after target review.
5. **Load/verification:** durable start/status/resume; complete counts plus full
   canonical properties/identity/endpoint comparison for P1. Bounded integrity
   diagnostics may be retained as incomplete but cannot substitute for full
   evidence. Missing/failed required evidence keeps the workflow incomplete/failed.
6. **Live qualification:** prepare dedicated sources and Mac CSV, run all base
   cases through the real GUI/production controller, cover the branch ledger,
   retain screenshots/action transcripts, CLI/ARM/guest IDs and checksummed
   verification artifacts. Do not silently bypass approval or remote execution
   via a CLI-only harness and call it a GUI pass.

Use an explicitly reviewed, commit-pinned development artifact for test-only
bootstrap when 2.4.0 is not released; no mutable branch builds on customer VMs,
unverified download or public release solely to bypass the gate. Its source,
version and checksum must be visible. Production installer trust remains strict.

## Review findings

- R1/R2 is not an end-to-end implementation: remote dispatch, target provisioning,
  mapping, upload, resizing and load/verify are missing. They are required here.
- Existing inventory is Neo4j-only. Other connectors need their own bounded
  assessment, not a fabricated Neo4j count response.
- CSV properties currently decode as strings. P1 numeric/boolean/array equivalence
  requires an opt-in typed mapping and fingerprint/schema updates.
- Deep verify currently exits zero for `incomplete` by compatibility design.
  Add opt-in strict signaling and inspect structured outcome/coverage in the GUI.
- Resource-group boundaries are not VNet boundaries. Creating/choosing an RG
  cannot prove reachability; no automatic peering/public exposure workaround.
- Managed identity bootstrap alone does not grant source/Blob/Key Vault data access.
- The prior P1 resource group is absent in the currently selected subscription;
  prepare a new owned environment rather than assuming retained fixtures exist.

## Evidence and final outcome

For every base/branch case retain: code/artifact version and hash, fixture/source
root, GUI path, redacted settings, deployment/VM identity before/after resize,
CLI job/generation/fingerprint, before/after source proof, counts, rejects,
complete digest/range comparison, duration, resource peaks and cost envelope.
Track `not-run / running / passed / failed / blocked` per case; missing rows are
not passes. Final release-readiness requires all mandatory rows passed, no
uncovered selectable branch, and the installed VSIX matching the tested build.
Stop/deallocate owned compute after evidence retention and close the cost window.
