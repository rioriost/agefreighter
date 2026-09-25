# R4 prerequisite: complete CSV capacity evidence

Status: implemented and locally validated. **Not yet exercised on Azure or
through the installed GUI. R4 target deployment/resize and R5 remain open.**

## Review and scope

The real Linux sample profile observed 10,000 vertices and no edges out of P1's
5.6 million records. Scaling that prefix by an exact row count cannot establish
a representative capacity estimate. The capacity combiner now retains such
estimates for review but never marks them deployable. A failed/incomplete report
cannot become deployable merely by containing `complete-stream-range` fields.

For CSV, extend the existing `inventory` command with a complete streaming read
using the real typed CSV connector, with zero rejected rows. Retain exact mapped
vertex/edge counts, per-label counts, decoded record widths and unique input
file bytes. Hash all mapped files before and after the scan, including file
metadata. A changed/missing/non-regular file, malformed record, cancellation,
timeout, close error or limit breach returns an error, not a partial exact report.
No target connection, target credentials, quarantine or migration is used.

Limits: 64 files, 10 GiB physical input, 100 million mapped records, the configured
source operation timeout capped at 30 minutes; existing parser record/field
limits remain. Gzip expansion is additionally bounded by time, per-record limits
and mapped record count. These are guided-trial bounds, not a new P3 claim.

Counts are per configured mapping, not distinct identities. No uniqueness or
cross-record endpoint-existence proof is claimed. Capacity storage multipliers
remain estimates requiring target/cost review; no throughput-based duration is
invented. File fingerprints include paths/metadata and are not portable canonical
graph digests or substitutes for the existing full-range fixture comparison.

## Compatibility and GUI gate

The Linux readiness receipt advertises `csv-inventory-v1`. The controller and
webview allow complete CSV inventory only with that capability, fresh matching
readiness, reviewed configuration and all mapped file seals verified. An old
guest omitting capabilities still supports its existing operations, but cannot
receive CSV inventory. Native approval discloses the full scan and bounds.

The currently approved/deallocated Azure VM still runs `6ff072c0db71`, without
this capability. Its artifact, workflow, CSVs and historical sample report were
not changed. A reviewed matching CLI/tools artifact and explicit readiness
refresh are required before Azure qualification; do not silently rewrite the
private workflow's pinned artifact or promote this local result to a GUI pass.

## Local P1 evidence

The new CLI was built from this implementation tree and invoked directly by the
test harness on the existing MacStudio P1 CSV fixture (not by the extension).
The report is [retained here](evidence/csv-inventory-local-20260906.json).

- Observed 1,600,000 vertices and 4,000,000 edges across all 18 mappings.
- Exact physical input bytes: 1,168,576,671; decoded mapped record widths:
  1,921,565,474 bytes. All configured maps reached EOF; before/after hashes matched.
- Report outcome `pass`, no errors/incomplete checks. The warning explicitly
  distinguishes inventory from migration/identity/endpoint verification.
- Measured on this Mac: 9.70 seconds wall time, maximum RSS 29,622,272 bytes
  (about 28.3 MiB), swaps zero. Build time was excluded. This is not a Linux VM
  throughput guarantee or a migration runtime estimate.
- Go app/runner/CLI tests with the race detector and `go vet` passed. The durable
  worker test executes the actual inventory implementation in a child process
  with real sealed CSV input; systemd transport is injected for portability.
- All 130 extension unit tests, typecheck, five actual-CLI configuration
  contracts and VSIX packaging passed. Tests cover old-guest blocking, upload
  seals, review gating, incomplete capacity rejection and prefix extrapolation.

Next: wire reviewed complete evidence into the private target plan, add target
network/what-if and identity-preserving same-VM resize orchestration, and then
qualify the matching guest build within the existing budget/window. Do not
create target resources from the retained incomplete sample profile.
