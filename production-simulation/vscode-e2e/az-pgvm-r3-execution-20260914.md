# AZ-PGVM native-float corrective GUI attempt

Status: fixed development artifact installed on the new private VM; installed-GUI
readiness and source health pass. The GUI is awaiting the read-only source
password for a new complete inventory. No source inventory, migration or full P1
qualification has completed in this attempt.

## Scope and preserved evidence

- The released v2.3.1 fix (`952b6b4`) was merged into guided development in
  `43490f6`; `9ef1696` adds the PostgreSQL native-float guest capability gate.
- Extension remains 2.4.0; runner-first flow and Azure Resources authentication
  remain. No 2.4.0 release or Marketplace publication was performed.
- AZ-PGVM r1/r2 targets, jobs, failed digests and guest evidence remain unchanged.
  Old PostgreSQL checkpoints are not resumable with the new fingerprint.
- Overall GUI qualification remains 3/9. This attempt must use a fresh target
  and job and match all 5,600,000 records / 64 canonical digest ranges.
- Renewed trial budget remains USD 800 and deadline
  `2026-09-16T07:14:35.311Z`; no P3 authorization is reused.

## Local validation and artifacts

All Go package tests passed. PostgreSQL/runner race tests passed. Live local
PostgreSQL 18 / AGE tests preserve exact native float serialization in COPY,
cursor and keyset, with pre-encoding on/off, arrays/domains, non-finite rejection
and legacy-checkpoint refusal. The first AGE test invocation used the wrong
local database role and failed authentication; rerunning with the container's
configured role passed all three modes. Temporary local test containers are
stopped again, not deleted.

Extension typecheck/build and 180 unit tests passed. The newly packaged VSIX
was installed and reloaded in the actual Mac VS Code:

- VSIX SHA-256: `95e6864210bc2ace043d2685ece28d9892b0f091dee51e9ba902dfd4f31c120c`.
- Linux version: `2.4.0-dev.9ef16968363b`.
- Commit: `9ef16968363b31214324f392553f7c8e88150272`.
- Archive SHA-256: `10a27dd02b53f070ad2529b21c40c5d382a6e67cb59d9175f98cc29531ff8997`.
- Archive bytes: 37,056,164.
- Local build manifest: `production-simulation/work/vscode-runner-build.2cDvou/manifest.json`.

## Fresh GUI workflow

- Workflow: `c275d043-de93-4b0a-b2b0-59cddd13c84f`.
- Name: `az-pgvm-p1-r3`.
- Azure discovery selected the retained PostgreSQL source VM in the trial group.
- All 18 mappings were entered into visible GUI fields; a read-only comparison
  of the saved form exactly matches `fixtures/postgresql-p1-mappings.json`.
- Configuration SHA-256 (JSON.stringify encoding):
  `691b33014f5aa54f0806d24269851cf7b442e32498959db4f59e71517185e426`.
- Existing public CA SHA-256:
  `0b196d1e310a5a68732879c7bf0aa8c9f1fc67bbfe436fe560a31de7afa05b68`.
- Transfer account: `afc275d043de934b0ab2b059`; dedicated workflow container.
- Runner: `af-c275d043de934b0ab2b0`, Japan East / zone 1 / B2s_v2;
  compute estimate USD 0.109/hour plus disk/network.

The transfer deployment succeeded. Effective public network access was initially
Disabled; within the previously approved storage-only exception, the new test
account received `SecurityControl=Ignore` and authenticated public HTTPS access.
Anonymous access and shared keys remain disabled; HTTPS-only and TLS 1.2 remain
required. No source firewall, VNet peering or public VM IP was added. Installed
GUI upload independently verified the pinned archive and recorded `ready`.
The VM preview and its container-only Blob Reader grant were reviewed; the GUI
submitted a new deployment once. Provisioning is not guest readiness.

## Live gates at resumption

Cost Management returned USD 17.5157291900634 for September 12 and
USD 15.4061238600271 for September 13 (USD 32.9218530500905 total in returned
rows). Billing lags; this is not the final total. The existing USD 400
accrued/non-compute reserve and the USD 800 ceiling remain in effect.

All eight previous trial VMs were deallocated. The unused Flexible Server source
was unexpectedly Ready after its earlier stopped state; recent activity only
showed resource-health changes, not a user start action. Its cause is not
asserted. It was stopped again and Stopped was confirmed. The other five
Flexible Servers remained Stopped. Old failed runners/targets were not started.

Only the retained PostgreSQL source VM was started for r3. Its initial fresh
health check showed disk 9%, zero swap, zero boot OOM events and valid certificate
chain / more than 96 hours remaining. Its intentionally restart-disabled
PostgreSQL container was stopped; an explicit start of that same container was
submitted without changing data, credentials or certificate validation.

## Readiness passed; private credential entry pending

The new VM deployment completed at `2026-09-14T04:59:04.242274Z` and the GUI
reconciled it without replay. Guest readiness operation
`091c9cda-423e-4306-ba8c-f0b7fdd9012a` was submitted at `05:00:23.758Z` and
completed successfully. Version, commit and archive SHA match the pinned build;
`postgresql-native-floats-v1` is advertised. Boot ID is
`842f6400-ed79-40c0-a7f8-c3fa38e67bcc`; idle=true, disk 3.48%, swap=0, OOM=0.

The source container start also succeeded. Current certificate validation
accepts both its DNS name and private IP, with more than 96 hours remaining;
source disk remains 9%, swap=0. No data or credential changes occurred.

The installed GUI re-reviewed the source, accepted the already-covered complete
inventory read approval and opened **Read-only source password**. The operator
must enter the existing read-only credential in that private input and press
Enter. No inventory intent exists yet; its operation is created only after
credential entry and fresh health admission. No password is stored in this
report, the source form or a generated LoadJob. The source and new runner remain
running within the renewed trial window; old failed resources remain stopped.
