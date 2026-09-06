# CSV-MAC completion execution sheet

Scope: complete the existing P1 CSV GUI path on the owned Linux runner and a
private PostgreSQL 18 Flexible Server with AGE. This is not a P3 rerun or an
authorization to modify another subscription's resources. The original USD 800
ceiling and 2026-09-09T08:55:00Z deadline remain authoritative.

## Sequence and gates

1. Read live ownership, stopped state, governance events, elapsed time and cost
   exposure. Preserve the 18 imported CSV files and all existing evidence.
2. Upgrade the idle development runner through an explicitly reviewed GUI
   operation. Upload content-addressed archive bytes, retain previous binaries
   and installation receipts, exclude active guest work, and bind to the current
   boot and old archive hash. Persist the new operation before PUT; reconcile by
   GET only. A failed or uncertain upgrade blocks other guest controls. Never
   edit the extension's private workflow file outside its controller.
3. Refresh matching readiness, review the mappings and approve complete CSV
   inventory. Import and independently verify the report. Require all 1.6M
   vertices / 4M edges and complete storage evidence; no sample extrapolation.
4. Review a private target plan in the same migration RG, region and zone.
   Use a dedicated delegated subnet and linked private DNS, not public access
   or a source firewall exception. Recheck service/SKU/quota and live prices;
   show duration, budget reserve and disk/network costs before mutation.
5. Save secret-reference-only LoadJob after folder selection. Deploy only
   reviewed new resources. Reconcile deployment by ID. Resize the same idle
   x64/SCSI VM only if the reviewed plan requires it; retain NIC, identity and
   persistent disk. Recheck guest identity/resources after restart.
6. Prepare AGE and prove TLS/network/target readiness. Explicitly start a durable
   load; retain its job ID, configuration fingerprint and target generation.
   Reconnect without replay and explicitly resume only the same job if needed.
7. Require complete counts verification with zero rejects plus independent full
   P1 typed-property/identity/endpoint digest comparison. A committed load or
   successful worker is not completion. Seal results, stop owned compute, and
   report CSV-MAC separately from the other eight unqualified branches.

## Initial live reads and implementation review

At 2026-09-06T02:53Z, the selected authorized subscription matched; the retained
runner was `PowerState/deallocated`. The dedicated RG had no target server.
Failed activity events since 02:00Z were empty. The regional capability read
advertised PostgreSQL 18 and Standard_D4ds_v5 in zones 1/2/3; this is not a
capacity reservation, price check or deployment approval.

The upgrade implementation uses a retained installation lock and workflow lease,
backs up binaries/markers, verifies the archive and both versions before switching,
and changes the workflow's pinned artifact only after a matching successful
receipt. Per-binary publication is atomic, but the pair is not a transaction:
a partial installation intentionally requires operator reconciliation and keeps
all locks/evidence. It never automatically rolls back or resumes a guest job.
The installer has a 20-minute bound, checks storage below 80%, and never deletes
CSV data, report evidence or a prior installation. Tests cover unknown submission,
wrong receipt identity, stale readiness, pending operations and command limits.

Private-network reference: [Microsoft's Flexible Server private access guidance](https://learn.microsoft.com/en-us/azure/postgresql/flexible-server/concepts-networking-private).
AGE preparation reference: [Microsoft's AGE extension guidance](https://github.com/MicrosoftDocs/azure-databases-docs/blob/main/articles/postgresql/azure-ai/generative-ai-age-overview.md).

## Target foundation and current execution gate

The local target controller now binds the imported complete CSV report to the
current source configuration, artifact, all file seals and exact per-label counts.
It builds a PostgreSQL 18/private delegated subnet/DNS-link/database/AGE-parameter
template in the same region, zone and migration RG. The initial path requires
the VNet in that RG; it fails rather than guessing a cross-group deployment.
Capacity plus 25% storage headroom, CIDR containment/non-overlap, bounded budget
and deadline, create-only what-if and persist-before-PUT/GET-only reconciliation
are covered by tests. Actual passwords are secure ARM parameters, not fields in
the template or retained record. HA is explicitly disabled for this P1 trial.

The subsequent controller adds a native target-review wizard and main-view
entry point. It binds imported source evidence, checks owned private NIC/VM
placement, service capabilities, total/family quota, subnet overlap and unique
current Linux/PostgreSQL compute prices. It exports a new secret-reference-only
LoadJob (JSON-form YAML 1.2) and reviewed plan only after folder selection.
Generated target credentials stay in VS Code SecretStorage and ARM secure
parameters, not the exported files. Submission is separately approved and
rechecked under the workflow lock. Failed/unknown submissions reconcile by GET.
HA is explicitly disabled for this bounded single-server trial; resize and R5
remain separate unimplemented gates. Local tests/typechecking/package pass
(142 tests); this target path is not yet live-qualified or publicly released.

The [documented quota endpoint](https://learn.microsoft.com/en-us/rest/api/postgresql/quota-usages/list?view=rest-postgresql-2025-08-01)
with stable `2025-08-01` returned `InvalidApiVersionParameter` in this subscription;
`2025-08-01-preview` returned `NoRegisteredProviderFound`. The RP-advertised
`2023-06-01-preview` succeeded: regional PostgreSQL cores 2/196 and Ddsv5-family
2/64 in Japan East. The controller uses this supported version and still fails
closed on missing/insufficient quota. The live PG D4ds_v5 compute meter was
USD 0.488/hour; quotes are fetched again before any deployment. A rate-limited
retail-price response is not treated as zero cost or permission to skip pricing.

## Actual GUI continuation on 2026-09-06

- Mac initially unlocked; installed VS Code 1.136.1 reloaded the reviewed VSIX
  and restored the existing CSV workflow without editing private metadata.
- VM started after ownership/window/governance checks. Boot:
  `365f83e4-2cde-44db-aa31-2ced80664ddc`.
- GUI-approved upgrade `75cd1825-edbf-4936-9285-1cc088e62ae4` submitted
  03:37:04Z; Azure success 03:37:25Z. Archive SHA-256
  `50873caae75cfe03d9f790147c9a1c9c0e87722b9769e90425f0a77593918bf1`.
  Old binaries and all input/report evidence were retained. Fresh readiness
  advertised `csv-inventory-v1` with matching commit/version/hash.
- GUI-approved complete inventory `e4b4cdcc-8403-4847-95a8-38213d7ce11c`
  ran 03:43:29.076722476Z–03:43:52.046389067Z. All 5.6M mapped rows passed;
  all file bytes/metadata matched before/after; zero errors/incomplete checks.
  Capacity high estimate is 29,052,316,636 bytes, not a measured target size.
- GUI approved export; Azure exported the report successfully at 03:46:44Z.
  Independent authenticated Blob read matched all 3,220 bytes and SHA-256
  `a285e77400c4aa705b36871f4134d0eedb5df03683b1b2c8d62b1c2b6168e837`.
  The Mac locked before GUI reconciliation/import. **Do not label the report
  GUI-imported yet**, replay inventory, or patch the private workflow file.
- Read-only guest check at 03:49:40Z: inactive/successful worker, exit 0, no
  active lease or loader, root filesystem 6%, swap 0, kernel OOM events 0.
  Peak RSS was not returned by this systemd version; do not invent a measured
  peak from its configured 4 GiB/no-swap limit.
- Ownership tags still matched; failed RG activity events since 03:24Z were
  empty. No target created. VM deallocation independently confirmed 03:52:23Z.
  Storage, NAT/IP, disk, CSVs and all failure/success evidence remain retained.
  Current-hour B2s_v2 exposure was less than USD 0.109; the original 800 USD/
  96-hour overall authorization is unchanged. Existing retained-resource charges
  continue and are not claimed to be a finalized Azure bill.

## Continuation after unlock

The existing full inventory was imported and displayed through the actual GUI
at approximately 04:56Z. Export/assessment were not replayed. After reloading
the installed target wizard, the same workflow was restored. The runner boot
changed to `16f8a00c-6821-48c8-a822-6447761161c5`; GUI readiness matched the pinned
installation. Read-only health at 05:03:21Z showed idle lease/loader, disk 6%,
swap 0 and no boot OOM events. Governance reads showed the prior owned
deallocation and resource-health transitions, no other control changes.

The GUI selected `afpg-83c6b829acdc4405aa2d`, PG18/D4ds_v5/128 GiB, delegated
`10.246.2.0/24` in the existing VNet, and later D4s_v5 for the same runner.
Current compute quote was USD 0.736/hour; a USD 100 accrued/non-compute reserve
was included under the original USD 800 ceiling and Sep 9 08:55Z deadline.
After explicit approval and folder selection, secret-reference-only LoadJob
and plan were saved in the ignored work directory. ARM target deployment
subsequently succeeded, with the server Ready in Japan East zone 1 and public
access Disabled. The AGE preload parameter requires a separately reconciled
restart. This does not establish AGE readiness or successful migration.

The new local R5 implementation seals a create-only job UUID before writes,
uses verified TLS to the owned Flexible Server, prepares AGE, loads with that
UUID, then requires complete counts verification. A fixed Linux worker and
same-VM resize controller preserve unknown outcomes without replay. Migration
freezes the source evidence and installed artifact. Explicit recovery of a
failed migration remains an operator gate, not an automatic retry. A new pinned
guest upgrade, complete re-inventory, resize/preload restart and live execution
are still required. Full P1 property/identity/endpoint digest is separate.

The Mac locked again before GUI target reconciliation. Azure's deployment is
Succeeded while the retained GUI target intent is still `submitted`; reconcile
by GET after unlock, never redeploy. The exact target plan hash is
`2f387200a96ae56f5420f29ec7a26ffb3443b4b4990534780a4360772f8879e7`.
At 05:23:29Z, a fresh read-only guest check again showed no active workflow or
loader, disk 6%, swap 0, and no OOM matches. `journalctl --grep` returned 1 with
empty stdout/stderr for no matches; the new health reader explicitly distinguishes
that case from permission/missing-journal errors. Two policy audit events were
read: the audit principal could not GET `Microsoft.Security/assessments` for the
new subnet. No denial or resource mutation was inferred, and no exception tag
was added outside the previously authorized storage account.

For cost containment during the GUI block, fresh ARM reads confirmed the VM
deallocated and Flexible Server Stopped. Neither operation deletes data or
evidence; retained storage, disk and network charges continue. The server's
seven-day automatic-start behavior does not extend the September 9 deadline.
The retained source guest is still c880a67, not the new local execution code.
The new implementation has 148 passing extension tests and passing Go app/CLI/
tools/runner tests. A parallel test run had a readiness child-process timeout;
the complete runner package passed on isolated rerun without relaxing the
production timeout. This is local test evidence, not an Azure migration pass.

## Saved restart checkpoint

- Implementation commit: `360bee443206ed8f36b04b0c64375d8326feb873`, pushed to
  `codex/2.4.0-guided-migration`. A final serial package run passed all four
  Go packages (runner, app, CLI, tools); extension packaging passed 148 tests.
- Linux/amd64 CLI and tools cross-build succeeded from that clean commit.
  Version: `2.4.0-dev.360bee443206`; archive: 37,015,749 bytes;
  SHA-256: `0e55422e668bc9d4e82b59547dd6d6763e388176345784172a5c3acfe8227923`.
  Local reviewed manifest: `../work/vscode-runner-build.F6cq0d/manifest.json`.
  This artifact is not yet installed on the guest and is not a public release.
- Updated VSIX installed in the Mac's VS Code. Installed/built extension bundle
  SHA-256 both equal `c04fda19cf5f3540fc3491d258a0da1496b543f0e86a214f1e532879d0bbe0d5`.
  Reloading and GUI validation remain pending because the Mac is locked.
- Next: unlock, reload, reconcile the existing target (GET, never redeploy),
  safely restart retained compute, approve the pinned guest upgrade, repeat
  complete inventory/import, apply preload/same-VM resize, then explicitly
  start migration/counts verification. Full independent P1 digest remains
  required before calling CSV-MAC qualified.
