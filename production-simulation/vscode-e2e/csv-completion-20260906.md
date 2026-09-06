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

Status: implementation/execution in progress. No new CSV-MAC migration pass yet.
