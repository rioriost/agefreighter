# AZ-PGVM corrective GUI attempt

Status: local source draft reviewed; new transfer storage/access grant awaits
action-time user approval. No Azure mutation or source read this turn.

## Preserved original

Original workflow `2595fb2d-df9d-4582-b8b8-237ae211ec1c`, migration job
`12a2462e-e5a3-4368-a356-54e292650051` and failed verifier
`c0efdd7b-fd18-49db-a872-bbd29d1736c2` remain unchanged. Their graph, source
data, configuration, reports and guest evidence are retained. The failed
verifier's active marker was not cleared. See the
[first execution sheet](az-pgvm-execution-20260913.md).

At 2026-09-14 00:33 UTC, the installed GUI selected the original target and
the full-P1 reconcile action. On the deallocated VM, the current ARM response
reports instanceView `Pending`, exitCode 0 and provisioningState `Succeeded`,
without the previous execution output. These do not supersede the retained
exit-1 failure evidence or prove success. The GUI keeps its local phase
`submitted`; no operation was replayed and no private state was manually patched.
Do not restart old compute merely to update this display.

## New source draft

- Workflow: `22f11b89-e943-4d56-9675-7331a78b6de7`.
- Name / proposed graph: `az-pgvm-p1-r2` / `az_pgvm_p1_r2`.
- Azure resource discovery selected the existing PostgreSQL source VM.
- Existing trial resource group, Japan East / zone 1 / existing runner subnet.
- Proposed discovery SKU: `Standard_B2s_v2`; no VM deployed yet.
- PostgreSQL 18, database `p1source`, existing read-only role and validated TLS.
- Selected public CA SHA-256:
  `0b196d1e310a5a68732879c7bf0aa8c9f1fc67bbfe436fe560a31de7afa05b68`.
- Saved generated configuration SHA-256 (JSON.stringify encoding):
  `f35bfa01b6f487ca3c4b84c501a40148f6ecaf2b6ff31d6305ad7beb83b6c7bc`.

All nine vertex and nine edge mappings were entered through visible input
fields. A read-only comparison of the saved form exactly matches
[the corrected fixture](fixtures/postgresql-p1-mappings.json), and the generated
configuration passes `assertP1Projection`. Every vertex explicitly includes
`source_key` and `external_id`; every edge explicitly includes `source_key` and
`relationship_id`, alongside the original typed properties and endpoints.
No password was entered or reset. No source read or migration has started.

## Current approval boundary

The installed GUI displays its native approval for new account
`af22f11b89e9434d56967573` and **Storage Blob Data Contributor for the signed-in
Azure user on that new account only**. The proposed authenticated HTTPS
endpoint is network-public; anonymous access and shared keys are disabled.
The source remains private. Standard LRS storage/request/egress charges apply.
The confirmation has not been accepted. This account and grant did not exist
in the saved draft's state; no storage deployment has been submitted.

The renewed USD 800 ceiling and `2026-09-16T07:14:35.311Z` deadline are unchanged.
The read-only Cost Management refresh returned HTTP 429 again; no fresh actual
total is claimed. Existing seven VMs are deallocated and five Flexible Servers
Stopped. No resource-group locks, failed activity events or policy-modify
events were returned for the checked interval starting September 13 23:00 UTC.

After approval, create/reconcile only this scoped storage; inspect live policy
effects before any transfer. A policy-disabled network must not be silently
re-enabled. Prepare the pinned qualification runner through the existing GUI
approval flow, deploy the private runner, validate guest and source health,
and perform a new complete inventory for the changed projection. Continue
through reviewed target deployment, same-VM resize, create-only migration,
strict counts and independent full canonical verification. None of these
later stages is complete; coverage remains 3/9.
