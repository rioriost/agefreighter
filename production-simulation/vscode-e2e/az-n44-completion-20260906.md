# AZ-N44 completion execution sheet

## Final outcome — PASS at 2026-09-06 12:59Z

Installed VS Code 1.136.1 completed the designed Azure Neo4j discovery and
migration path. The source was Neo4j 4.4.48 on an Azure VM in Japan East zone
1. Complete inventory used Neo4j's transactional count store and returned
1,600,000 vertices plus 4,000,000 edges with no errors or incomplete checks.

The GUI created a private PostgreSQL 18 Flexible Server with AGE in the same
region and zone, then resized the same discovery VM from Standard_B2s_v2 to
Standard_D4s_v5. The retained disk, NIC and managed identity passed the
canonical preservation check. No second runner was substituted.

Migration job `45e2d8bb-641b-424d-9074-d55e14b6ac2a` ran from
12:33:18.94271862Z to 12:37:25.384258304Z. All 5,600,000 records were read and
committed, including exactly 1,600,000 vertices and 4,000,000 edges. Per-label
physical and identity counts agreed, and rejects were zero. The independently
imported counts report passed with SHA-256
`587685bad7ac7d7c8767dc0a20176098fa85d8f0b473af68b0d81e94e1324b95`.

The isolated read-only verifier then regenerated the frozen P1 fixture and
compared every typed property, identity and endpoint across all 64 canonical
ranges. Expected and actual record counts were 5,600,000, and both canonical
roots were
`bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.
The 23,214-byte verifier report has SHA-256
`0b8ca5db3bc3b4bb78c2afd4bdb24fda1edcb40aa976498054eb1697d47d3fed`.

All three compute services were stopped after qualification: the source and
runner VMs are deallocated, and the Flexible Server is Stopped. Target storage
was 13.96%, runner storage 3.52%, swap zero and boot OOM count zero before the
final verifier. No failed Azure activity event was present from migration start
through shutdown. Resources, graphs and raw reports remain retained; nothing
was deleted. Flexible Server's seven-day automatic restart behavior still
applies while the retained server exists.

This qualifies **AZ-N44 only**. It demonstrates the complete Azure-resource
selection, private discovery VM, same-VM resize, target deployment, Neo4j 4.4
migration and full P1 verification branch. It does not qualify Neo4j 5.x,
PostgreSQL, Cosmos DB or IP-and-port-only source branches. The committed
[redacted evidence](evidence/az-n44-qualified-20260906.json) contains no
credentials, SAS URLs or local paths; full raw evidence remains private.
