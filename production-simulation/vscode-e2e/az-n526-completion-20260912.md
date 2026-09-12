# AZ-N526 completion execution sheet

## Final outcome — PASS at 2026-09-12 13:16Z

Installed VS Code 1.136.1 completed the designed Azure Neo4j discovery and
migration path. The source was Neo4j 5.26.30 on an Azure VM in Japan East zone
1. Complete inventory returned exactly 1,600,000 vertices and 4,000,000 edges.

The GUI used the existing private PostgreSQL 18 Flexible Server with AGE in the
same region and zone, and the same discovery VM previously resized from
Standard_B2s_v2 to Standard_D4s_v5. The retained disk, NIC and managed identity
matched the reviewed preservation digest; no substitute runner was introduced.

Before the accepted run, two create-only attempts failed before graph or
metadata creation because the retained Neo4j credential was rejected. Their
empty-target diagnostics remain in GUI history and were never resumed or
replayed. The native credential was recovered with Neo4j's system-database
recovery procedure on an unpublished, loopback-only container. Authentication
was restored, the original container resumed, authenticated reads returned the
unchanged exact source counts, and a checksummed pre-change system-database
backup remains on the guest. Temporary password material and the secret-bearing
Azure command were removed after the VS Code SecretStorage handoff.

Fresh job `313f6dca-680b-4379-9eac-a7539cb95792` ran from
2026-09-12T12:50:37.853654116Z to 2026-09-12T12:56:46.579313305Z. It committed
all 5,600,000 records: exactly 1,600,000 vertices and 4,000,000 edges, with zero
rejects. The GUI imported the 9,619-byte migration report, verified its SHA-256
`4816412aece55c3c70170779503c6ba0477b2ac903c6b4ea154e1a68afd678dc`,
and displayed a passing exact-count result.

The isolated read-only P1 verifier was built from commit
`19026db1930a7893ac4fb30f8647e1c277fe9920`. It regenerated the frozen fixture
and compared typed properties, identities and endpoints across all 64 canonical
ranges. Expected and actual record counts were 5,600,000, and both roots were
`bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.
VS Code imported and displayed `P1 full canonical digest: PASS`. The 23,215-byte
report has SHA-256
`fe3f6e23e19230c8f54de8615bcbe4bf15932bacbc516ed878515c0cb2d90bff`.

The final GUI guest check reported the runner idle with 5.18% storage use, zero
swap and zero boot OOM events. Azure Monitor reported 12.79% target storage use.
No failed Azure activity event was present from migration start through
shutdown. The source and runner VMs are deallocated and the Flexible Server is
Stopped. Graph data, source data, backups, raw reports and guest evidence remain
retained. Flexible Server's seven-day automatic restart behavior still applies.

This qualifies **AZ-N526 only**. It demonstrates the complete Azure-resource
selection, private discovery VM, same-VM resize, target deployment, Neo4j 5.26
migration and full P1 verification branch. It does not qualify PostgreSQL,
Cosmos DB or IP-and-port-only source branches. The committed
[redacted evidence](evidence/az-n526-qualified-20260912.json) contains no
credentials, SAS URLs or local private paths.
