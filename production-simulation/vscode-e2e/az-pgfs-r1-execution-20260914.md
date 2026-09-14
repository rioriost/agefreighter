# AZ-PGFS r1 installed-GUI qualification

Status at 2026-09-14T09:09Z: source configuration saved and reviewed in the
installed VS Code GUI. Dedicated transfer-storage approval is pending.
No assessment, migration or target deployment has started. Overall qualified
route coverage remains 5/9.

## Scope and preserved source

- Workflow: `29558917-403e-4a76-aaa0-de07122ea9c6`.
- Source selection: PostgreSQL / Azure, then the approved subscription,
  existing trial resource group and ARM-discovered Flexible Server candidate.
- Retained PostgreSQL 18 source: `afpg-p1-source-20260907`, database `p1source`,
  read-only user `agefreighter_reader`; no credential is included here.
- Private networking and TLS certificate validation remain enabled. No source
  public endpoint or firewall change was made; the Linux runner uses the
  existing trial VNet's runner subnet.
- Source data was not recreated. The frozen P1 fixture has 1,600,000 vertices
  and 4,000,000 edges; fresh GUI inventory and final verification remain required.
- All 18 saved GUI mappings exactly match `fixtures/postgresql-p1-mappings.json`.
- SHA-256 of the JSON-serialized reviewed source configuration:
  `b1d422780ada2953cfd6b252331a86c97c94138df6111568ad247ad3bf89f8f4`.

## Gates and current handoff

At the initial preflight all ten trial VMs were deallocated, all eight Flexible
Servers were stopped, no resource-group locks were present, and the recent
activity query returned no policy actions. The renewed ceiling remains USD 800
and the deadline remains `2026-09-16T07:14:35.311Z`; neither is reset by this route.
The last returned, delayed cost was USD 36.7436452084183, not final billing.

Only the retained AZ-PGFS source was started and subsequently observed Ready.
A concurrent expiry-tag update returned `ServerIsBusy`; its old expiry tag must
be reconciled separately once a fresh read succeeds. Do not replay the start
or infer that the tag was updated. All other sources, targets and runners remain
preserved. Source compute and retained storage charges continue while waiting.

The GUI is paused at creation of storage account `af29558917403e4a76aaa0de`
in the existing trial group. Its explicit confirmation describes a grant of
Storage Blob Data Contributor to the signed-in user on this new account only,
an HTTPS network-public endpoint, disabled anonymous access and disabled shared
keys. This grant has **not** been submitted. No source server is exposed.

## Remaining sequence

1. Approve and reconcile the scoped transfer storage, then upload the already
   reviewed Linux artifact (no new build or release).
2. Preview and provision the private B2s_v2 discovery runner; verify guest
   readiness, artifact, storage, idle state and swap/OOM.
3. Obtain the source password privately through VS Code; run complete inventory
   and import the checksummed result through the GUI.
4. Review sizing, save the LoadJob and provision a fresh private AGE target;
   resize the same runner and refresh readiness.
5. Run migration and strict complete verification, then compare all 64 P1
   digest ranges and the canonical root through the GUI and independently.
6. Only on full qualification, update route coverage; preserve evidence and
   stop/deallocate this route's compute resources.

The intended Linux artifact remains `2.4.0-dev.9ef16968363b`, archive SHA-256
`10a27dd02b53f070ad2529b21c40c5d382a6e67cb59d9175f98cc29531ff8997`.
The expected P1 canonical root remains
`bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70`.
