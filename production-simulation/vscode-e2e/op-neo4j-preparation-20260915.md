# IP-only Neo4j qualification preparation

Status: reviewed preparation, not GUI qualification. Coverage remains 7/9.

## Scope and gates

Use the existing dedicated P1 subscription/resource group. Renewed deadline is
2026-09-16T07:14:35.311Z, ceiling USD 800, conservative accrued/noncompute
reserve USD 400. Do not reset the window. Stop source/runner/target compute after
each route. Preserve all original sources, failed runs and qualification evidence.

## OP-N44 isolated source

The original `af-n44-source` is deallocated; its 128-GiB Linux Gen2 OS disk
contains the qualified P1 fixture. Create a cold disk copy and attach it to a
new specialized VM `af-op-n44-source`, D8s_v5, Japan East zone 1, without a
managed identity, public IP or new role assignment. Reuse the existing restricted
Neo4j source subnet, with verified available private IP `10.246.3.5`.
The subnet permits Bolt only from the runner subnet and denies other inbound
traffic. Original disk, VM, certificate and authentication remain untouched.

The original ARM model has no custom data, user data or VM extensions. A copied
Docker service may start, so the clone is fixture preparation only until inspected.
The copied account database is retained; do not extract its password or auth store.
Retain the cloned original container/certificate, stop it, and create an isolated
container against the copied P1 data with a new locally generated IP-SAN TLS
certificate. Export only the public CA. Retain private keys in root-owned guest
storage. Verify TLS against the literal IP, not insecure `+ssc`, and keep the
database read-only. Check free space, swap/OOM, image version and active services.

At 2026-09-15T01:16Z the source was stopped, the IP was available, no RG locks
were present, and no unexplained successful governance operation appeared in
the recent activity log. DSv5 quota was 68/100, regional total 70/101.
Azure retail Linux D8s_v5 is USD 0.496/hour. With the reviewed target/runner
USD 0.736/hour, a serial route costs USD 1.232/hour compute; even 30 hours is
USD 36.96 plus the USD 400 reserve and retained-resource charges. This is a
conservative planning estimate, not a fresh billing total (billing is delayed).

## Review and acceptance

Cold-copy avoids live database copying and mutation of a previously qualified
source. A specialized clone can retain hostname/host keys/local logs: it has no
public ingress or identity grant, and is not a production deployment template.
Inspect it before assessment; do not generalize the original disk or replay the
old bootstrap (which contains destructive container initialization).

Fixture-harness ARM operations are separate from the installed extension path.
The customer workflow must receive only IP, port, public CA and privately entered
credentials. It must not discover source resources, NICs or DNS through Azure.
Use GUI assessment, target review/deploy, same-runner resize, migration, strict
complete-count verification and the unchanged full 64-range canonical digest.
Source readiness, counts, CLI success or ARM success alone do not qualify a route.

OP-N526 follows the same isolation design only after its source/disk/network and
budget gates are independently checked. No second source is started speculatively.

## Execution / handoff

The cold copy and private specialized VM were created successfully. Original
`af-n44-source` remained deallocated. The reviewed preparation script passed
`bash -n` and ShellCheck. Guest checks showed 6% disk use, zero swap, no container
OOM and no unexpected application service. The copied old container is stopped
and retained; `af-op-n44-neo4j44` uses the copied database with TLS required and
read-only configuration. Both certificate-chain/IP verification and the actual
Bolt TLS handshake passed. Authenticated GUI assessment is still required.

Public CA SHA-256:
`91354fb6eb52e3c338b7c4333d6b8c325c9bcc9800c511ef04137cb344d901cd`.
The local public bundle is retained privately under
`production-simulation/work/op-n44-20260915/ca.crt`; no private key left the guest.
Server certificate SAN is `10.246.3.5`, valid September 15–22.

Installed GUI workflow `75e4e508-4f38-467b-b3c1-07ed05607603` selects Neo4j /
on-premises and a Japan East zone 1 B2s_v2 runner. Source resource discovery was
not used. The form has `op-n44-p1-r1`, namespace `p1`, host `10.246.3.5`, port
7687, database/user `neo4j`, stable vertex/edge property `source_key`; these fields
still need review/save with the CA selected. The release preview correctly stops
because public 2.4.0 release/checksums do not exist; reuse the separately reviewed
pinned development artifact through the GUI test-artifact flow, not a fake release.

Current GUI confirmation is for new transfer account `af75e4e5084f38467bb3c107`
and **Storage Blob Data Contributor on that account only** for the signed-in user.
No storage creation or role grant has been submitted. This action needs scoped
confirmation. The source clone is being deallocated while waiting; no runner or
target for this route has been deployed. Resume by reconciling the dialog, selecting
the public CA and fixed artifact, and later restarting only this clone before
authenticated assessment. All prior evidence remains retained.
