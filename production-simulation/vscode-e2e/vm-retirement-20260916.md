# Retired runner VMs — September 16, 2026 JST

Status: **complete** — 13 retired VMs deleted, all 13 original OS disks and
NICs preserved and independently verified. Six reusable VMs remain deallocated.

## Authorization and selection

The user explicitly requested deletion of trial VMs that will not be reused,
keeping OS disks where possible. This supersedes the earlier VM-object retention
requirement only for the exact retired set below. Subscription remains
`67c417f3-5a13-446c-afb9-40cd87f2fdb7`; resource group remains
`rg-af-vscode-p1-20260905-a`. No quota increase is requested by this cleanup.

All 13 selected VMs are former D4s_v5 migration runners. Their persisted GUI
migration and last guest command are finished, their VMs are deallocated, and
they are not used by the current recovery workflow. Successful and failed
historical qualifications remain historical; no completed graph is resumed.

| Retired VM | Retained OS disk |
| --- | --- |
| af-09bb0173608f4ab1bf62 | af-09bb0173608f4ab1bf62-os |
| af-1b2c7189b77141e69086 | af-1b2c7189b77141e69086-os |
| af-1f480fe1490d4789bc18 | af-1f480fe1490d4789bc18-os |
| af-22f11b89e9434d569675 | af-22f11b89e9434d569675-os |
| af-2595fb2ddf9d4582b8b8 | af-2595fb2ddf9d4582b8b8-os |
| af-29558917403e4a76aaa0 | af-29558917403e4a76aaa0-os |
| af-31ce478995344bd6bcac | af-31ce478995344bd6bcac-os |
| af-53625ae3b1554821bfc3 | af-53625ae3b1554821bfc3-os |
| af-75e4e5084f38467bb3c1 | af-75e4e5084f38467bb3c1-os |
| af-7b79f05d1dc140a6b3dc | af-7b79f05d1dc140a6b3dc-os |
| af-83c6b829acdc4405aa2d | af-83c6b829acdc4405aa2d-os |
| af-c275d043de934b0ab2b0 | af-c275d043de934b0ab2b0-os |
| af-d138f4e4bcf340fea876 | af-d138f4e4bcf340fea876-os |

Keep the current CSV recovery runner `af-54da6ddd27d245e0bb68`, together with
`af-n44-source`, `af-n526-source`, `af-op-n44-source`, `af-op-n526-source` and
`af-pgvm-source`. These five source fixtures are reusable for outstanding
network-source and negative/recovery coverage. All six remain stopped.

## Preservation procedure

1. Read exact ownership, power state, disk/NIC relationships and delete options.
   Require managed persistent OS disks and `Detach` for OS disk and each NIC;
   no selected VM has a data disk. Require no resource lock, deallocated power
   state and a finished final GUI operation. Historical ARM commands can retain
   Pending/Running even on a stopped VM; archive these states verbatim and do
   not start a retired guest to reconcile or replay them.
2. Archive VM configuration, OS disk identity, NIC configuration, all managed
   Run Command definitions/results, extensions and persisted workflow metadata.
   Keep raw bundles private outside Git, with mode 0600 inside a 0700 directory;
   publish only names, counts and SHA-256 seals. Do not export SecretStorage.
3. Recheck unchanged VM bindings and retention options immediately before each
   exact VM deletion. Do not use group deletion, force deletion or broad globs.
4. Verify each VM is absent while its original OS disk unique ID survives in
   `Unattached` state and its NIC survives detached. Verify retained VM set and
   PostgreSQL targets, then refresh actual regional/family quota usage.

Raw private archive directory:
`production-simulation/work/retired-vm-archive-20260916.UtmpRI/`.
OS disks retain guest files, configurations, logs and credentials; they remain
sensitive and billable. Disk contents were not rehashed block-by-block by this
procedure. Restoring a VM from a specialized OS disk is possible, but a new VM
has a new Azure VM identity/system-assigned principal; RBAC and boot-bound GUI
workflow state must be reviewed again. Do not claim transparent same-job resume
or replay a retired workflow from its old GUI record.

VM-associated command/extension ARM resources and the system-assigned identity
are removed with the VM. Their archival evidence does not recreate the old
identity. NICs, networks, transfer accounts/blobs, local workflow records,
source VMs and all PostgreSQL targets are outside this deletion scope.

## Archive seal

All 13 bundles include 270 managed Run Command resources (current ARM views),
13 VM/disk/NIC configurations and their local workflow records. SHA-256 of the
private manifest: `71279848f439c6561c461b14544271f960e96435c49218506953726a5edcb370`.
Independent file hashing verified every bundle against that manifest before
the first delete. Some command execution views are Pending with no start/end
time despite finished GUI operations; those values were preserved, not rewritten
as success. Historical local reports and on-disk guest evidence remain retained.
The manifest contains each original disk unique ID for post-delete comparison.

## Verified result

The last exact VM deletion was checked at `2026-09-15T19:29:06.824655Z`
(September 16 JST). A separate read-only verification then confirmed:

- Exactly the six keep-list VMs remain, all deallocated; no new VM was created.
- All 13 deleted VMs' original 64-GiB OS disks survive with matching unique IDs,
  in Unattached state (832 GiB retained). All associated NICs survive detached.
- The group's total 19 OS disks and 20 NICs remain. No disk, NIC, subnet,
  storage account/blob or PostgreSQL server was deleted by this operation.
- All 14 PostgreSQL servers remain Stopped, preserving their graphs/results.
- Japan East regional vCPU usage fell **100/101 → 48/101**; DSv5 family usage
  fell **96/100 → 44/100**. The current recovery runner's planned four-vCPU
  size now fits these observed quotas. No quota increase was submitted or is
  currently necessary; fresh preflight is still required before deployment.

Private post-verification report SHA-256:
`4d87672b8d89ac24d73c7240499cbc3f83d50593de6c6ac9ba6a7aff883b955a`.
Per-VM deletion/readback receipts and sealed archives are retained beside it.
Disk retention charges continue. No trial was started or marked qualified by
this cleanup. Keep historical GUI records as evidence, not reusable live VMs.
