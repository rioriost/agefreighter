import { randomUUID } from "node:crypto";
import { object, RunnerArtifact, RunnerRecord } from "./runner";
import { RunnerControl } from "./runnerLifecycle";
import { developmentArtifact, developmentDownload } from "./runnerDevelopment";

export interface RunnerUpgrade {
  operation: string;
  commandId: string;
  phase: "submitted" | "unknown" | "finished" | "failed";
  previous: RunnerArtifact;
  artifact: RunnerArtifact;
  bootId: string;
  submittedAt: string;
}

export function assertUpgradeIdle(record: RunnerRecord): void {
  if (record.phase !== "provisioned" || record.upgrade && record.upgrade.phase !== "finished" ||
      record.guestCommand && ["submitted", "unknown"].includes(record.guestCommand.phase) ||
      record.assessment && !["finished", "failed"].includes(record.assessment.phase) ||
      record.csvTransfers?.some(x => x.phase !== "verified") ||
      record.reportTransfers?.some(x => x.phase !== "imported")) throw new Error("Reconcile every retained operation before upgrading this runner.");
  const ready = record.guestReady, age = ready ? Date.now() - Date.parse(ready.checkedAt) : NaN;
  if (!ready || !Number.isFinite(age) || age < 0 || age > 300000 || ready.archiveSha256 !== record.artifact.sha256 || ready.cliVersion !== record.artifact.version) throw new Error("Fresh matching guest readiness is required before an upgrade.");
}

/** No secrets or mutable download URLs. Old binaries and installation evidence
 * remain on disk. A partial switch fails readiness and requires reconciliation;
 * it is never retried automatically. The workflow lease excludes guest jobs. */
export function upgradeScript(record: RunnerRecord, upgrade: RunnerUpgrade): string {
  const artifact = developmentArtifact(record, { schemaVersion: 1, platform: "linux-amd64", commit: upgrade.artifact.development?.commit,
    version: upgrade.artifact.version, sha256: upgrade.artifact.sha256, bytes: upgrade.artifact.development?.bytes });
  if (JSON.stringify(artifact) !== JSON.stringify(upgrade.artifact) || !/^[a-f0-9-]{36}$/.test(record.id) || !/^[a-f0-9-]{36}$/.test(upgrade.operation) ||
      !/^[a-f0-9-]{36}$/.test(upgrade.bootId) || !/^[a-f0-9]{64}$/.test(upgrade.previous.sha256)) throw new Error("Invalid reviewed upgrade identity.");
  return `#!/bin/bash
set -euo pipefail
set +x
umask 077
base=/var/lib/agefreighter
test "$(cat /proc/sys/kernel/random/boot_id)" = '${upgrade.bootId}'
test "$(cat "$base/evidence/archive.sha256")" = '${upgrade.previous.sha256}'
test -f "$base/bootstrap.complete"
# Exclusive installation lock remains after any failure for operator review.
mkdir "$base/upgrade.lock"
root="$base/workflows/${record.id}"
(set -o noclobber; printf '%s' '${upgrade.operation}' > "$root/active")
for lease in "$base"/workflows/*/active; do
  test "$lease" = "$root/active" || { echo 'Another guest workflow is active'; exit 1; }
done
test "$(df -P "$base" | awk 'NR==2 {gsub(/%/,"",$5); print $5}')" -lt 80
install -d -m 0700 "$base/upgrades"
work="$base/upgrades/${upgrade.operation}"
mkdir -m 0700 "$work"
cp -p /usr/local/bin/agefreighter "$work/previous-agefreighter"
cp -p /usr/local/bin/agefreighter-tools "$work/previous-agefreighter-tools"
cp -p "$base/evidence/archive.sha256" "$work/previous-archive.sha256"
cp -p "$base/evidence/version.txt" "$work/previous-version.txt"
${developmentDownload(artifact)}
printf '%s  %s\\n' '${artifact.sha256}' "$work/archive.tar.gz" | sha256sum --check --status
tar -xOzf "$work/archive.tar.gz" agefreighter > "$work/agefreighter"
tar -xOzf "$work/archive.tar.gz" agefreighter-tools > "$work/agefreighter-tools"
chmod 0755 "$work/agefreighter" "$work/agefreighter-tools"
"$work/agefreighter" version > "$work/version.txt"
"$work/agefreighter-tools" version > "$work/tools-version.txt"
grep -F '${artifact.version}' "$work/version.txt" >/dev/null
grep -F '${artifact.development!.commit}' "$work/version.txt" >/dev/null
sed 's/^agefreighter-tools /agefreighter /' "$work/tools-version.txt" | cmp -s - "$work/version.txt"
test "$(cat /proc/sys/kernel/random/boot_id)" = '${upgrade.bootId}'
# Per-file atomic publication; the archive marker changes only after both.
install -m 0755 "$work/agefreighter" '/usr/local/bin/agefreighter.${upgrade.operation}'
install -m 0755 "$work/agefreighter-tools" '/usr/local/bin/agefreighter-tools.${upgrade.operation}'
mv '/usr/local/bin/agefreighter.${upgrade.operation}' /usr/local/bin/agefreighter
mv '/usr/local/bin/agefreighter-tools.${upgrade.operation}' /usr/local/bin/agefreighter-tools
cp "$work/version.txt" "$base/evidence/version.txt"
printf '%s\\n' '${artifact.sha256}' > "$work/archive.sha256"
cp "$work/archive.sha256" "$base/evidence/archive.sha256"
sync
/usr/local/bin/agefreighter-tools runner dispatch <<'AF_READY' > "$work/readiness.json"
{"version":1,"workflow":"${record.id}","operation":"${upgrade.operation}","action":"ready"}
AF_READY
test "$(cat "$root/active")" = '${upgrade.operation}'
rm "$root/active"
rmdir "$base/upgrade.lock"
cat "$work/readiness.json"
`;
}

/** Caller holds the workflow lock after artifact upload and modal approval. */
export async function submitUpgrade(control: RunnerControl, record: RunnerRecord, artifact: RunnerArtifact): Promise<RunnerRecord> {
  assertUpgradeIdle(record);
  if (artifact.sha256 === record.artifact.sha256) throw new Error("This artifact is already installed.");
  if ((await control.list(record.input.subscriptionId, `${record.vmId}/runCommands?api-version=2024-07-01`)).length >= 25) throw new Error("Archive completed command evidence before upgrading: Azure command limit reached.");
  const operation = randomUUID();
  const upgrade: RunnerUpgrade = { operation, commandId: `${record.vmId}/runCommands/af-${operation}`, phase: "submitted", previous: record.artifact, artifact,
    bootId: record.guestReady!.bootId, submittedAt: new Date().toISOString() };
  const script = upgradeScript(record, upgrade);
  if ((await control.request(record.input.subscriptionId, `${upgrade.commandId}?api-version=2024-07-01`)).status !== 404) throw new Error("Upgrade command collision.");
  const next: RunnerRecord = { ...record, upgrade, upgradeHistory: [...record.upgradeHistory ?? [], ...record.upgrade ? [record.upgrade] : []] };
  if (next.upgradeHistory!.length > 16) throw new Error("Upgrade evidence history limit reached.");
  delete next.guestReady;
  await control.persist(next);
  try {
    const response = await control.request(record.input.subscriptionId, `${upgrade.commandId}?api-version=2024-07-01`, "PUT", {
      location: record.input.region, properties: { source: { script }, timeoutInSeconds: 1200, asyncExecution: true }
    });
    if (response.status < 200 || response.status >= 300) throw new Error();
  } catch { next.upgrade = { ...upgrade, phase: "unknown" }; await control.persist(next); }
  return next;
}

/** GET only. A timeout or missing command never permits replay or rollback. */
export async function refreshUpgrade(control: RunnerControl, record: RunnerRecord): Promise<RunnerRecord> {
  const upgrade = record.upgrade;
  if (!upgrade || upgrade.phase === "finished" || upgrade.phase === "failed") return record;
  if (upgrade.commandId !== `${record.vmId}/runCommands/af-${upgrade.operation}`) throw new Error("Upgrade command identity changed.");
  const response = await control.request(record.input.subscriptionId, `${upgrade.commandId}?api-version=2024-07-01&$expand=instanceView`);
  if (response.status === 404) return record;
  const properties = object(object(response.value).properties), view = properties.instanceView ? object(properties.instanceView) : {};
  if (!["Succeeded", "Failed", "Canceled", "TimedOut"].includes(String(view.executionState))) return record;
  let accepted = false;
  try {
    const value = object(JSON.parse(String(view.output)));
    accepted = view.executionState === "Succeeded" && view.exitCode === 0 && value.version === 1 && value.ready === true && value.bootId === upgrade.bootId &&
      value.os === "linux" && value.architecture === "amd64" && value.archiveSha256 === upgrade.artifact.sha256 && value.cliVersion === upgrade.artifact.version && value.commit === upgrade.artifact.development?.commit;
  } catch { /* No arbitrary guest diagnostics are surfaced. */ }
  const next: RunnerRecord = { ...record, upgrade: { ...upgrade, phase: accepted ? "finished" : "failed" } };
  if (accepted) next.artifact = upgrade.artifact;
  delete next.guestReady;
  await control.persist(next);
  return next;
}
