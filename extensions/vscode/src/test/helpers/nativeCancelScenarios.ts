/** Synthetic prerequisites only. No Azure or VS Code dependency. These IDs are
 * a bounded subset of the frozen 30-decision B09 inventory, not new cases. */
import { randomUUID } from "node:crypto";
import { RunnerRecord, sourceWorkflowDraft } from "../../core/runner";
import { targetPreview } from "../../core/runnerTarget";
import { guestDispatchScript } from "../../core/runnerGuest";
import { retainReadinessReceipt } from "../../core/runnerReceipts";

export const nativeCancelCases = [
  { id: "A18", name: "execution.costApproval", action: "Review a new cost authorization (no Azure mutation)", title: "Record this new cost authorization?", module: "runnerExecutionPanel", exported: "continueRunnerExecution" },
  { id: "A19", name: "execution.preloadRestart", action: "Apply / reconcile AGE preload restart", title: "Apply the AGE preload configuration?", module: "runnerExecutionPanel", exported: "continueRunnerExecution" },
  { id: "A29", name: "receipt.remove", action: "", title: "Archive and permanently remove this one Azure readiness control record?", module: "runnerReceiptRemovalPanel", exported: "manageReadinessRemoval" }
] as const;
export type NativeCancelCase = typeof nativeCancelCases[number];

export function cancellationFixture(scenario: NativeCancelCase, now = Date.now()): {
  record: RunnerRecord; responses: Map<string, unknown>; inputValues: string[]; selectionIndexes: number[];
} {
  const id = randomUUID(), operation = randomUUID();
  let record = sourceWorkflowDraft(id, { subscriptionId: id, resourceGroup: "b09-isolated-synthetic", region: "japaneast", zone: "1", size: "Standard_B2s_v2",
    subnetId: `/subscriptions/${id}/resourceGroups/b09-isolated-synthetic/providers/Microsoft.Network/virtualNetworks/fixture/subnets/runner`,
    source: { type: "csv", location: "local" } });
  record.phase = "provisioned";
  record.artifact = { version: "2.4.0", sha256: "a".repeat(64), url: "https://example.invalid/synthetic-not-an-artifact" };
  const responses = new Map<string, unknown>();
  if (scenario.id === "A29") {
    const oldTime = new Date(now - 60_000).toISOString(), currentTime = new Date(now - 10_000).toISOString();
    record.guestCommand = { id: `${record.vmId}/runCommands/af-${operation}`, operation, action: "ready", phase: "finished", submittedAt: oldTime };
    record.guestReady = { bootId: id, cliVersion: "2.4.0", archiveSha256: record.artifact.sha256, commit: "synthetic-fixture", checkedAt: oldTime,
      health: { idle: true, storageUsedPercent: 12, swapUsedBytes: 0, oomEvents: 0 } };
    record = retainReadinessReceipt(record);
    const currentOperation = randomUUID();
    record.guestCommand = { ...record.guestCommand!, id: `${record.vmId}/runCommands/af-${currentOperation}`, operation: currentOperation, submittedAt: currentTime };
    record.guestReady = { ...record.guestReady!, checkedAt: currentTime };
    record = retainReadinessReceipt(record);
    responses.set(record.vmId, { id: record.vmId, location: "japaneast", zones: ["1"], tags: { application: "agefreighter", workflow: id, purpose: "discovery-and-migration" },
      properties: { vmId: id, provisioningState: "Succeeded", instanceView: { statuses: [{ code: "PowerState/running" }] } } });
    for (const receipt of record.readinessReceipts!) responses.set(receipt.command.id, { id: receipt.command.id, location: "japaneast", properties: {
      source: { script: guestDispatchScript }, timeoutInSeconds: 60, asyncExecution: false, provisioningState: "Succeeded", instanceView: {
        executionState: "Succeeded", exitCode: 0, startTime: receipt.command.submittedAt, endTime: new Date(Date.parse(receipt.command.submittedAt) + 1000).toISOString(), error: "",
        output: JSON.stringify({ version: 1, ready: true, os: "linux", architecture: "amd64", bootId: id, cliVersion: "2.4.0",
          archiveSha256: record.artifact.sha256, commit: "synthetic-fixture", health: receipt.readiness.health })
      }
    } });
    return { record, responses, inputValues: [], selectionIndexes: [0, 0] };
  }
  record.target = targetPreview(record, { serverName: "b09-synthetic-target", subnetCIDR: "10.0.2.0/24", postgresSKU: "Standard_D4ds_v5", postgresTier: "GeneralPurpose",
    storageGiB: 128, loaderSize: "Standard_D4s_v5", hourlyUSD: 2, additionalReserveUSD: 50, budgetUSD: 800, deadline: new Date(now + 3600_000).toISOString() },
  { operation, reportSHA256: "b".repeat(64), configurationSHA256: "c".repeat(64), artifactSHA256: record.artifact.sha256, sourceType: "csv",
    rows: "3", vertices: "2", edges: "1", storageHighBytes: "1024", labels: { "v.Person": 2, "e.KNOWS": 1 } });
  record.target.phase = "provisioned";
  responses.set(record.target.serverId, { tags: { workflow: id, application: "agefreighter", purpose: "migration-target" }, properties: { state: "Ready" } });
  responses.set(`${record.target.serverId}/configurations/shared_preload_libraries`, { properties: { value: "pg_stat_statements,age", isConfigPendingRestart: true } });
  return { record, responses, inputValues: [new Date(now + 3600_000).toISOString(), "800", "50"], selectionIndexes: [] };
}
