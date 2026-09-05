import { randomUUID } from "node:crypto";
import { object, RunnerRecord } from "./runner";
import { RunnerControl } from "./runnerLifecycle";
import { dispatchGuest, reconcileGuest } from "./runnerGuest";
import { csvCapability } from "./runnerBlob";
import { CSVManifest, validateCSVManifest } from "../guided/csvTransfer";
import { verifyTransferStorage } from "./runnerReportStorage";

export function csvAssessmentReady(record: RunnerRecord): boolean {
  if (record.input.source.type !== "csv" || !record.sourceDraft) return false;
  const csv = object(object(record.sourceDraft.configuration.source).csv), rows = [...(Array.isArray(csv.vertices) ? csv.vertices : []), ...(Array.isArray(csv.edges) ? csv.edges : [])];
  return rows.length > 0 && rows.every(raw => {
    const row = object(raw), prefix = `/var/lib/agefreighter/workflows/${record.id}/uploads/`;
    return record.csvTransfers?.some(item => item.phase === "verified" && row.path === `${prefix}${item.file}.csv`);
  });
}

export async function startCSVImport(control: RunnerControl, record: RunnerRecord, manifest: CSVManifest, capability: string): Promise<RunnerRecord> {
  // A CSVTransfer is structurally a CSVManifest, but includes controller-only
  // phase/history fields. The strict Go decoder must receive only the wire DTO.
  manifest = validateCSVManifest(manifest); csvCapability(capability, record.id, manifest.file, manifest.sha256);
  const retained = record.csvTransfers?.find(item => item.file === manifest.file);
  if (record.input.source.type !== "csv" || !record.sourceFiles?.some(item => item.id === manifest.file) || retained?.phase !== "uploaded" || retained.bytes !== manifest.bytes || retained.sha256 !== manifest.sha256) throw new Error("Only a reviewed uploaded CSV can be imported.");
  if (record.csvTransfers?.some(item => ["submitted", "unknown", "interrupted"].includes(item.phase))) throw new Error("Reconcile the retained CSV import before starting another.");
  await verifyTransferStorage(control, record);
  const operation = randomUUID();
  return dispatchGuest(control, { ...record, csvTransfers: record.csvTransfers!.map(item => item.file === manifest.file ? { ...item, operation, phase: "submitted" } : item) },
    { version: 1, workflow: record.id, operation, action: "import-csv", import: { ...manifest, url: capability } });
}

export async function refreshCSVImport(control: RunnerControl, record: RunnerRecord): Promise<RunnerRecord> {
  const transfer = record.csvTransfers?.find(item => ["submitted", "unknown"].includes(item.phase));
  if (!transfer?.operation) throw new Error("No retained CSV import to refresh.");
  // A proven decoder rejection precedes operation creation or source reads.
  // Retain the rejected command and permit a NEW user-approved attempt, never
  // replaying the failed request. All other failures stay reconciliation-only.
  const command = record.guestCommand;
  if (command?.phase === "failed" && command.action === "import-csv" && command.operation === transfer.operation &&
      command.id.startsWith(`${record.vmId}/runCommands/af-`) && /^[a-f0-9-]{36}$/.test(command.id.slice(`${record.vmId}/runCommands/af-`.length))) {
    const response = await control.request(record.input.subscriptionId, `${command.id}?api-version=2024-07-01&$expand=instanceView`);
    const view = response.status === 200 ? object(object(object(response.value).properties).instanceView) : {};
    if (view.executionState === "Failed" && view.exitCode === 1 && view.output === "" && view.error === "invalid runner request\n") {
      const next: RunnerRecord = { ...record, csvTransfers: record.csvTransfers!.map(item => item.file === transfer.file ? {
        ...item, phase: "uploaded", operation: undefined,
        rejectedImports: [...item.rejectedImports ?? [], { operation: command.operation, commandId: command.id, reason: "protocol-rejected-before-execution" as const }]
      } : item) };
      await control.persist(next); return next;
    }
  }
  if (!record.guestCommand || !["submitted", "unknown"].includes(record.guestCommand.phase)) return dispatchGuest(control, record, { version: 1, workflow: record.id, operation: transfer.operation, action: "status" });
  if (record.guestCommand.operation !== transfer.operation || !["import-csv", "status"].includes(record.guestCommand.action)) throw new Error("Reconcile the other guest command first.");
  const checked = await reconcileGuest(control, record);
  if (!checked.result) return checked.record;
  const result = object(checked.result);
  if (result.action !== "import-csv" || result.fileId !== transfer.file || result.fileBytes !== transfer.bytes || result.fileSha256 !== transfer.sha256 || !["accepted", "running", "finished", "failed", "interrupted"].includes(String(result.phase))) throw new Error("CSV receipt does not match the independently reviewed manifest.");
  if (result.phase === "finished" && result.exitCode !== 0) throw new Error("CSV import lacks a successful full-hash seal.");
  const phase = result.phase === "finished" ? "verified" : result.phase === "failed" ? "failed" : result.phase === "interrupted" ? "interrupted" : "submitted";
  const next: RunnerRecord = { ...checked.record, csvTransfers: checked.record.csvTransfers!.map(item => item.file === transfer.file ? { ...item, phase } : item) };
  await control.persist(next); return next;
}
