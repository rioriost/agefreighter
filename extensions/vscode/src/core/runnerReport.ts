import { object, RunnerRecord } from "./runner";
import { dispatchGuest, reconcileGuest } from "./runnerGuest";
import { RunnerControl } from "./runnerLifecycle";
import { downloadReport, reportCapability, reportManifest, ReportManifest, ReportTransfer } from "./runnerBlob";
import { verifyReportStorage } from "./runnerReportStorage";

/** Caller approves an owned private destination and holds the workflow lock.
 * Capability minting/storage provisioning are separate; never accept a webview SAS. */
export async function startReportExport(control: RunnerControl, record: RunnerRecord, createCapability: string): Promise<RunnerRecord> {
  if (record.phase !== "provisioned") throw new Error("The report exporter requires the retained provisioned runner.");
  const assessment = record.assessment;
  if (!assessment || !["finished", "failed"].includes(assessment.phase) || !assessment.reportSHA256 || !assessment.reportBytes) throw new Error("A terminal assessment and independently retained report manifest are required.");
  const manifest = reportManifest({ operation: assessment.operation, sha256: assessment.reportSHA256, bytes: assessment.reportBytes });
  if (record.reportTransfers?.some(value => value.operation === manifest.operation)) throw new Error("An export intent already exists. Reconcile or import its destination; never replay it.");
  if ((record.reportTransfers?.length ?? 0) >= 17) throw new Error("Report retention limit reached.");
  const url = reportCapability(createCapability, record.id, manifest.operation, "c");
  const transfer: ReportTransfer = { ...manifest, blob: `${url.origin}${url.pathname}`, phase: "submitted" };
  await verifyReportStorage(control, record, transfer.blob);
  const next = await dispatchGuest(control, { ...record, reportTransfers: [...record.reportTransfers ?? [], transfer] },
    { version: 1, workflow: record.id, operation: manifest.operation, action: "export-report", export: { url: createCapability, sha256: manifest.sha256, bytes: manifest.bytes } });
  if (next.guestCommand?.phase === "unknown") {
    next.reportTransfers = next.reportTransfers!.map(item => item.operation === manifest.operation ? { ...item, phase: "unknown" } : item);
    await control.persist(next);
  }
  return next;
}

/** GET-only reconciliation never creates another export, even after an absent receipt. */
export async function refreshReportExport(control: RunnerControl, record: RunnerRecord): Promise<RunnerRecord> {
  const command = record.guestCommand;
  const transfer = record.reportTransfers?.find(value => value.operation === command?.operation);
  if (!command || command.action !== "export-report" || !transfer) throw new Error("No retained report export command.");
  const checked = await reconcileGuest(control, record);
  let phase = transfer.phase;
  if (phase !== "imported") {
    const receipt = checked.result === undefined ? undefined : object(checked.result);
    phase = receipt?.exported === true && receipt.sha256 === transfer.sha256 && receipt.bytes === transfer.bytes ? "exported" : "unknown";
  }
  const next = { ...checked.record, reportTransfers: checked.record.reportTransfers!.map(value => value.operation === transfer.operation ? { ...value, phase } : value) };
  await control.persist(next); return next;
}

/** Import is safe after a lost PUT acknowledgement: exact bytes must match the
 * independently retained manifest. It does not clear an uncertain ARM command. */
export async function importReport(control: RunnerControl, record: RunnerRecord, operation: string, readCapability: string,
  retain: (workflow: string, manifest: ReportManifest, text: string) => Promise<void>, fetcher: typeof fetch = fetch): Promise<RunnerRecord> {
  const transfer = record.reportTransfers?.find(value => value.operation === operation);
  if (!transfer) throw new Error("No retained export destination.");
  const assessment = [record.assessment, ...record.assessmentHistory ?? []].find(value => value?.operation === operation);
  if (!assessment || assessment.reportSHA256 !== transfer.sha256 || assessment.reportBytes !== transfer.bytes) throw new Error("Export no longer matches independent assessment evidence.");
  const url = reportCapability(readCapability, record.id, operation, "r");
  if (`${url.origin}${url.pathname}` !== transfer.blob) throw new Error("Report capability points to a different retained destination.");
  await verifyReportStorage(control, record, transfer.blob);
  const text = await downloadReport(readCapability, record.id, transfer, fetcher);
  await retain(record.id, reportManifest(transfer), text);
  const next: RunnerRecord = { ...record, reportTransfers: record.reportTransfers!.map(value => value.operation === operation ? { ...value, phase: "imported" } : value) };
  await control.persist(next); return next;
}
