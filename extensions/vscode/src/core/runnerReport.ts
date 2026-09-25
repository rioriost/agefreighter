import { object, RunnerRecord } from "./runner";
import { assertIdleHealth, dispatchGuest, reconcileGuest } from "./runnerGuest";
import { RunnerControl } from "./runnerLifecycle";
import { downloadReport, reportCapability, reportManifest, ReportManifest, ReportTransfer } from "./runnerBlob";
import { verifyReportStorage } from "./runnerReportStorage";

/** Caller approves an owned private destination and holds the workflow lock.
 * Capability minting/storage provisioning are separate; never accept a webview SAS. */
export async function startReportExport(control: RunnerControl, record: RunnerRecord, createCapability: string, operation?:string): Promise<RunnerRecord> {
  if (record.phase !== "provisioned") throw new Error("The report exporter requires the retained provisioned runner.");
  const assessment = operation === undefined ? record.assessment : [record.assessment, record.migration, record.postgresCatalog].find(value => value?.operation === operation);
  if (!assessment || !["finished", "failed"].includes(assessment.phase) || !assessment.reportSHA256 || !assessment.reportBytes) throw new Error("A terminal assessment and independently retained report manifest are required.");
  const manifest = reportManifest({ operation: assessment.operation, sha256: assessment.reportSHA256, bytes: assessment.reportBytes });
  if (record.reportTransfers?.some(value => value.operation === manifest.operation)) throw new Error("An export intent already exists. Reconcile or import its destination; never replay it.");
  if ((record.reportTransfers?.length ?? 0) >= 17) throw new Error("Report retention limit reached.");
  const url = reportCapability(createCapability, record.id, manifest.operation, "c");
  const transfer: ReportTransfer = { ...manifest, blob: `${url.origin}${url.pathname}`, phase: "submitted" };
  await verifyReportStorage(control, record, transfer.blob);
  await requireRunnerPower(control, record, "PowerState/running");
  const prior = record.rejectedReportExports?.filter(item => item.transfer.operation === manifest.operation) ?? [];
  if (prior.length) {
    if (prior.some(item => item.transfer.sha256 !== manifest.sha256 || item.transfer.bytes !== manifest.bytes || item.transfer.blob !== transfer.blob)) throw new Error("Recovered export must preserve the original report seal and destination.");
    assertIdleHealth(record);
  }
  const next = await dispatchGuest(control, { ...record, reportTransfers: [...record.reportTransfers ?? [], transfer] },
    { version: 1, workflow: record.id, operation: manifest.operation, action: "export-report", export: { url: createCapability, sha256: manifest.sha256, bytes: manifest.bytes } });
  if (next.guestCommand?.phase === "unknown") {
    next.reportTransfers = next.reportTransfers!.map(item => item.operation === manifest.operation ? { ...item, phase: "unknown" } : item);
    await control.persist(next);
  }
  return next;
}

async function requireRunnerPower(control: RunnerControl, record: RunnerRecord, expected: string): Promise<void> {
  const response = await control.request(record.input.subscriptionId, `${record.vmId}/instanceView?api-version=2024-07-01`);
  const statuses = object(response.value ?? {}).statuses;
  const power = Array.isArray(statuses) ? statuses.map(s => object(s).code).filter(c => typeof c === "string" && c.startsWith("PowerState/")) : [];
  if (response.status !== 200 || power.length !== 1 || power[0] !== expected) throw new Error(`Report transfer requires verified ${expected}; no export was submitted.`);
}

export function canRetainRejectedReportExport(record: RunnerRecord): boolean {
  const c = record.guestCommand, a = record.assessment;
  return record.phase === "provisioned" && !record.target && !record.migration &&
    c?.action === "export-report" && c.phase === "unknown" &&
    c.failure === "Azure returned HTTP 409; reconcile before retrying." &&
    a?.phase === "finished" && a.operation === c.operation &&
    record.reportTransfers?.some(t => t.operation === c.operation && t.phase === "unknown") === true;
}

/** Explicit, local evidence retention only. Never retry an ambiguous transport,
 * live command or existing blob. A fresh export requires separate approval and
 * fresh idle readiness after the operator starts the stopped runner. */
export async function retainRejectedReportExport(control: RunnerControl, record: RunnerRecord, readCapability: string,
  fetcher: typeof fetch = fetch, now = Date.now()): Promise<RunnerRecord> {
  if (!canRetainRejectedReportExport(record)) throw new Error("Only a retained HTTP 409 pre-target report export can be reviewed here.");
  const command = record.guestCommand!, assessment = record.assessment!;
  const transfer = record.reportTransfers!.find(t => t.operation === command.operation)!;
  const age = now - Date.parse(command.submittedAt);
  // Original capabilities expire within 15 minutes of submission validation.
  // Allow an additional five-minute margin before clearing a rejected intent.
  if (!Number.isFinite(age) || age < 1200000 || (record.rejectedReportExports?.length ?? 0) >= 16) throw new Error("Wait at least twenty minutes for the original capability to expire and preserve bounded export history before review.");
  if (!command.id.startsWith(`${record.vmId}/runCommands/af-`) ||
    !/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/.test(command.id.slice(`${record.vmId}/runCommands/af-`.length))) throw new Error("Report command is not owned by this runner.");
  reportManifest(transfer);
  if (assessment.reportSHA256 !== transfer.sha256 || assessment.reportBytes !== transfer.bytes) throw new Error("Independent report seal changed.");
  const url = reportCapability(readCapability, record.id, command.operation, "r", now);
  if (`${url.origin}${url.pathname}` !== transfer.blob) throw new Error("Report destination changed.");
  await verifyReportStorage(control, record, transfer.blob);
  await requireRunnerPower(control, record, "PowerState/deallocated");
  if ((await control.request(record.input.subscriptionId, `${command.id}?api-version=2024-07-01&$expand=instanceView`)).status !== 404) throw new Error("Retained export command is present or uncertain; reconcile it without replay.");
  let missing = false;
  try {
    const response = await fetcher(readCapability, { method: "HEAD", redirect: "error", signal: AbortSignal.timeout(30000) });
    missing = response.status === 404 && response.headers.get("x-ms-error-code") === "BlobNotFound";
  } catch { throw new Error("Report destination absence could not be verified."); }
  if (!missing) throw new Error("Report blob exists or is uncertain; import/reconcile it without replay.");
  const next: RunnerRecord = { ...record,
    rejectedReportExports: [...record.rejectedReportExports ?? [], { command: { ...command }, transfer: { ...transfer }, retainedAt: new Date(now).toISOString(), reason: "http409-command-and-blob-absent" }],
    reportTransfers: record.reportTransfers!.filter(t => t.operation !== command.operation),
    guestCommand: { ...command, phase: "failed" } };
  delete next.guestReady;
  await control.persist(next);
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
  const assessment = [record.assessment, ...record.assessmentHistory ?? [],record.migration,record.postgresCatalog].find(value => value?.operation === operation);
  if (!assessment || assessment.reportSHA256 !== transfer.sha256 || assessment.reportBytes !== transfer.bytes) throw new Error("Export no longer matches independent assessment evidence.");
  const url = reportCapability(readCapability, record.id, operation, "r");
  if (`${url.origin}${url.pathname}` !== transfer.blob) throw new Error("Report capability points to a different retained destination.");
  await verifyReportStorage(control, record, transfer.blob);
  const text = await downloadReport(readCapability, record.id, transfer, fetcher);
  await retain(record.id, reportManifest(transfer), text);
  const next: RunnerRecord = { ...record, reportTransfers: record.reportTransfers!.map(value => value.operation === operation ? { ...value, phase: "imported" } : value) };
  await control.persist(next); return next;
}
