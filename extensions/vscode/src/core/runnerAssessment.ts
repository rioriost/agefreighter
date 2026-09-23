import { createHash, randomUUID } from "node:crypto";
import { catalogActive, object, RunnerRecord } from "./runner";
import { RunnerControl } from "./runnerLifecycle";
import { assertIdleHealth, assertPostgreSQLTypePreservation, dispatchGuest, reconcileGuest } from "./runnerGuest";
import { csvAssessmentReady } from "./runnerCSV";
import { assertCosmosAccessCurrent, cosmosAccessReady } from "./runnerCosmosAccess";

export interface Assessment {
  operation: string; action: "profile" | "inventory"; phase: "submitted" | "unknown" | "accepted" | "running" | "finished" | "failed" | "interrupted";
  configurationSHA256: string; bootId: string; guestConfigurationSHA256?: string; reportSHA256?: string; reportBytes?: number;
}
const sha = /^[a-f0-9]{64}$/;
/** After interactive credential entry, refresh only VM health, never source reads.
 * Caller holds the workflow lock. Credentials are deliberately not an argument.
 * Preserve an uncertain command for reconciliation; never resubmit it here.
 */
export async function ensureAssessmentReadiness(control: RunnerControl, record: RunnerRecord, cancelled: () => boolean = () => false): Promise<RunnerRecord> {
  const checkCancelled = () => { if (cancelled()) throw new Error("Source assessment cancelled; no source read was submitted."); };
  checkCancelled();
  if (record.guestCommand && ["submitted", "unknown"].includes(record.guestCommand.phase)) throw new Error("Reconcile the pending guest command before source assessment.");
  if (record.guestCommand?.phase === "bootstrap-pending") throw new Error("Linux bootstrap is still pending; explicitly check readiness in the runner panel. No source read was submitted.");
  if (record.phase !== "provisioned" || !record.guestReady || assessmentActive(record) || record.migration) throw new Error("Review the provisioned runner and retained source operation first.");
  const boot = record.guestReady.bootId;
  const age = Date.now() - Date.parse(record.guestReady.checkedAt);
  // Leave a full minute for the following protected dispatch.
  if (Number.isFinite(age) && age >= 0 && age <= 240000) { assertIdleHealth(record); return record; }
  let current = await dispatchGuest(control, record, { version: 1, workflow: record.id, operation: randomUUID(), action: "ready" }, checkCancelled);
  for (let attempt = 0; attempt < 20; attempt++) {
    checkCancelled();
    current = (await reconcileGuest(control, current)).record;
    if (current.guestCommand?.phase === "failed") throw new Error("Linux readiness failed; no source read was submitted. Reconcile the runner.");
    if (current.guestCommand?.phase === "bootstrap-pending") throw new Error("Linux bootstrap is still pending; explicitly check readiness in the runner panel. No source read was submitted.");
    if (current.guestCommand?.phase === "finished") {
      if (current.guestReady?.bootId !== boot) throw new Error("The runner rebooted while awaiting input. Review the new boot before source reads.");
      assertIdleHealth(current); checkCancelled(); return current;
    }
    if (attempt < 19) await control.sleep(3000);
  }
  throw new Error("Linux readiness is still pending; reconcile it in the runner panel. No source read was submitted and credentials were not retained.");
}

export function assessmentActive(record: RunnerRecord): boolean {
  return catalogActive(record) || record.assessment !== undefined && (record.assessment.phase !== "finished" || !record.assessment.reportSHA256);
}

/** Explicit operator reconciliation only; preserves evidence and never starts a worker. */
export function retainFailedAssessment(record: RunnerRecord, operation: string, now = Date.now()): RunnerRecord {
  const a = record.assessment, ready = record.guestReady, command = record.guestCommand;
  if (record.phase !== "provisioned" || record.target || record.migration || !a || a.operation !== operation || a.phase !== "failed") throw new Error("Only this failed pre-target assessment can be retained for a fresh attempt.");
  // A terminal failure belongs to its historical boot. Deallocation/restart or
  // an approved idle upgrade must not make its evidence impossible to archive.
  // Require a newly reconciled, idle *current* installation instead. This does
  // not resume the old worker or permit unknown/interrupted operations to retry.
  if (command?.action !== "ready" || command.phase !== "finished" || !ready ||
      !/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/.test(ready.bootId) ||
      ready.checkedAt !== command.submittedAt ||
      (record.upgrade && record.upgrade.phase !== "finished") ||
      (record.artifact.development && ready.commit !== record.artifact.development.commit)) {
    throw new Error("Refresh successful Linux readiness for the current installation before retaining the failure.");
  }
  assertIdleHealth(record, now);
  const assessmentHistory = [...record.assessmentHistory ?? [], a];
  if (assessmentHistory.length > 16) throw new Error("Assessment history limit reached; retain evidence and review the workflow before continuing.");
  return { ...record, assessmentHistory, assessment: undefined };
}

/** Caller holds the workflow lock, reviewed the form and approved source reads. */
export async function startAssessment(control: RunnerControl, record: RunnerRecord, action: "profile" | "inventory", secrets: Record<string, string>, cancelled: () => boolean = () => false): Promise<RunnerRecord> {
  const checkCancelled = () => { if (cancelled()) throw new Error("Source assessment cancelled or workspace trust changed; no source read was submitted."); };
  checkCancelled();
  if(record.migration)throw new Error("The retained migration freezes source evidence; reconcile it instead of starting another assessment.");
  assertPostgreSQLTypePreservation(record);
  if (!record.sourceDraft?.canAssess || assessmentActive(record)) throw new Error("A reviewed source and a workflow without a retained assessment are required.");
  if (!cosmosAccessReady(record)) throw new Error("Grant and verify Cosmos Data Reader access for this runner first.");
  if (record.input.source.type === "csv" && !csvAssessmentReady(record)) throw new Error("Every mapped CSV requires an independently verified guest upload seal.");
  const inventoryCapability=`${record.input.source.type}-inventory-v1`;
  if (action === "inventory" && !record.guestReady?.capabilities?.includes(inventoryCapability)) throw new Error("The installed guest does not advertise complete inventory for this source. Use a reviewed matching runner artifact and refresh readiness; no request was submitted.");
  if (object(record.sourceDraft.configuration.source).type !== record.input.source.type) throw new Error("Source type changed after review.");
  await assertCosmosAccessCurrent(control, record);
  // Cosmos admission performs asynchronous ARM identity/role reads. Trust or
  // panel lifetime can change during either read, after the panel's own check.
  checkCancelled();
  const operation = randomUUID();
  const assessmentHistory = [...record.assessmentHistory ?? [], ...record.assessment ? [record.assessment] : []];
  if (assessmentHistory.length > 16) throw new Error("Assessment history limit reached; retain evidence and review the workflow before continuing.");
  const assessment: Assessment = { operation, action, phase: "submitted", bootId: record.guestReady?.bootId ?? "", configurationSHA256: createHash("sha256").update(JSON.stringify(record.sourceDraft.configuration)).digest("hex") };
  return dispatchGuest(control, { ...record, assessmentHistory, assessment }, { version: 1, workflow: record.id, operation, action, configuration: record.sourceDraft.configuration, secrets }, checkCancelled);
}

/** One bounded control step. Repeated clicks reconcile instead of re-running. */
export async function refreshAssessment(control: RunnerControl, record: RunnerRecord): Promise<RunnerRecord> {
  const assessment = record.assessment;
  if (!assessment) throw new Error("No retained source assessment.");
  const pending = record.guestCommand && ["submitted", "unknown"].includes(record.guestCommand.phase);
  if (!pending) return dispatchGuest(control, record, { version: 1, workflow: record.id, operation: assessment.operation, action: "status" });
  if (record.guestCommand!.operation !== assessment.operation || !["profile", "inventory", "status"].includes(record.guestCommand!.action)) throw new Error("Reconcile the other pending guest control first.");
  const checked = await reconcileGuest(control, record);
  if (!checked.result) return checked.record;
  const value = object(checked.result);
  if (value.action !== assessment.action || value.bootId !== assessment.bootId || typeof value.configSha256 !== "string" || !sha.test(value.configSha256) || assessment.guestConfigurationSHA256 && assessment.guestConfigurationSHA256 !== value.configSha256 || !["accepted", "running", "finished", "failed", "interrupted"].includes(String(value.phase))) throw new Error("Guest assessment identity or configuration evidence changed.");
  if (value.phase === "finished" && (value.exitCode !== 0 || !value.reportBytes || !value.reportSha256)) throw new Error("Finished worker lacks a successful exit and report manifest.");
  const next: RunnerRecord = { ...checked.record, assessment: { ...assessment, phase: value.phase as Assessment["phase"], guestConfigurationSHA256: value.configSha256 } };
  if (value.reportBytes !== undefined || value.reportSha256 !== undefined) {
    if (!Number.isSafeInteger(value.reportBytes) || Number(value.reportBytes) < 1 || Number(value.reportBytes) > 4 * 1024 * 1024 || typeof value.reportSha256 !== "string" || !sha.test(value.reportSha256)) throw new Error("Invalid guest report manifest.");
    next.assessment!.reportBytes = Number(value.reportBytes); next.assessment!.reportSHA256 = value.reportSha256;
    if (assessment.reportSHA256 && (assessment.reportSHA256 !== value.reportSha256 || assessment.reportBytes !== value.reportBytes)) throw new Error("Retained assessment report changed.");
  }
  await control.persist(next);
  return next;
}
