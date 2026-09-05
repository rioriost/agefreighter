import { createHash, randomUUID } from "node:crypto";
import { object, RunnerRecord } from "./runner";
import { RunnerControl } from "./runnerLifecycle";
import { dispatchGuest, reconcileGuest } from "./runnerGuest";

export interface Assessment {
  operation: string; action: "profile" | "inventory"; phase: "submitted" | "unknown" | "accepted" | "running" | "finished" | "failed" | "interrupted";
  configurationSHA256: string; bootId: string; guestConfigurationSHA256?: string; reportSHA256?: string; reportBytes?: number;
}
const sha = /^[a-f0-9]{64}$/;
export function assessmentActive(record: RunnerRecord): boolean {
  return record.assessment !== undefined && (record.assessment.phase !== "finished" || !record.assessment.reportSHA256);
}

/** Caller holds the workflow lock, reviewed the form and approved source reads. */
export async function startAssessment(control: RunnerControl, record: RunnerRecord, action: "profile" | "inventory", secrets: Record<string, string>): Promise<RunnerRecord> {
  if (!record.sourceDraft?.canAssess || assessmentActive(record)) throw new Error("A reviewed source and a workflow without a retained assessment are required.");
  if (action === "inventory" && record.input.source.type !== "neo4j") throw new Error("Exact inventory currently supports Neo4j only.");
  if (object(record.sourceDraft.configuration.source).type !== record.input.source.type) throw new Error("Source type changed after review.");
  const operation = randomUUID();
  const assessmentHistory = [...record.assessmentHistory ?? [], ...record.assessment ? [record.assessment] : []];
  if (assessmentHistory.length > 16) throw new Error("Assessment history limit reached; retain evidence and review the workflow before continuing.");
  const assessment: Assessment = { operation, action, phase: "submitted", bootId: record.guestReady?.bootId ?? "", configurationSHA256: createHash("sha256").update(JSON.stringify(record.sourceDraft.configuration)).digest("hex") };
  return dispatchGuest(control, { ...record, assessmentHistory, assessment }, { version: 1, workflow: record.id, operation, action, configuration: record.sourceDraft.configuration, secrets });
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
