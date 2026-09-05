import { createHash, randomUUID } from "node:crypto";
import { object, RunnerRecord } from "./runner";
import { RunnerControl } from "./runnerLifecycle";

export interface GuestCommand {
  id: string;
  operation: string;
  action: "ready" | "profile" | "inventory" | "status" | "report";
  phase: "submitted" | "unknown" | "finished" | "failed";
  submittedAt: string;
}
export interface GuestReadiness { bootId: string; cliVersion: string; archiveSha256: string; commit: string; checkedAt: string }
export interface GuestRequest { version: 1; workflow: string; operation: string; action: GuestCommand["action"]; expectedBootId?: string; configuration?: unknown; secrets?: Record<string, string>; offset?: number }

const uuid = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/;

// Constant script: credentials/configuration never occur in source, public
// parameters, resource IDs, status records, or command-line arguments.
export const guestDispatchScript = `#!/bin/bash
set -euo pipefail
set +x
umask 077
printf '%s' "$AF_RUNNER_REQUEST" | base64 --decode | /usr/local/bin/agefreighter-tools runner dispatch
`;

/** Caller holds the workflow lock and has obtained approval for source reads. */
export async function dispatchGuest(control: RunnerControl, record: RunnerRecord, request: GuestRequest): Promise<RunnerRecord> {
  if (record.phase !== "provisioned") throw new Error("The runner VM must be provisioned first.");
  if (record.guestCommand && ["submitted", "unknown"].includes(record.guestCommand.phase)) throw new Error("Reconcile the pending guest command; do not resubmit it.");
  if (request.version !== 1 || request.workflow !== record.id || !/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/.test(request.operation) || !["ready", "profile", "inventory", "status", "report"].includes(request.action)) throw new Error("Invalid guest request identity or action.");
  if (["ready", "status", "report"].includes(request.action) && (request.configuration !== undefined || request.secrets !== undefined || request.expectedBootId !== undefined)) throw new Error("Read-only guest controls cannot contain source credentials.");
  const assessment = ["profile", "inventory"].includes(request.action);
  if (assessment) {
    const ready = record.guestReady;
    const age = ready ? Date.now() - Date.parse(ready.checkedAt) : NaN;
    if (!ready || !uuid.test(ready.bootId) || !Number.isFinite(age) || age < 0 || age > 5 * 60 * 1000 || ready.cliVersion !== record.artifact.version || ready.archiveSha256 !== record.artifact.sha256 || request.configuration === undefined) throw new Error("Verify fresh guest readiness and review source configuration first.");
  }
  // Bind execution to the verified boot, not a value supplied by a webview.
  const payload = JSON.stringify(assessment ? { ...request, expectedBootId: record.guestReady!.bootId } : request);
  if (Buffer.byteLength(payload) > 1024 * 1024) throw new Error("Guest request is too large.");
  const command: GuestCommand = { id: `${record.vmId}/runCommands/af-${randomUUID()}`, operation: request.operation, action: request.action, phase: "submitted", submittedAt: new Date().toISOString() };
  if ((await control.request(record.input.subscriptionId, `${command.id}?api-version=2024-07-01`)).status !== 404) throw new Error("Guest command resource already exists.");
  const submitted: RunnerRecord = { ...record, guestCommand: command };
  await control.persist(submitted);
  try {
    const response = await control.request(record.input.subscriptionId, `${command.id}?api-version=2024-07-01`, "PUT", {
      location: record.input.region,
      properties: { source: { script: guestDispatchScript }, protectedParameters: [{ name: "AF_RUNNER_REQUEST", value: Buffer.from(payload).toString("base64") }], timeoutInSeconds: 60, asyncExecution: false }
    });
    if (response.status < 200 || response.status >= 300) throw new Error();
    return submitted;
  } catch {
    const unknown: RunnerRecord = { ...submitted, guestCommand: { ...command, phase: "unknown" } };
    await control.persist(unknown);
    return unknown;
  }
}

/** GET only. A failed or absent control response never repeats a source read. */
export async function reconcileGuest(control: RunnerControl, record: RunnerRecord): Promise<{ record: RunnerRecord; result?: unknown }> {
  const command = record.guestCommand;
  if (!command) throw new Error("No retained guest command to reconcile.");
  if (!command.id.startsWith(`${record.vmId}/runCommands/af-`) || !uuid.test(command.id.slice(`${record.vmId}/runCommands/af-`.length))) throw new Error("Guest command does not belong to this VM.");
  const response = await control.request(record.input.subscriptionId, `${command.id}?api-version=2024-07-01&$expand=instanceView`);
  if (response.status === 404) return { record };
  const properties = object(object(response.value).properties);
  const view = properties.instanceView ? object(properties.instanceView) : {};
  if (!["Succeeded", "Failed", "Canceled", "TimedOut"].includes(String(view.executionState))) return { record };
  let next: RunnerRecord = { ...record, guestCommand: { ...command, phase: "failed" } };
  let result: unknown;
  if (view.executionState === "Succeeded" && view.exitCode === 0 && typeof view.output === "string" && Buffer.byteLength(view.output) < 4096) {
    try {
      result = JSON.parse(view.output);
      const value = object(result);
      if (value.version !== 1) throw new Error();
      if (command.action === "ready") {
        if (value.ready !== true || value.os !== "linux" || value.architecture !== "amd64" || value.cliVersion !== record.artifact.version || value.archiveSha256 !== record.artifact.sha256 || typeof value.bootId !== "string" || !uuid.test(value.bootId) || typeof value.commit !== "string") throw new Error();
        // Re-reading an old ARM response must never refresh its validity.
        next.guestReady = { bootId: value.bootId, cliVersion: value.cliVersion, archiveSha256: value.archiveSha256, commit: value.commit, checkedAt: command.submittedAt };
      } else if (value.operation !== command.operation || command.action !== "report" && value.workflow !== record.id) throw new Error();
      next.guestCommand = { ...command, phase: "finished" };
    } catch { result = undefined; }
  }
  if (command.action === "ready" && next.guestCommand?.phase !== "finished") delete next.guestReady;
  await control.persist(next);
  return { record: next, result };
}

/** Assemble all chunks and verify the independently retained artifact hash. */
export function assembleGuestReport(operation: string, chunks: unknown[], expectedSHA256: string): string {
  let offset = 0, total: number | undefined;
  const parts: Buffer[] = [];
  if (!chunks.length || chunks.length > 2800) throw new Error("Incomplete guest report.");
  for (const raw of chunks) {
    const chunk = object(raw);
    if (chunk.version !== 1 || chunk.operation !== operation || chunk.offset !== offset || chunk.sha256 !== expectedSHA256 || !Number.isSafeInteger(chunk.total) || Number(chunk.total) < 1 || Number(chunk.total) > 4 * 1024 * 1024 || total !== undefined && chunk.total !== total || typeof chunk.data !== "string") throw new Error("Guest report chunk identity mismatch.");
    total = Number(chunk.total);
    const data = Buffer.from(chunk.data, "base64");
    if (!data.length || data.length > 1536 || data.toString("base64") !== chunk.data) throw new Error("Invalid guest report chunk.");
    parts.push(data); offset += data.length;
  }
  const data = Buffer.concat(parts);
  if (offset !== total || createHash("sha256").update(data).digest("hex") !== expectedSHA256) throw new Error("Incomplete or changed guest report.");
  return new TextDecoder("utf-8", { fatal: true }).decode(data);
}
