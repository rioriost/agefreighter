import { createHash } from "node:crypto";
import { object, RunnerRecord } from "./runner";
import type { GuestCommand, GuestReadiness } from "./runnerGuest";
import { ReportManifest } from "./runnerBlob";

/** Only successful readiness controls are eligible. This is not a job report. */
export interface ReadinessReceipt {
  version: 1;
  workflow: string;
  vmId: string;
  command: GuestCommand;
  readiness: GuestReadiness;
  sha256: string;
}

export const commandCapacityMessage = "This extension's 25 managed Run Command limit has been reached. Use AGEFreighter: Archive runner readiness receipts to retain available evidence for operator review. No command is removed automatically; no new request was submitted.";

function hash(value: unknown): string {
  return createHash("sha256").update(JSON.stringify(value)).digest("hex");
}

/** Project validated fields, never public/protected ARM parameters or raw output. */
export function sealReadinessReceipt(record: RunnerRecord): ReadinessReceipt {
  const c = record.guestCommand, r = record.guestReady;
  const uuid = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/;
  const prefix = `${record.vmId}/runCommands/af-`;
  if (!uuid.test(record.id) || !c || !r || c.action !== "ready" || c.phase !== "finished" ||
      !c.id.startsWith(prefix) || !uuid.test(c.id.slice(prefix.length)) || !uuid.test(c.operation) ||
      !Number.isFinite(Date.parse(c.submittedAt)) || c.submittedAt !== r.checkedAt ||
      !uuid.test(r.bootId) || r.cliVersion !== record.artifact.version || r.archiveSha256 !== record.artifact.sha256 ||
      !/^[a-f0-9]{64}$/.test(r.archiveSha256) || !/^[a-zA-Z0-9.+_-]{1,128}$/.test(r.cliVersion) ||
      !/^[a-zA-Z0-9._-]{1,128}$/.test(r.commit) ||
      (record.artifact.development && r.commit !== record.artifact.development.commit)) {
    throw new Error("No validated readiness receipt is available for this workflow.");
  }
  if (r.capabilities && (r.capabilities.length > 32 || r.capabilities.some(x => !/^[a-z0-9-]{1,64}$/.test(x)))) throw new Error("Invalid readiness capabilities.");
  const h = r.health;
  if (h && (typeof h.idle !== "boolean" || !Number.isFinite(h.storageUsedPercent) || h.storageUsedPercent < 0 || h.storageUsedPercent > 100 ||
      !Number.isSafeInteger(h.swapUsedBytes) || h.swapUsedBytes < 0 || !Number.isSafeInteger(h.oomEvents) || h.oomEvents < 0)) throw new Error("Invalid readiness health.");
  const body = {
    version: 1 as const, workflow: record.id, vmId: record.vmId,
    command: { id: c.id, operation: c.operation, action: "ready" as const, phase: "finished" as const, submittedAt: c.submittedAt },
    readiness: { bootId: r.bootId, cliVersion: r.cliVersion, archiveSha256: r.archiveSha256, commit: r.commit, checkedAt: r.checkedAt,
      ...(r.capabilities ? { capabilities: [...r.capabilities] } : {}),
      ...(h ? { health: { idle: h.idle, storageUsedPercent: h.storageUsedPercent, swapUsedBytes: h.swapUsedBytes, oomEvents: h.oomEvents } } : {}) }
  };
  return { ...body, sha256: hash(body) };
}

/** Idempotent reconciliation cannot rewrite a previous successful receipt. */
export function retainReadinessReceipt(record: RunnerRecord): RunnerRecord {
  const receipt = sealReadinessReceipt(record);
  const old = record.readinessReceipts?.find(x => x.command.id === receipt.command.id);
  if (old && old.sha256 !== receipt.sha256) throw new Error("Successful readiness evidence changed; retain it for operator review.");
  return old ? record : { ...record, readinessReceipts: [...record.readinessReceipts ?? [], receipt] };
}

/** Hash-verified archive for operator review; never authorizes ARM deletion. */
export function readinessArchive(record: RunnerRecord, receipt: ReadinessReceipt): { text: string; manifest: ReportManifest } {
  if (receipt.version !== 1 || receipt.workflow !== record.id || receipt.vmId !== record.vmId ||
      !record.readinessReceipts?.some(x => x.command.id === receipt.command.id && x.sha256 === receipt.sha256)) throw new Error("Receipt is not retained by this workflow.");
  // Validate the receipt against its historical installation, not a later upgrade.
  const sealed = sealReadinessReceipt({ ...record, guestCommand: receipt.command, guestReady: receipt.readiness,
    artifact: { version: receipt.readiness.cliVersion, sha256: receipt.readiness.archiveSha256, url: "" } });
  if (sealed.sha256 !== receipt.sha256 || JSON.stringify(sealed) !== JSON.stringify(receipt)) throw new Error("Retained readiness evidence failed integrity validation.");
  const text = JSON.stringify({ kind: "agefreighter-readiness-receipt-v1", receipt: sealed });
  // Use the ARM command UUID, not the guest operation (which may be reused).
  const operation = receipt.command.id.slice(receipt.command.id.lastIndexOf("/af-") + 4);
  return { text, manifest: { operation, bytes: Buffer.byteLength(text), sha256: createHash("sha256").update(text).digest("hex") } };
}

/** Exclude the receipt ledger itself when finding still-referenced evidence. */
export function readinessReceiptReferenced(record: RunnerRecord, receipt: ReadinessReceipt): boolean {
  if (record.guestReady?.checkedAt === receipt.readiness.checkedAt && record.guestReady.bootId === receipt.readiness.bootId) return true;
  const { readinessReceipts: _receipts, ...state } = record;
  const identifiers = new Set([receipt.command.id.toLowerCase(), receipt.command.operation.toLowerCase()]);
  const contains = (value: unknown): boolean => typeof value === "string" ? identifiers.has(value.toLowerCase()) :
    Array.isArray(value) ? value.some(contains) : value && typeof value === "object" ? Object.values(object(value)).some(contains) : false;
  return contains(state);
}
