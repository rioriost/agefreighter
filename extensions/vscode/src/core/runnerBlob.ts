import { createHash } from "node:crypto";
import { object } from "./runner";

const uuid = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/;
const sha = /^[a-f0-9]{64}$/;
export const maxReportBytes = 4 * 1024 * 1024;
export interface ReportManifest { operation: string; sha256: string; bytes: number }
export interface ReportTransfer extends ReportManifest {
  /** Public identity only, never the SAS query string. */
  blob: string;
  phase: "submitted" | "unknown" | "exported" | "imported";
}

/** Match the guest's strict single-blob, user-delegation capability policy. */
export function reportCapability(raw: string, workflow: string, operation: string, permission: "r" | "c", now = Date.now()): URL {
  const invalid = () => new Error("A short-lived HTTPS user-delegation capability for the exact report blob is required.");
  let url: URL;
  try { url = new URL(raw); } catch { throw invalid(); }
  const path = `/af-${workflow}/reports/${operation}.json`;
  if (raw.length > 4096 || /[\s\\\u0000-\u001f\u007f]/.test(raw) || !uuid.test(workflow) || !uuid.test(operation) ||
    url.protocol !== "https:" || url.username || url.password || url.hash ||
    !/^[a-z0-9]{3,24}\.blob\.core\.windows\.net$/.test(url.host) || url.pathname !== path ||
    raw.split("?")[0] !== `${url.origin}${path}`) throw invalid();
  const q = url.searchParams, allowed = new Set(["sv", "spr", "sr", "sp", "st", "se", "sig", "skoid", "sktid", "skt", "ske", "sks", "skv"]);
  for (const [name, value] of q) if (!allowed.has(name) || !value || q.getAll(name).length !== 1) throw invalid();
  if (q.get("sr") !== "b" || q.get("sp") !== permission || q.get("spr") !== "https" || !q.get("sig") ||
    !uuid.test(q.get("skoid") ?? "") || !uuid.test(q.get("sktid") ?? "") || q.get("sks") !== "b" || !q.get("sv") || !q.get("skv")) throw invalid();
  const dates = ["st", "se", "skt", "ske"].map(name => {
    const value = q.get(name) ?? "";
    if (!/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)$/.test(value)) throw invalid();
    return Date.parse(value);
  });
  const [start, end, keyStart, keyEnd] = dates as [number, number, number, number];
  if (!Number.isFinite(now) || dates.some(value => !Number.isFinite(value)) || start > now || start < now - 300000 || end <= now ||
    end > now + 900000 || end - start > 1200000 || keyStart > start || keyEnd < end) throw invalid();
  return url;
}

export function reportManifest(value: ReportManifest): ReportManifest {
  if (!uuid.test(value.operation) || !sha.test(value.sha256) || !Number.isSafeInteger(value.bytes) || value.bytes < 1 || value.bytes > maxReportBytes) throw new Error("Invalid retained report manifest.");
  return { operation: value.operation, sha256: value.sha256, bytes: value.bytes };
}

/** Retain original JSON bytes: parsing/re-serialization can lose int64 values. */
export function verifyReportBytes(data: Uint8Array, expected: ReportManifest): string {
  reportManifest(expected);
  if (data.byteLength !== expected.bytes || createHash("sha256").update(data).digest("hex") !== expected.sha256) throw new Error("Report length or SHA-256 does not match independent guest evidence.");
  try {
    const text = new TextDecoder("utf-8", { fatal: true }).decode(data);
    object(JSON.parse(text));
    return text;
  } catch { throw new Error("Verified report is not a UTF-8 JSON object."); }
}

/** One bounded GET; no redirects, retries, ARM bearer token, or URL diagnostics. */
export async function downloadReport(raw: string, workflow: string, expected: ReportManifest, fetcher: typeof fetch = fetch): Promise<string> {
  reportManifest(expected);
  const url = reportCapability(raw, workflow, expected.operation, "r");
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  try {
    const response = await fetcher(url, { method: "GET", redirect: "error", signal: AbortSignal.timeout(25000), headers: { "x-ms-version": "2023-11-03", "accept-encoding": "identity" } });
    if (response.status !== 200 || !response.body || response.headers.has("content-encoding") && response.headers.get("content-encoding") !== "identity" ||
      response.headers.has("content-length") && response.headers.get("content-length") !== String(expected.bytes)) {
      await response.body?.cancel(); throw new Error();
    }
    reader = response.body.getReader();
    const parts: Uint8Array[] = []; let bytes = 0;
    while (true) {
      const part = await reader.read(); if (part.done) break;
      bytes += part.value.byteLength; if (bytes > expected.bytes) throw new Error();
      parts.push(part.value);
    }
    return verifyReportBytes(Buffer.concat(parts), expected);
  } catch { throw new Error("Report download could not be verified; evidence remains incomplete. No source operation was replayed."); }
  finally { try { await reader?.cancel(); } catch { /* Never expose transport errors containing the SAS. */ } reader?.releaseLock(); }
}


