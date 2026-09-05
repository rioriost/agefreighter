import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { open, type FileHandle } from "node:fs/promises";
import type { TokenCredential } from "@azure/core-auth";
import { RunnerRecord } from "../core/runner";
import { reportStorageNames } from "../core/runnerReportStorage";

export interface CSVManifest { file: string; bytes: number; sha256: string }
export interface CSVTransfer extends CSVManifest {
  phase: "prepared" | "uploaded" | "submitted" | "unknown" | "verified" | "failed" | "interrupted";
  operation?: string;
}
const blockBytes = 8 * 1024 * 1024;
export const maxCSVBytes = 2 * 1024 * 1024 * 1024;

export function validateCSVManifest(value: CSVManifest): CSVManifest {
  if (!/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/.test(value.file) ||
    !/^[a-f0-9]{64}$/.test(value.sha256) || !Number.isSafeInteger(value.bytes) || value.bytes < 1 || value.bytes > maxCSVBytes) throw new Error("CSV transfer requires a file UUID, SHA-256 and size between 1 byte and 2 GiB.");
  return { file: value.file, bytes: value.bytes, sha256: value.sha256 };
}

async function regular(path: string): Promise<FileHandle> {
  const file = await open(path, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
  try { const info = await file.stat(); if (!info.isFile() || info.size < 1 || info.size > maxCSVBytes) throw new Error(); }
  catch { await file.close(); throw new Error("Select a regular nonempty CSV file no larger than 2 GiB."); }
  return file;
}

/** Streamed pre-approval inventory, no contents/credentials in retained state. */
export async function inspectCSV(file: string, path: string): Promise<CSVManifest> {
  const handle = await regular(path);
  try {
    const hash = createHash("sha256"), buffer = Buffer.alloc(blockBytes); let bytes = 0;
    while (true) { const part = await handle.read(buffer, 0, buffer.length, bytes); if (!part.bytesRead) break;
      bytes += part.bytesRead; if (bytes > maxCSVBytes) throw new Error("CSV grew beyond its transfer limit."); hash.update(buffer.subarray(0, part.bytesRead)); }
    return validateCSVManifest({ file, bytes, sha256: hash.digest("hex") });
  } finally { await handle.close(); }
}

/** Content-addressed blocks make explicit retry safe. No blob is overwritten.
 * A remote metadata match is only upload reconciliation, never guest verification.
 * Retries require another explicit call; tokens and diagnostics are not retained. */
export async function uploadCSV(record: RunnerRecord, path: string, manifest: CSVManifest, credential: TokenCredential,
  fetcher: typeof fetch = fetch, progress: (bytes: number) => void = () => {}): Promise<void> {
  validateCSVManifest(manifest);
  return uploadImmutable(record, path, manifest, credential, `uploads/${manifest.file}/${manifest.sha256}.csv`, "text/csv; charset=utf-8", fetcher, progress);
}

export async function uploadRunnerArchive(record: RunnerRecord, path: string, manifest: CSVManifest, credential: TokenCredential,
  fetcher: typeof fetch = fetch): Promise<void> {
  validateCSVManifest(manifest);
  if (manifest.bytes > 128 * 1024 * 1024 || manifest.file !== record.id) throw new Error("Invalid development archive manifest.");
  return uploadImmutable(record, path, manifest, credential, `artifacts/${manifest.sha256}.tar.gz`, "application/gzip", fetcher, () => {});
}

async function uploadImmutable(record: RunnerRecord, path: string, manifest: CSVManifest, credential: TokenCredential, suffix: string, contentType: string,
  fetcher: typeof fetch, progress: (bytes: number) => void): Promise<void> {
  const names = reportStorageNames(record), url = `${names.origin}/${names.container}/${suffix}`;
  const request = async (suffix: string, method: string, body?: Uint8Array | string, extra: Record<string, string> = {}) => {
    try {
      const token = await credential.getToken("https://storage.azure.com/.default"); if (!token) throw new Error();
      const response = await fetcher(url + suffix, { method, body: body as NonNullable<Parameters<typeof fetch>[1]>["body"], redirect: "error", signal: AbortSignal.timeout(60000),
        headers: { authorization: `Bearer ${token.token}`, "x-ms-version": "2023-11-03", ...extra } });
      await response.body?.cancel(); return response;
    } catch { throw new Error("CSV transfer acknowledgement is uncertain. Explicit retry reconciles the same content-addressed destination without overwriting a committed file."); }
  };
  const same = (response: Response) => response.status === 200 && response.headers.get("content-length") === String(manifest.bytes) && response.headers.get("x-ms-meta-sha256") === manifest.sha256;
  const handle = await regular(path);
  try {
    const head = await request("", "HEAD");
    if (head.status !== 404 && !same(head)) throw new Error("CSV destination is unavailable or conflicts with the approved manifest.");
    const hash = createHash("sha256"), buffer = Buffer.alloc(blockBytes), blocks: string[] = []; let offset = 0;
    while (offset < manifest.bytes) {
      const count = Math.min(buffer.length, manifest.bytes - offset), part = await handle.read(buffer, 0, count, offset);
      if (part.bytesRead !== count) throw new Error("Local CSV changed after review; no block list was committed.");
      const data = buffer.subarray(0, count); hash.update(data);
      const id = Buffer.from(String(blocks.length).padStart(8, "0")).toString("base64"); blocks.push(id);
      if (head.status === 404 && (await request(`?comp=block&blockid=${encodeURIComponent(id)}`, "PUT", data)).status !== 201) throw new Error("CSV block was not acknowledged; retry requires explicit approval.");
      offset += count; progress(offset);
    }
    if ((await handle.stat()).size !== manifest.bytes || hash.digest("hex") !== manifest.sha256) throw new Error("Local CSV changed after review; no block list was committed.");
    if (head.status === 200) return;
    const response = await request("?comp=blocklist", "PUT", `<?xml version="1.0" encoding="utf-8"?><BlockList>${blocks.map(id => `<Latest>${id}</Latest>`).join("")}</BlockList>`,
      { "content-type": "application/xml", "x-ms-blob-content-type": contentType, "x-ms-meta-sha256": manifest.sha256, "If-None-Match": "*" });
    if (response.status !== 201 && !(response.status === 412 && same(await request("", "HEAD")))) throw new Error("CSV commit is not confirmed. Reconcile with an explicitly approved upload retry.");
  } finally { await handle.close(); }
}
