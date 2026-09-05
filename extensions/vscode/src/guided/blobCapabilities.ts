import { BlobServiceClient, BlobSASPermissions, generateBlobSASQueryParameters, SASProtocol, type IHttpClient, type UserDelegationKey } from "@azure/storage-blob";
import type { TokenCredential } from "@azure/core-auth";
import { csvCapability, reportCapability } from "../core/runnerBlob";
import { RunnerRecord } from "../core/runner";
import { reportStorageNames } from "../core/runnerReportStorage";
import { CSVManifest, validateCSVManifest } from "./csvTransfer";

export async function issueCSVCapability(record: RunnerRecord, manifest: CSVManifest, credential: TokenCredential, fetcher: typeof fetch = fetch): Promise<string> {
  validateCSVManifest(manifest);
  const names = reportStorageNames(record);
  const client = new BlobServiceClient(names.origin, credential, { retryOptions: { maxTries: 1 }, httpClient: delegationTransport(names.origin, fetcher), audience: "https://storage.azure.com/.default" });
  try {
    const now = Date.now(), key = await client.getUserDelegationKey(new Date(now - 120000), new Date(now + 720000));
    const sas = generateBlobSASQueryParameters({ containerName: names.container, blobName: `uploads/${manifest.file}/${manifest.sha256}.csv`,
      permissions: BlobSASPermissions.parse("r"), protocol: SASProtocol.Https, version: "2023-11-03", startsOn: new Date(now - 60000), expiresOn: new Date(now + 600000) }, key, names.account);
    const url = `${names.origin}/${names.container}/uploads/${manifest.file}/${manifest.sha256}.csv?${sas}`;
    csvCapability(url, record.id, manifest.file, manifest.sha256); return url;
  } catch { throw new Error("The existing Azure login could not issue a short-lived CSV read capability. No alternate login or shared key is used."); }
}

/** Native fetch transport for the single key request: fixed host, no redirects,
 * retries, response streams, service diagnostics, or unbounded XML bodies. */
export function delegationTransport(origin: string, fetcher: typeof fetch = fetch): IHttpClient {
  return { async sendRequest(request) {
    if (request.url !== `${origin}/?restype=service&comp=userdelegationkey` || request.method !== "POST" || typeof request.body !== "string" || request.body.length > 1024) throw new Error("Unexpected storage credential request.");
    try {
      const response = await fetcher(request.url, { method: "POST", body: request.body, headers: request.headers.toJson(), redirect: "error", signal: AbortSignal.timeout(25000) });
      if (!response.ok || !response.body) { await response.body?.cancel(); throw new Error(); }
      const reader = response.body.getReader(), parts: Uint8Array[] = []; let count = 0;
      try { while (true) { const part = await reader.read(); if (part.done) break; count += part.value.length; if (count > 16384) throw new Error(); parts.push(part.value); } }
      finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
      const headers = request.headers.clone(); for (const name of headers.headerNames()) headers.remove(name);
      response.headers.forEach((value, name) => headers.set(name, value));
      return { request, status: response.status, headers, bodyAsText: new TextDecoder("utf-8", { fatal: true }).decode(Buffer.concat(parts)) };
    } catch { throw new Error("Storage data-plane access could not be verified. Check the approved user role and network; no account-key fallback is used."); }
  } };
}

export function signReportCapability(record: RunnerRecord, operation: string, permission: "r" | "c", key: UserDelegationKey, now = Date.now()): string {
  const names = reportStorageNames(record);
  const sas = generateBlobSASQueryParameters({ containerName: names.container, blobName: `reports/${operation}.json`,
    permissions: BlobSASPermissions.parse(permission), protocol: SASProtocol.Https, version: "2023-11-03",
    startsOn: new Date(now - 60000), expiresOn: new Date(now + 600000) }, key, names.account);
  const url = `${names.origin}/${names.container}/reports/${operation}.json?${sas}`;
  reportCapability(url, record.id, operation, permission, now); return url;
}

/** No key cache/persistence, desktop tokens never leave the storage audience. */
export async function issueReportCapability(record: RunnerRecord, operation: string, permission: "r" | "c", credential: TokenCredential, fetcher: typeof fetch = fetch): Promise<string> {
  const names = reportStorageNames(record);
  const client = new BlobServiceClient(names.origin, credential, { retryOptions: { maxTries: 1 }, httpClient: delegationTransport(names.origin, fetcher), audience: "https://storage.azure.com/.default" });
  try {
    const now = Date.now(), key = await client.getUserDelegationKey(new Date(now - 120000), new Date(now + 720000));
    return signReportCapability(record, operation, permission, key);
  } catch { throw new Error("A short-lived report capability could not be issued using the existing Azure login. Check storage data permissions and propagation; no alternate login or shared key is used."); }
}
