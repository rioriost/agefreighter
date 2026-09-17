/** Opt-in live Azure transport qualification. Never imported by the extension.
 * Reuses only the approved isolated CSV trial account. Does not edit GUI state,
 * grant access, start compute, delete files or overwrite existing Blobs.
 * Run fault and reconcile separately, with the same newly allocated file UUID.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFile, writeFile, access } from "node:fs/promises";
import { join } from "node:path";
import { inspectCSV, uploadCSV } from "../src/guided/csvTransfer";
import { reportStorageNames } from "../src/core/runnerReportStorage";
import type { RunnerRecord } from "../src/core/runner";

const subscription = "67c417f3-5a13-446c-afb9-40cd87f2fdb7";
const workflow = "bd3b6680-1e18-4d78-8f36-f43467a09a0a";
const group = "rg-af-vscode-p1-20260905-a";
const deadline = Date.parse("2026-09-20T07:14:35.311Z");
const expectedSHA = "0ecaaca3879b11a4bc76835c23f37cda9bfac7d5ed457ad50937170876d861fe";
const statePath = `/Users/rifujita/Library/Application Support/Code/User/globalStorage/rioriost.agefreighter/runner-v2/${workflow}.json`;
const source = "/Users/rifujita/Git_Managed/agefreighter/production-simulation/work/vscode-p1-portable-20260905/Supplier.csv";
const digest = (data: Uint8Array) => createHash("sha256").update(data).digest("hex");
const az = (args: string[]) => JSON.parse(execFileSync("az", [...args, "--subscription", subscription, "-o", "json"], { encoding: "utf8", timeout: 60000, stdio: ["ignore", "pipe", "pipe"] }));
const save = (path: string, value: unknown) => writeFile(path, JSON.stringify(value, null, 2) + "\n", { flag: "wx", mode: 0o600 });

async function main() {
  const [phase, file, output, confirmation] = process.argv.slice(2);
  assert.ok(phase === "fault" || phase === "reconcile");
  assert.equal(confirmation, "--confirm-isolated-azure-write");
  assert.ok(file && /^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/.test(file));
  assert.ok(output?.startsWith("/Users/rifujita/Git_Managed/agefreighter/production-simulation/work/"));
  assert.ok(Date.now() < deadline, "Trial deadline expired");
  const raw = await readFile(statePath), record: RunnerRecord = JSON.parse(raw.toString());
  assert.equal(record.id, workflow); assert.equal(record.input.subscriptionId, subscription);
  assert.equal(record.input.resourceGroup, group); assert.equal(record.input.source.type, "csv");
  assert.ok(!record.sourceFiles?.some(f => f.id === file), "Never use a GUI-owned file identity");
  const names = reportStorageNames(record);
  const account = az(["storage", "account", "show", "-g", group, "-n", names.account]);
  assert.equal(account.tags.workflow, workflow); assert.equal(account.tags.application, "agefreighter");
  assert.equal(account.tags.purpose, "artifact-transfer"); assert.equal(account.provisioningState, "Succeeded");
  assert.equal(account.publicNetworkAccess, "Enabled"); assert.equal(account.enableHttpsTrafficOnly, true);
  assert.equal(account.allowSharedKeyAccess, false); assert.equal(account.allowBlobPublicAccess, false);
  assert.equal(account.minimumTlsVersion, "TLS1_2");
  const token = az(["account", "get-access-token", "--resource", "https://storage.azure.com/"]);
  const credential = { getToken: async () => ({ token: token.accessToken, expiresOnTimestamp: Date.now() + 300000 }) };
  const headers = { authorization: `Bearer ${token.accessToken}`, "x-ms-version": "2023-11-03" };
  const manifest = await inspectCSV(file, source);
  assert.equal(manifest.bytes, 8797607); assert.equal(manifest.sha256, expectedSHA);
  const url = `${names.origin}/${names.container}/uploads/${file}/${manifest.sha256}.csv`;
  const calls: { method: string; kind: string; status: number }[] = [];
  const trace: typeof fetch = async (input, init) => {
    const u = new URL(String(input)); assert.equal(u.origin + u.pathname, url);
    assert.ok(Date.now() < deadline, "Deadline expired before request");
    const method = String(init?.method), kind = u.searchParams.get("comp") ?? "blob";
    if (phase === "reconcile") assert.equal(method, "HEAD", "Reconciliation must issue no PUT");
    const response = await fetch(input, init);
    calls.push({ method, kind, status: response.status });
    if (phase === "fault" && method === "PUT" && kind === "blocklist") {
      assert.equal(response.status, 201, "Fault only after actual Azure commit success");
      await save(join(output, "commit-witness.json"), { at: new Date().toISOString(), workflow, manifest, etag: response.headers.get("etag"), status: response.status, requestId: response.headers.get("x-ms-request-id") });
      await response.body?.cancel();
      throw new Error("Injected post-commit acknowledgement loss");
    }
    return response;
  };
  if (phase === "fault") {
    for (const name of ["commit-witness.json", "fault-result.json"]) {
      await assert.rejects(access(join(output, name)), { code: "ENOENT" });
    }
    const head = await fetch(url, { method: "HEAD", headers, redirect: "error", signal: AbortSignal.timeout(60000) });
    await head.body?.cancel(); assert.equal(head.status, 404, "Only a fresh never-committed test Blob may be faulted");
    await assert.rejects(uploadCSV(record, source, manifest, credential, trace), /acknowledgement is uncertain/);
    const witness = JSON.parse(await readFile(join(output, "commit-witness.json"), "utf8"));
    assert.deepEqual(witness.manifest, manifest);
    assert.deepEqual(calls.map(c => [c.method, c.kind, c.status]), [["HEAD", "blob", 404], ["PUT", "block", 201], ["PUT", "block", 201], ["PUT", "blocklist", 201]]);
    assert.equal(digest(await readFile(statePath)), digest(raw), "GUI state must remain untouched");
    const result = { at: new Date().toISOString(), evidenceLevel: "live Azure production transport with injected response loss; not installed GUI", phase, workflow, manifest, expectedUncertainError: true, implicitRetry: false, calls, guiStateUnchanged: true };
    await save(join(output, "fault-result.json"), result); console.log(JSON.stringify(result));
  } else {
    const fault = JSON.parse(await readFile(join(output, "fault-result.json"), "utf8"));
    const witness = JSON.parse(await readFile(join(output, "commit-witness.json"), "utf8"));
    assert.equal(fault.workflow, workflow); assert.deepEqual(fault.manifest, manifest);
    await uploadCSV(record, source, manifest, credential, trace);
    assert.deepEqual(calls, [{ method: "HEAD", kind: "blob", status: 200 }]);
    const response = await fetch(url, { headers, redirect: "error", signal: AbortSignal.timeout(60000) });
    assert.equal(response.status, 200); assert.equal(response.headers.get("etag"), witness.etag);
    const bytes = new Uint8Array(await response.arrayBuffer());
    assert.equal(bytes.length, manifest.bytes); assert.equal(digest(bytes), manifest.sha256);
    assert.equal(digest(await readFile(statePath)), digest(raw));
    const result = { at: new Date().toISOString(), evidenceLevel: fault.evidenceLevel, phase, workflow, manifest, calls, zeroRetryPUTs: true, unchangedETag: witness.etag, fullReadbackMatches: true, guiStateUnchanged: true };
    await save(join(output, "reconcile-result.json"), result); console.log(JSON.stringify(result));
  }
}
main().catch(() => { console.error("Qualification stopped; inspect retained bounded evidence. No automatic retry or deletion."); process.exitCode = 1; });
