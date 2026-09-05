import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { downloadReport, reportCapability, ReportManifest, verifyReportBytes } from "../../core/runnerBlob";
import { importReport, refreshReportExport, startReportExport } from "../../core/runnerReport";
import { RunnerRecord } from "../../core/runner";
import { RunnerControl } from "../../core/runnerLifecycle";
import { reportStorageNames, verifyReportStorage } from "../../core/runnerReportStorage";
import { RunnerStore } from "../../guided/runnerStore";
import { verifyPrivateWindowsPath } from "../../guided/privateDirectory";

const workflow = "11111111-1111-4111-8111-111111111111", operation = "22222222-2222-4222-8222-222222222222";
const payload = '{"command":"profile","integer":9223372036854775807,"name":"工場 🏭","padding":"' + "x".repeat(100000) + '"}';
const manifest: ReportManifest = { operation, sha256: createHash("sha256").update(payload).digest("hex"), bytes: Buffer.byteLength(payload) };
function capability(permission: "r" | "c", now = Date.now()): string {
  const q = new URLSearchParams({ sv: "2023-11-03", spr: "https", sr: "b", sp: permission,
    st: new Date(now - 60000).toISOString(), se: new Date(now + 600000).toISOString(), sig: "SECRET-CAPABILITY",
    skoid: workflow, sktid: operation, skt: new Date(now - 3600000).toISOString(), ske: new Date(now + 3600000).toISOString(), sks: "b", skv: "2023-11-03" });
  return `https://af${workflow.replaceAll("-", "").slice(0, 22)}.blob.core.windows.net/af-${workflow}/reports/${operation}.json?${q}`;
}
function fixture() {
  const record: RunnerRecord = { schemaVersion: 2, id: workflow, phase: "provisioned", input: { subscriptionId: workflow, resourceGroup: "test", region: "japaneast", zone: "1", subnetId: "subnet", size: "Standard_B2s_v2", source: { type: "neo4j", location: "on-premises" } }, artifact: { version: "2.4.0", sha256: "a".repeat(64), url: "https://example.invalid/archive" }, vmId: `/subscriptions/${workflow}/resourceGroups/test/providers/Microsoft.Compute/virtualMachines/runner`, deploymentId: "deployment", template: {}, previewHash: "hash", expiresAt: "", updatedAt: "", hourlyComputeUSD: .1,
    assessment: { operation, action: "profile", phase: "finished", configurationSHA256: "b".repeat(64), bootId: workflow, reportSHA256: manifest.sha256, reportBytes: manifest.bytes } };
  const events: string[] = [], saved: RunnerRecord[] = [], bodies: unknown[] = [];
  let result: unknown, failure = false;
  const control: RunnerControl = { sleep: async () => {}, list: async () => [], persist: async r => { events.push("persist"); saved.push(structuredClone(r)); }, request: async (_sub, path, method = "GET", body) => {
    events.push(method + path);
    const names = reportStorageNames(record);
    if (path === `${names.id}?api-version=2023-05-01`) return { status: 200, value: { id: names.id, location: record.input.region, tags: { application: "agefreighter", workflow, purpose: "artifact-transfer" }, properties: { provisioningState: "Succeeded", supportsHttpsTrafficOnly: true, allowBlobPublicAccess: false, allowSharedKeyAccess: false, minimumTlsVersion: "TLS1_2", primaryEndpoints: { blob: `${names.origin}/` } } } };
    if (path === `${names.containerId}?api-version=2023-05-01`) return { status: 200, value: { id: names.containerId, properties: { publicAccess: "None" } } };
    if (method === "PUT") { bodies.push(body); if (failure) throw new Error("SECRET-CAPABILITY"); return { status: 201, value: {} }; }
    return result === undefined ? { status: 404, value: {} } : { status: 200, value: { properties: { instanceView: { executionState: "Succeeded", exitCode: 0, output: JSON.stringify(result) } } } };
  } };
  return { record, control, events, saved, bodies, result: (value: unknown) => { result = value; }, fail: () => { failure = true; } };
}
const validFetch: typeof fetch = async (_url, init) => {
  assert.equal(init?.redirect, "error"); assert.equal(init?.method, "GET"); assert.ok(init?.signal);
  assert.ok(!JSON.stringify(init?.headers).includes("authorization"));
  return new Response(payload, { headers: { "content-length": String(manifest.bytes) } });
};

test("blob capabilities bind host, workflow, operation, expiry and one user-delegation permission", () => {
  const now = Date.now(), valid = capability("c", now);
  assert.equal(reportCapability(valid, workflow, operation, "c", now).searchParams.get("sp"), "c");
  for (const raw of [valid.replace("https:", "http:"), valid.replace(/af[a-f0-9]+\.blob\.core\.windows\.net/, "localhost"),
    valid.replace(".windows.net", ".windows.net.attacker.invalid"), valid.replace(".windows.net", ".windows.net:443"),
    valid.replace("/reports/", "/other/"), valid.replace("sp=c", "sp=cw"), valid.replace("sr=b", "sr=c"), valid + "&sp=c", valid + "&comp=block",
    valid + "#fragment", valid.replace(".json?", "%2Ejson?"), valid.replace("sks=b", "sks=f"), valid.replace("skoid=", "skoid=x"),
    capability("r", now), capability("c", now + 3600000), capability("c", now - 3600000),
    valid.replace("/reports/", "/reports/../reports/"), valid.replace("https://", "https://user:pass@"), "\n" + valid]) {
    assert.throws(() => reportCapability(raw, workflow, operation, "c", now), error => error instanceof Error && !error.message.includes("SECRET-CAPABILITY"));
  }
});

test("full report transfer preserves int64 and Unicode bytes with one bounded GET", async () => {
  let calls = 0;
  const text = await downloadReport(capability("r"), workflow, manifest, async (...args) => { calls++; return validFetch(...args); });
  assert.equal(calls, 1); assert.equal(text, payload); assert.match(text, /9223372036854775807/);
  assert.equal(verifyReportBytes(Buffer.from(payload), manifest), payload);
});

test("download fails closed on truncated, changed, oversized, redirected or leaking transport results", async () => {
  const responses = [() => new Response(payload.slice(0, -1)), () => new Response(payload + "extra"),
    () => new Response(payload.replace("工場", "変更")), () => new Response("SECRET-CAPABILITY", { status: 403 }),
    () => new Response("", { status: 302, headers: { location: "https://attacker.invalid" } }),
    () => new Response(payload, { headers: { "content-encoding": "gzip" } }),
    () => new Response(payload, { headers: { "content-length": "1" } })];
  for (const response of responses) {
    let calls = 0;
    await assert.rejects(downloadReport(capability("r"), workflow, manifest, async () => { calls++; return response(); }), error => error instanceof Error && !error.message.includes("SECRET-CAPABILITY"));
    assert.equal(calls, 1);
  }
  await assert.rejects(downloadReport(capability("r"), workflow, manifest, async url => { throw new Error(String(url)); }), error => error instanceof Error && !error.message.includes("SECRET-CAPABILITY"));
});

test("non-object JSON and invalid UTF-8 cannot become imported evidence despite matching hashes", () => {
  for (const bytes of [Buffer.from("[]"), Buffer.from("null"), Buffer.from("not json"), Buffer.from([0xff])]) {
    assert.throws(() => verifyReportBytes(bytes, { operation, sha256: createHash("sha256").update(bytes).digest("hex"), bytes: bytes.length }));
  }
});

test("export intent persists before PUT with capability only in protected parameters", async () => {
  const f = fixture(); const r = await startReportExport(f.control, f.record, capability("c"));
  assert.equal(f.events[3], "persist"); assert.ok(f.events[4]?.startsWith("PUT"));
  assert.equal(r.reportTransfers?.[0]?.phase, "submitted"); assert.ok(!JSON.stringify(f.saved).includes("SECRET-CAPABILITY"));
  const body = f.bodies[0] as { properties: { protectedParameters: { value: string }[]; parameters?: unknown } };
  assert.equal(body.properties.parameters, undefined);
  assert.match(Buffer.from(body.properties.protectedParameters[0]!.value, "base64").toString(), /SECRET-CAPABILITY/);
  await assert.rejects(startReportExport(f.control, r, capability("c")), /intent already exists/);
  assert.equal(r.phase, "provisioned"); // An export is never a migration pass.
});

test("lost PUT acknowledgement reconciles GET-only, and independent import does not clear pending ARM state", async () => {
  const f = fixture(); f.fail(); const r = await startReportExport(f.control, f.record, capability("c"));
  assert.equal(r.reportTransfers?.[0]?.phase, "unknown");
  const count = f.events.filter(e => e.startsWith("PUT")).length;
  const checked = await refreshReportExport(f.control, r);
  let retained = false;
  const imported = await importReport(f.control, checked, operation, capability("r"), async (id, m, text) => { assert.equal(id, workflow); assert.deepEqual(m, manifest); assert.equal(text, payload); retained = true; }, validFetch);
  assert.ok(retained); assert.equal(imported.reportTransfers?.[0]?.phase, "imported");
  assert.equal(imported.guestCommand?.phase, "unknown"); assert.equal(f.events.filter(e => e.startsWith("PUT")).length, count);
  assert.ok(!JSON.stringify(f.saved).includes("SECRET-CAPABILITY"));
});

test("receipt requires correct identity/hash/length; no receipt alone imports a report", async () => {
  const f = fixture(); const r = await startReportExport(f.control, f.record, capability("c"));
  const receipt = { version: 1, workflow, operation, sha256: manifest.sha256, bytes: manifest.bytes, exported: true };
  for (const change of [{ sha256: "b".repeat(64) }, { bytes: manifest.bytes + 1 }, { workflow: operation }, { exported: false }]) {
    f.result({ ...receipt, ...change }); assert.equal((await refreshReportExport(f.control, r)).reportTransfers?.[0]?.phase, "unknown");
  }
  f.result(receipt); const exported = await refreshReportExport(f.control, r); assert.equal(exported.reportTransfers?.[0]?.phase, "exported");
  await assert.rejects(importReport(f.control, exported, operation, capability("r").replace(/af[a-f0-9]+\.blob/, "otherstorage.blob"), async () => { assert.fail("wrong destination retained"); }, validFetch));
  await assert.rejects(importReport(f.control, exported, operation, capability("r"), async () => { throw new Error("disk full"); }, validFetch), /disk full/);
  assert.equal(exported.reportTransfers?.[0]?.phase, "exported");
});

test("private report retention publishes atomically, allows identical recovery, rejects replacement", async () => {
  const root = await mkdtemp(join(tmpdir(), "af-report-store-"));
  try {
    const store = new RunnerStore(root);
    await store.retainReport(workflow, manifest, payload);
    await store.retainReport(workflow, manifest, payload);
    assert.equal(await store.readReport(workflow, manifest), payload);
    const path = join(root, `${workflow}.report-${operation}.json`);
    if (process.platform === "win32") await verifyPrivateWindowsPath(path);
    assert.deepEqual(await store.list(), []);
    assert.equal((await readdir(root)).length, 1);
    const changed = payload.replace("工場", "変更"), other = { ...manifest, sha256: createHash("sha256").update(changed).digest("hex") };
    await assert.rejects(store.retainReport(workflow, other, changed));
    assert.equal(await readFile(path, "utf8"), payload);
    await writeFile(path, "changed");
    await assert.rejects(store.readReport(workflow, manifest));
    await assert.rejects(store.retainReport(workflow, manifest, payload));
    assert.equal((await readdir(root)).length, 1);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("export rejects missing, foreign, anonymously readable or shared-key-enabled storage before dispatch", async () => {
  for (const mutation of ["missing", "owner", "region", "https", "public", "sharedkey", "container", "endpoint", "Disabled", "SecuredByPerimeter"]) {
    const f = fixture(), request = f.control.request;
    f.control.request = async (sub, path, method, body) => {
      const result = await request(sub, path, method, body);
      if (path.includes("/storageAccounts/")) {
        const value = result.value as { location: string; tags: { workflow: string }; properties: Record<string, unknown> };
        if (mutation === "missing") return { status: 404, value: {} };
        if (path.includes("/containers/")) { if (mutation === "container") value.properties.publicAccess = "Blob"; }
        else {
          if (mutation === "owner") value.tags.workflow = operation;
          if (mutation === "region") value.location = "westus";
          if (mutation === "https") value.properties.supportsHttpsTrafficOnly = false;
          if (mutation === "public") value.properties.allowBlobPublicAccess = true;
          if (mutation === "sharedkey") value.properties.allowSharedKeyAccess = true;
          if (["Disabled", "SecuredByPerimeter"].includes(mutation)) value.properties.publicNetworkAccess = mutation;
          if (mutation === "endpoint") value.properties.primaryEndpoints = { blob: "https://attacker.invalid/" };
        }
      }
      return result;
    };
    await assert.rejects(startReportExport(f.control, f.record, capability("c")));
    assert.ok(f.events.every(event => event.startsWith("GET")));
    assert.equal(f.saved.length, 0);
  }
  const f = fixture();
  await assert.rejects(verifyReportStorage(f.control, f.record, "https://other.blob.core.windows.net/foreign"));
  assert.equal(f.events.length, 0);
});
