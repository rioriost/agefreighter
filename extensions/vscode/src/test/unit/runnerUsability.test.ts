import assert from "node:assert/strict";
import test from "node:test";
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { Script } from "node:vm";
import { transformSync } from "esbuild";
import { catalogFixture } from "../catalogFixtures";
import { RunnerControl } from "../../core/runnerLifecycle";
import { RunnerRecord } from "../../core/runner";
import { discoverComputeSubnets } from "../../core/runnerPlacement";
import { sourceReportSummary } from "../../core/runnerSourceReport";
import { boundedWatch } from "../../core/boundedWatch";

function load(file: string, modules: Record<string, unknown>) {
  const output = { exports: {} as any }, native = createRequire(__filename);
  const code = transformSync(readFileSync(join(__dirname, "../../", file), "utf8"), { loader: "ts", format: "cjs" }).code;
  new Script(code).runInNewContext({ module: output, exports: output.exports, Error, Date, Buffer,
    require: (name: string) => name in modules ? modules[name] : name.startsWith("node:") ? native(name) : (() => { throw Error("Unexpected dependency " + name); })() });
  return output.exports;
}

test("subnet discovery marks the source and excludes delegated, reserved and other-region subnets without selecting one", async () => {
  const r = catalogFixture().record, sub = r.input.subscriptionId;
  const vnet = `/subscriptions/${sub}/resourceGroups/network/providers/Microsoft.Network/virtualNetworks/demo`;
  const source = `/subscriptions/${sub}/resourceGroups/source/providers/Microsoft.Compute/virtualMachines/source`;
  const nic = `/subscriptions/${sub}/resourceGroups/source/providers/Microsoft.Network/networkInterfaces/source`;
  const subnet = (name: string, delegated = false) => ({ name, id: `${vnet}/subnets/${name}`, properties: {
    addressPrefix: "10.0.1.0/24", delegations: delegated ? [{}] : [], provisioningState: "Succeeded"
  } });
  const control: RunnerControl = { persist: async () => { throw Error("Read only"); }, sleep: async () => {},
    request: async (_s, path, method = "GET") => {
      assert.equal(method, "GET");
      return { status: 200, value: { properties: path.startsWith(source + "?") ? { networkProfile: { networkInterfaces: [{ id: nic }] } }
        : { ipConfigurations: [{ properties: { subnet: { id: `${vnet}/subnets/source` } } }] } } };
    },
    list: async () => [{ id: vnet, name: "demo", location: r.input.region, properties: { subnets: [subnet("source"), subnet("runner"), subnet("target", true), subnet("GatewaySubnet")] } },
      { location: "other-region", properties: { subnets: [subnet("foreign")] } }]
  };
  const values = await discoverComputeSubnets(control, sub, r.input.region, source);
  assert.deepEqual(values.map(s => [s.name, s.containsSource]), [["runner", false], ["source", true]]);
  assert.equal(values[0]!.resourceGroup, "network");
  await assert.rejects(discoverComputeSubnets(control, "", r.input.region), /subscription and region/);
});

test("report outcome, completeness and exact counts are distinct from a matching hash", () => {
  const r = catalogFixture().record;
  r.input.source.type = "neo4j";
  r.sourceDraft = { form: {} as any, configuration: { source: { type: "neo4j" } }, warnings: [], canAssess: true };
  const doc = { schemaVersion: 1, command: "inventory", agefreighterVersion: r.artifact.version, outcome: "pass", errors: [], incompleteChecks: [],
    checks: [{ id: "source-counts", status: "pass" }], sections: [{ title: "Source inventory", fields:
      Object.entries({ vertices: "100000", edges: "250000", totalRows: "350000", countMethod: "neo4j-transactional-count-store" }).map(([name, value]) => ({ name, value, status: "pass" })) }] };
  const seal = (text: string) => {
    const sha256 = createHash("sha256").update(text).digest("hex"), bytes = Buffer.byteLength(text);
    r.assessment = { operation: r.id, action: "inventory", phase: "finished", bootId: r.id, configurationSHA256: createHash("sha256").update(JSON.stringify(r.sourceDraft!.configuration)).digest("hex"), reportSHA256: sha256, reportBytes: bytes };
    r.reportTransfers = [{ operation: r.id, sha256, bytes, phase: "imported", blob: "owned" }];
    return sourceReportSummary(r, text);
  };
  assert.deepEqual(seal(JSON.stringify(doc)), { title: "Complete source inventory passed", detail: "Next: review the private target and migration sizing. No target or migration has been approved.", vertices: "100000", edges: "250000", canReviewTarget: true });
  assert.equal(seal(JSON.stringify({ ...doc, outcome: "incomplete" })).canReviewTarget, false);
  assert.equal(seal(JSON.stringify({ ...doc, command: "profile" })).title, "Sample assessment passed");
  assert.equal(seal(JSON.stringify({ ...doc, errors: ["missing proof"] })).canReviewTarget, false);
});

test("report transfer reconciles a pending status receipt before one export and never replays a source command", async () => {
  let r = catalogFixture().record, reconciles = 0, exports = 0;
  const manifest = { operation: r.id, sha256: "a".repeat(64), bytes: 3 };
  r.assessment = { operation: r.id, action: "inventory", phase: "finished", bootId: r.id, configurationSHA256: "b".repeat(64), reportSHA256: manifest.sha256, reportBytes: 3 };
  r.guestCommand = { id: "pending-status", operation: r.id, action: "status", phase: "submitted", submittedAt: new Date().toISOString() };
  const flow = load("runnerReportFlow.ts", {
    "./core/boundedWatch": { boundedWatch },
    "./core/runnerGuest": { reconcileGuest: async (_c: unknown, current: RunnerRecord) => {
      reconciles++; r = { ...current, guestCommand: { ...current.guestCommand!, phase: reconciles === 1 ? "submitted" : "finished" } }; return { record: r };
    } },
    "./core/runnerReport": { startReportExport: async (_c: unknown, current: RunnerRecord) => {
      exports++; assert.equal(reconciles, 2); r = { ...current, reportTransfers: [{ ...manifest, phase: "imported", blob: "owned" }] }; return r;
    } }
  });
  await flow.transferApprovedReport({ sleep: async () => {} }, { read: async () => r, exclusive: async (_id: string, fn: () => unknown) => fn() }, r.id, manifest, async () => "protected");
  assert.equal(exports, 1); assert.equal(reconciles, 2);
});

for (const approved of [false, true]) test(`required storage consent ${approved ? "accepted" : "cancelled"} never replays deployment`, async () => {
  let r = catalogFixture().record, submits = 0, reviews = 0, checks = 0, locked = false;
  const flow = load("runnerStorageFlow.ts", {
    "./core/boundedWatch": { boundedWatch },
    "./core/runnerStorageLifecycle": {
      storageDraft: () => ({ phase: "previewed", hash: "approved" }),
      submitStorage: async (_c: unknown, current: RunnerRecord) => { submits++; r = { ...current, storageDeployment: { ...current.storageDeployment!, phase: "submitted" } }; return r; },
      refreshStorage: async (_c: unknown, current: RunnerRecord) => { r = { ...current, storageDeployment: { ...current.storageDeployment!, phase: "ready" } }; return r; }
    },
    "./core/runnerReportStorage": { verifyTransferStorage: async () => { checks++; } }
  });
  const store = { read: async () => r, exclusive: async (_id: string, fn: () => unknown) => { locked = true; try { return await fn(); } finally { locked = false; } } };
  const control = { persist: async (value: RunnerRecord) => { r = value; }, sleep: async () => { assert.equal(locked, false); } };
  const run = () => flow.prepareRequiredStorage(control, store, r.id, async () => r.id, async () => { reviews++; return approved; }, () => false, async () => {});
  const result = await run();
  assert.equal(submits, approved ? 1 : 0);
  assert.equal(result?.storageDeployment.phase, approved ? "ready" : undefined);
  if (approved) { await run(); assert.equal(submits, 1); assert.equal(reviews, 1); assert.equal(checks, 2); }
});

test("status monitoring polls receipts promptly but spaces new status dispatches by two minutes", async () => {
  let r = catalogFixture().record, refreshes = 0, ticks = 0;
  r.assessment = { operation: r.id, action: "inventory", phase: "running", bootId: r.id, configurationSHA256: "a".repeat(64) };
  r.guestCommand = { operation: r.id, id: "status", action: "status", phase: "finished", submittedAt: new Date().toISOString() };
  const flow = load("runnerWatch.ts", { "./core/runnerTarget": {}, "./core/runnerLifecycle": {}, "./core/runnerGuest": {}, "./core/boundedWatch": { boundedWatch },
    "./core/runnerCatalog": {}, "./core/runnerExecution": {},
    "./core/runnerAssessment": { refreshAssessment: async () => { refreshes++; return r; } },
    vscode: { workspace: { isTrusted: true }, ProgressLocation: { Notification: 1 }, window: { withProgress: async (_o: unknown, fn: any) => fn({}, { isCancellationRequested: false }) } }
  });
  const result = await flow.watchRetainedOperation({ sleep: async (ms: number) => { assert.equal(ms, 15000); ticks++; } },
    { read: async () => r, exclusive: async (_id: string, fn: () => unknown) => fn() }, r.id, "assessment", () => ticks === 3);
  assert.equal(result, undefined); assert.equal(refreshes, 0);
});
