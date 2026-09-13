import assert from "node:assert/strict";
import test from "node:test";
import { RunnerRecord } from "../../core/runner";
import { RunnerControl } from "../../core/runnerLifecycle";
import { buildSourceDraft } from "../../core/runnerSource";
import { assessmentActive, refreshAssessment, startAssessment, retainFailedAssessment } from "../../core/runnerAssessment";
import { workflow, sourceForm, csvFile } from "../sourceFixtures";

function fixture() {
  const record: RunnerRecord = { schemaVersion: 2, id: workflow, phase: "provisioned", input: { subscriptionId: workflow, resourceGroup: "test", region: "japaneast", zone: "1", subnetId: "subnet", size: "Standard_B2s_v2", source: { type: "neo4j", location: "on-premises" } }, artifact: { version: "2.4.0", sha256: "a".repeat(64), url: "https://example.invalid/archive" }, vmId: `/subscriptions/${workflow}/resourceGroups/test/providers/Microsoft.Compute/virtualMachines/runner`, deploymentId: "deployment", template: {}, previewHash: "hash", expiresAt: "", updatedAt: "", hourlyComputeUSD: 0.1,
    guestReady: { bootId: workflow, cliVersion: "2.4.0", archiveSha256: "a".repeat(64), commit: "commit", checkedAt: new Date().toISOString(), capabilities: ["neo4j-inventory-v1", "neo4j-migration-v1"] },
    sourceDraft: buildSourceDraft({ type: "neo4j", location: "on-premises" }, sourceForm, workflow) };
  const saved: RunnerRecord[] = [], requests: { method: string; path: string }[] = [];
  let result: unknown, fail = false;
  const control: RunnerControl = { sleep: async () => {}, list: async () => [], persist: async r => { saved.push(structuredClone(r)); }, request: async (_sub, path, method = "GET") => {
    requests.push({ method, path });
    if (method === "PUT") { if (fail) throw new Error("lost response"); return { status: 201, value: {} }; }
    return result === undefined ? { status: 404, value: {} } : { status: 200, value: { properties: { instanceView: { executionState: "Succeeded", exitCode: 0, output: JSON.stringify(result) } } } };
  } };
  return { record, control, saved, requests, set: (value: unknown) => { result = value; }, fail: () => { fail = true; } };
}
test("assessment intent is durable before dispatch; on-prem source needs only runner ARM calls", async () => {
  const f = fixture(); const r = await startAssessment(f.control, f.record, "profile", { AGEFREIGHTER_SOURCE_PASSWORD: "private" });
  assert.equal(r.assessment?.phase, "submitted"); assert.equal(f.saved[0]?.assessment?.operation, r.guestCommand?.operation);
  assert.ok(!JSON.stringify(f.saved).includes('"private"'));
  assert.ok(f.requests.every(request => request.path.startsWith(f.record.vmId + "/runCommands/")));
  await assert.rejects(startAssessment(f.control, r, "profile", {}), /retained assessment/);
});

test("failed source reconciliation is explicit, idle-gated and retains evidence without dispatch", () => {
  const f = fixture(), now = Date.now();
  const a = { operation: workflow, action: "inventory" as const, phase: "failed" as const, bootId: workflow, configurationSHA256: "c".repeat(64) };
  f.record.assessment = a;
  f.record.guestCommand = { id: "ready-command", operation: "ready-op", action: "ready", phase: "finished", submittedAt: new Date(now).toISOString() };
  f.record.guestReady!.checkedAt = new Date(now).toISOString();
  f.record.guestReady!.health = { idle: true, swapUsedBytes: 0, oomEvents: 0, storageUsedPercent: 4 };
  const result = retainFailedAssessment(f.record, workflow, now);
  assert.equal(result.assessment, undefined);
  assert.deepEqual(result.assessmentHistory, [a]);
  assert.deepEqual(f.record.assessment, a);
  assert.equal(f.requests.length, 0);
  for (const mutate of [
    (r: RunnerRecord) => { r.assessment!.phase = "running"; },
    (r: RunnerRecord) => { r.assessment!.phase = "unknown"; },
    (r: RunnerRecord) => { r.assessment!.phase = "interrupted"; },
    (r: RunnerRecord) => { r.guestCommand!.phase = "submitted"; },
    (r: RunnerRecord) => { r.guestCommand!.action = "status"; },
    (r: RunnerRecord) => { r.guestReady!.checkedAt = new Date(now - 300001).toISOString(); },
    (r: RunnerRecord) => { r.guestReady!.bootId = "changed"; },
    (r: RunnerRecord) => { r.guestReady!.health!.idle = false; },
    (r: RunnerRecord) => { r.guestReady!.health!.oomEvents = 1; },
    (r: RunnerRecord) => { r.guestReady!.health!.swapUsedBytes = 1; },
    (r: RunnerRecord) => { r.guestReady!.health!.storageUsedPercent = 80; },
    (r: RunnerRecord) => { r.target = {} as NonNullable<RunnerRecord["target"]>; },
    (r: RunnerRecord) => { r.migration = {} as NonNullable<RunnerRecord["migration"]>; },
  ]) { const r = structuredClone(f.record); mutate(r); assert.throws(() => retainFailedAssessment(r, workflow, now)); }
  assert.throws(() => retainFailedAssessment(f.record, "different", now));
});
test("lost assessment acknowledgement is reconciled without duplicate source reads", async () => {
  const f = fixture(); f.fail(); const r = await startAssessment(f.control, f.record, "profile", {});
  assert.equal(r.guestCommand?.phase, "unknown"); const count = f.requests.filter(r => r.method === "PUT").length;
  await refreshAssessment(f.control, r); await refreshAssessment(f.control, r);
  assert.equal(f.requests.filter(r => r.method === "PUT").length, count);
  assert.equal(assessmentActive(r), true);
});
test("worker status binds operation and configuration; finished is evidence availability, not migration pass", async () => {
  const f = fixture(); const started = await startAssessment(f.control, f.record, "profile", {});
  const state = { version: 1, workflow, operation: started.assessment!.operation, action: "profile", phase: "finished", bootId: workflow, exitCode: 0, configSha256: "b".repeat(64), reportBytes: 1024, reportSha256: "c".repeat(64) };
  f.set(state); const complete = await refreshAssessment(f.control, started);
  assert.equal(complete.assessment?.phase, "finished"); assert.equal(complete.assessment?.reportSHA256, state.reportSha256);
  assert.equal(complete.phase, "provisioned"); // Never a completed migration.
  f.set(undefined); const inventory = await startAssessment(f.control, complete, "inventory", {});
  assert.equal(inventory.assessmentHistory?.[0]?.operation, started.assessment?.operation);
  assert.notEqual(inventory.assessment?.operation, started.assessment?.operation);
  f.set({ ...state, operation: inventory.assessment?.operation, action: "inventory", configSha256: "changed" });
  await assert.rejects(refreshAssessment(f.control, inventory), /configuration evidence changed/);
});
test("unfinished, changed or unsupported source drafts never dispatch an assessment", async () => {
  const f = fixture(); f.record.sourceDraft!.canAssess = false;
  await assert.rejects(startAssessment(f.control, f.record, "profile", {}));
  f.record.sourceDraft!.canAssess = true; f.record.input.source.type = "postgresql";
  await assert.rejects(startAssessment(f.control, f.record, "inventory", {}));
  await assert.rejects(startAssessment(f.control, f.record, "profile", {}));
  assert.equal(f.requests.length, 0);
});

test("CSV inventory requires verified files and an advertised guest capability", async () => {
  const f = fixture(); f.record.input.source = { type: "csv", location: "local" };
  f.record.sourceDraft = buildSourceDraft(f.record.input.source, { ...sourceForm, mappings: sourceForm.mappings.map(m => ({ ...m, collection: csvFile.id })) }, workflow, [csvFile]);
  f.record.sourceDraft.canAssess = true;
  f.record.csvTransfers = [{ file: csvFile.id, bytes: 10, sha256: "b".repeat(64), phase: "verified" }];
  await assert.rejects(startAssessment(f.control, f.record, "inventory", {}), /does not advertise/);
  assert.equal(f.requests.length, 0);
  f.record.guestReady!.capabilities = ["csv-inventory-v1"];
  f.record.csvTransfers[0]!.phase = "uploaded";
  await assert.rejects(startAssessment(f.control, f.record, "inventory", {}), /verified/);
  f.record.csvTransfers[0]!.phase = "verified";
  const result = await startAssessment(f.control, f.record, "inventory", {});
  assert.equal(result.assessment?.action, "inventory");
});
