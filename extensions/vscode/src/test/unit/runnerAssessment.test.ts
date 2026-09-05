import assert from "node:assert/strict";
import test from "node:test";
import { RunnerRecord } from "../../core/runner";
import { RunnerControl } from "../../core/runnerLifecycle";
import { buildSourceDraft } from "../../core/runnerSource";
import { assessmentActive, refreshAssessment, startAssessment } from "../../core/runnerAssessment";
import { workflow, sourceForm } from "../sourceFixtures";

function fixture() {
  const record: RunnerRecord = { schemaVersion: 2, id: workflow, phase: "provisioned", input: { subscriptionId: workflow, resourceGroup: "test", region: "japaneast", zone: "1", subnetId: "subnet", size: "Standard_B2s_v2", source: { type: "neo4j", location: "on-premises" } }, artifact: { version: "2.4.0", sha256: "a".repeat(64), url: "https://example.invalid/archive" }, vmId: `/subscriptions/${workflow}/resourceGroups/test/providers/Microsoft.Compute/virtualMachines/runner`, deploymentId: "deployment", template: {}, previewHash: "hash", expiresAt: "", updatedAt: "", hourlyComputeUSD: 0.1,
    guestReady: { bootId: workflow, cliVersion: "2.4.0", archiveSha256: "a".repeat(64), commit: "commit", checkedAt: new Date().toISOString() },
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
