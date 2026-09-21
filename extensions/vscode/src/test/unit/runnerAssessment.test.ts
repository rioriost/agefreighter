import assert from "node:assert/strict";
import test from "node:test";
import { RunnerRecord } from "../../core/runner";
import { RunnerControl } from "../../core/runnerLifecycle";
import { buildSourceDraft } from "../../core/runnerSource";
import { assessmentActive, ensureAssessmentReadiness, refreshAssessment, startAssessment, retainFailedAssessment } from "../../core/runnerAssessment";
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
    (r: RunnerRecord) => { r.guestReady!.archiveSha256 = "b".repeat(64); },
    (r: RunnerRecord) => { r.guestReady!.cliVersion = "old"; },
    (r: RunnerRecord) => { r.guestCommand!.submittedAt = new Date(now - 1).toISOString(); },
    (r: RunnerRecord) => { r.guestReady!.health!.storageUsedPercent = -1; },
    (r: RunnerRecord) => { r.guestReady!.health!.idle = false; },
    (r: RunnerRecord) => { r.guestReady!.health!.oomEvents = 1; },
    (r: RunnerRecord) => { r.guestReady!.health!.swapUsedBytes = 1; },
    (r: RunnerRecord) => { r.guestReady!.health!.storageUsedPercent = 80; },
    (r: RunnerRecord) => { r.target = {} as NonNullable<RunnerRecord["target"]>; },
    (r: RunnerRecord) => { r.migration = {} as NonNullable<RunnerRecord["migration"]>; },
  ]) { const r = structuredClone(f.record); mutate(r); assert.throws(() => retainFailedAssessment(r, workflow, now)); }
  assert.throws(() => retainFailedAssessment(f.record, "different", now));
  for (const delta of [-300001, 1]) {
    const r = structuredClone(f.record);
    r.guestReady!.checkedAt = r.guestCommand!.submittedAt = new Date(now + delta).toISOString();
    assert.throws(() => retainFailedAssessment(r, workflow, now));
  }
  const full = structuredClone(f.record); full.assessmentHistory = Array(16).fill(a);
  assert.throws(() => retainFailedAssessment(full, workflow, now), /history limit/);
});

test("terminal failure survives a reboot and approved upgrade without dispatch or history rewriting", () => {
  const f = fixture(), now = Date.now(), boot = "62fafaf5-90cd-46ce-89f1-76e14e0e674f";
  const a = { operation: workflow, action: "inventory" as const, phase: "failed" as const,
    bootId: workflow, configurationSHA256: "c".repeat(64), guestConfigurationSHA256: "d".repeat(64) };
  f.record.assessment = a;
  f.record.guestCommand = { id: "ready-command", operation: "ready-op", action: "ready", phase: "finished", submittedAt: new Date(now).toISOString() };
  const previous = structuredClone(f.record.artifact);
  f.record.artifact = { ...previous, version: "2.4.0-dev.fixed", sha256: "b".repeat(64), development: { commit: "1".repeat(40), bytes: 100 } };
  f.record.upgrade = { operation: workflow, commandId: "upgrade-command", phase: "finished", previous, artifact: f.record.artifact, bootId: boot, submittedAt: new Date(now - 1000).toISOString() };
  f.record.guestReady = { ...f.record.guestReady!, bootId: boot, cliVersion: f.record.artifact.version,
    archiveSha256: f.record.artifact.sha256, commit: "1".repeat(40), checkedAt: new Date(now).toISOString(),
    health: { idle: true, storageUsedPercent: 4, swapUsedBytes: 0, oomEvents: 0 } };
  const before = structuredClone(f.record);
  const next = retainFailedAssessment(f.record, workflow, now);
  assert.equal(next.assessment, undefined);
  assert.deepEqual(next.assessmentHistory, [a]);
  assert.equal(next.assessmentHistory![0]!.bootId, workflow);
  assert.equal(next.guestReady!.bootId, boot);
  assert.deepEqual(f.record, before);
  assert.equal(f.requests.length, 0);
  for (const phase of ["submitted", "unknown", "failed"] as const) {
    const r = structuredClone(f.record); r.upgrade!.phase = phase;
    assert.throws(() => retainFailedAssessment(r, workflow, now));
  }
  f.record.guestReady.commit = "old";
  assert.throws(() => retainFailedAssessment(f.record, workflow, now));
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
test("PostgreSQL inventory refuses the legacy capability before any intent or cloud request", async () => {
  const f=fixture();f.record.input.source.type="postgresql";
  f.record.guestReady!.capabilities=["postgresql-inventory-v1","postgresql-migration-v1"];
  for(const action of ["profile","inventory"] as const)await assert.rejects(startAssessment(f.control,f.record,action,{}),/native floating-point preservation/);
  assert.equal(f.saved.length,0);assert.equal(f.requests.length,0);
});

function readinessFixture() {
  const f = fixture();
  f.record.guestReady!.checkedAt = new Date(Date.now() - 600000).toISOString();
  f.record.guestReady!.health = { idle: true, storageUsedPercent: 4, swapUsedBytes: 0, oomEvents: 0 };
  const result = { version: 1, ready: true, os: "linux", architecture: "amd64",
    ...f.record.guestReady, health: { ...f.record.guestReady!.health } };
  const payloads: Record<string, unknown>[] = [];
  let reads = 0;
  f.control.request = async (_sub, _path, method = "GET", body) => {
    if (method === "PUT") {
      const p = (body as { properties: { protectedParameters: { value: string }[] } }).properties.protectedParameters[0]!;
      payloads.push(JSON.parse(Buffer.from(p.value, "base64").toString("utf8")));
      return { status: 201, value: {} };
    }
    if (!payloads.length) return { status: 404, value: {} };
    reads++;
    return { status: 200, value: { properties: { instanceView: { executionState: "Succeeded", exitCode: 0, output: JSON.stringify(result) } } } };
  };
  return { ...f, result, payloads, reads: () => reads };
}

test("credential delay refreshes health once without any source payload, then permits original inventory", async () => {
  const f = readinessFixture();
  const r = await ensureAssessmentReadiness(f.control, f.record);
  assert.equal(r.guestCommand?.phase, "finished");
  assert.equal(r.guestReady?.bootId, f.record.guestReady?.bootId);
  assert.ok(Date.parse(r.guestReady!.checkedAt) > Date.parse(f.record.guestReady!.checkedAt));
  assert.equal(f.payloads.length, 1);
  assert.deepEqual(Object.keys(f.payloads[0]!).sort(), ["action", "operation", "version", "workflow"]);
  assert.equal(f.payloads[0]!.action, "ready");
  assert.equal(r.assessment, undefined);
  assert.ok(f.saved.every(s => s.assessment === undefined));
});

test("fresh idle readiness is reused, but unhealthy evidence is not bypassed", async () => {
  const f = readinessFixture(); f.record.guestReady!.checkedAt = new Date().toISOString();
  assert.equal(await ensureAssessmentReadiness(f.control, f.record), f.record);
  assert.equal(f.payloads.length, 0);
  f.record.guestReady!.health!.idle = false;
  await assert.rejects(ensureAssessmentReadiness(f.control, f.record), /idle worker/);
});

test("readiness after input rejects changed boot, installation or unsafe health", async () => {
  for (const change of [
    (r: ReturnType<typeof readinessFixture>["result"]) => { r.bootId = "22222222-2222-4222-8222-222222222222"; },
    (r: ReturnType<typeof readinessFixture>["result"]) => { r.archiveSha256 = "b".repeat(64); },
    (r: ReturnType<typeof readinessFixture>["result"]) => { r.health.storageUsedPercent = 80; },
    (r: ReturnType<typeof readinessFixture>["result"]) => { r.health.swapUsedBytes = 1; },
    (r: ReturnType<typeof readinessFixture>["result"]) => { r.health.oomEvents = 1; },
    (r: ReturnType<typeof readinessFixture>["result"]) => { r.health.idle = false; },
  ]) {
    const f = readinessFixture(); change(f.result);
    await assert.rejects(ensureAssessmentReadiness(f.control, f.record));
    assert.equal(f.payloads.length, 1); assert.equal(f.payloads[0]!.action, "ready");
    assert.ok(f.saved.every(s => s.assessment === undefined));
  }
});

test("pending readiness is bounded and never replayed after timeout or cancellation", async () => {
  const f = readinessFixture(); let puts = 0, gets = 0;
  f.control.request = async (_s, _p, method = "GET") => {
    if (method === "PUT") { puts++; return { status: 201, value: {} }; }
    gets++; return puts ? { status: 200, value: { properties: { instanceView: { executionState: "Pending" } } } } : { status: 404, value: {} };
  };
  await assert.rejects(ensureAssessmentReadiness(f.control, f.record), /still pending/);
  assert.equal(puts, 1); assert.equal(gets, 21);
  assert.equal(f.saved.at(-1)?.guestCommand?.phase, "submitted");
  await assert.rejects(ensureAssessmentReadiness(f.control, f.saved.at(-1)!), /pending guest/);
  await assert.rejects(ensureAssessmentReadiness(f.control, f.record, () => true), /cancelled/);
  assert.equal(puts, 1);
});

test("bootstrap pending during credential-wait readiness never starts assessment or retries",async()=>{
  const f=readinessFixture();let puts=0;
  f.control.request=async(_s,_p,method="GET")=>{
    if(method==="PUT"){puts++;return {status:201,value:{}};}
    return puts?{status:200,value:{properties:{instanceView:{executionState:"Succeeded",exitCode:0,output:JSON.stringify({version:1,ready:false,bootstrap:"pending"})}}}}:{status:404,value:{}};
  };
  await assert.rejects(ensureAssessmentReadiness(f.control,f.record),/bootstrap is still pending/);
  assert.equal(puts,1);assert.equal(f.saved.at(-1)?.guestReady,undefined);
  await assert.rejects(ensureAssessmentReadiness(f.control,f.saved.at(-1)!),/bootstrap is still pending/);
  assert.equal(puts,1);
});

test("closing the panel during readiness prevents source dispatch", async () => {
  const f = readinessFixture();
  await assert.rejects(ensureAssessmentReadiness(f.control, f.record, () => f.reads() > 0), /cancelled/);
  assert.equal(f.payloads.length, 1); assert.equal(f.payloads[0]!.action, "ready");
  assert.ok(f.saved.every(s => s.assessment === undefined));
});

test("post-inventory readiness refresh preserves the target and completed inventory for first migration", async () => {
  const f = readinessFixture();
  f.record.assessment = { operation: workflow, action: "inventory", phase: "finished", bootId: workflow,
    configurationSHA256: "b".repeat(64), reportSHA256: "c".repeat(64), reportBytes: 100 };
  f.record.target = { phase: "provisioned", hash: "plan-unchanged" } as RunnerRecord["target"];
  f.record.resize = { phase: "finished" } as RunnerRecord["resize"];
  const r = await ensureAssessmentReadiness(f.control, f.record);
  assert.deepEqual(r.assessment, f.record.assessment);
  assert.deepEqual(r.target, f.record.target);
  assert.deepEqual(r.resize, f.record.resize);
  assert.equal(r.migration, undefined);
  assert.equal(f.payloads.length, 1); assert.equal(f.payloads[0]!.action, "ready");
  const alreadyStarted = { ...r, migration: { phase: "submitted" } } as RunnerRecord;
  await assert.rejects(ensureAssessmentReadiness(f.control, alreadyStarted), /retained source operation/);
  assert.equal(f.payloads.length, 1);
});
