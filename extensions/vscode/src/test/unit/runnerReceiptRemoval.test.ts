import assert from "node:assert/strict";
import test from "node:test";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { RunnerStore } from "../../guided/runnerStore";
import { sourceWorkflowDraft, RunnerRecord } from "../../core/runner";
import { guestDispatchScript, guestReadinessScript } from "../../core/runnerGuest";
import { retainReadinessReceipt } from "../../core/runnerReceipts";
import { previewReceiptRemoval, submitReceiptRemoval, reconcileReceiptRemoval, RemovalIO } from "../../core/runnerReceiptRemoval";
import { verifyReportBytes } from "../../core/runnerBlob";

const id = "11111111-1111-4111-8111-111111111111", op = "22222222-2222-4222-8222-222222222222", binding = "b".repeat(64);
export function removalFixture() {
  let record = sourceWorkflowDraft(id, { subscriptionId: id, resourceGroup: "trial", region: "japaneast", zone: "1", size: "Standard_B2s_v2", subnetId: "unused", source: { type: "csv", location: "local" } });
  record.phase = "provisioned";
  record.artifact = { version: "2.4.0", sha256: "a".repeat(64), url: "unused" };
  record.guestCommand = { id: `${record.vmId}/runCommands/af-${op}`, operation: op, action: "ready", phase: "finished", submittedAt: "2026-09-18T05:00:00Z" };
  record.guestReady = { bootId: id, cliVersion: "2.4.0", archiveSha256: record.artifact.sha256, commit: "abcdef", checkedAt: record.guestCommand.submittedAt,
    health: { idle: true, storageUsedPercent: 12, swapUsedBytes: 0, oomEvents: 0 } };
  record = retainReadinessReceipt(record);
  const receipt = record.readinessReceipts![0]!;
  const output = JSON.stringify({ version: 1, ready: true, os: "linux", architecture: "amd64", bootId: id, cliVersion: "2.4.0", archiveSha256: record.artifact.sha256, commit: "abcdef", health: receipt.readiness.health });
  const currentOperation = "33333333-3333-4333-8333-333333333333";
  const checkedAt = new Date(Date.now() - 10_000).toISOString();
  record.guestCommand = { ...receipt.command, id: `${record.vmId}/runCommands/af-${currentOperation}`, operation: currentOperation, submittedAt: checkedAt };
  record.guestReady = { ...receipt.readiness, checkedAt };
  record = retainReadinessReceipt(record);
  const vm: any = { id: record.vmId, location: "japaneast", zones: ["1"], tags: { application: "agefreighter", workflow: id, purpose: "discovery-and-migration" }, properties: { vmId: id, provisioningState: "Succeeded", instanceView: { statuses: [{ code: "PowerState/running" }] } } };
  const command: any = { id: receipt.command.id, location: "japaneast", properties: { source: { script: guestDispatchScript }, timeoutInSeconds: 60, asyncExecution: false, provisioningState: "Succeeded",
    instanceView: { executionState: "Succeeded", exitCode: 0, startTime: "2026-09-18T05:00:01Z", endTime: "2026-09-18T05:00:02Z", error: "", output } } };
  const currentCommand = structuredClone(command);
  currentCommand.id = record.guestCommand!.id;
  currentCommand.properties.instanceView.startTime = checkedAt;
  currentCommand.properties.instanceView.endTime = new Date(Date.parse(checkedAt) + 1_000).toISOString();
  const events: string[] = [], archives = new Map<string, string>(), saved: RunnerRecord[] = [];
  let absent = false, deny = false, loseReply = false, failArchive = false, failPersist = false;
  const io: RemovalIO = {
    control: { sleep: async () => {}, list: async () => { throw Error("No list expected"); },
      persist: async r => { events.push("persist"); if (failPersist) throw Error("disk failed"); saved.push(structuredClone(r)); },
      request: async (_sub, path, method = "GET") => { assert.equal(method, "GET");
        if (path.startsWith(currentCommand.id + "?")) { events.push("GET-current-readiness"); return { status: 200, value: currentCommand }; }
        events.push(path.includes("runCommands") ? "GET-command" : "GET-vm");
        return path.includes("runCommands") ? { status: absent ? 404 : 200, value: command } : { status: 200, value: vm }; } },
    guard: async expected => { events.push("guard"); assert.equal(expected, binding); if (deny) throw Error("account/trust changed"); },
    retain: async (m, text) => { events.push("archive"); if (failArchive) throw Error("archive failed"); verifyReportBytes(Buffer.from(text), m); archives.set(m.operation, text); },
    read: async m => { events.push("read-archive"); const text = archives.get(m.operation); if (!text) throw Error("archive missing"); return verifyReportBytes(Buffer.from(text), m); },
    remove: async (commandId, account) => { assert.equal(commandId, receipt.command.id); assert.equal(account, binding); events.push("DELETE"); if (loseReply) throw Error("sensitive service diagnostics"); }
  };
  return { record, receipt, vm, command, currentCommand, events, saved, archives, io, binding,
    preview: () => previewReceiptRemoval(io, record, receipt.command.id, binding),
    absent: () => { absent = true; }, deny: () => { deny = true; }, loseReply: () => { loseReply = true; },
    failArchive: () => { failArchive = true; }, failPersist: () => { failPersist = true; } };
}

test("preview is read-only; approved removal archives before single-use intent and DELETE", async () => {
  const f = removalFixture(), plan = await f.preview();
  assert.ok(f.events.every(e => e === "guard" || e.startsWith("GET")));
  const next = await submitReceiptRemoval(f.io, f.record, plan, true);
  assert.ok(f.events.indexOf("archive") < f.events.indexOf("persist"));
  assert.ok(f.events.indexOf("read-archive") < f.events.indexOf("persist"));
  assert.ok(f.events.indexOf("persist") < f.events.indexOf("DELETE"));
  assert.equal(next.readinessRemovals![0]!.phase, "submitted");
  const after = f.events.length;
  const stillThere = await reconcileReceiptRemoval(f.io, next, plan.commandId);
  assert.strictEqual(stillThere, next);
  f.absent();
  const finished = await reconcileReceiptRemoval(f.io, next, plan.commandId);
  assert.equal(finished.readinessRemovals![0]!.phase, "absent");
  assert.ok(!f.events.slice(after).includes("DELETE"));
});

test("cancel never archives, persists, or contacts Azure again", async () => {
  const f = removalFixture(), plan = await f.preview(), before = f.events.length;
  assert.strictEqual(await submitReceiptRemoval(f.io, f.record, plan, false), f.record);
  assert.equal(f.events.length, before);
});

test("running removal archives independent current health proof and cannot renew its freshness", async () => {
  const f = removalFixture(), plan = await f.preview(), archive = JSON.parse(plan.text);
  assert.equal(archive.kind, "agefreighter-readiness-removal-v2");
  assert.deepEqual(archive.receipt, f.receipt);
  assert.deepEqual(archive.observation.currentReadiness.receipt, f.record.readinessReceipts![1]);
  assert.equal(archive.observation.currentReadiness.command.id, f.currentCommand.id);
  assert.equal(archive.observation.vm.power, "PowerState/running");
  assert.equal(plan.expiresAt, Date.parse(f.record.guestReady!.checkedAt) + 300_000);
  const next = await submitReceiptRemoval(f.io, f.record, plan, true);
  assert.deepEqual(next.readinessReceipts, f.record.readinessReceipts);
  assert.deepEqual(next.guestReady, f.record.guestReady);
  assert.deepEqual(next.guestCommand, f.record.guestCommand);
});

for (const fault of ["missing", "stale", "future", "busy", "disk", "swap", "oom", "artifact", "different-boot", "unsealed",
  "status-pointer", "pending", "missing-output", "changed-output", "future-end", "unknown-history", "deployment", "unreconciled-removal"] as const) {
  test(`independent fresh readiness gate refuses ${fault}`, async () => {
    const f = removalFixture(), ready = f.record.guestReady!, live = f.currentCommand.properties.instanceView;
    if (fault === "missing") delete f.record.guestReady;
    if (fault === "stale") ready.checkedAt = new Date(Date.now() - 301_000).toISOString();
    if (fault === "future") ready.checkedAt = new Date(Date.now() + 60_000).toISOString();
    if (fault === "busy") ready.health = { ...ready.health!, idle: false };
    if (fault === "disk") ready.health = { ...ready.health!, storageUsedPercent: 80 };
    if (fault === "swap") ready.health = { ...ready.health!, swapUsedBytes: 1 };
    if (fault === "oom") ready.health = { ...ready.health!, oomEvents: 1 };
    if (fault === "artifact") f.record.artifact.sha256 = "c".repeat(64);
    if (fault === "different-boot") {
      ready.bootId = op;
      f.record.readinessReceipts = [f.receipt];
      Object.assign(f.record, retainReadinessReceipt(f.record));
      live.output = JSON.stringify({ ...JSON.parse(live.output), bootId: op });
    }
    if (fault === "unsealed") f.record.readinessReceipts = [f.receipt];
    if (fault === "status-pointer") f.record.guestCommand!.action = "status";
    if (fault === "pending") live.executionState = "Pending";
    if (fault === "missing-output") delete live.output;
    if (fault === "changed-output") live.output = live.output.replace("abcdef", "fedcba");
    if (fault === "future-end") live.endTime = new Date(Date.now() + 60_000).toISOString();
    if (fault === "unknown-history") f.record.assessmentHistory = [{ phase: "unknown" } as any];
    if (fault === "deployment") f.record.storageDeployment = { phase: "submitted" } as any;
    if (fault === "unreconciled-removal") f.record.readinessRemovals = [{ commandId: "another", phase: "unknown" } as any];
    await assert.rejects(f.preview());
    assert.ok(!f.events.some(e => ["archive", "persist", "DELETE"].includes(e)));
  });
}

test("latest readiness cannot be selected for removal even when its live proof succeeds", async () => {
  const f = removalFixture();
  await assert.rejects(previewReceiptRemoval(f.io, f.record, f.currentCommand.id, binding), /still referenced/);
  assert.ok(!f.events.includes("DELETE"));
});

test("historical Pending without timestamps or output stays blocked despite fresh current idle proof", async () => {
  const f = removalFixture();
  f.command.properties.instanceView = { executionState: "Pending", exitCode: 0 };
  await assert.rejects(f.preview(), /pending or not proven successful/);
  assert.ok(!f.events.some(e => ["archive", "persist", "DELETE"].includes(e)));
});

for (const stage of ["approval", "archive"] as const) for (const changed of ["candidate-pending", "current-pending", "current-metadata", "current-receipt", "vm-stopped"] as const) {
  test(`${changed} during ${stage} prevents deletion`, async () => {
    const f = removalFixture(), plan = await f.preview();
    const change = () => {
      if (changed === "candidate-pending") f.command.properties.instanceView = { executionState: "Pending", exitCode: 0 };
      if (changed === "current-pending") f.currentCommand.properties.instanceView.executionState = "Pending";
      if (changed === "current-metadata") f.currentCommand.tags = { changed: "yes" };
      if (changed === "current-receipt") f.record.readinessReceipts![1]!.sha256 = "c".repeat(64);
      if (changed === "vm-stopped") f.vm.properties.instanceView.statuses[0].code = "PowerState/deallocated";
    };
    if (stage === "approval") change();
    else { const retain = f.io.retain; f.io.retain = async (m, t) => { await retain(m, t); change(); }; }
    await assert.rejects(submitReceiptRemoval(f.io, f.record, plan, true));
    assert.ok(!f.events.includes("DELETE"));
    assert.ok(!f.events.includes("persist"));
  });
}

test("bootstrap-aware successful readiness has its exact script bound without admitting pending output", async () => {
  const f=removalFixture();f.command.properties.source.script=guestReadinessScript;
  const plan=await f.preview();
  assert.equal(JSON.parse(plan.text).observation.command.scriptSHA256,
    (await import("node:crypto")).createHash("sha256").update(JSON.stringify(guestReadinessScript)).digest("hex"));
  f.command.properties.instanceView.output=JSON.stringify({version:1,ready:false,bootstrap:"pending"});
  await assert.rejects(f.preview(),/Unexpected readiness/);
  assert.ok(!f.events.includes("DELETE"));
});

for (const fault of ["referenced", "readiness", "active", "unknown", "deallocated-vm", "owner", "vm-instance", "pending", "updating", "failed", "wrong-output", "unknown-field", "foreign", "script", "late-run", "secret-parameter"] as const) {
  test(`removal admission refuses ${fault}`, async () => {
    const f = removalFixture(), p = f.command.properties;
    if (fault === "referenced") f.record.guestCommand = f.receipt.command;
    if (fault === "readiness") f.record.guestReady = f.receipt.readiness;
    if (fault === "active") f.record.migration = { phase: "running" } as any;
    if (fault === "unknown") f.record.upgrade = { phase: "unknown" } as any;
    if (fault === "deallocated-vm") f.vm.properties.instanceView.statuses[0].code = "PowerState/deallocated";
    if (fault === "owner") f.vm.tags.workflow = op;
    if (fault === "vm-instance") delete f.vm.properties.vmId;
    if (fault === "pending") p.instanceView.executionState = "Pending";
    if (fault === "updating") p.provisioningState = "Updating";
    if (fault === "failed") p.instanceView.exitCode = 1;
    if (fault === "wrong-output") p.instanceView.output = p.instanceView.output.replace("abcdef", "fedcba");
    if (fault === "unknown-field") p.instanceView.output = JSON.stringify({ ...JSON.parse(p.instanceView.output), password: "not-archivable" });
    if (fault === "foreign") f.command.id += "-other";
    if (fault === "script") p.source.script += "\necho changed";
    if (fault === "late-run") { p.instanceView.startTime = "2026-09-19T05:00:00Z"; p.instanceView.endTime = "2026-09-19T05:00:01Z"; }
    if (fault === "secret-parameter") p.parameters = [{ name: "password", value: "secret" }];
    await assert.rejects(f.preview());
    assert.ok(!f.events.some(e => ["archive", "persist", "DELETE"].includes(e)));
  });
}

for (const fault of ["expired", "state", "account", "vm-replaced", "tags", "output", "archive", "archive-corrupt", "persist", "changed-during-archive"] as const) {
  test(`post-approval ${fault} fails without DELETE`, async () => {
    const f = removalFixture(), plan = await f.preview();
    if (fault === "expired") plan.expiresAt = Date.now() - 1;
    if (fault === "state") f.record.updatedAt = "changed";
    if (fault === "account") f.deny();
    if (fault === "vm-replaced") f.vm.properties.vmId = op;
    if (fault === "tags") f.vm.tags.changed = "yes";
    if (fault === "output") f.command.properties.instanceView.output += " ";
    if (fault === "archive") f.failArchive();
    if (fault === "archive-corrupt") f.io.read = async () => "corrupt";
    if (fault === "persist") f.failPersist();
    if (fault === "changed-during-archive") { const retain = f.io.retain; f.io.retain = async (m, t) => { await retain(m, t); f.vm.properties.vmId = op; }; }
    await assert.rejects(submitReceiptRemoval(f.io, f.record, plan, true));
    assert.ok(!f.events.includes("DELETE"));
  });
}

test("lost delete acknowledgement is sanitized, survives reload and never replays", async () => {
  const f = removalFixture(), plan = await f.preview(); f.loseReply();
  const next = await submitReceiptRemoval(f.io, f.record, plan, true);
  assert.equal(next.readinessRemovals![0]!.phase, "unknown");
  assert.ok(!JSON.stringify(next).includes("sensitive"));
  await assert.rejects(submitReceiptRemoval(f.io, next, plan, true));
  await assert.rejects(previewReceiptRemoval(f.io, next, plan.commandId, f.binding));
  const restored = JSON.parse(JSON.stringify(next));
  await reconcileReceiptRemoval(f.io, restored, plan.commandId);
  f.absent(); await reconcileReceiptRemoval(f.io, restored, plan.commandId);
  assert.equal(f.events.filter(e => e === "DELETE").length, 1);
});

test("crash immediately after durable intent cannot cause a DELETE on recovery", async () => {
  const f = removalFixture(), plan = await f.preview(), persist = f.io.control.persist;
  f.io.control.persist = async r => { await persist(r); throw Error("simulated host crash"); };
  await assert.rejects(submitReceiptRemoval(f.io, f.record, plan, true));
  assert.ok(!f.events.includes("DELETE"));
  f.io.control.persist = persist;
  await reconcileReceiptRemoval(f.io, f.saved[0]!, plan.commandId);
  assert.ok(!f.events.includes("DELETE"));
});

test("404 without the retained matching archive cannot establish completion", async () => {
  const f = removalFixture(), plan = await f.preview();
  const next = await submitReceiptRemoval(f.io, f.record, plan, true);
  f.absent(); f.archives.clear(); const before = f.events.length;
  await assert.rejects(reconcileReceiptRemoval(f.io, next, plan.commandId));
  assert.ok(!f.events.slice(before).includes("GET-command"));
  assert.equal(next.readinessRemovals![0]!.phase, "submitted");
});

test("resource reappearance invalidates an earlier absence observation without deletion", async () => {
  const f = removalFixture(), plan = await f.preview();
  const next = await submitReceiptRemoval(f.io, f.record, plan, true);
  next.readinessRemovals![0]!.phase = "absent";
  const observed = await reconcileReceiptRemoval(f.io, next, plan.commandId);
  assert.equal(observed.readinessRemovals![0]!.phase, "unknown");
  assert.equal(f.events.filter(e => e === "DELETE").length, 1);
});

test("legacy v1 intent still reconciles by GET with compute stopped and no fresh readiness", async () => {
  const f = removalFixture(), plan = await f.preview(), archive = JSON.parse(plan.text);
  archive.kind = "agefreighter-readiness-removal-v1";
  delete archive.observation.currentReadiness;
  archive.observation.vm.power = "PowerState/deallocated";
  const text = JSON.stringify(archive), manifest = { ...plan.manifest, bytes: Buffer.byteLength(text), sha256: createHash("sha256").update(text).digest("hex") };
  await f.io.retain(manifest, text);
  f.record.readinessRemovals = [{ commandId: plan.commandId, receiptSHA256: plan.receiptSHA256, archive: manifest,
    accountBinding: binding, submittedAt: new Date().toISOString(), phase: "unknown" }];
  f.vm.properties.instanceView.statuses[0].code = "PowerState/deallocated";
  delete f.record.guestReady;
  f.absent(); const before = f.events.length;
  const next = await reconcileReceiptRemoval(f.io, f.record, plan.commandId);
  assert.equal(next.readinessRemovals![0]!.phase, "absent");
  assert.deepEqual(f.events.slice(before), ["guard", "read-archive", "GET-command", "persist"]);
  assert.ok(!f.events.includes("DELETE"));
});

test("a legacy v1 review cannot authorize a new removal", async () => {
  const f = removalFixture(), plan = await f.preview(), archive = JSON.parse(plan.text);
  archive.kind = "agefreighter-readiness-removal-v1";
  plan.text = JSON.stringify(archive);
  plan.manifest = { ...plan.manifest, bytes: Buffer.byteLength(plan.text), sha256: createHash("sha256").update(plan.text).digest("hex") };
  await assert.rejects(submitReceiptRemoval(f.io, f.record, plan, true), /approved evidence/);
  assert.ok(!f.events.some(e => ["archive", "persist", "DELETE"].includes(e)));
});

test("malformed guest output never leaks its contents into the UI error", async () => {
  const f = removalFixture(); f.command.properties.instanceView.output = "SECRET-never-display";
  await assert.rejects(f.preview(), e => e instanceof Error && !e.message.includes("SECRET"));
});

test("trust/account failure after intent persists unknown without dispatch", async () => {
  const f = removalFixture(), plan = await f.preview(), persist = f.io.control.persist;
  f.io.control.persist = async r => { await persist(r); f.deny(); };
  const result = await submitReceiptRemoval(f.io, f.record, plan, true);
  assert.equal(result.readinessRemovals![0]!.phase, "unknown");
  assert.ok(!f.events.includes("DELETE"));
});

test("new store instance recovers a durable uncertain intent and archive with GET only", {skip: process.platform === "win32"}, async () => {
  const root = await mkdtemp(join(tmpdir(), "af-removal-reload-"));
  try {
    const f = removalFixture(), store = new RunnerStore(root), plan = await f.preview();
    f.io.control.persist = async r => { await store.write(r); await store.syncEvidenceDirectory(); };
    f.io.retain = async (m, t) => { await store.retainReport(id, m, t); await store.syncEvidenceDirectory(); };
    f.io.read = m => store.readReport(id, m);
    f.loseReply(); await submitReceiptRemoval(f.io, f.record, plan, true);
    const reopened = new RunnerStore(root), restored = await reopened.read(id);
    assert.equal(restored.readinessRemovals![0]!.phase, "unknown");
    f.io.read = m => reopened.readReport(id, m); f.absent();
    const next = await reconcileReceiptRemoval(f.io, restored, plan.commandId);
    assert.equal(next.readinessRemovals![0]!.phase, "absent");
    assert.equal(f.events.filter(e => e === "DELETE").length, 1);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("Windows archive durability failure prevents removal intent and DELETE while retaining evidence", {skip: process.platform !== "win32"}, async () => {
  const root = await mkdtemp(join(tmpdir(), "af-removal-unsupported-"));
  try {
    const f = removalFixture(), store = new RunnerStore(root), plan = await f.preview();
    await store.write(f.record);
    const before = await readFile(join(root, `${id}.json`));
    f.io.control.persist = async () => assert.fail("No removal intent before durable archive");
    f.io.retain = async (m, text) => {await store.retainReport(id, m, text); await store.syncEvidenceDirectory();};
    f.io.read = m => store.readReport(id, m);
    await assert.rejects(submitReceiptRemoval(f.io, f.record, plan, true), {code: "EPERM", syscall: "fsync"});
    assert.equal(f.events.filter(e => e === "DELETE").length, 0);
    assert.deepEqual(await readFile(join(root, `${id}.json`)), before);
    assert.equal((await new RunnerStore(root).read(id)).readinessRemovals, undefined);
    assert.equal(await store.readReport(id, plan.manifest), plan.text);
  } finally { await rm(root, {recursive: true, force: true}); }
});
