import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { RunnerStore } from "../../guided/runnerStore";
import { sourceWorkflowDraft, RunnerRecord } from "../../core/runner";
import { guestDispatchScript } from "../../core/runnerGuest";
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
  delete record.guestCommand; delete record.guestReady;
  const vm: any = { id: record.vmId, location: "japaneast", zones: ["1"], tags: { application: "agefreighter", workflow: id, purpose: "discovery-and-migration" }, properties: { vmId: id, provisioningState: "Succeeded", instanceView: { statuses: [{ code: "PowerState/deallocated" }] } } };
  const command: any = { id: receipt.command.id, location: "japaneast", properties: { source: { script: guestDispatchScript }, timeoutInSeconds: 60, asyncExecution: false, provisioningState: "Succeeded",
    instanceView: { executionState: "Succeeded", exitCode: 0, startTime: "2026-09-18T05:00:01Z", endTime: "2026-09-18T05:00:02Z", error: "", output } } };
  const events: string[] = [], archives = new Map<string, string>(), saved: RunnerRecord[] = [];
  let absent = false, deny = false, loseReply = false, failArchive = false, failPersist = false;
  const io: RemovalIO = {
    control: { sleep: async () => {}, list: async () => { throw Error("No list expected"); },
      persist: async r => { events.push("persist"); if (failPersist) throw Error("disk failed"); saved.push(structuredClone(r)); },
      request: async (_sub, path, method = "GET") => { assert.equal(method, "GET"); events.push(path.includes("runCommands") ? "GET-command" : "GET-vm");
        return path.includes("runCommands") ? { status: absent ? 404 : 200, value: command } : { status: 200, value: vm }; } },
    guard: async expected => { events.push("guard"); assert.equal(expected, binding); if (deny) throw Error("account/trust changed"); },
    retain: async (m, text) => { events.push("archive"); if (failArchive) throw Error("archive failed"); verifyReportBytes(Buffer.from(text), m); archives.set(m.operation, text); },
    read: async m => { events.push("read-archive"); const text = archives.get(m.operation); if (!text) throw Error("archive missing"); return verifyReportBytes(Buffer.from(text), m); },
    remove: async (commandId, account) => { assert.equal(commandId, receipt.command.id); assert.equal(account, binding); events.push("DELETE"); if (loseReply) throw Error("sensitive service diagnostics"); }
  };
  return { record, receipt, vm, command, events, saved, archives, io, binding,
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

for (const fault of ["referenced", "readiness", "active", "unknown", "running-vm", "owner", "vm-instance", "pending", "updating", "failed", "wrong-output", "unknown-field", "foreign", "script", "late-run", "secret-parameter"] as const) {
  test(`removal admission refuses ${fault}`, async () => {
    const f = removalFixture(), p = f.command.properties;
    if (fault === "referenced") f.record.guestCommand = f.receipt.command;
    if (fault === "readiness") f.record.guestReady = f.receipt.readiness;
    if (fault === "active") f.record.migration = { phase: "running" } as any;
    if (fault === "unknown") f.record.upgrade = { phase: "unknown" } as any;
    if (fault === "running-vm") f.vm.properties.instanceView.statuses[0].code = "PowerState/running";
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

test("new store instance recovers a durable uncertain intent and archive with GET only", async () => {
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
