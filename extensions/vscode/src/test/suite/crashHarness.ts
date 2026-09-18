import assert from "node:assert/strict";
import * as vscode from "vscode";
import { mkdir, open, readFile, stat } from "node:fs/promises";
import { isAbsolute, join, resolve } from "node:path";
import { createHash } from "node:crypto";
import { RunnerStore, RunnerLockedError } from "../../guided/runnerStore";
import { sourceWorkflowDraft } from "../../core/runner";
import { retainReadinessReceipt } from "../../core/runnerReceipts";
import { guestDispatchScript } from "../../core/runnerGuest";
import { RemovalIO, previewReceiptRemoval, submitReceiptRemoval, reconcileReceiptRemoval } from "../../core/runnerReceiptRemoval";

const id = "11111111-1111-4111-8111-111111111111", operation = "22222222-2222-4222-8222-222222222222";
const binding = "b".repeat(64);
const sha = (text: string) => createHash("sha256").update(text).digest("hex");

export async function run(): Promise<void> {
  const root = process.env.AF_CRASH_ROOT!, phase = process.env.AF_CRASH_PHASE, point = process.env.AF_CRASH_POINT;
  // This runner may kill only its own disposable Extension Host, never a PID
  // discovered in the operator's normal VS Code or any external service.
  assert.ok(root && isAbsolute(root) && root.includes("af-host-crash-"));
  assert.equal(vscode.env.appHost, "desktop");
  assert.equal(vscode.extensions.getExtension("rioriost.agefreighter")?.extensionPath, resolve(__dirname, "../../.."),
    "Must be the disposable development Extension Host, not the installed extension");
  assert.ok(phase === "crash" || phase === "recover");
  assert.ok(point === "before-dispatch" || point === "after-dispatch");
  await mkdir(root, { recursive: true, mode: 0o700 });
  const store = new RunnerStore(join(root, "runner-v2"));
  async function sealed(name: string, value: unknown): Promise<void> {
    const file = await open(join(root, name), "wx", 0o600);
    try { await file.writeFile(JSON.stringify(value)); await file.sync(); } finally { await file.close(); }
    const dir = await open(root, "r"); try { await dir.sync(); } finally { await dir.close(); }
  }
  let record = sourceWorkflowDraft(id, { subscriptionId: id, resourceGroup: "trial", region: "japaneast", zone: "1", size: "Standard_B2s_v2", subnetId: "unused", source: { type: "csv", location: "local" } });
  record.phase = "provisioned";
  record.artifact = { version: "2.4.0", sha256: "a".repeat(64), url: "unused" };
  record.guestCommand = { id: `${record.vmId}/runCommands/af-${operation}`, operation, action: "ready", phase: "finished", submittedAt: "2026-09-18T05:00:00Z" };
  record.guestReady = { bootId: id, cliVersion: "2.4.0", archiveSha256: record.artifact.sha256, commit: "abcdef", checkedAt: record.guestCommand.submittedAt, health: { idle: true, storageUsedPercent: 12, swapUsedBytes: 0, oomEvents: 0 } };
  record = retainReadinessReceipt(record);
  const receipt = record.readinessReceipts![0]!;
  delete record.guestCommand; delete record.guestReady;
  const vm = { id: record.vmId, location: "japaneast", zones: ["1"], tags: { application: "agefreighter", workflow: id, purpose: "discovery-and-migration" }, properties: { vmId: id, provisioningState: "Succeeded", instanceView: { statuses: [{ code: "PowerState/deallocated" }] } } };
  const command = { id: receipt.command.id, location: "japaneast", properties: { source: { script: guestDispatchScript }, timeoutInSeconds: 60, asyncExecution: false, provisioningState: "Succeeded", instanceView: {
    executionState: "Succeeded", exitCode: 0, startTime: "2026-09-18T05:00:01Z", endTime: "2026-09-18T05:00:02Z", error: "", output: JSON.stringify({ version: 1, ready: true, os: "linux", architecture: "amd64", bootId: id, cliVersion: "2.4.0", archiveSha256: record.artifact!.sha256, commit: "abcdef", health: receipt.readiness.health }) } } };
  const events: string[] = [];
  async function crash(): Promise<never> {
    const saved = await store.read(id), intent = saved.readinessRemovals![0]!;
    const archive = await store.readReport(id, intent.archive);
    await sealed("crash.json", { point, signal: "SIGKILL", pid: process.pid, vscode: vscode.version, events,
      recordSHA256: sha(await readFile(join(root, "runner-v2", `${id}.json`), "utf8")), archiveSHA256: sha(archive) });
    process.kill(process.pid, "SIGKILL");
    return new Promise<never>(() => {});
  }
  // No AzureSession, credential lookup, network client or real ARM transport.
  const io: RemovalIO = {
    guard: async expected => { assert.equal(expected, binding); },
    retain: async (m, text) => { await store.retainReport(id, m, text); await store.syncEvidenceDirectory(); },
    read: m => store.readReport(id, m),
    control: {
      sleep: async () => {}, list: async () => { throw Error("Unexpected list"); },
      persist: async r => { await store.write(r); await store.syncEvidenceDirectory(); if (phase === "crash" && point === "before-dispatch") await crash(); },
      request: async (_sub, path, method = "GET") => {
        assert.equal(method, "GET"); events.push(path.includes("runCommands") ? "GET-command" : "GET-vm");
        return path.includes("runCommands") ? { status: phase === "recover" && point === "after-dispatch" ? 404 : 200, value: command } : { status: 200, value: vm };
      }
    },
    remove: async () => { assert.equal(phase, "crash", "Recovery must never dispatch"); events.push("synthetic-DELETE"); await crash(); }
  };
  if (phase === "crash") {
    await store.write(record);
    const plan = await previewReceiptRemoval(io, record, receipt.command.id, binding);
    await store.exclusive(id, () => submitReceiptRemoval(io, record, plan, true));
    throw Error("Crash injection was not reached");
  }
  const marker = JSON.parse(await readFile(join(root, "crash.json"), "utf8"));
  assert.notEqual(process.pid, marker.pid);
  const restored = await store.read(id), intent = restored.readinessRemovals![0]!;
  assert.equal(intent.phase, "submitted");
  assert.equal(sha(await readFile(join(root, "runner-v2", `${id}.json`), "utf8")), marker.recordSHA256);
  assert.equal(sha(await store.readReport(id, intent.archive)), marker.archiveSHA256);
  assert.equal(marker.events.filter((e: string) => e === "synthetic-DELETE").length, point === "after-dispatch" ? 1 : 0);
  await assert.rejects(store.exclusive(id, async () => { throw Error("Must not enter crash-locked action"); }), RunnerLockedError);
  assert.ok((await stat(join(root, "runner-v2", `${id}.lock`))).isFile());
  assert.equal(events.length, 0, "The native lock boundary permits no request");
  // Separately test the pure recovery controller against read-only fixtures.
  // Do NOT remove the crash lock or persist over it to simulate operator approval.
  io.control.persist = async () => { events.push("inert-persist"); };
  const result = await reconcileReceiptRemoval(io, restored, intent.commandId);
  assert.equal(result.readinessRemovals![0]!.phase, point === "after-dispatch" ? "absent" : "submitted");
  assert.deepEqual(events, point === "after-dispatch" ? ["GET-command", "inert-persist"] : ["GET-command"]);
  assert.equal(sha(await readFile(join(root, "runner-v2", `${id}.json`), "utf8")), marker.recordSHA256);
  await sealed("recovery.json", { pass: true, point, pid: process.pid, vscode: vscode.version, crashLockPreserved: true, archiveSHA256: marker.archiveSHA256, recoveryEvents: events, cloudRequests: 0 });
}
