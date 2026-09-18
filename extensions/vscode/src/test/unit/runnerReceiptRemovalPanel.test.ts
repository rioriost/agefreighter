import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { Script } from "node:vm";
import { transformSync } from "esbuild";

const id = "11111111-1111-4111-8111-111111111111", binding = "b".repeat(64);
const code = transformSync(readFileSync(join(__dirname, "../../runnerReceiptRemovalPanel.ts"), "utf8"), { loader: "ts", format: "cjs" }).code;
function fixture() {
  const commandId = "/a-vm/runCommands/af-22222222-2222-4222-8222-222222222222";
  let record: any = { id, vmId: "a-vm", input: { subscriptionId: id, resourceGroup: "trial" }, readinessReceipts: [{ command: { id: commandId } }] };
  let answer: string | undefined = "Archive and remove this record", duringConfirm = () => {}, currentBinding = binding, failSync = false;
  const workspace = { isTrusted: true }, events: string[] = [], messages: string[] = [];
  const store = { list: async () => [record], read: async () => structuredClone(record), exclusive: async (_id: string, fn: () => unknown) => { events.push("lock"); return fn(); },
    retainReport: async () => { events.push("archive"); }, readReport: async () => "archive", syncEvidenceDirectory: async () => { events.push("sync-dir"); if (failSync) throw Error("sync unsupported"); } };
  const control = { persist: async (r: any) => { events.push("persist"); record = r; } };
  const azure = { runnerAccountBinding: async () => currentBinding, removeRunnerReadiness: async () => { events.push("DELETE"); } };
  const plan = { commandId, receiptSHA256: "a".repeat(64) };
  const modules: Record<string, unknown> = { vscode: { workspace, window: {
    showQuickPick: async (items: any[]) => items[0],
    showWarningMessage: async () => { events.push("confirm"); duringConfirm(); return answer; },
    showInformationMessage: async (m: string) => { messages.push(m); }
  } }, "./core/runnerReceiptRemoval": {
    previewReceiptRemoval: async () => { events.push("preview"); return plan; },
    submitReceiptRemoval: async (io: any, r: any) => {
      await io.guard(binding); await io.retain({}, "archive"); await io.control.persist(r); await io.remove(commandId, binding);
      return r;
    },
    reconcileReceiptRemoval: async (io: any, r: any) => { await io.guard(binding); events.push("GET-only"); return r; }
  } };
  const output = { exports: { manageReadinessRemoval: async (_control: unknown, _store: unknown, _azure: unknown) => {} } }, native = createRequire(__filename);
  new Script(code).runInNewContext({ module: output, exports: output.exports, Error, require: (n: string) => n in modules ? modules[n] : n.startsWith("node:") ? native(n) : {} });
  return { run: () => output.exports.manageReadinessRemoval(control, store, azure), workspace, events, messages,
    cancel: () => { answer = undefined; }, duringConfirm: (fn: () => void) => { duringConfirm = fn; }, changeAccount: () => { currentBinding = "c".repeat(64); }, failSync: () => { failSync = true; },
    intent: (phase: string) => { record.readinessRemovals = [{ commandId, accountBinding: binding, phase }]; } };
}

test("native cancellation does not archive, persist or delete", async () => {
  const f = fixture(); f.cancel(); await f.run(); assert.deepEqual(f.events, ["preview", "confirm"]);
});
for (const fault of ["trust", "account", "directory-sync"] as const) test(`native removal blocks ${fault} after confirmation`, async () => {
  const f = fixture(); f.duringConfirm(() => { if (fault === "trust") f.workspace.isTrusted = false; if (fault === "account") f.changeAccount(); if (fault === "directory-sync") f.failSync(); });
  await assert.rejects(f.run()); assert.ok(!f.events.includes("DELETE")); assert.equal(f.messages.length, 0);
});
test("native approved path uses workflow lock and fsyncs archive and intent", async () => {
  const f = fixture(); await f.run(); assert.deepEqual(f.events, ["preview", "confirm", "lock", "archive", "sync-dir", "persist", "sync-dir", "DELETE"]);
  assert.ok(f.messages[0]?.includes("No deletion-complete claim"));
});
test("retained uncertain intent selects GET-only reconciliation, not confirmation/dispatch", async () => {
  const f = fixture(); f.intent("unknown"); await f.run(); assert.deepEqual(f.events, ["lock", "GET-only"]);
  assert.ok(f.messages[0]?.includes("not confirmed"));
});

const azureCode = transformSync(readFileSync(join(__dirname, "../../guided/azure.ts"), "utf8"), { loader: "ts", format: "cjs" }).code;
test("Azure removal transport rejects wrong scopes/accounts/trust and uses only the exact command", async () => {
  const workspace = { isTrusted: true }, native = createRequire(__filename);
  const modules: Record<string, unknown> = { vscode: { workspace }, "@microsoft/vscode-azext-azureauth": { VSCodeAzureSubscriptionProvider: class {} } };
  const output: any = { exports: {} };
  new Script(azureCode).runInNewContext({ module: output, exports: output.exports, Error, require: (n: string) => n in modules ? modules[n] : n.startsWith("node:") ? native(n) : {} });
  const azure = new output.exports.AzureSession(), events: any[] = [];
  azure.runnerAccountBinding = async () => binding;
  azure.runnerRequest = async (...args: unknown[]) => { events.push(args); return { status: 202 }; };
  const command = `/subscriptions/${id}/resourceGroups/trial/providers/Microsoft.Compute/virtualMachines/af-${"a".repeat(20)}/runCommands/af-${id}`;
  for (const path of [command.replace("/trial/", "/trial/../"), command.replace(`/subscriptions/${id}`, "/subscriptions/other"), command.split("/runCommands")[0], command + "?bad=true"]) {
    await assert.rejects(azure.removeRunnerReadiness(id, path, binding));
  }
  await assert.rejects(azure.removeRunnerReadiness(id, command, "different-account"));
  workspace.isTrusted = false; await assert.rejects(azure.removeRunnerReadiness(id, command, binding));
  assert.equal(events.length, 0);
  workspace.isTrusted = true; await azure.removeRunnerReadiness(id, command, binding);
  assert.deepEqual(events[0], [id, command + "?api-version=2024-07-01", "DELETE"]);
});
