import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { Script } from "node:vm";
import { transformSync } from "esbuild";
import { sourceWorkflowDraft } from "../../core/runner";
import * as receipts from "../../core/runnerReceipts";

const id = "11111111-1111-4111-8111-111111111111", op = "22222222-2222-4222-8222-222222222222";
const code = transformSync(readFileSync(join(__dirname, "../../runnerReceiptsPanel.ts"), "utf8"), { loader: "ts", format: "cjs" }).code;
function fixture() {
  let record = sourceWorkflowDraft(id, { subscriptionId: id, resourceGroup: "trial", region: "japaneast", zone: "1", size: "Standard_B2s_v2", subnetId: "unused", source: { type: "csv", location: "local" } });
  record.artifact = { version: "2.4.0", sha256: "a".repeat(64), url: "unused" };
  record.guestCommand = { id: `${record.vmId}/runCommands/af-${op}`, operation: op, action: "ready", phase: "finished", submittedAt: "2026-09-18T05:00:00Z" };
  record.guestReady = { bootId: op, cliVersion: "2.4.0", archiveSha256: record.artifact.sha256, commit: "abcdef", checkedAt: record.guestCommand.submittedAt };
  record = receipts.retainReadinessReceipt(record);
  let pick = 0, cancelAt = 0, onSecond = () => {}, retained = "", rejectArchive = false;
  const events: string[] = [], messages: string[] = [];
  const workspace = { isTrusted: true, openTextDocument: async (o: { content: string }) => { events.push("open"); return o; } };
  const store = { list: async () => [record], read: async () => structuredClone(record), exclusive: async (_id: string, action: () => unknown) => action(),
    retainReport: async (_id: string, _manifest: unknown, text: string) => { if (rejectArchive) throw Error("archive write failed"); events.push("retain"); retained = text; },
    readReport: async () => { events.push("read-back"); return retained; } };
  const modules: Record<string, unknown> = { "./core/runnerReceipts": receipts, vscode: { workspace, window: {
    showQuickPick: async (items: unknown[]) => { pick++; if (pick === 2) onSecond(); return pick === cancelAt ? undefined : items[0]; },
    showInformationMessage: async (m: string) => { messages.push(m); }, showTextDocument: async () => { events.push("show"); }
  } } };
  const output = { exports: { archiveRunnerReadiness: async (_store: unknown) => {} } }, native = createRequire(__filename);
  new Script(code).runInNewContext({ module: output, exports: output.exports, Error, require: (n: string) => n in modules ? modules[n] : n.startsWith("node:") ? native(n) : {} });
  return { run: () => output.exports.archiveRunnerReadiness(store), record: () => record, workspace, events, messages,
    cancel: (n: number) => { cancelAt = n; }, duringSecond: (fn: () => void) => { onSecond = fn; }, failArchive: () => { rejectArchive = true; } };
}

test("native archive action works without Azure adapters, VM start, target or local CLI", async () => {
  const f = fixture(); await f.run();
  assert.deepEqual(f.events, ["retain", "read-back", "open", "show"]);
  assert.ok(f.messages.some(m => m.includes("Nothing") || m.includes("No Azure command")));
  assert.equal(f.record().target, undefined);
});
for (const change of ["cancel-workflow", "cancel-receipt", "trust", "vm", "subscription", "receipt", "archive-write"] as const) {
  test(`archive UI fails closed on ${change} without showing successful export`, async () => {
    const f = fixture();
    if (change === "cancel-workflow") f.cancel(1);
    if (change === "cancel-receipt") f.cancel(2);
    f.duringSecond(() => {
      if (change === "trust") f.workspace.isTrusted = false;
      if (change === "vm") f.record().vmId += "-changed";
      if (change === "subscription") f.record().input.subscriptionId = op;
      if (change === "receipt") f.record().readinessReceipts = [];
      if (change === "archive-write") f.failArchive();
    });
    if (change.startsWith("cancel")) await f.run(); else await assert.rejects(f.run());
    assert.deepEqual(f.events, []);
    assert.equal(f.messages.length, 0);
  });
}

test("legacy workflows without a receipt ledger are not adopted", async () => {
  const f = fixture(); delete f.record().readinessReceipts;
  await f.run(); assert.deepEqual(f.events, []);
  assert.ok(f.messages[0]?.includes("not automatically adopted"));
});
