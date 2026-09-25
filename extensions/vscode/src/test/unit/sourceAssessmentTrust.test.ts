import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { Script } from "node:vm";
import { transformSync } from "esbuild";
import { object, RunnerRecord, sourceWorkflowDraft } from "../../core/runner";
import { buildSourceDraft, sourceSecrets } from "../../core/runnerSource";
import { sourceForm, workflow } from "../sourceFixtures";

// Actual production panel handler; all desktop/cloud adapters are inert.
// The core cancellation boundaries have separate actual-controller tests.
const code = transformSync(readFileSync(join(__dirname, "../../runnerSourcePanel.ts"), "utf8"), { loader: "ts", format: "cjs" }).code;
function fixture() {
  const account = `/subscriptions/${workflow}/resourceGroups/trial/providers/Microsoft.DocumentDB/databaseAccounts/fixture`;
  let record = sourceWorkflowDraft(workflow, { subscriptionId: workflow, resourceGroup: "trial", region: "japaneast", zone: "1", size: "Standard_B2s_v2", subnetId: "unused", source: { type: "cosmos-nosql", location: "azure", resourceId: account } });
  record.phase = "provisioned";
  record.guestReady = { bootId: workflow } as RunnerRecord["guestReady"];
  const workspace = { isTrusted: true }, messages: Record<string, any>[] = [];
  let receive = async (_message: unknown) => {}, onConfirm = () => {}, onLock = () => {}, onReady = () => {};
  let confirmations = 0, readiness = 0, starts = 0, writes = 0, cancellationDuringReadiness: boolean | undefined;
  const store = { read: async () => structuredClone(record), write: async (r: RunnerRecord) => { writes++; record = structuredClone(r); },
    exclusive: async (_id: string, fn: () => unknown) => { onLock(); return fn(); } };
  const modules: Record<string, unknown> = {
    vscode: { workspace, ViewColumn: { One: 1 }, ProgressLocation: { Notification: 15 }, window: {
      createWebviewPanel: () => ({ onDidDispose: () => {}, webview: { html: "", postMessage: async (m: Record<string, any>) => { messages.push(m); }, onDidReceiveMessage: (fn: typeof receive) => { receive = fn; return { dispose: () => {} }; } } }),
      showWarningMessage: async () => { confirmations++; onConfirm(); return "Approve source reads"; },
      withProgress: async (_options: unknown, fn: () => unknown) => fn()
    } },
    "./core/runner": { object }, "./core/runnerSource": { buildSourceDraft, sourceSecrets },
    "./core/runnerAssessment": { assessmentActive: () => false,
      ensureAssessmentReadiness: async (_control: unknown, r: RunnerRecord, cancelled: () => boolean) => { readiness++; onReady(); cancellationDuringReadiness = cancelled(); return r; },
      startAssessment: async (_control: unknown, r: RunnerRecord, _action: unknown, _secrets: unknown, cancelled: () => boolean) => { assert.equal(cancelled(), false); starts++; return r; } },
    "./runnerWatch": { watchRetainedOperation: async () => {} }, "./core/runnerSourceView": { runnerSourceHTML: () => "inert" },
    "./core/runnerReport": { canRetainRejectedReportExport: () => false }
  };
  const output = { exports: { openRunnerSource: (_context: unknown, _control: unknown, _store: unknown, _id: string) => {} } }, native = createRequire(__filename);
  new Script(code).runInNewContext({ module: output, exports: output.exports, Error, require: (name: string) => name in modules ? modules[name] : name.startsWith("node:") ? native(name) : {} });
  output.exports.openRunnerSource({ subscriptions: [] }, {}, store, workflow);
  return { workspace, messages, review: () => receive({ action: "review", form: { ...sourceForm, host: "fixture.documents.azure.com", database: "p1" } }),
    run: () => receive({ action: "assess", method: "profile" }),
    duringConfirm: (fn: () => void) => { onConfirm = fn; }, duringLock: (fn: () => void) => { onLock = fn; }, duringReady: (fn: () => void) => { onReady = fn; },
    counts: () => ({ confirmations, readiness, starts, writes, cancellationDuringReadiness }) };
}

for (const boundary of ["before approval", "during approval", "waiting for lock", "during readiness"] as const) test(`source panel trust loss ${boundary} cannot dispatch`, async () => {
  const f = fixture(); await f.review();
  const revoke = () => { f.workspace.isTrusted = false; };
  if (boundary === "before approval") revoke();
  if (boundary === "during approval") f.duringConfirm(revoke);
  if (boundary === "waiting for lock") f.duringLock(revoke);
  if (boundary === "during readiness") f.duringReady(revoke);
  await f.run();
  const counts = f.counts();
  assert.equal(counts.confirmations, boundary === "before approval" ? 0 : 1);
  assert.equal(counts.readiness, boundary === "during readiness" ? 1 : 0);
  if (boundary === "during readiness") assert.equal(counts.cancellationDuringReadiness, true);
  assert.equal(counts.starts, 0); assert.equal(counts.writes, 1, "only the earlier local review may persist");
  assert.ok(f.messages.some(m => m.kind === "error" && /Trust|trust changed/.test(m.text)));
});

test("trusted reviewed source proceeds once and passes its live cancellation callback", async () => {
  const f = fixture(); await f.review(); await f.run();
  assert.deepEqual(f.counts(), { confirmations: 1, readiness: 1, starts: 1, writes: 1, cancellationDuringReadiness: false });
  assert.ok(!f.messages.some(m => m.kind === "error"));
});
