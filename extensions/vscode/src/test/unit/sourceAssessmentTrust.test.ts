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
import { reportStorageNames } from "../../core/runnerReportStorage";
import { storageDraft } from "../../core/runnerStorageLifecycle";
import { sourceReportSummary } from "../../core/runnerSourceReport";
import { escapeHTML } from "../../core/report";
import { createHash } from "node:crypto";
import type { TargetReviewFeedback } from "../../runnerTargetPanel";

// Actual production panel handler; all desktop/cloud adapters are inert.
// The core cancellation boundaries have separate actual-controller tests.
const code = transformSync(readFileSync(join(__dirname, "../../runnerSourcePanel.ts"), "utf8"), { loader: "ts", format: "cjs" }).code;
function fixture(storageAccepted = true, autoFinish = false) {
  const account = `/subscriptions/${workflow}/resourceGroups/trial/providers/Microsoft.DocumentDB/databaseAccounts/fixture`;
  let record = sourceWorkflowDraft(workflow, { subscriptionId: workflow, resourceGroup: "trial", region: "japaneast", zone: "1", size: "Standard_B2s_v2", subnetId: "unused", source: { type: "cosmos-nosql", location: "azure", resourceId: account } });
  record.phase = "provisioned";
  record.guestReady = { bootId: workflow } as RunnerRecord["guestReady"];
  record.storageDeployment = { ...storageDraft(record, workflow), phase: "ready" };
  const workspace = { isTrusted: true }, messages: Record<string, any>[] = [];
  let receive = async (_message: unknown) => {}, onConfirm = () => {}, onLock = () => {}, onReady = () => {};
  let confirmations = 0, readiness = 0, starts = 0, writes = 0, cancellationDuringReadiness: boolean | undefined;
  let transfers = 0, views = 0;
  let report = JSON.stringify({ command: "profile", outcome: "incomplete" });
  let targetReview = async (_id: string, _feedback: TargetReviewFeedback) => {};
  const store = { read: async () => structuredClone(record), write: async (r: RunnerRecord) => { writes++; record = structuredClone(r); },
    readReport: async () => report,
    exclusive: async (_id: string, fn: () => unknown) => { onLock(); return fn(); } };
  const modules: Record<string, unknown> = {
    vscode: { workspace, ViewColumn: { One: 1 }, ProgressLocation: { Notification: 15 }, window: {
      createWebviewPanel: () => { views++; return { onDidDispose: () => {}, webview: { html: "", postMessage: async (m: Record<string, any>) => { messages.push(m); }, onDidReceiveMessage: (fn: typeof receive) => { receive = fn; return { dispose: () => {} }; } } }; },
      showWarningMessage: async () => { confirmations++; onConfirm(); return "Approve source reads"; },
      withProgress: async (_options: unknown, fn: (p: unknown, t: unknown) => unknown) => fn({}, { isCancellationRequested: false })
    } },
    "./core/runner": { object }, "./core/runnerSource": { buildSourceDraft, sourceSecrets },
    "./core/runnerAssessment": { assessmentActive: () => false,
      ensureAssessmentReadiness: async (_control: unknown, r: RunnerRecord, cancelled: () => boolean) => { readiness++; onReady(); cancellationDuringReadiness = cancelled(); return r; },
      startAssessment: async (_control: unknown, r: RunnerRecord, _action: unknown, _secrets: unknown, cancelled: () => boolean, autoReport: unknown) => {
        assert.equal(cancelled(), false); starts++;
        assert.deepEqual(JSON.parse(JSON.stringify(autoReport)), { storageId: reportStorageNames(record).id, deploymentHash: record.storageDeployment!.hash });
        if (autoFinish) record = { ...r, assessment: { operation: workflow, action: "profile", phase: "finished", bootId: workflow,
          configurationSHA256: createHash("sha256").update(JSON.stringify(r.sourceDraft!.configuration)).digest("hex"),
          reportSHA256: createHash("sha256").update(report).digest("hex"), reportBytes: Buffer.byteLength(report),
          autoReport: autoReport as NonNullable<RunnerRecord["assessment"]>["autoReport"] } };
        return autoFinish ? record : r;
      } },
    "./runnerWatch": { watchRetainedOperation: async () => autoFinish ? record : undefined }, "./core/runnerSourceView": { runnerSourceHTML: () => "inert" },
    "./core/runnerReport": { canRetainRejectedReportExport: () => false },
    "./core/runnerReportStorage": { reportStorageNames },
    "./runnerStorageFlow": { prepareRequiredStorage: async () => storageAccepted ? record : undefined },
    "./core/runnerSourceReport": { sourceReportSummary }, "./core/report": { escapeHTML },
    "./runnerReportFlow": { transferApprovedReport: async () => {
      transfers++; const a = record.assessment!;
      record = { ...record, reportTransfers: [{ operation: a.operation, sha256: a.reportSHA256!, bytes: a.reportBytes!, phase: "imported", blob: "owned" }] }; return record;
    } }
  };
  const output = { exports: { openRunnerSource: (_context: unknown, _control: unknown, _store: unknown, _id: string, _services: unknown, _reviewTarget: typeof targetReview) => {} } }, native = createRequire(__filename);
  new Script(code).runInNewContext({ module: output, exports: output.exports, Error, Date, Buffer, require: (name: string) => name in modules ? modules[name] : name.startsWith("node:") ? native(name) : {} });
  output.exports.openRunnerSource({ subscriptions: [] }, {}, store, workflow, {}, (id, feedback) => targetReview(id, feedback));
  return { workspace, messages, review: () => receive({ action: "review", form: { ...sourceForm, host: "fixture.documents.azure.com", database: "p1" } }),
    run: () => receive({ action: "assess", method: "profile" }),
    reviewTarget: () => receive({ action: "reviewTarget" }),
    stopWatch: () => receive({ action: "stopWatch" }),
    setTargetReview: (fn: typeof targetReview) => { targetReview = fn; },
    prepareTarget: (outcome = "pass") => {
      record.input.source = { type: "neo4j", location: "on-premises" };
      record.sourceDraft = { form: sourceForm, configuration: { source: { type: "neo4j" } }, warnings: [], canAssess: true };
      report = JSON.stringify({ schemaVersion: 1, command: "inventory", agefreighterVersion: record.artifact.version, outcome, errors: [], incompleteChecks: [],
        checks: [{ id: "source-counts", status: "pass" }], sections: [{ title: "Source inventory", fields:
          Object.entries({ vertices: "100000", edges: "250000", totalRows: "350000", countMethod: "neo4j-transactional-count-store" }).map(([name, value]) => ({ name, value, status: "pass" })) }] });
      const sha256 = createHash("sha256").update(report).digest("hex"), bytes = Buffer.byteLength(report);
      record.assessment = { operation: workflow, action: "inventory", phase: "finished", bootId: workflow,
        configurationSHA256: createHash("sha256").update(JSON.stringify(record.sourceDraft.configuration)).digest("hex"), reportSHA256: sha256, reportBytes: bytes };
      record.reportTransfers = [{ operation: workflow, sha256, bytes, phase: "imported", blob: "owned" }];
      return record;
    },
    duringConfirm: (fn: () => void) => { onConfirm = fn; }, duringLock: (fn: () => void) => { onLock = fn; }, duringReady: (fn: () => void) => { onReady = fn; },
    counts: () => ({ confirmations, readiness, starts, writes, cancellationDuringReadiness }), transferCount: () => transfers, viewCount: () => views };
}

test("target review failure returns to the source panel and clears busy without a native notification", async () => {
  const f = fixture(); f.prepareTarget();
  let calls = 0;
  f.setTargetReview(async id => { assert.equal(id, workflow); calls++; throw Error("Azure VM provisioning is still Updating after Linux readiness."); });
  await f.reviewTarget();
  assert.equal(calls, 1);
  assert.equal(f.messages.at(-2)?.kind, "error");
  assert.match(f.messages.at(-2)!.text, /still Updating/);
  assert.equal(f.messages.at(-1)?.kind, "busy"); assert.equal(f.messages.at(-1)?.value, false);
  f.setTargetReview(async () => { calls++; });
  await f.reviewTarget();
  assert.equal(calls, 2, "failure releases the host-side busy guard for an explicit retry");
  assert.equal(f.counts().starts, 0); assert.equal(f.counts().writes, 0);
});

test("an ineligible report cannot invoke the direct target-review callback", async () => {
  const f = fixture(); f.prepareTarget("incomplete");
  f.setTargetReview(async () => assert.fail("Incomplete report must not advance"));
  await f.reviewTarget();
  assert.equal(f.messages.at(-2)?.kind, "error");
  assert.equal(f.messages.at(-1)?.kind, "busy"); assert.equal(f.messages.at(-1)?.value, false);
});

test("source target progress and stop-monitoring are wired through the direct callback", async () => {
  const f = fixture(), r = f.prepareTarget();
  f.setTargetReview(async (_id, feedback) => {
    assert.equal(feedback.cancelled!(), false);
    await feedback.progress!(r, "Creating private target", true);
    assert.equal(f.messages.at(-1)!.text, "Creating private target");
    await f.stopWatch();
    assert.equal(feedback.cancelled!(), true);
    const count = f.messages.length;
    await feedback.progress!(r, "Late target result", false);
    assert.equal(f.messages.length, count);
  });
  await f.reviewTarget();
  assert.equal(f.messages.at(-1)!.kind, "busy"); assert.equal(f.messages.at(-1)!.value, false);
  assert.equal(f.counts().starts, 0); assert.equal(f.counts().writes, 0);
});

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

test("cancelling required storage preparation never starts the approved source read", async () => {
  const f = fixture(false); await f.review(); await f.run();
  assert.equal(f.counts().starts, 0); assert.equal(f.counts().readiness, 0);
  assert.ok(f.messages.some(m => m.kind === "progress" && /cancelled/.test(m.text)));
});

test("an approved finished assessment automatically imports and opens its actual incomplete outcome", async () => {
  const f = fixture(true, true); await f.review(); await f.run();
  await new Promise<void>(resolve => setImmediate(resolve));
  assert.equal(f.transferCount(), 1); assert.equal(f.viewCount(), 2);
  const result = f.messages.find(m => m.kind === "reportResult");
  assert.equal(result?.summary.canReviewTarget, false);
  assert.equal(result?.summary.title, "Source assessment incomplete");
  assert.ok(!f.messages.some(m => m.kind === "error"), JSON.stringify(f.messages));
});
