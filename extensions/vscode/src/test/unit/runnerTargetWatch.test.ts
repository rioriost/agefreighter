import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { Script } from "node:vm";
import { join } from "node:path";
import { transformSync } from "esbuild";
import { RunnerRecord } from "../../core/runner";
import { RunnerControl } from "../../core/runnerLifecycle";
import * as target from "../../core/runnerTarget";
import * as drafts from "../../core/targetDraft";
import { boundedWatch } from "../../core/boundedWatch";
import { executionActionState } from "../../core/runnerExecutionActions";
import type { TargetReviewFeedback } from "../../runnerTargetPanel";
import { otherCancellationFixture, otherNativeCancelCases } from "../helpers/nativeCancelOtherScenarios";

function load<T>(file: string, modules: Record<string, unknown>, clock: { now(): number }) {
  const output = { exports: {} }, native = createRequire(__filename);
  const code = transformSync(readFileSync(join(__dirname, "../../", file), "utf8"), { loader: "ts", format: "cjs" }).code;
  new Script(code).runInNewContext({ module: output, exports: output.exports, Date: clock, Error, Buffer,
    require: (name: string) => name in modules ? modules[name] : name.startsWith("node:") ? native(name) : assert.fail("Unexpected dependency " + name) });
  return output.exports as T;
}

function fixture(states = ["Running", "Succeeded"]) {
  let record = otherCancellationFixture(otherNativeCancelCases.find(c => c.id === "A21")!).record;
  record.target!.phase = "submitted";
  delete record.targetRestart; delete record.resize;
  let now = Date.now(), locked = false, cancelled = false, reads = 0, sleeps = 0, progressWindows = 0;
  let duringRead = () => {}, duringSleep = () => {}, duringLock = () => {};
  const workspace = { isTrusted: true }, token = { isCancellationRequested: false };
  class Clock extends Date { static override now() { return now; } }
  const messages: string[] = [], updates: RunnerRecord[] = [];
  const vscode = { workspace, ProgressLocation: { Notification: 1 }, window: {
    withProgress: async <T>(options: { cancellable: boolean }, fn: (progress: { report: (value: { message: string }) => void }, cancellation: typeof token) => Promise<T>) => {
      progressWindows++; assert.equal(options.cancellable, true);
      return fn({ report: ({ message }) => { messages.push(message); } }, token);
    },
    showInformationMessage: async (text: string) => { messages.push(text); },
    showWarningMessage: async () => assert.fail("Monitoring cannot authorize a repair or new deployment")
  } };
  const control: RunnerControl = {
    persist: async r => { record = r; },
    request: async (_sub, path, method = "GET", body) => {
      assert.equal(method, "GET"); assert.equal(body, undefined);
      assert.equal(path, record.target!.deploymentId + "?api-version=2022-09-01");
      reads++; duringRead();
      return { status: 200, value: { properties: { provisioningState: states[Math.min(reads - 1, states.length - 1)] } } };
    },
    list: async () => assert.fail("Unexpected ARM list"),
    sleep: async ms => { assert.equal(locked, false); assert.equal(ms, 15000); sleeps++; now += ms; duringSleep(); }
  };
  const store = { read: async () => structuredClone(record), write: control.persist, readReport: async () => "{}",
    exclusive: async <T>(_id: string, run: () => Promise<T>) => {
      assert.equal(locked, false); locked = true;
      try { duringLock(); return await run(); } finally { locked = false; }
    }
  };
  type Watch = (control: RunnerControl, storage: typeof store, workflow: string, cancelled: () => boolean, progress: (r: RunnerRecord) => Promise<void>) => Promise<RunnerRecord | undefined>;
  const watchers = load<{ watchTargetState: Watch }>("runnerWatch.ts", {
    vscode, "./core/runnerTarget": target, "./core/runnerLifecycle": {}, "./core/runnerGuest": {},
    "./core/runnerAssessment": {}, "./core/runnerCatalog": {}, "./core/runnerExecution": {},
    "./core/boundedWatch": { boundedWatch: <T>(options: Parameters<typeof boundedWatch<T>>[0]) => boundedWatch({ ...options, now: () => now }) }
  }, Clock);
  type Review = (context: unknown, control: RunnerControl, storage: typeof store, azure: unknown, workflow: string, feedback?: TargetReviewFeedback) => Promise<void>;
  const panelModules = { vscode, "./runnerWatch": watchers, "./runnerTargetInputs": {}, "./core/targetDraft": drafts,
    "./core/runnerTarget": target, "./core/runnerTargetPreflight": {}, "./core/runnerAssessment": {} };
  const panel = load<{ reviewRunnerTarget: Review }>("runnerTargetPanel.ts", panelModules, Clock);
  const progress = async (r: RunnerRecord) => { updates.push(r); };
  return { control, store, workspace, token, updates, messages, panelModules, Clock, watchers,
    current: () => record, stats: () => ({ reads, sleeps, progressWindows }),
    replace: (r: RunnerRecord) => { record = r; }, cancel: () => { cancelled = true; },
    duringRead: (fn: () => void) => { duringRead = fn; }, duringSleep: (fn: () => void) => { duringSleep = fn; },
    duringLock: (fn: () => void) => { duringLock = fn; }, advance: (ms: number) => { now += ms; },
    run: () => watchers.watchTargetState(control, store, record.id, () => cancelled, progress),
    review: (feedback?: TargetReviewFeedback) => panel.reviewRunnerTarget({}, control, store, {}, record.id, feedback)
  };
}

test("target watcher uses only GETs, releases the lock between checks and enables only the next legitimate step", async () => {
  const f = fixture();
  assert.equal(executionActionState(f.current(), "preload").enabled, false);
  const result = await f.run();
  assert.equal(result?.target?.phase, "provisioned");
  assert.deepEqual(f.stats(), { reads: 2, sleeps: 1, progressWindows: 1 });
  assert.deepEqual(f.updates.map(r => r.target?.phase), ["submitted", "provisioned"]);
  assert.equal(executionActionState(f.updates[0], "preload").enabled, false);
  assert.equal(executionActionState(result, "preload").enabled, true);
  for (const action of ["resize", "readiness", "start", "verify"] as const) assert.equal(executionActionState(result, action).enabled, false);
  assert.match(f.messages[0]!, /Creating.*Step 5-1 stays unavailable/);
  assert.match(f.messages.at(-1)!, /5-1.*Migration has not started/);
});

test("unknown submissions reconcile without replay and completed preload is not presented as the next step", async () => {
  const f = fixture(["Succeeded"]);
  f.current().target!.phase = "unknown";
  f.current().targetRestart = { phase: "finished", submittedAt: new Date().toISOString() };
  const result = await f.run();
  assert.equal(result?.target?.phase, "provisioned"); assert.equal(f.stats().reads, 1);
  assert.match(f.messages.at(-1)!, /remaining steps.*Migration has not started/);
  assert.doesNotMatch(f.messages.at(-1)!, /Continue with 5-1/);
});

for (const state of ["Failed", "Canceled"]) test(`terminal target ${state} stops without repair or replay`, async () => {
  const f = fixture([state]), result = await f.run();
  assert.equal(result?.target?.phase, "failed"); assert.equal(f.stats().reads, 1); assert.equal(f.stats().sleeps, 0);
  assert.equal(executionActionState(result, "preload").enabled, false);
  assert.match(f.messages.at(-1)!, /failed.*blocked/);
});

for (const state of ["Running", "unexpected"]) test(`pending target ${state} is bounded and explicitly resumable without replay`, async () => {
  const f = fixture([state]), feedback: { text: string; active: boolean }[] = [];
  await f.review({ progress: async (_r, text, active) => { feedback.push({ text, active }); } });
  assert.equal(f.stats().reads, 120); assert.equal(f.stats().sleeps, 119);
  assert.equal(f.current().target?.phase, state === "Running" ? "submitted" : "unknown");
  assert.match(feedback.at(-1)!.text, /reached its time limit.*resume monitoring/);
  assert.equal(feedback.at(-1)!.active, false);
  assert.equal(executionActionState(f.current(), "preload").enabled, false);
  await f.run(); assert.equal(f.stats().reads, 240);
});

test("a slow GET reaching the deadline ends the watch without another check or wait", async () => {
  const f = fixture(["Running"]);
  f.duringRead(() => f.advance(30 * 60000));
  const result = await f.run();
  assert.equal(result?.target?.phase, "submitted"); assert.equal(f.stats().reads, 1); assert.equal(f.stats().sleeps, 0);
});

for (const boundary of ["before", "lock", "request", "sleep", "native", "trust"] as const) test(`target monitoring cancellation at ${boundary} cannot start work or publish late progress`, async () => {
  const f = fixture(["Running"]);
  if (boundary === "before") f.cancel();
  if (boundary === "lock") f.duringLock(f.cancel);
  if (boundary === "request") f.duringRead(f.cancel);
  if (boundary === "sleep") f.duringSleep(f.cancel);
  if (boundary === "native") f.token.isCancellationRequested = true;
  if (boundary === "trust") f.duringRead(() => { f.workspace.isTrusted = false; });
  assert.equal(await f.run(), undefined);
  assert.equal(f.stats().reads, ["request", "sleep", "trust"].includes(boundary) ? 1 : 0);
  assert.equal(f.updates.length, boundary === "sleep" ? 1 : 0);
});

for (const field of ["vmId", "artifact", "deploymentId", "serverId", "hash", "phase", "removed"] as const) test(`changed ${field} stops target monitoring before another ARM read`, async () => {
  const f = fixture(["Running"]);
  f.duringSleep(() => {
    const r = f.current();
    if (field === "removed") delete r.target;
    else if (field === "vmId") r.vmId += "-other";
    else if (field === "artifact") r.artifact.sha256 = "f".repeat(64);
    else if (field === "phase") r.target!.phase = "previewed";
    else r.target![field] += "-other";
  });
  await assert.rejects(f.run(), /identity changed/); assert.equal(f.stats().reads, 1);
});

test("target read failures surface immediately without a success-shaped status or retry", async () => {
  const f = fixture();
  f.duringRead(() => { throw Error("ARM permission denied"); });
  await assert.rejects(f.review(), /ARM permission denied/);
  assert.equal(f.stats().reads, 1); assert.equal(f.updates.length, 0);
  assert.equal(f.current().target?.phase, "submitted");
});

test("target review forwards pending and completed progress, and native cancellation remains pending", async () => {
  const f = fixture(), updates: { phase?: string; text: string; active: boolean }[] = [];
  await f.review({ progress: async (r, text, active) => { updates.push({ phase: r.target?.phase, text, active }); } });
  assert.ok(updates.some(u => u.phase === "submitted" && u.active));
  assert.equal(updates.at(-1)!.phase, "provisioned"); assert.equal(updates.at(-1)!.active, false);
  assert.match(updates.at(-1)!.text, /5-1/);
  const cancelled = fixture();
  cancelled.token.isCancellationRequested = true;
  await cancelled.review();
  assert.equal(cancelled.stats().reads, 0); assert.equal(cancelled.current().target?.phase, "submitted");
  assert.match(cancelled.messages.at(-1)!, /Monitoring stopped.*not cancelled/);
});

test("a newly approved target is monitored after releasing the submission lock, without a second submission", async () => {
  const f = fixture(), r = f.current(), plan = r.target!;
  delete r.target;
  f.replace(drafts.retainTargetDraft(r, drafts.targetDraftBinding(r), plan.input, "/inert-approved-folder"));
  let submissions = 0, files = 0;
  type Review = (context: unknown, control: RunnerControl, store: typeof f.store, azure: unknown, id: string) => Promise<void>;
  const panel = load<{ reviewRunnerTarget: Review }>("runnerTargetPanel.ts", {
    ...f.panelModules,
    vscode: { ...f.panelModules.vscode, window: { ...f.panelModules.vscode.window,
      showQuickPick: async () => "Reuse saved target inputs", showWarningMessage: async () => "Save plan and approve target deployment" } },
    "node:fs/promises": { open: async () => { files++; return { writeFile: async () => {}, sync: async () => {}, close: async () => {} }; } },
    "./runnerTargetInputs": { pickTargetSubnet: async () => plan.input.subnetCIDR, pickTargetDeadline: async () => plan.input.deadline },
    "./core/runnerAssessment": { ensureAssessmentReadiness: async (_c: unknown, current: RunnerRecord) => current },
    "./core/runnerTargetPreflight": { preflightTarget: async () => {}, targetComputeRate: () => plan.input.hourlyUSD },
    "./core/runnerTarget": { ...target, sourceTargetEvidence: () => plan.evidence, targetPreview: () => ({ ...plan, phase: "previewed" }),
      submitTarget: async (_c: unknown, current: RunnerRecord) => { submissions++; await f.control.persist({ ...current, target: { ...plan, phase: "submitted" } }); } }
  }, f.Clock);
  await panel.reviewRunnerTarget({ secrets: { get: async () => "inert-target-password-never-sent" } }, f.control, f.store, { retailRates: async () => [] }, r.id);
  assert.equal(submissions, 1); assert.equal(files, 2); assert.equal(f.stats().reads, 2);
  assert.equal(f.current().target?.phase, "provisioned");
});
