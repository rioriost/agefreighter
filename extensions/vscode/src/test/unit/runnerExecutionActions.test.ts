import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { Script } from "node:vm";
import { transformSync } from "esbuild";
import { RunnerRecord } from "../../core/runner";
import { RunnerControl } from "../../core/runnerLifecycle";
import * as actions from "../../core/runnerExecutionActions";
import * as execution from "../../core/runnerExecution";
import { GuestHealth } from "../../core/runnerGuest";
import { otherCancellationFixture, otherNativeCancelCases } from "../helpers/nativeCancelOtherScenarios";
import { boundedWatch } from "../../core/boundedWatch";
import type { ExecutionProgress } from "../../runnerExecutionPanel";

function fixture() {
  const f = otherCancellationFixture(otherNativeCancelCases.find(c => c.id === "A21")!);
  f.record.targetRestart = { phase: "finished", submittedAt: new Date().toISOString() };
  return f;
}

test("required steps guide a resized runner to readiness and never enable replay", () => {
  const { record: r } = fixture();
  delete r.guestReady;
  assert.equal(actions.executionActionState(r, "readiness").enabled, true);
  assert.equal(actions.executionActionState(r, "start").enabled, false);
  assert.match(actions.executionActionState(r, "start").detail, /5-3/);
  delete r.resize;
  assert.match(actions.executionActionState(r, "readiness").detail, /5-2/);
  assert.equal(actions.executionActionState(r, "resize").enabled, true);
  delete r.targetRestart;
  assert.match(actions.executionActionState(r, "resize").detail, /5-1/);
  assert.equal(actions.executionActionState(r, "preload").enabled, true);
});

test("freshness, unhealthy guests, capability, budget and pending commands block new migration", () => {
  const { record: r } = fixture(), now = Date.parse(r.guestReady!.checkedAt);
  assert.equal(actions.executionActionState(r, "start", now + 300000).enabled, true);
  assert.equal(actions.executionActionState(r, "start", now + 300001).enabled, false);
  for (const field of ["idle", "storageUsedPercent", "swapUsedBytes", "oomEvents"] as const) {
    const unhealthy = structuredClone(r);
    if (field === "idle") unhealthy.guestReady!.health!.idle = false;
    else unhealthy.guestReady!.health![field] = field === "storageUsedPercent" ? 80 : 1;
    assert.equal(actions.executionActionState(unhealthy, "start", now).enabled, false, field);
    assert.notEqual(actions.executionActionState(unhealthy, "readiness", now).complete, true, field);
  }
  const incapable = structuredClone(r); incapable.guestReady!.capabilities = [];
  assert.match(actions.executionActionState(incapable, "start", now).detail, /lacks migration support/);
  r.target!.input.deadline = new Date(now).toISOString();
  assert.match(actions.executionActionState(r, "start", now).detail, /5-5/);
  assert.equal(actions.executionActionState(r, "renew", now).enabled, true);
  r.guestCommand = { id: "retained", operation: r.id, action: "export-report", phase: "unknown", submittedAt: new Date(now).toISOString() };
  assert.equal(actions.executionActionState(r, "start", now).enabled, false);
  assert.match(actions.executionActionState(r, "readiness", now).detail, /5-8/);
  r.guestCommand.action = "ready";
  assert.equal(actions.executionActionState(r, "readiness", now).enabled, true);
  assert.match(actions.executionActionState(r, "readiness", now).detail, /same receipt/);
});

test("running, failed, imported and passing reports have distinct migration and verification states", () => {
  const { record: r } = fixture();
  r.migration = { operation: r.id, jobId: r.id, phase: "running", startedAt: new Date().toISOString(), bootId: r.id,
    artifactSHA256: r.artifact.sha256, cliVersion: r.artifact.version, evidence: r.target!.evidence };
  for (const phase of ["running", "failed", "interrupted", "finished"] as const) {
    r.migration.phase = phase;
    assert.equal(actions.executionActionState(r, "start").enabled, false);
    assert.equal(actions.executionActionState(r, "readiness").enabled, false);
    assert.equal(actions.executionActionState(r, "migrationRefresh").enabled, true);
    assert.equal(actions.executionActionState(r, "verify").enabled, false);
    assert.equal(actions.executionActionState(r, "recoveryReady").enabled, ["failed", "interrupted"].includes(phase));
  }
  r.migration.reportSHA256 = "a".repeat(64); r.migration.reportBytes = 1;
  assert.equal(actions.executionActionState(r, "verify").enabled, true);
  assert.equal(actions.executionActionState(r, "verify").complete, false);
  r.migration.verification = { outcome: "fail", summary: "Fixture counts differ" };
  assert.equal(actions.executionActionState(r, "verify").complete, false);
  assert.match(actions.executionActionState(r, "verify").detail, /Counts: fail/);
  r.migration.verification.outcome = "pass";
  assert.equal(actions.executionActionState(r, "verify").complete, true);
});

function readinessFixture() {
  const f = fixture();
  let current = f.record;
  const calls: string[] = [], capabilities = current.guestReady!.capabilities;
  let terminal = true, cancel = false, fail = false;
  let health: GuestHealth = { idle: true, storageUsedPercent: 12, swapUsedBytes: 0, oomEvents: 0 };
  delete current.guestReady;
  const control: RunnerControl = {
    list: async () => [],
    sleep: async () => {},
    persist: async r => { current = r; },
    request: async (_s, path, method = "GET", body) => {
      calls.push(method);
      if (method === "PUT") {
        const payload = body as { properties: { protectedParameters: { value: string }[] } };
        const request = JSON.parse(Buffer.from(payload.properties.protectedParameters[0]!.value, "base64").toString("utf8"));
        assert.equal(request.action, "ready");
        assert.equal(request.configuration, undefined); assert.equal(request.secrets, undefined);
        return { status: 200, value: {} };
      }
      if (!path.includes("$expand")) return { status: 404, value: {} };
      return { status: 200, value: { properties: { instanceView: {
        executionState: terminal ? fail ? "Failed" : "Succeeded" : "Running", exitCode: fail ? 1 : 0,
        output: JSON.stringify({ version: 1, ready: true, os: "linux", architecture: "amd64",
          bootId: "22222222-2222-4222-8222-222222222222", cliVersion: current.artifact.version, archiveSha256: current.artifact.sha256,
          commit: "synthetic", capabilities, health })
      } } } };
    }
  };
  return { control, calls, current: () => current, cancelled: () => cancel, cancel: () => { cancel = true; },
    fail: () => { fail = true; }, pending: (value: boolean) => { terminal = !value; }, health: (value: GuestHealth) => { health = value; } };
}

test("post-resize readiness establishes the new boot with one source-free check", async () => {
  const f = readinessFixture();
  const result = await execution.checkMigrationReadiness(f.control, f.current());
  assert.equal(result.guestReady?.bootId, "22222222-2222-4222-8222-222222222222");
  assert.equal(actions.executionActionState(result, "start").enabled, true);
  assert.equal(result.migration, undefined);
  assert.equal(f.calls.filter(method => method === "PUT").length, 1);
});

test("bounded readiness polls retain pending evidence and a second click reconciles without replay", async () => {
  const f = readinessFixture(); f.pending(true);
  const pending = await execution.checkMigrationReadiness(f.control, f.current());
  assert.equal(pending.guestReady, undefined); assert.equal(pending.guestCommand?.phase, "submitted");
  f.pending(false);
  const result = await execution.checkMigrationReadiness(f.control, f.current());
  assert.equal(result.guestCommand?.id, pending.guestCommand?.id);
  assert.equal(result.guestCommand?.phase, "finished");
  assert.equal(f.calls.filter(method => method === "PUT").length, 1);
});

test("readiness rejects unrelated pending commands, cancellation and failed health without starting migration", async () => {
  const other = readinessFixture();
  other.current().guestCommand = { id: "pending", operation: other.current().id, action: "inventory", phase: "unknown", submittedAt: new Date().toISOString() };
  await assert.rejects(execution.checkMigrationReadiness(other.control, other.current()), /5-8/);
  assert.deepEqual(other.calls, []);
  const cancelled = readinessFixture(); cancelled.cancel();
  await assert.rejects(execution.checkMigrationReadiness(cancelled.control, cancelled.current(), cancelled.cancelled), /stopped/);
  assert.deepEqual(cancelled.calls, []);
  const during = readinessFixture();
  const persist = during.control.persist;
  during.control.persist = async r => { await persist(r); during.cancel(); };
  await assert.rejects(execution.checkMigrationReadiness(during.control, during.current(), during.cancelled), /stopped/);
  assert.equal(during.calls.includes("PUT"), false);
  assert.equal(during.current().guestCommand?.phase, "submitted");
  for (const field of ["command", "disk", "swap", "oom", "worker"]) {
    const f = readinessFixture();
    if (field === "command") f.fail();
    else f.health({ idle: field !== "worker", storageUsedPercent: field === "disk" ? 80 : 12, swapUsedBytes: field === "swap" ? 1 : 0, oomEvents: field === "oom" ? 1 : 0 });
    await assert.rejects(execution.checkMigrationReadiness(f.control, f.current()), /readiness|Readiness/);
    assert.equal(f.current().migration, undefined);
  }
});

type Controller = (context: unknown, control: RunnerControl, store: unknown, azure: unknown, workflow: string, action: actions.ExecutionAction,
  update?: (record: RunnerRecord, progress?: ExecutionProgress) => Promise<void>, cancelled?: () => boolean) => Promise<void>;
function controllerFixture() {
  const f = fixture(), native = createRequire(__filename);
  let record = f.record, approval: string | undefined, preflights = 0, starts = 0, transfers = 0, verifies = 0, checks = 0, locked = false;
  const workspace = { isTrusted: true }, token = { isCancellationRequested: false };
  let clock = Date.now(), cancelled = false, lostAcknowledgement = false, serverReads = 0, confirmations = 0;
  let states = [{state: "Ready", pending: true}, {state: "Restarting", pending: true}, {state: "Ready", pending: false}];
  let duringSleep = () => {}, duringRequest = (_method: string, _path: string) => {}, duringLock = () => {};
  const requests: {method: string; path: string}[] = [], waits: number[] = [], notifications: string[] = [];
  const updates: {record: RunnerRecord; progress?: ExecutionProgress}[] = [];
  class Clock extends Date { static override now() { return clock; } }
  const control: RunnerControl = { request: async (_s, path, method = "GET") => {
      requests.push({method, path}); duringRequest(method, path);
      const server = record.target!.serverId;
      if (path === `${server}/restart?api-version=2024-08-01` && method === "POST") {
        if (lostAcknowledgement) throw Error("Restart acknowledgement lost");
        return {status: 202, value: {}};
      }
      assert.equal(method, "GET");
      if (path === `${server}?api-version=2024-08-01`) {
        const state = states[Math.min(serverReads++, states.length - 1)]!;
        return {status: 200, value: {tags: {workflow: record.id, application: "agefreighter", purpose: "migration-target"}, properties: {state: state.state}}};
      }
      if (path === `${server}/configurations/shared_preload_libraries?api-version=2024-08-01`)
        return {status: 200, value: {properties: {value: "pg_stat_statements,age", isConfigPendingRestart: states[Math.min(serverReads - 1, states.length - 1)]!.pending}}};
      throw Error("Unexpected live Azure call");
    },
    list: async () => { throw Error("Unexpected live Azure list"); },
    sleep: async ms => { assert.equal(locked, false); waits.push(ms); clock += ms; duringSleep(); },
    persist: async r => { record = structuredClone(r); } };
  const store = { read: async () => structuredClone(record), readReport: async () => "fixture", exclusive: async (_id: string, action: () => Promise<RunnerRecord>) => {
    duringLock();
    assert.equal(locked, false); locked = true; try { return await action(); } finally { locked = false; }
  } };
  const modules: Record<string, unknown> = {
    vscode: { workspace, ProgressLocation: { Notification: 1 }, window: {
      showQuickPick: async () => { throw Error("Direct step must not open an action picker"); },
      showWarningMessage: async () => { confirmations++; return approval; },
      withProgress: async (_o: unknown, run: (progress: {report: (value: {message: string}) => void}, token: { isCancellationRequested: boolean }) => Promise<unknown>) =>
        run({report: ({message}) => { notifications.push(message); }}, token),
      showInformationMessage: async () => {}
    } },
    "./core/runnerExecutionActions": actions,
    "./core/runnerExecution": {
      applyTargetPreload: execution.applyTargetPreload,
      migrationPreflight: async () => { preflights++; return record.target!.evidence; },
      checkMigrationReadiness: async () => { assert.equal(locked, true); checks++; return record; },
      startMigration: async () => { assert.equal(locked, true); starts++; return record; },
      verifyMigrationReport: (r: RunnerRecord) => { assert.equal(locked, true); verifies++; return r; }
    },
    "./core/runnerAssessment": { ensureAssessmentReadiness: async () => { assert.equal(locked, true); return record; } },
    "./sourceCredentialPanel": { sourceCredential: async () => "inert-source-password" },
    "./core/runnerTargetPreflight": { targetComputeRate: () => record.target!.input.hourlyUSD },
    "./runnerReportFlow": { transferApprovedReport: async () => {
      transfers++; record = { ...record, reportTransfers: [{ operation: record.migration!.operation,
        sha256: record.migration!.reportSHA256!, bytes: record.migration!.reportBytes!, blob: "inert", phase: "imported" }] }; return record;
    } },
    "./runnerWatch": { watchRetainedOperation: async () => {} }
  };
  for (const name of ["./core/runnerResize", "./core/runnerTarget", "./core/report", "./core/runnerDiagnostic", "./p1QualificationPanel", "./p1DiagnosticPanel", "./core/runnerSource", "./core/runnerResume", "./migrationVerificationPanel", "./core/resizeAuthorization", "./runnerTargetInputs", "./core/runnerGuest"]) modules[name] = {};
  modules["./core/boundedWatch"] = {boundedWatch: <T>(options: Parameters<typeof boundedWatch<T>>[0]) => boundedWatch({...options, now: () => clock})};
  const output = { exports: {} as { continueRunnerExecution: Controller } };
  const code = transformSync(readFileSync(join(__dirname, "../../runnerExecutionPanel.ts"), "utf8"), { loader: "ts", format: "cjs" }).code;
  new Script(code).runInNewContext({ module: output, exports: output.exports, Error, Date: Clock, Buffer,
    require: (name: string) => name in modules ? modules[name] : name.startsWith("node:") ? native(name) : (() => { throw Error("Unexpected dependency " + name); })() });
  return { record: () => record, workspace, token, requests, waits, notifications, updates, approve: () => { approval = "Approve this step"; },
    setStates: (values: typeof states) => { states = values; serverReads = 0; },
    loseAcknowledgement: () => { lostAcknowledgement = true; },
    duringSleep: (fn: () => void) => { duringSleep = fn; }, duringRequest: (fn: typeof duringRequest) => { duringRequest = fn; },
    duringLock: (fn: () => void) => { duringLock = fn; }, cancel: () => { cancelled = true; },
    advance: (ms: number) => { clock += ms; }, confirmations: () => confirmations,
    counts: () => ({ preflights, starts, transfers, verifies, checks }),
    run: (action: actions.ExecutionAction) => output.exports.continueRunnerExecution({ secrets: { get: async () => "inert-target-password" } }, control, store, { retailRates: async () => [] }, record.id, action,
      async (r, progress) => { updates.push({record: structuredClone(r), progress}); }, () => cancelled) };
}

function preloadFixture() {
  const f = controllerFixture();
  delete f.record().targetRestart; delete f.record().resize;
  return f;
}

test("preload shows restarting while the single approved POST is pending, then advances only after Ready and applied", async () => {
  const f = preloadFixture(); f.approve();
  f.duringRequest(method => {
    if (method === "POST") {
      assert.match(f.updates.at(-1)!.progress!.text, /Restarting PostgreSQL/);
      assert.equal(f.updates.at(-1)!.progress!.active, true);
    }
  });
  await f.run("preload");
  assert.equal(f.requests.filter(r => r.method === "POST").length, 1);
  assert.equal(f.requests.filter(r => r.method === "GET").length, 6);
  assert.deepEqual(f.waits, [15000]);
  assert.equal(f.record().targetRestart?.phase, "finished");
  const pending = f.updates.find(update => update.record.targetRestart?.phase === "submitted")!;
  assert.equal(actions.executionActionState(pending.record, "resize").enabled, false);
  assert.equal(actions.executionActionState(f.record(), "resize").enabled, true);
  assert.equal(actions.executionActionState(f.record(), "start").enabled, false);
  const complete = f.updates.filter(update => update.progress).at(-1)!;
  assert.equal(complete.progress!.active, false); assert.match(complete.progress!.text, /preload is applied.*PostgreSQL is Ready/);
  assert.equal(f.record().migration, undefined); assert.equal(f.record().resize, undefined);
  assert.equal(f.counts().starts, 0);
});

test("lost restart acknowledgement is shown as uncertain and never replayed", async () => {
  const f = preloadFixture(); f.approve(); f.loseAcknowledgement();
  await f.run("preload");
  assert.ok(f.updates.some(u => u.record.targetRestart?.phase === "unknown" && u.progress?.text.includes("uncertain")));
  assert.equal(f.requests.filter(r => r.method === "POST").length, 1);
  assert.equal(f.record().targetRestart?.phase, "finished");
});

for (const phase of ["submitted", "unknown"] as const) test(`retained ${phase} restart is monitored with GETs and no new approval`, async () => {
  const f = preloadFixture();
  f.record().targetRestart = {phase, submittedAt: new Date().toISOString()};
  await f.run("preload");
  assert.equal(f.confirmations(), 0);
  assert.ok(f.requests.every(r => r.method === "GET"));
  assert.equal(f.record().targetRestart?.phase, "finished");
});

test("no restart occurs when approval is cancelled or preload is already applied", async () => {
  const cancelled = preloadFixture();
  await cancelled.run("preload");
  assert.equal(cancelled.record().targetRestart, undefined);
  assert.ok(cancelled.requests.every(r => r.method === "GET"));
  assert.match(cancelled.notifications.at(-1)!, /No restart was submitted/);
  const applied = preloadFixture(); applied.approve(); applied.setStates([{state: "Ready", pending: false}]);
  await applied.run("preload");
  assert.equal(applied.record().targetRestart?.phase, "finished");
  assert.ok(applied.requests.every(r => r.method === "GET"));
});

test("preload monitoring is bounded, keeps pending state and can reconnect without another POST", async () => {
  const f = preloadFixture(); f.approve(); f.setStates([{state: "Ready", pending: true}]);
  await f.run("preload");
  assert.equal(f.requests.filter(r => r.method === "GET").length, 242);
  assert.equal(f.waits.length, 119);
  assert.equal(f.record().targetRestart?.phase, "submitted");
  assert.match(f.notifications.at(-1)!, /time limit.*Use 5-1/);
  assert.equal(f.updates.filter(u => u.progress).at(-1)!.progress!.active, false);
  f.setStates([{state: "Ready", pending: false}]);
  await f.run("preload");
  assert.equal(f.record().targetRestart?.phase, "finished");
  assert.equal(f.requests.filter(r => r.method === "POST").length, 1); assert.equal(f.confirmations(), 1);
});

test("neither Ready with pending preload nor a restarting server with applied preload marks completion", async () => {
  for (const state of [{state: "Ready", pending: true}, {state: "Restarting", pending: false}]) {
    const f = preloadFixture(); f.record().targetRestart = {phase: "submitted", submittedAt: new Date().toISOString()};
    f.setStates([state]); f.duringSleep(f.cancel);
    await f.run("preload");
    assert.equal(f.record().targetRestart?.phase, "submitted");
    assert.equal(actions.executionActionState(f.record(), "resize").enabled, false);
    assert.match(f.notifications.at(-1)!, /Monitoring stopped.*Use 5-1/);
    assert.ok(f.requests.every(r => r.method === "GET"));
  }
});

for (const boundary of ["before", "lock", "request", "post", "sleep", "trust"] as const) test(`preload monitor cancellation at ${boundary} preserves evidence without another restart`, async () => {
  const f = preloadFixture(); f.approve();
  if (boundary === "before") f.token.isCancellationRequested = true;
  if (boundary === "lock") f.duringLock(() => { f.token.isCancellationRequested = true; });
  if (boundary === "request") f.duringRequest(() => { f.token.isCancellationRequested = true; });
  if (boundary === "post") f.duringRequest(method => { if (method === "POST") f.token.isCancellationRequested = true; });
  if (boundary === "sleep") f.duringSleep(f.cancel);
  if (boundary === "trust") f.duringRequest(() => { f.workspace.isTrusted = false; });
  await f.run("preload");
  assert.equal(f.requests.filter(r => r.method === "POST").length, ["post", "sleep"].includes(boundary) ? 1 : 0);
  assert.equal(f.record().targetRestart?.phase, ["post", "sleep"].includes(boundary) ? "submitted" : undefined);
  assert.equal(f.updates.filter(u => u.progress).at(-1)!.progress!.active, false);
  assert.equal(f.counts().starts, 0);
});

for (const field of ["server", "plan", "receipt", "removed", "vm", "artifact"] as const) test(`preload monitoring rejects changed ${field} before the next request`, async () => {
  const f = preloadFixture(); f.approve();
  f.duringSleep(() => {
    if (field === "server") f.record().target!.serverId += "-other";
    if (field === "plan") f.record().target!.input.budgetUSD++;
    if (field === "receipt") f.record().targetRestart!.submittedAt = "changed";
    if (field === "removed") delete f.record().targetRestart;
    if (field === "vm") f.record().vmId += "-other";
    if (field === "artifact") f.record().artifact.sha256 = "f".repeat(64);
  });
  await assert.rejects(f.run("preload"), /scope or retained receipt changed/);
  assert.equal(f.requests.length, 5);
});

test("preload failures and invalidated readiness are surfaced without a new restart", async () => {
  for (const state of ["Failed", "Stopped", "Dropping", "Inaccessible"]) {
    const f = preloadFixture(); f.record().targetRestart = {phase: "submitted", submittedAt: new Date().toISOString()};
    f.setStates([{state, pending: true}]);
    await assert.rejects(f.run("preload"), /not restarting or Ready/);
    assert.equal(f.waits.length, 0); assert.ok(f.requests.every(r => r.method === "GET"));
  }
  const changed = preloadFixture();
  changed.record().targetRestart = {phase: "finished", submittedAt: new Date().toISOString()};
  changed.setStates([{state: "Restarting", pending: true}, {state: "Ready", pending: false}]);
  await changed.run("preload");
  assert.ok(changed.updates.some(u => u.record.targetRestart?.phase === "unknown"));
  assert.equal(changed.record().targetRestart?.phase, "finished");
  assert.ok(changed.requests.every(r => r.method === "GET"));
});

test("slow preload status requests respect the watch deadline and expired budgets remain blocked", async () => {
  const f = preloadFixture(); f.approve(); f.setStates([{state: "Ready", pending: true}]);
  let reads = 0;
  f.duringRequest(method => { if (method === "GET" && ++reads === 3) f.advance(30 * 60000); });
  await f.run("preload");
  assert.equal(reads, 4); assert.equal(f.waits.length, 0);
  assert.match(f.notifications.at(-1)!, /time limit/);
  const expired = preloadFixture(); expired.approve();
  expired.duringSleep(() => { expired.record().target!.input.deadline = "2020-01-01T00:00:00Z"; });
  await assert.rejects(expired.run("preload"), /scope or retained receipt changed/);
  assert.equal(expired.requests.filter(r => r.method === "POST").length, 1);
});

test("direct start preserves native approval, while cancellation and trust loss start nothing", async () => {
  const f = controllerFixture();
  await f.run("start"); assert.equal(f.counts().starts, 0); assert.equal(f.counts().preflights, 1);
  f.approve(); await f.run("start"); assert.equal(f.counts().starts, 1);
  f.workspace.isTrusted = false;
  await assert.rejects(f.run("start"), /Trust/); assert.equal(f.counts().starts, 1);
});

test("direct readiness does not fall back to the menu or migration, and stale direct starts are blocked", async () => {
  const f = controllerFixture(); delete f.record().guestReady;
  await f.run("readiness"); assert.equal(f.counts().checks, 1);
  await assert.rejects(f.run("start"), /5-3/);
  assert.equal(f.counts().preflights, 0); assert.equal(f.counts().starts, 0);
});

test("direct verification transfers and evaluates retained evidence without loading or readiness checks", async () => {
  const f = controllerFixture(), r = f.record();
  r.migration = { operation: r.id, jobId: r.id, phase: "finished", startedAt: new Date().toISOString(), bootId: r.id,
    artifactSHA256: r.artifact.sha256, cliVersion: r.artifact.version, evidence: r.target!.evidence, reportSHA256: "a".repeat(64), reportBytes: 1 };
  await f.run("verify"); assert.equal(f.counts().transfers, 0);
  f.approve(); await f.run("verify");
  assert.deepEqual(f.counts(), { preflights: 0, starts: 0, checks: 0, transfers: 1, verifies: 1 });
  await assert.rejects(f.run("start"), /already finished/);
});
