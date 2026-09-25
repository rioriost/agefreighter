import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { Script } from "node:vm";
import { transformSync } from "esbuild";

const id = "11111111-1111-4111-8111-111111111111";
const code = transformSync(readFileSync(join(__dirname, "../../runnerLockRecoveryPanel.ts"), "utf8"), { loader: "ts", format: "cjs" }).code;
function fixture() {
  const events: string[] = [], messages: string[] = [];
  let empty = false, cancelPick = false, cancelConfirm = false, recoveryError = "", reviewError = "", afterPick = () => {}, afterConfirm = () => {};
  const workspace = { isTrusted: true };
  const review = { workflowId: id, owner: { pid: 12345, createdAt: "2026-09-23T00:00:00Z" }, lockSHA256: "a".repeat(64), recordSHA256: "b".repeat(64) };
  const store = {
    list: async () => { events.push("list"); return empty ? [] : [{ id, input: { resourceGroup: "trial" } }]; },
    reviewCrashLock: async (selected: string) => { events.push("review"); assert.equal(selected, id); if (reviewError) throw Error(reviewError); return review; },
    recoverCrashLock: async (actual: unknown, approved: boolean, guard: () => void) => {
      assert.strictEqual(actual, review); events.push(approved ? "approved" : "cancelled");
      if (!approved) return; guard(); if (recoveryError) throw Error(recoveryError); events.push("archive-and-recover");
    }
  };
  const modules: Record<string, unknown> = { vscode: { workspace, window: {
    showQuickPick: async (items: unknown[]) => { afterPick(); return cancelPick ? undefined : items[0]; },
    showWarningMessage: async (_message: string, options: { modal: boolean; detail: string }, choice: string) => {
      events.push("confirm"); assert.equal(options.modal, true); assert.match(options.detail, /remote operation may still be running/);
      assert.ok(options.detail.includes(id)); assert.ok(options.detail.includes(review.lockSHA256)); assert.ok(options.detail.includes(review.recordSHA256));
      afterConfirm(); return cancelConfirm ? undefined : choice;
    },
    showInformationMessage: async (m: string) => { messages.push(m); }
  } } };
  const output = { exports: { reviewRunnerCrashLock: async (_store: unknown) => {} } }, native = createRequire(__filename);
  new Script(code).runInNewContext({ module: output, exports: output.exports, Error, require: (name: string) => name in modules ? modules[name] : name.startsWith("node:") ? native(name) : {} });
  return { run: () => output.exports.reviewRunnerCrashLock(store), events, messages, workspace, empty: () => { empty = true; },
    cancelPick: () => { cancelPick = true; }, cancelConfirm: () => { cancelConfirm = true; },
    afterPick: (fn: () => void) => { afterPick = fn; }, afterConfirm: (fn: () => void) => { afterConfirm = fn; },
    failRecovery: (message: string) => { recoveryError = message; }, failReview: (message: string) => { reviewError = message; } };
}

test("native recovery uses explicit modal approval and no remote adapters", async () => {
  const f = fixture(); await f.run();
  assert.deepEqual(f.events, ["list", "review", "confirm", "approved", "archive-and-recover"]);
  assert.equal(f.messages.length, 1); assert.match(f.messages[0]!, /No remote action was sent/);
});
test("empty workflow store provides guidance without recovery", async () => {
  const f = fixture(); f.empty(); await f.run(); assert.deepEqual(f.events, ["list"]); assert.match(f.messages[0]!, /No retained/);
});
for (const point of ["selection", "confirmation"] as const) test(`cancel ${point} makes no local recovery write`, async () => {
  const f = fixture(); if (point === "selection") f.cancelPick(); else f.cancelConfirm(); await f.run();
  assert.deepEqual(f.events, point === "selection" ? ["list"] : ["list", "review", "confirm", "cancelled"]); assert.equal(f.messages.length, 0);
});
for (const point of ["initial", "selection", "confirmation"] as const) test(`lost trust at ${point} blocks local recovery`, async () => {
  const f = fixture(); if (point === "initial") f.workspace.isTrusted = false;
  if (point === "selection") f.afterPick(() => { f.workspace.isTrusted = false; });
  if (point === "confirmation") f.afterConfirm(() => { f.workspace.isTrusted = false; });
  await assert.rejects(f.run(), /Trust this workspace/); assert.ok(!f.events.includes("archive-and-recover")); assert.equal(f.messages.length, 0);
});
for (const reason of ["changed lock", "expired review", "live owner", "uncertain owner"] as const) test(`panel propagates ${reason} refusal without success`, async () => {
  const f = fixture(); if (reason.endsWith("owner")) f.failReview(reason); else f.failRecovery(reason);
  await assert.rejects(f.run(), new RegExp(reason)); assert.ok(!f.events.includes("archive-and-recover")); assert.equal(f.messages.length, 0);
});
