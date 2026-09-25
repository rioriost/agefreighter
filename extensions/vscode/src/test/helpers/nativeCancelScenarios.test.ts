import assert from "node:assert/strict";
import test from "node:test";
import { cancellationFixture, nativeCancelCases } from "./nativeCancelScenarios";
import { RunnerControl } from "../../core/runnerLifecycle";
import { renewTargetAuthorization, targetBudget } from "../../core/runnerTarget";
import { applyTargetPreload } from "../../core/runnerExecution";
import { previewReceiptRemoval } from "../../core/runnerReceiptRemoval";

// Fixture-contract checks only. These do not invoke VS Code, fabricate a native
// Cancel result, or claim any native/signed-in qualification.
function inertControl(responses: Map<string, unknown>) {
  const requests: string[] = [];
  const control: RunnerControl = {
    persist: async () => { assert.fail("Fixture admission must not persist"); },
    sleep: async () => { assert.fail("Unexpected wait"); },
    list: async () => { assert.fail("Unexpected list"); },
    request: async (_subscription, path, method = "GET") => {
      assert.equal(method, "GET"); requests.push(path);
      const resource = path.split("?")[0]!; assert.ok(responses.has(resource));
      return { status: 200, value: structuredClone(responses.get(resource)) };
    }
  };
  return { control, requests };
}

test("native cancellation fixture subset stays within frozen uncredited IDs", () => {
  assert.deepEqual(nativeCancelCases.map(scenario => scenario.id), ["A18", "A19", "A29"]);
  assert.equal(new Set(nativeCancelCases.map(scenario => scenario.title)).size, 3);
});

test("A18 cost fixture passes production review validation without changing its baseline", () => {
  const fixture = cancellationFixture(nativeCancelCases[0]), before = JSON.stringify(fixture.record);
  targetBudget(fixture.record.target!.input);
  const reviewed = renewTargetAuthorization(fixture.record, { deadline: fixture.inputValues[0]!, budgetUSD: Number(fixture.inputValues[1]), additionalReserveUSD: Number(fixture.inputValues[2]), hourlyUSD: 2 });
  assert.equal(reviewed.costAuthorizations?.length, 1);
  assert.equal(JSON.stringify(fixture.record), before);
});

test("A19 synthetic target still needs restart and read-only reconciliation cannot claim it applied", async () => {
  const fixture = cancellationFixture(nativeCancelCases[1]), io = inertControl(fixture.responses);
  const original = JSON.stringify(fixture.record);
  const result = await applyTargetPreload(io.control, fixture.record, false);
  assert.strictEqual(result, fixture.record); assert.equal(result.targetRestart, undefined);
  assert.equal(io.requests.length, 2); assert.equal(JSON.stringify(fixture.record), original);
});

test("A29 two synthetic same-boot controls reach real production removal preview using three GETs", async () => {
  const fixture = cancellationFixture(nativeCancelCases[2]), io = inertControl(fixture.responses), original = JSON.stringify(fixture.record);
  const plan = await previewReceiptRemoval({ control: io.control, guard: async () => {},
    retain: async () => { assert.fail("No preview archive"); }, read: async () => { throw Error("No preview archive read"); },
    remove: async () => { assert.fail("No preview deletion"); }
  }, fixture.record, fixture.record.readinessReceipts![0]!.command.id, "b".repeat(64));
  assert.equal(plan.workflow, fixture.record.id); assert.equal(io.requests.length, 3);
  assert.equal(JSON.stringify(fixture.record), original);
  assert.equal(JSON.parse(plan.text).kind, "agefreighter-readiness-removal-v2");
});
