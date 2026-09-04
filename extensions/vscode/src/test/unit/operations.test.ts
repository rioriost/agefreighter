import assert from "node:assert/strict";
import test from "node:test";
import {
  buildReadArguments,
  buildTerminalArguments,
  isConnectedReadOperation,
  validateJobId
} from "../../core/operations";

const job = "/workspace/jobs/migration file.yaml";
const id = "11111111-2222-4333-8444-555555555555";

test("read commands preserve paths as individual arguments", () => {
  assert.deepEqual(buildReadArguments("validate", job), [
    "validate", "--format", "json", job
  ]);
  assert.deepEqual(buildReadArguments("report", job, id), [
    "report", "--format", "json", "--target", job, id
  ]);
});

test("terminal commands preserve paths as individual arguments", () => {
  assert.deepEqual(buildTerminalArguments("load", job), ["load", job]);
  assert.deepEqual(buildTerminalArguments("resume", job, id), [
    "resume", "--job", job, id
  ]);
});

test("durable job IDs are strict UUIDs", () => {
  assert.equal(validateJobId(id), id);
  assert.throws(() => validateJobId("$(touch unsafe)"), /valid durable/);
  assert.throws(() => validateJobId(undefined), /valid durable/);
});

test("static validation and planning do not connect", () => {
  assert.equal(isConnectedReadOperation("validate"), false);
  assert.equal(isConnectedReadOperation("plan"), false);
  assert.equal(isConnectedReadOperation("doctor"), true);
});
