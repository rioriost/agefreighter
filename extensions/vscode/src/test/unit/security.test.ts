import assert from "node:assert/strict";
import test from "node:test";
import { redactForModel, redactText } from "../../core/security";

test("redacts sensitive keys recursively", () => {
  const result = redactForModel({
    job: "safe",
    source: {
      connection: "postgres://user:password@example/db",
      queryText: "MATCH (n) RETURN n",
      count: 3
    },
    token: "abc"
  });
  assert.deepEqual(result, {
    job: "safe",
    source: {
      connection: "[redacted]",
      queryText: "[redacted]",
      count: 3
    },
    token: "[redacted]"
  });
});

test("redacts URL credentials and bounds text", () => {
  assert.equal(
    redactText("failed postgres://alice:hunter2@example.test/db"),
    "failed postgres://[redacted]@example.test/db"
  );
  assert.match(redactText("x".repeat(20), 4), /^xxxx\n\[truncated\]$/);
});

test("replaces oversized model evidence", () => {
  const result = redactForModel({ safe: "x".repeat(100) }, {
    maxDepth: 2,
    maxArrayItems: 2,
    maxStringLength: 100,
    maxSerializedBytes: 10
  });
  assert.deepEqual(result, {
    truncated: true,
    reason: "Bounded AGEFreighter evidence exceeded the model context limit. Open the full local report in VS Code."
  });
});
