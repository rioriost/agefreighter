import assert from "node:assert/strict";
import test from "node:test";
import { migrationVerificationView } from "../../core/runnerVerificationView";
import { VerificationDecision } from "../../core/runnerVerification";

test("failed and incomplete decisions never produce a verified tab or PASS heading", () => {
  for (const outcome of ["fail", "incomplete"] as const) {
    const view = migrationVerificationView({outcome, summary: "Not complete"}, '{"outcome":"pass"}');
    assert.doesNotMatch(view.title, /verified/i);
    assert.match(view.html, /<h1>Verification: (FAILED|INCOMPLETE) — migration is not complete<\/h1>/);
    assert.doesNotMatch(view.html, /<h1>[^<]*PASS/);
  }
});
test("counts PASS is explicitly narrower than full qualification", () => {
  const view = migrationVerificationView({outcome: "pass", summary: "Counts agree"}, "{}");
  assert.equal(view.title, "AGEFreighter counts verified");
  assert.match(view.html, /Full property-digest qualification remains a separate check/);
});
test("untrusted report and summary are escaped and cannot change the verdict", () => {
  const payload = '<script>alert(1)</script><h1>PASS</h1>';
  const view = migrationVerificationView({outcome: "incomplete", summary: payload}, payload);
  assert.doesNotMatch(view.html, /<script>|<h1>PASS/);
  assert.match(view.html, /&lt;script&gt;/);
  assert.match(view.html, /default-src 'none'/);
});
test("an unexpected runtime decision defaults to incomplete", () => {
  const view = migrationVerificationView({outcome: "unknown", summary: ""} as unknown as VerificationDecision, "{}");
  assert.equal(view.title, "AGEFreighter verification incomplete");
});
