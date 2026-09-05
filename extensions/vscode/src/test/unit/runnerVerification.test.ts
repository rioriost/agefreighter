import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { assessCountsVerification, VerificationExpectation } from "../../core/runnerVerification";

const expected: VerificationExpectation = { jobId: "11111111-1111-4111-8111-111111111111", fingerprint: "a".repeat(64), cliVersion: "2.4.0", startedAt: "2026-09-05T00:00:00Z", labels: { "v.Person": 2, "e.KNOWS": 1 } };
const now = Date.parse("2026-09-05T00:05:00Z");
function document() {
  return { schemaVersion: 1, command: "verify", agefreighterVersion: "2.4.0", generatedAt: "2026-09-05T00:01:00Z", outcome: "pass", job: { id: expected.jobId, configFingerprint: expected.fingerprint },
    checks: ["job-status", "graph-generation", "generation-ownership"].map(id => ({ id, status: "pass", summary: "checked" })), errors: [] as unknown[], warnings: [], incompleteChecks: [] as string[],
    sections: [{ title: "Per-label counts", fields: [...Object.entries(expected.labels).map(([name, rows]) => ({ name, status: "pass", value: `counterCompleteness=complete,counterProvenance=lifecycle,identityCoverage=full,acceptedRows=${rows},committedRows=${rows},livePhysicalRows=${rows},liveIdentityRows=${rows},storedPhysicalComparison=verified,physicalIdentityEquality=verified,committedBytes=100,rejectedRows=0` })), { name: "unclassified.rejects", status: "pass", value: "0" }] }] };
}
function assess(doc: unknown, exitCode = 0) {
  const reportJSON = JSON.stringify(doc);
  return assessCountsVerification(expected, { exitCode, reportJSON, sha256: createHash("sha256").update(reportJSON).digest("hex") }, now);
}
test("only matching complete verification with exact source/target counts and zero rejects passes", () => {
  assert.equal(assess(document()).outcome, "pass");
  assert.equal(assess(document(), 1).outcome, "incomplete");
});
test("incomplete CLI exit zero and unknown checks never mean completion", () => {
  const doc = document(); doc.outcome = "incomplete"; assert.equal(assess(doc).outcome, "incomplete");
  doc.outcome = "pass"; doc.incompleteChecks.push("bounded-integrity"); assert.equal(assess(doc).outcome, "incomplete");
  doc.incompleteChecks = []; doc.checks[0]!.status = "unknown"; assert.equal(assess(doc).outcome, "incomplete");
});
test("wrong job, fingerprint, version, time, command and schema fail closed", () => {
  for (const change of [
    { job: { id: "wrong", configFingerprint: expected.fingerprint } }, { job: { id: expected.jobId, configFingerprint: "b".repeat(64) } },
    { agefreighterVersion: "2.3.0" }, { generatedAt: "2026-09-04T23:59:00Z" }, { generatedAt: "2026-09-06T00:00:00Z" }, { command: "report" }, { schemaVersion: 2 }, { sections: [] }, { checks: [] }
  ]) assert.equal(assess({ ...document(), ...change }).outcome, "incomplete");
});
test("missing or duplicate labels, unknown coverage and tampered evidence fail closed", () => {
  const doc = document(); doc.sections[0]!.fields.push(doc.sections[0]!.fields[0]!); assert.equal(assess(doc).outcome, "incomplete");
  const valid = JSON.stringify(document()); assert.equal(assessCountsVerification(expected, { reportJSON: valid, exitCode: 0, sha256: "0".repeat(64) }, now).outcome, "incomplete");
  assert.equal(assessCountsVerification({ ...expected, labels: {} }, { reportJSON: valid, exitCode: 0, sha256: "0".repeat(64) }, now).outcome, "incomplete");
});
test("source count mismatch, property check failure and nonzero rejects cannot pass", () => {
  const mismatch = document(); mismatch.sections[0]!.fields[0]!.value = mismatch.sections[0]!.fields[0]!.value.replace("livePhysicalRows=2", "livePhysicalRows=1"); assert.equal(assess(mismatch).outcome, "fail");
  const rejects = document(); rejects.sections[0]!.fields[2]!.value = "1"; assert.equal(assess(rejects).outcome, "fail");
  const failed = document(); failed.checks[0]!.status = "fail"; assert.equal(assess(failed).outcome, "fail");
});
