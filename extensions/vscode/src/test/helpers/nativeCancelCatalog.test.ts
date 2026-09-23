import assert from "node:assert/strict";
import test from "node:test";
import { buildSourceDraft } from "../../core/runnerSource";
import { assertPreparationWrite, integratedCancellationFixture, integratedNativeCancelCases, selectNativeCancelCases } from "./nativeCancelCatalog";

test("integrated choices are exactly20 frozen uncredited decisions; explicit17 filter excludes proven3", () => {
  assert.deepEqual(integratedNativeCancelCases.map(s => s.id), ["A01","A02","A03","A04","A05","A06","A07","A08","A10","A12","A13","A14","A15","A17","A18","A19","A21","A22","A28","A29"]);
  const remaining = integratedNativeCancelCases.filter(s => !["A18", "A19", "A29"].includes(s.id)).map(s => s.id);
  assert.equal(selectNativeCancelCases(remaining.join(",")).length, 17);
  assert.equal(selectNativeCancelCases().length, 20);
  for (const invalid of ["A18,A18", "A09", "A01,", "unknown", " A01"]) assert.throws(() => selectNativeCancelCases(invalid));
});

test("all integrated fixture entries retain explicit warning and count-bounded preparation", () => {
  for (const scenario of integratedNativeCancelCases) {
    const f = integratedCancellationFixture(scenario);
    assert.ok(f.title.length > 10); assert.ok(f.record.id); assert.equal(f.preparationWrites, ["A14","A15","A17"].includes(scenario.id) ? 1 : 0);
    assert.equal(f.requiresDevelopmentOptIn, ["A02","A03","A28"].includes(scenario.id));
    assert.ok(f.files.every(file => !file.text.includes("#!/")));
  }
});

test("pre-modal write allowlist rejects all undeclared changes, duplicates and any post-modal write", () => {
  for (const scenario of integratedNativeCancelCases) {
    const f = integratedCancellationFixture(scenario), original = structuredClone(f.record);
    const expected = structuredClone(original);
    if (scenario.id === "A14" || scenario.id === "A15") expected.sourceDraft = buildSourceDraft(original.input.source, f.prepareMessages[0]!.form, original.id, original.sourceFiles, original.sourceCA);
    const check = (record = expected, count = 0, opened = false) => assertPreparationWrite(scenario, f, original, record, count, opened);
    if (f.preparationWrites) {
      assert.doesNotThrow(() => check());
      for (const mutate of [(r: typeof expected) => { r.updatedAt = "changed"; }, (r: typeof expected) => { r.phase = "unknown"; },
        (r: typeof expected) => { r.artifact.sha256 = "x".repeat(64); }]) {
        const changed = structuredClone(expected); mutate(changed); assert.throws(() => check(changed));
      }
    } else assert.throws(() => check());
    assert.throws(() => check(expected, 1)); assert.throws(() => check(expected, 0, true));
  }
});
