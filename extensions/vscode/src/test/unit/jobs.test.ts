import assert from "node:assert/strict";
import test from "node:test";
import { looksLikeAgefreighterJob } from "../../core/jobs";

const bytes = (value: string): Uint8Array => new TextEncoder().encode(value);

test("recognizes YAML and JSON LoadJobs", () => {
  assert.equal(looksLikeAgefreighterJob(bytes(`
apiVersion: agefreighter.io/v2
kind: LoadJob
metadata:
  name: example
`)), true);
  assert.equal(looksLikeAgefreighterJob(bytes(JSON.stringify({
    apiVersion: "agefreighter.io/v2",
    kind: "LoadJob"
  }))), true);
});

test("rejects unrelated, malformed, empty, and oversized files", () => {
  assert.equal(looksLikeAgefreighterJob(bytes("kind: Deployment")), false);
  assert.equal(looksLikeAgefreighterJob(bytes("{")), false);
  assert.equal(looksLikeAgefreighterJob(new Uint8Array()), false);
  assert.equal(looksLikeAgefreighterJob(new Uint8Array(1024 * 1024 + 1)), false);
});
