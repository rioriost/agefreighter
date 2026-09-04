import assert from "node:assert/strict";
import test from "node:test";
import { assertPersistableState, GuidedState } from "../../core/guided";

const state: GuidedState = {
  schemaVersion: 1,
  revision: 1,
  id: "safe-id",
  phase: "draft",
  createdAt: "2026-09-04T00:00:00Z",
  updatedAt: "2026-09-04T00:00:00Z",
  source: {
    type: "neo4j",
    host: "source.internal",
    port: 7687,
    database: "neo4j",
    sourceId: "source-1",
    placement: "on-premises",
    placementConfidence: "declared"
  }
};

test("allows non-secret workflow state", () => {
  assert.equal(assertPersistableState(state), state);
});

test("rejects credential-shaped workflow state", () => {
  const unsafe = { ...state, password: "secret" } as GuidedState;
  assert.throws(() => assertPersistableState(unsafe), /prohibited/);
});
