import assert from "node:assert/strict";
import test from "node:test";
import {
  buildNeo4jDraftYAML,
  combineCapacityAndInventory,
  extractCapacityEvidence,
  extractInventoryEvidence,
  normalizeNeo4jInput
} from "../../core/guided";

const input = {
  name: "supply-chain",
  host: "neo4j.internal",
  port: 7687,
  encrypted: true,
  database: "neo4j",
  sourceId: "source-1",
  namespace: "supply",
  username: "migration",
  vertexKeyProperty: "id",
  edgeKeyProperty: "id"
};

test("normalizes and validates guided Neo4j input", () => {
  assert.equal(normalizeNeo4jInput({ ...input, name: " Supply-Chain " }).name, "supply-chain");
  assert.throws(() => normalizeNeo4jInput({ ...input, host: "neo4j://bad" }), /Host/);
  assert.throws(() => normalizeNeo4jInput({ ...input, port: 70000 }), /Port/);
  assert.throws(() => normalizeNeo4jInput({ ...input, database: "bad\nvalue" }), /Database/);
});

test("builds a secret-reference-only Neo4j draft", () => {
  const yaml = buildNeo4jDraftYAML(input, "/private/source-password", "/private/target-dsn");
  assert.match(yaml, /uri: "neo4j\+s:\/\/neo4j\.internal:7687"/);
  assert.match(yaml, /file: "\/private\/source-password"/);
  assert.match(yaml, /file: "\/private\/target-dsn"/);
  assert.doesNotMatch(yaml, /password: [^\n]+/);
  assert.throws(() => buildNeo4jDraftYAML(input, "relative", "/target"), /absolute/);
});

test("keeps bounded profile estimates non-deployable", () => {
  const result = extractCapacityEvidence({
    sections: [{ title: "Capacity indicators", fields: [
      { name: "method", value: "sampled-lower-bound-range" },
      { name: "estimatedTargetRows", value: ">=10000" },
      { name: "recommendedStorageBytesRange", value: "100000..500000" }
    ] }]
  });
  assert.equal(result.targetRows, 10000n);
  assert.equal(result.targetRowsLowerBound, true);
  assert.equal(result.deployable, false);
});

test("accepts complete profile capacity evidence", () => {
  const result = extractCapacityEvidence({
    sections: [{ title: "Capacity indicators", fields: [
      { name: "method", value: "complete-stream-range" },
      { name: "estimatedTargetRows", value: "560000000" },
      { name: "recommendedStorageBytesRange", value: "1000000000..4000000000" }
    ] }]
  });
  assert.equal(result.recommendedStorageHigh, 4000000000n);
  assert.equal(result.deployable, true);
  assert.equal(result.reason, undefined);
});

test("scales a bounded profile only with an exact consistent inventory", () => {
  const inventory = extractInventoryEvidence({
    outcome: "pass",
    sections: [{ title: "Source inventory", fields: [
      { name: "countMethod", value: "neo4j-transactional-count-store" },
      { name: "vertices", value: "160000000" },
      { name: "edges", value: "400000000" },
      { name: "totalRows", value: "560000000" }
    ] }]
  });
  const result = combineCapacityAndInventory({
    method: "sampled-lower-bound-range",
    targetRows: 10000n,
    targetRowsLowerBound: true,
    recommendedStorageLow: 1000000n,
    recommendedStorageHigh: 4000000n,
    deployable: false
  }, inventory);
  assert.equal(result.targetRows, 560000000n);
  assert.equal(result.recommendedStorageHigh, 224000000000n);
  assert.equal(result.deployable, true);
  assert.equal(result.method, "exact-counts-scaled-bounded-profile");
});

test("rejects inconsistent inventory evidence", () => {
  assert.throws(() => extractInventoryEvidence({
    outcome: "pass",
    sections: [{ title: "Source inventory", fields: [
      { name: "countMethod", value: "neo4j-transactional-count-store" },
      { name: "vertices", value: "1" },
      { name: "edges", value: "2" },
      { name: "totalRows", value: "4" }
    ] }]
  }), /consistent exact totals/);
});
