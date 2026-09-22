// Explicit integration test: AGEFREIGHTER_TEST_BINARY is required. The guided
// extension itself never launches this local binary.
import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { buildSourceDraft } from "../core/runnerSource";
import { sourceForm, workflow, csvFile } from "./sourceFixtures";
import { SourceKind, SourceLocation } from "../core/runner";
import { assertP1Projection } from "../core/p1Qualification";

test("other-cloud PostgreSQL frozen P1 mappings pass the actual Go validator and P1 projection gate", async () => {
  const binary = process.env.AGEFREIGHTER_TEST_BINARY;
  assert.ok(binary, "AGEFREIGHTER_TEST_BINARY must reference the locally built test CLI");
  const mappings = JSON.parse(await readFile("../../production-simulation/vscode-e2e/fixtures/postgresql-p1-mappings.json", "utf8"));
  const draft = buildSourceDraft({ type: "postgresql", location: "other-cloud" }, {
    ...sourceForm, name: "othercloud-pg-p1-r1", host: "192.0.2.20", port: 5432,
    database: "p1source", username: "agefreighter_reader", mappings
  }, workflow);
  assertP1Projection(draft.configuration);
  const directory = await mkdtemp(join(tmpdir(), "af-pg-other-cloud-contract-")), file = join(directory, "generated.json");
  await writeFile(file, JSON.stringify(draft.configuration), { mode: 0o600 });
  const result = spawnSync(binary, ["validate", file, "--format", "json"], { encoding: "utf8", timeout: 10000 });
  assert.equal(result.status, 0, result.stderr + result.stdout);
});

const paths: readonly { name: string; type: SourceKind; location: SourceLocation; cosmosFormat?: "gremlin"; gremlinPropertyTypes?: string }[] = [
  { name: "azure-neo4j", type: "neo4j", location: "azure" },
  { name: "on-premises-neo4j", type: "neo4j", location: "on-premises" },
  { name: "other-cloud-neo4j", type: "neo4j", location: "other-cloud" },
  { name: "azure-postgresql", type: "postgresql", location: "azure" },
  { name: "on-premises-postgresql", type: "postgresql", location: "on-premises" },
  { name: "other-cloud-postgresql", type: "postgresql", location: "other-cloud" },
  { name: "azure-cosmos-explicit", type: "cosmos-nosql", location: "azure" },
  { name: "azure-cosmos-gremlin", type: "cosmos-nosql", location: "azure", cosmosFormat: "gremlin" },
  { name: "azure-cosmos-gremlin-typed", type: "cosmos-nosql", location: "azure", cosmosFormat: "gremlin", gremlinPropertyTypes: "score=float64,distance_km=float64" },
  { name: "local-csv", type: "csv", location: "local" }
];

for (const path of paths) {
  test(`${path.name} field-generated configuration passes the actual Go validator`, async () => {
    const binary = process.env.AGEFREIGHTER_TEST_BINARY;
    assert.ok(binary, "AGEFREIGHTER_TEST_BINARY must reference the locally built test CLI");
    const type: SourceKind = path.type;
    const form = { ...sourceForm, ...(type === "cosmos-nosql" ? { host: "account.documents.azure.com" } : {}),
      ...(path.cosmosFormat ? { cosmosFormat: path.cosmosFormat } : {}),
      ...(path.gremlinPropertyTypes ? { gremlinPropertyTypes: path.gremlinPropertyTypes } : {}),
      ...(type === "csv" ? { mappings: sourceForm.mappings.map(m => ({ ...m, collection: csvFile.id, properties: m.kind === "vertex" ? "age=age:int64,tags=tags:string[]" : "" })) } : {}) };
    const draft = buildSourceDraft({ type, location: path.location }, form, workflow, [csvFile]);
    const directory = await mkdtemp(join(tmpdir(), "af-source-contract-")), file = join(directory, "generated.json");
    await writeFile(file, JSON.stringify(draft.configuration), { mode: 0o600 });
    const result = spawnSync(binary, ["validate", file, "--format", "json"], { encoding: "utf8", timeout: 10000 });
    assert.equal(result.status, 0, result.stderr + result.stdout);
  });
}

for (const nullValue of ["\\N", "", "NULL"]) {
  test(`CSV all eight explicit property types and null ${JSON.stringify(nullValue)} pass the actual Go validator`, async () => {
    const binary = process.env.AGEFREIGHTER_TEST_BINARY;
    assert.ok(binary, "AGEFREIGHTER_TEST_BINARY must reference the locally built test CLI");
    const edgeFile = { id: "33333333-3333-4333-8333-333333333333", name: "edges.csv" };
    const properties = "s=s:string,i=i:int64,f=f:float64,b=b:boolean,sa=sa:string[],ia=ia:int64[],fa=fa:float64[],ba=ba:boolean[]";
    const form = { ...sourceForm, nullValue, mappings: sourceForm.mappings.map(m => ({ ...m, collection: m.kind === "vertex" ? csvFile.id : edgeFile.id, properties })) };
    const draft = buildSourceDraft({ type: "csv", location: "local" }, form, workflow, [csvFile, edgeFile]);
    const directory = await mkdtemp(join(tmpdir(), "af-csv-choices-")), file = join(directory, "generated.json");
    await writeFile(file, JSON.stringify(draft.configuration), { mode: 0o600 });
    const result = spawnSync(binary, ["validate", file, "--format", "json"], { encoding: "utf8", timeout: 10000 });
    assert.equal(result.status, 0, result.stderr + result.stdout);
  });
}
