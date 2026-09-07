// Explicit integration test: AGEFREIGHTER_TEST_BINARY is required. The guided
// extension itself never launches this local binary.
import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { buildSourceDraft } from "../core/runnerSource";
import { sourceForm, workflow, csvFile } from "./sourceFixtures";
import { SourceKind, SourceLocation } from "../core/runner";

const paths: readonly { name: string; type: SourceKind; location: SourceLocation; cosmosFormat?: "gremlin" }[] = [
  { name: "azure-neo4j", type: "neo4j", location: "azure" },
  { name: "on-premises-neo4j", type: "neo4j", location: "on-premises" },
  { name: "other-cloud-neo4j", type: "neo4j", location: "other-cloud" },
  { name: "azure-postgresql", type: "postgresql", location: "azure" },
  { name: "on-premises-postgresql", type: "postgresql", location: "on-premises" },
  { name: "other-cloud-postgresql", type: "postgresql", location: "other-cloud" },
  { name: "azure-cosmos-explicit", type: "cosmos-nosql", location: "azure" },
  { name: "azure-cosmos-gremlin", type: "cosmos-nosql", location: "azure", cosmosFormat: "gremlin" },
  { name: "local-csv", type: "csv", location: "local" }
];

for (const path of paths) {
  test(`${path.name} field-generated configuration passes the actual Go validator`, async () => {
    const binary = process.env.AGEFREIGHTER_TEST_BINARY;
    assert.ok(binary, "AGEFREIGHTER_TEST_BINARY must reference the locally built test CLI");
    const type: SourceKind = path.type;
    const form = { ...sourceForm, ...(type === "cosmos-nosql" ? { host: "account.documents.azure.com" } : {}),
      ...(path.cosmosFormat ? { cosmosFormat: path.cosmosFormat } : {}),
      ...(type === "csv" ? { mappings: sourceForm.mappings.map(m => ({ ...m, collection: csvFile.id, properties: m.kind === "vertex" ? "age=age:int64,tags=tags:string[]" : "" })) } : {}) };
    const draft = buildSourceDraft({ type, location: path.location }, form, workflow, [csvFile]);
    const directory = await mkdtemp(join(tmpdir(), "af-source-contract-")), file = join(directory, "generated.json");
    await writeFile(file, JSON.stringify(draft.configuration), { mode: 0o600 });
    const result = spawnSync(binary, ["validate", file, "--format", "json"], { encoding: "utf8", timeout: 10000 });
    assert.equal(result.status, 0, result.stderr + result.stdout);
  });
}
