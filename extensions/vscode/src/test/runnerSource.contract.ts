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
import { SourceKind } from "../core/runner";

for (const kind of ["neo4j", "postgresql", "cosmos-nosql", "cosmos-gremlin", "csv"] as const) {
  test(`${kind} GUI-generated configuration passes the actual Go validator`, async () => {
    const binary = process.env.AGEFREIGHTER_TEST_BINARY;
    assert.ok(binary, "AGEFREIGHTER_TEST_BINARY must reference the locally built test CLI");
    const type: SourceKind = kind === "cosmos-gremlin" ? "cosmos-nosql" : kind;
    const form = { ...sourceForm, ...(type === "cosmos-nosql" ? { host: "account.documents.azure.com" } : {}),
      ...(kind === "cosmos-gremlin" ? { cosmosFormat: "gremlin" } : {}),
      ...(kind === "csv" ? { mappings: sourceForm.mappings.map(m => ({ ...m, collection: csvFile.id, properties: m.kind === "vertex" ? "age=age:int64,tags=tags:string[]" : "" })) } : {}) };
    const draft = buildSourceDraft({ type, location: type === "csv" ? "local" : "azure" }, form, workflow, [csvFile]);
    const directory = await mkdtemp(join(tmpdir(), "af-source-contract-")), file = join(directory, "generated.json");
    await writeFile(file, JSON.stringify(draft.configuration), { mode: 0o600 });
    const result = spawnSync(binary, ["validate", file, "--format", "json"], { encoding: "utf8", timeout: 10000 });
    assert.equal(result.status, 0, result.stderr + result.stdout);
  });
}
