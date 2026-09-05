import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { csvFilesInFolder } from "../../guided/csvSelection";

test("CSV folder selection is shallow, CSV-only and bounded", async () => {
  const dir = await mkdtemp(join(tmpdir(), "agefreighter-csv-selection-"));
  try {
    await assert.rejects(csvFilesInFolder(dir), /No regular CSV/);
    await writeFile(join(dir, "nodes.csv"), "id\n1\n");
    await writeFile(join(dir, "not-a-csv.json"), "{}");
    await mkdir(join(dir, "nested.csv"));
    await writeFile(join(dir, "nested.csv", "edges.csv"), "id\n2\n");
    assert.deepEqual(await csvFilesInFolder(dir), [join(dir, "nodes.csv")]);
    for (let i=0;i<64;i++) await writeFile(join(dir, `${i}.csv`), "id\n");
    await assert.rejects(csvFilesInFolder(dir), /at most 64/);
  } finally { await rm(dir, { recursive: true, force: true }); }
});
