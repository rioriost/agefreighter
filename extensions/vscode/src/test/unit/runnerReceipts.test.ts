import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { sourceWorkflowDraft } from "../../core/runner";
import { retainReadinessReceipt, readinessArchive, readinessReceiptReferenced } from "../../core/runnerReceipts";
import { RunnerStore } from "../../guided/runnerStore";
import { reconcileGuest } from "../../core/runnerGuest";
import { RunnerControl } from "../../core/runnerLifecycle";

const id = "11111111-1111-4111-8111-111111111111", op = "22222222-2222-4222-8222-222222222222";
function fixture() {
  const record = sourceWorkflowDraft(id, { subscriptionId: id, resourceGroup: "trial", region: "japaneast", zone: "1", size: "Standard_B2s_v2", subnetId: "unused", source: { type: "csv", location: "local" } });
  record.phase = "provisioned";
  record.artifact = { version: "2.4.0", sha256: "a".repeat(64), url: "https://example.invalid/archive" };
  record.guestCommand = { id: `${record.vmId}/runCommands/af-${op}`, operation: id, action: "ready", phase: "finished", submittedAt: "2026-09-18T05:00:00Z" };
  record.guestReady = { bootId: op, cliVersion: "2.4.0", archiveSha256: record.artifact.sha256, commit: "abcdef", checkedAt: record.guestCommand.submittedAt,
    capabilities: ["csv-migration-v1"], health: { idle: true, storageUsedPercent: 12, swapUsedBytes: 0, oomEvents: 0 } };
  return record;
}

test("successful readiness is immutable, idempotent and retained across later controls/upgrades", () => {
  const record = retainReadinessReceipt(fixture());
  assert.equal(record.readinessReceipts?.length, 1);
  assert.strictEqual(retainReadinessReceipt(record), record);
  const receipt = record.readinessReceipts![0]!;
  record.guestReady!.bootId = id;
  assert.throws(() => retainReadinessReceipt(record), /evidence changed/);
  delete record.guestReady; delete record.guestCommand;
  record.artifact.sha256 = "b".repeat(64);
  assert.equal(readinessArchive(record, receipt).manifest.operation, op);
});

test("archive is schema-projected and excludes arbitrary source/configuration and ARM secrets", () => {
  const record = fixture();
  (record.guestReady as any).password = "never-copy-this";
  (record.guestReady!.health as any).password = "never-copy-this";
  record.guestCommand!.failure = "never-copy-this";
  const retained = retainReadinessReceipt(record);
  const archive = readinessArchive(retained, retained.readinessReceipts![0]!);
  assert.ok(!archive.text.includes("never-copy-this"));
  assert.equal(archive.manifest.bytes, Buffer.byteLength(archive.text));
});

for (const change of ["action", "failed", "unknown", "foreign", "missing", "time", "health", "hash", "version"] as const) {
  test(`invalid readiness is never sealed: ${change}`, () => {
    const record = fixture();
    if (change === "action") record.guestCommand!.action = "inventory";
    if (change === "failed" || change === "unknown") record.guestCommand!.phase = change;
    if (change === "foreign") record.guestCommand!.id = `/another-vm/runCommands/af-${op}`;
    if (change === "missing") delete record.guestReady;
    if (change === "time") record.guestReady!.checkedAt = "2020-01-01";
    if (change === "health") record.guestReady!.health!.storageUsedPercent = NaN;
    if (change === "hash") record.guestReady!.archiveSha256 = "b".repeat(64);
    if (change === "version") record.guestReady!.cliVersion += "\nsecret";
    assert.throws(() => retainReadinessReceipt(record));
  });
}

test("archive refuses altered, foreign and unretained receipts", () => {
  const record = retainReadinessReceipt(fixture()), receipt = record.readinessReceipts![0]!;
  for (const change of ["workflow", "vmId", "sha256", "extra", "output"]) {
    const altered = structuredClone(receipt);
    if (change === "output") altered.readiness.commit = "changed";
    else (altered as any)[change] = "changed";
    assert.throws(() => readinessArchive(record, altered));
  }
  assert.throws(() => readinessArchive({ ...record, readinessReceipts: [] }, receipt));
});

test("current and historical operation references remain protected", () => {
  const record = fixture(); record.guestCommand!.operation = op;
  const retained = retainReadinessReceipt(record), receipt = retained.readinessReceipts![0]!;
  assert.equal(readinessReceiptReferenced(retained, receipt), true);
  delete retained.guestCommand; delete retained.guestReady;
  assert.equal(readinessReceiptReferenced(retained, receipt), false);
  (retained as any).assessmentHistory = [{ operation: op }];
  assert.equal(readinessReceiptReferenced(retained, receipt), true);
});

test("production reconcile captures only successful ready controls, including lost-response recovery", async () => {
  for (const state of ["Succeeded", "Failed", "Pending"]) {
    const record = fixture(), methods: string[] = [], writes: unknown[] = [];
    const ready = record.guestReady!;
    record.guestCommand!.phase = "unknown"; delete record.guestReady;
    const control: RunnerControl = { sleep: async () => {}, list: async () => [], persist: async r => { writes.push(r); }, request: async (_s, _p, method = "GET") => {
      methods.push(method);
      return { status: 200, value: { properties: { instanceView: { executionState: state, exitCode: 0, output: JSON.stringify({ version: 1, ready: true, os: "linux", architecture: "amd64", ...ready }) } } } };
    } };
    const result = await reconcileGuest(control, record);
    assert.equal(result.record.readinessReceipts?.length ?? 0, state === "Succeeded" ? 1 : 0);
    assert.deepEqual(methods, ["GET"]);
    assert.equal(writes.length, state === "Pending" ? 0 : 1);
  }
});

test("durable archive is create-only, hash-verified, and survives state replacement", async () => {
  const root = await mkdtemp(join(tmpdir(), "af-readiness-"));
  try {
    const store = new RunnerStore(root), record = retainReadinessReceipt(fixture());
    const archive = readinessArchive(record, record.readinessReceipts![0]!);
    await store.write(record);
    await store.retainReport(record.id, archive.manifest, archive.text);
    await store.syncEvidenceDirectory();
    await store.retainReport(record.id, archive.manifest, archive.text);
    await store.write({ ...record, guestCommand: undefined, guestReady: undefined });
    assert.equal(await store.readReport(record.id, archive.manifest), archive.text);
    assert.equal(await readFile(join(root, `${id}.report-${op}.json`), "utf8"), archive.text);
    await assert.rejects(store.retainReport(record.id, archive.manifest, archive.text + " "));
    assert.equal(await store.readReport(record.id, archive.manifest), archive.text);
  } finally { await rm(root, { recursive: true, force: true }); }
});
