import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, readFile, readdir, rename, rm, stat, symlink, writeFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { sourceWorkflowDraft } from "../../core/runner";
import { RunnerLockedError, RunnerStore } from "../../guided/runnerStore";
import { RunnerLockEnvironment, RunnerLockOwner, runnerLockEnvironment } from "../../guided/runnerLock";

const id = "11111111-1111-4111-8111-111111111111";
const token = "22222222-2222-4222-8222-222222222222";
const boot = `darwin:${token}`;
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "af-lock-review-"));
  let now = Date.now(), status: "alive" | "absent" | "uncertain" = "absent", identity: string | undefined = boot;
  let onBoot: () => Promise<void> = async () => {};
  const environment: RunnerLockEnvironment = { now: () => now, ownerStatus: () => status, bootIdentity: async () => { await onBoot(); return identity; } };
  const store = new RunnerStore(root, environment);
  const record = sourceWorkflowDraft(id, { subscriptionId: id, resourceGroup: "test", region: "japaneast", zone: "1", size: "Standard_B2s_v2", subnetId: "unused", source: { type: "csv", location: "local" } });
  const owner: RunnerLockOwner = { version: 1, workflowId: id, token, pid: 12345, bootIdentity: boot, createdAt: new Date(now).toISOString() };
  const lock = join(root, `${id}.lock`), recordPath = join(root, `${id}.json`);
  await store.write(record); await writeFile(lock, JSON.stringify(owner), { mode: 0o600 });
  return { root, store, owner, lock, record, recordPath, environment, cleanup: () => rm(root, { recursive: true, force: true }),
    time: (value: number) => { now = value; }, status: (value: typeof status) => { status = value; },
    identity: (value: string | undefined) => { identity = value; }, onBoot: (fn: () => Promise<void>) => { onBoot = fn; } };
}

test("explicit recovery archives reviewed metadata privately and preserves workflow/reports without replay", async () => {
  const f = await fixture();
  try {
    const report = join(f.root, `${id}.report-${token}.json`);
    await writeFile(report, "sealed-report", { mode: 0o600 });
    const before = await readFile(f.recordPath), originalLock = await readFile(f.lock, "utf8");
    const review = await f.store.reviewCrashLock(id);
    assert.equal((await readdir(f.root)).length, 3, "Review is read-only");
    await assert.rejects(f.store.exclusive(id, async () => assert.fail("No implicit recovery")), RunnerLockedError);
    await f.store.recoverCrashLock(review, true);
    await assert.rejects(stat(f.lock), { code: "ENOENT" });
    assert.deepEqual(await readFile(f.recordPath), before); assert.equal(await readFile(report, "utf8"), "sealed-report");
    const archives = (await readdir(f.root)).filter(name => name.includes(".recovered-lock-"));
    assert.equal(archives.length, 1);
    const archivePath = join(f.root, archives[0]!); const archive = JSON.parse(await readFile(archivePath, "utf8"));
    assert.equal(archive.originalLock, originalLock); assert.deepEqual(archive.review, review);
    assert.equal(archive.scope, "local-lock-only-no-remote-action");
    if (process.platform !== "win32") assert.equal((await stat(archivePath)).mode & 0o777, 0o600);
    assert.deepEqual(await f.store.list(), [f.record], "Recovery archive is not treated as a workflow");
    await assert.rejects(f.store.recoverCrashLock(review, true), RunnerLockedError);
    let entered = false;
    await f.store.exclusive(id, async () => { entered = true; const owner = JSON.parse(await readFile(f.lock, "utf8")); assert.equal(owner.pid, process.pid); assert.equal(owner.bootIdentity, boot); });
    assert.equal(entered, true);
  } finally { await f.cleanup(); }
});

test("cancellation is inert and invalidates the one-use review", async () => {
  const f = await fixture(); try {
    const before = await readdir(f.root), bytes = await readFile(f.lock); const review = await f.store.reviewCrashLock(id);
    await f.store.recoverCrashLock(review, false);
    assert.deepEqual(await readdir(f.root), before); assert.deepEqual(await readFile(f.lock), bytes);
    await assert.rejects(f.store.recoverCrashLock(review, true), RunnerLockedError);
  } finally { await f.cleanup(); }
});

for (const kind of ["live", "uncertain", "foreign-boot", "unknown-boot", "legacy", "malformed", "null", "version", "workflow", "pid", "token", "oversize", "symlink"] as const) {
  test(`review refuses ${kind} lock and preserves it`, async () => {
    const f = await fixture(); try {
      if (kind === "live") f.status("alive");
      if (kind === "uncertain") f.status("uncertain");
      if (kind === "foreign-boot") f.identity("another-boot");
      if (kind === "unknown-boot") f.identity(undefined);
      if (kind === "legacy") await writeFile(f.lock, "");
      if (kind === "malformed") await writeFile(f.lock, "not-json");
      if (kind === "null") await writeFile(f.lock, "null");
      if (kind === "version") await writeFile(f.lock, JSON.stringify({ ...f.owner, version: 2 }));
      if (kind === "workflow") await writeFile(f.lock, JSON.stringify({ ...f.owner, workflowId: token }));
      if (kind === "pid") await writeFile(f.lock, JSON.stringify({ ...f.owner, pid: -1 }));
      if (kind === "token") await writeFile(f.lock, JSON.stringify({ ...f.owner, token: {} }));
      if (kind === "oversize") await writeFile(f.lock, "a".repeat(4097));
      if (kind === "symlink") { await rename(f.lock, f.lock + ".original"); await symlink(f.lock + ".original", f.lock); }
      const before = await readFile(f.lock);
      await assert.rejects(f.store.reviewCrashLock(id), RunnerLockedError);
      assert.deepEqual(await readFile(f.lock), before);
    } finally { await f.cleanup(); }
  });
}

for (const kind of ["expired", "backward-clock", "tampered-review", "record", "same-inode-lock", "replacement-lock", "live-after-review", "unknown-after-review", "gate", "archive-failure", "durability-failure", "guard"] as const) {
  test(`approved recovery refuses ${kind} changes and keeps the lock`, async () => {
    const f = await fixture(); try {
      const review = await f.store.reviewCrashLock(id);
      if (kind === "expired") f.time(review.expiresAt + 1);
      if (kind === "backward-clock") f.time(review.reviewedAt - 1);
      if (kind === "tampered-review") review.owner.pid++;
      if (kind === "record") await f.store.write({ ...f.record, phase: "provisioned" });
      if (kind === "same-inode-lock") await writeFile(f.lock, JSON.stringify({ ...f.owner, token: id }));
      if (kind === "replacement-lock") { await rename(f.lock, f.lock + ".original"); await writeFile(f.lock, JSON.stringify(f.owner)); }
      if (kind === "live-after-review") f.status("alive");
      if (kind === "unknown-after-review") f.status("uncertain");
      if (kind === "gate") await writeFile(join(f.root, `${id}.lock-gate`), "");
      if (kind === "archive-failure") await writeFile(join(f.root, `${id}.recovered-lock-${review.reviewToken}.json`), "existing evidence");
      if (kind === "durability-failure") f.store.syncEvidenceDirectory = async () => { throw Error("Cannot sync"); };
      const before = await readFile(f.lock);
      await assert.rejects(f.store.recoverCrashLock(review, true, () => { if (kind === "guard") throw Error("Trust lost"); }));
      assert.deepEqual(await readFile(f.lock), before);
    } finally { await f.cleanup(); }
  });
}

test("recovery rechecks metadata after the durable archive and refuses an in-place change", async () => {
  const f = await fixture(); try {
    const review = await f.store.reviewCrashLock(id); let checks = 0;
    f.onBoot(async () => { if (++checks === 2) await writeFile(f.lock, JSON.stringify({ ...f.owner, token: id })); });
    await assert.rejects(f.store.recoverCrashLock(review, true), RunnerLockedError);
    assert.equal(JSON.parse(await readFile(f.lock, "utf8")).token, id);
    assert.equal((await readdir(f.root)).filter(n => n.includes(".recovered-lock-")).length, 1);
  } finally { await f.cleanup(); }
});

test("shared recovery gate blocks a fresh acquisition and concurrent recovery", async () => {
  const f = await fixture(); try {
    const review = await f.store.reviewCrashLock(id), second = new RunnerStore(f.root, f.environment), otherReview = await second.reviewCrashLock(id);
    let release!: () => void, entered!: () => void;
    const paused = new Promise<void>(resolve => { release = resolve; }), ready = new Promise<void>(resolve => { entered = resolve; });
    f.onBoot(async () => { entered(); await paused; });
    const recovery = f.store.recoverCrashLock(review, true); await ready;
    await assert.rejects(second.exclusive(id, async () => assert.fail("Acquired while recovering")), RunnerLockedError);
    await assert.rejects(second.recoverCrashLock(otherReview, true), RunnerLockedError);
    release(); await recovery;
  } finally { await f.cleanup(); }
});

test("normal finally never unlinks a replacement owner's lock", async () => {
  const f = await fixture(); try {
    await rm(f.lock);
    await assert.rejects(f.store.exclusive(id, async () => {
      await rename(f.lock, f.lock + ".original"); await writeFile(f.lock, "new-owner");
    }), RunnerLockedError);
    assert.equal(await readFile(f.lock, "utf8"), "new-owner");
  } finally { await f.cleanup(); }
});

test("normal release waits for a transient contender gate without stranding its live-owner lock", async () => {
  const f = await fixture(); try {
    await rm(f.lock);
    const gate = join(f.root, `${id}.lock-gate`);
    let finish!: () => void, entered!: () => void;
    const done = new Promise<void>(resolve => { finish = resolve; }), ready = new Promise<void>(resolve => { entered = resolve; });
    const action = f.store.exclusive(id, async () => { entered(); await done; });
    await ready; await writeFile(gate, "contender", { flag: "wx" });
    finish(); await new Promise(resolve => setTimeout(resolve, 50));
    assert.equal(await readFile(gate, "utf8"), "contender", "Release never removes the contender gate");
    await rm(gate); await action;
    await assert.rejects(stat(f.lock), { code: "ENOENT" });
    await assert.rejects(stat(gate), { code: "ENOENT" });
  } finally { await f.cleanup(); }
});

test("production liveness accepts only ESRCH and refuses permission or unknown failures", () => {
  const kill = process.kill;
  try {
    for (const code of ["ESRCH", "EPERM", "EIO"]) {
      process.kill = (() => { throw Object.assign(new Error("probe"), { code }); }) as typeof process.kill;
      assert.equal(runnerLockEnvironment.ownerStatus(12345), code === "ESRCH" ? "absent" : "uncertain");
    }
  } finally { process.kill = kill; }
});

test("actual exited local process lock requires explicit recovery; current process stays live", { skip: process.platform !== "darwin" && process.platform !== "linux" }, async () => {
  const f = await fixture(); try {
    await rm(f.lock);
    const modulePath = resolve(__dirname, "../../guided/runnerStore.ts");
    const child = spawnSync(process.execPath, ["-e", "require('tsx/cjs'); const {RunnerStore}=require(process.argv[1]); new RunnerStore(process.argv[2]).exclusive(process.argv[3],async()=>process.exit(23)).catch(()=>process.exit(24));", modulePath, f.root, id], { cwd: resolve(__dirname, "../../..") });
    assert.equal(child.status, 23, child.stderr.toString());
    assert.equal(runnerLockEnvironment.ownerStatus(process.pid), "alive");
    const store = new RunnerStore(f.root), review = await store.reviewCrashLock(id);
    assert.equal(review.owner.pid, child.pid);
    await assert.rejects(store.exclusive(id, async () => assert.fail("Automatic replay")), RunnerLockedError);
    await store.recoverCrashLock(review, true); await assert.rejects(stat(f.lock), { code: "ENOENT" });
  } finally { await f.cleanup(); }
});
