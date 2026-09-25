import { link, lstat, open, readdir, readFile, rename, unlink } from "node:fs/promises";
import { join } from "node:path";
import { createHash, randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { RunnerRecord } from "../core/runner";
import { preparePrivateDirectory } from "./privateDirectory";
import { reportManifest, ReportManifest, verifyReportBytes } from "../core/runnerBlob";
import { runnerLockEnvironment, RunnerLockEnvironment, RunnerLockOwner, RunnerLockReview } from "./runnerLock";

export class RunnerLockedError extends Error {}

/** Separate atomic records avoid globalState lost updates between VS Code windows. */
export class RunnerStore {
  private preparing?: Promise<void>;
  private readonly lockReviews = new Map<string, RunnerLockReview>();
  constructor(private readonly root: string, private readonly lockEnvironment: RunnerLockEnvironment = runnerLockEnvironment) {}
  private async prepare(): Promise<void> {
    this.preparing ??= preparePrivateDirectory(this.root);
    try { await this.preparing; } catch (error) { this.preparing = undefined; throw error; }
  }
  private path(id: string, suffix = ".json"): string {
    if (!/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/.test(id)) throw new Error("Invalid runner workflow ID.");
    return join(this.root, id + suffix);
  }
  async read(id: string): Promise<RunnerRecord> {
    await this.prepare();
    const record = JSON.parse(await readFile(this.path(id), "utf8")) as RunnerRecord;
    if (record.schemaVersion !== 2 || record.id !== id) throw new Error("Invalid retained workflow schema.");
    return record;
  }
  async list(): Promise<RunnerRecord[]> {
    await this.prepare();
    const entries = await readdir(this.root);
    return Promise.all(entries.filter(name => /^[a-f0-9-]{36}\.json$/.test(name)).map(name => this.read(name.slice(0, -5))));
  }
  async write(record: RunnerRecord): Promise<void> {
    await this.prepare();
    const destination = this.path(record.id);
    const temporary = this.path(record.id, `.${randomUUID()}.tmp`);
    const file = await open(temporary, "wx", 0o600);
    try { await file.writeFile(JSON.stringify(record)); await file.sync(); } finally { await file.close(); }
    await rename(temporary, destination);
  }
  async readReport(id: string, manifest: ReportManifest): Promise<string> {
    await this.prepare(); reportManifest(manifest);
    const path = this.path(id, `.report-${manifest.operation}.json`);
    const info = await lstat(path);
    if (!info.isFile() || info.isSymbolicLink() || info.size !== manifest.bytes) throw new Error("Retained report file changed.");
    const file = await open(path, "r");
    try {
      const data = Buffer.alloc(manifest.bytes + 1);
      let offset = 0;
      while (offset < data.length) {
        const result = await file.read(data, offset, data.length - offset, null);
        if (!result.bytesRead) break;
        offset += result.bytesRead;
      }
      return verifyReportBytes(data.subarray(0, offset), manifest);
    } finally { await file.close(); }
  }
  /** Atomic no-replace publication. Re-import accepts only identical evidence. */
  async retainReport(id: string, manifest: ReportManifest, text: string): Promise<void> {
    await this.prepare(); reportManifest(manifest);
    verifyReportBytes(Buffer.from(text, "utf8"), manifest);
    const destination = this.path(id, `.report-${manifest.operation}.json`);
    const temporary = this.path(id, `.${randomUUID()}.tmp`);
    const file = await open(temporary, "wx", 0o600);
    try {
      await file.writeFile(text, "utf8"); await file.sync(); await file.close();
      try { await link(temporary, destination); }
      catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
        if (await this.readReport(id, manifest) !== text) throw new Error("Retained report cannot be replaced.");
      }
    } finally { await file.close(); await unlink(temporary); }
  }
  async exclusive<T>(id: string, action: () => Promise<T>): Promise<T> {
    await this.prepare();
    const lockPath = this.path(id, ".lock");
    const lock = await this.lockGate(id, async () => {
      let file;
      try { file = await open(lockPath, "wx", 0o600); }
      catch { throw new RunnerLockedError("Another window may own this runner operation. Use Review Interrupted Runner Lock for explicit local review; uncertain or legacy crash locks remain blocked."); }
      try {
        const owner: RunnerLockOwner = { version: 1, workflowId: id, token: randomUUID(), pid: process.pid,
          bootIdentity: await this.lockEnvironment.bootIdentity() ?? null, createdAt: new Date(this.lockEnvironment.now()).toISOString() };
        await file.writeFile(JSON.stringify(owner)); await file.sync();
        return file;
      } catch (error) { await file.close(); throw error; } // Partial locks remain fail-closed.
    });
    try { return await action(); }
    finally {
      // Never remove a replacement lock if an external actor changed the path.
      try { await this.lockGate(id, () => this.unlinkOwned(lockPath, lock), true); }
      finally { await lock.close(); }
    }
  }
  private async unlinkOwned(path: string, file: Awaited<ReturnType<typeof open>>): Promise<void> {
    const owned = await file.stat(), current = await lstat(path);
    if (!current.isFile() || current.isSymbolicLink() || current.dev !== owned.dev || current.ino !== owned.ino) throw new RunnerLockedError("Runner lock identity changed; it has been preserved for operator review.");
    await unlink(path);
  }
  /** Both acquisition and recovery honor this short local gate. A crashed gate
   * is ambiguous and remains blocked; it is never automatically reclaimed. */
  private async lockGate<T>(id: string, action: () => Promise<T>, waitForRelease = false): Promise<T> {
    const path = this.path(id, ".lock-gate");
    let file;
    // A rejected contender may briefly hold the gate while the owner finishes.
    // Release waits at most two seconds, never removing an uncertain gate.
    for (let attempt = 0; !file; attempt++) {
      try { file = await open(path, "wx", 0o600); }
      catch (error) {
        if (!waitForRelease || attempt >= 100 || (error as NodeJS.ErrnoException).code !== "EEXIST") throw new RunnerLockedError("Runner lock review or acquisition is busy or interrupted. Its gate requires operator investigation.");
        await new Promise(resolve => setTimeout(resolve, 20));
      }
    }
    try { return await action(); }
    finally { try { await this.unlinkOwned(path, file); } finally { await file.close(); } }
  }
  private async lockSnapshot(id: string): Promise<{ owner: RunnerLockOwner; text: string; device: number; inode: number; size: number; modified: number; changed: number; lockSHA256: string; recordSHA256: string }> {
    const path = this.path(id, ".lock"), before = await lstat(path);
    if (!before.isFile() || before.isSymbolicLink() || before.size > 4096 || before.size === 0) throw new RunnerLockedError("Legacy, linked or invalid runner lock requires operator investigation; recovery is refused.");
    const file = await open(path, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
    let text: string;
    try {
      const info = await file.stat();
      if (info.dev !== before.dev || info.ino !== before.ino || !info.isFile() || info.size > 4096) throw new RunnerLockedError("Runner lock changed during review.");
      text = await file.readFile("utf8");
    } finally { await file.close(); }
    let owner: RunnerLockOwner;
    try { owner = JSON.parse(text) as RunnerLockOwner; }
    catch { throw new RunnerLockedError("Invalid runner lock metadata; recovery is refused."); }
    if (!owner || owner.version !== 1 || owner.workflowId !== id || !/^[a-f0-9-]{36}$/.test(owner.token) ||
        !Number.isSafeInteger(owner.pid) || owner.pid <= 0 || owner.pid > 2147483647 ||
        typeof owner.createdAt !== "string" || !Number.isFinite(Date.parse(owner.createdAt)) ||
        typeof owner.bootIdentity !== "string" || owner.bootIdentity.length > 200) throw new RunnerLockedError("Unknown runner lock owner; recovery is refused.");
    const boot = await this.lockEnvironment.bootIdentity();
    if (!boot || owner.bootIdentity !== boot) throw new RunnerLockedError("Runner lock belongs to an unknown or different boot/PID namespace; recovery is refused.");
    // PID reuse is conservative: any process at the PID blocks recovery, even
    // if it is not the original owner. Only ESRCH is admissible.
    if (this.lockEnvironment.ownerStatus(owner.pid) !== "absent") throw new RunnerLockedError("Runner lock owner is still running or its status is uncertain; recovery is refused.");
    const recordPath = this.path(id), recordInfo = await lstat(recordPath);
    if (!recordInfo.isFile() || recordInfo.isSymbolicLink()) throw new RunnerLockedError("Retained workflow is linked or invalid; recovery is refused.");
    const record = await readFile(recordPath);
    const retained = JSON.parse(record.toString("utf8")) as RunnerRecord;
    if (retained.schemaVersion !== 2 || retained.id !== id) throw new RunnerLockedError("Invalid retained workflow schema; recovery is refused.");
    const after = await lstat(path);
    if (!after.isFile() || after.isSymbolicLink() || after.dev !== before.dev || after.ino !== before.ino || after.size !== before.size || after.mtimeMs !== before.mtimeMs || after.ctimeMs !== before.ctimeMs) throw new RunnerLockedError("Runner lock changed during review.");
    const hash = (data: string | Buffer) => createHash("sha256").update(data).digest("hex");
    return { owner, text, device: before.dev, inode: before.ino, size: before.size, modified: before.mtimeMs, changed: before.ctimeMs, lockSHA256: hash(text), recordSHA256: hash(record) };
  }
  /** Read-only preparation. Does not authorize recovery or any remote operation. */
  async reviewCrashLock(id: string): Promise<RunnerLockReview> {
    await this.prepare();
    const snapshot = await this.lockSnapshot(id), reviewedAt = this.lockEnvironment.now();
    for (const [token, review] of this.lockReviews) if (review.expiresAt < reviewedAt) this.lockReviews.delete(token);
    if (this.lockReviews.size >= 64) throw new RunnerLockedError("Too many pending lock reviews. Finish or cancel an existing review first.");
    const review: RunnerLockReview = { reviewToken: randomUUID(), workflowId: id, owner: snapshot.owner,
      device: snapshot.device, inode: snapshot.inode, lockSHA256: snapshot.lockSHA256, recordSHA256: snapshot.recordSHA256,
      reviewedAt, expiresAt: reviewedAt + 5 * 60_000 };
    this.lockReviews.set(review.reviewToken, structuredClone(review));
    return review;
  }
  /** Explicit local recovery only. Preserve original lock evidence durably,
   * leave record/report bytes intact, and never reconnect or replay an action. */
  async recoverCrashLock(review: RunnerLockReview, approved: boolean, guard: () => void = () => {}): Promise<void> {
    const saved = this.lockReviews.get(review.reviewToken);
    this.lockReviews.delete(review.reviewToken);
    if (!approved) return;
    if (!saved || JSON.stringify(saved) !== JSON.stringify(review)) throw new RunnerLockedError("Runner lock review changed or is unavailable. Review it again.");
    const checkTime = () => {
      const now = this.lockEnvironment.now();
      if (now < saved.reviewedAt || now > saved.expiresAt) throw new RunnerLockedError("Runner lock review expired. Review it again.");
    };
    checkTime(); guard();
    await this.lockGate(saved.workflowId, async () => {
      const validate = async () => {
        checkTime(); guard();
        const current = await this.lockSnapshot(saved.workflowId);
        if (current.device !== saved.device || current.inode !== saved.inode || current.lockSHA256 !== saved.lockSHA256 || current.recordSHA256 !== saved.recordSHA256) throw new RunnerLockedError("Runner workflow or lock changed during review. Recovery is refused.");
        return current;
      };
      const current = await validate();
      const archivePath = this.path(saved.workflowId, `.recovered-lock-${saved.reviewToken}.json`);
      const archive = await open(archivePath, "wx", 0o600);
      try {
        await archive.writeFile(JSON.stringify({ version: 1, kind: "operator-reviewed-runner-lock", review: saved, originalLock: current.text,
          approvedAt: new Date(this.lockEnvironment.now()).toISOString(), scope: "local-lock-only-no-remote-action" }));
        await archive.sync();
      } finally { await archive.close(); }
      await this.syncEvidenceDirectory();
      const checked = await validate();
      checkTime(); guard();
      // The shared gate prevents participating windows from replacing the path.
      // Recheck identity after the final asynchronous read, before unlinking.
      const final = await lstat(this.path(saved.workflowId, ".lock"));
      if (!final.isFile() || final.isSymbolicLink() || final.dev !== saved.device || final.ino !== saved.inode || final.size !== checked.size || final.mtimeMs !== checked.modified || final.ctimeMs !== checked.changed ||
          this.lockEnvironment.ownerStatus(saved.owner.pid) !== "absent") throw new RunnerLockedError("Runner lock ownership changed; recovery is refused.");
      guard();
      await unlink(this.path(saved.workflowId, ".lock"));
      await this.syncEvidenceDirectory();
    });
  }
  /** Destructive control cleanup requires durable directory entries as well as
   * fsynced file contents. Unsupported hosts fail closed before cloud deletion. */
  async syncEvidenceDirectory(): Promise<void> {
    await this.prepare();
    const directory = await open(this.root, "r");
    try { await directory.sync(); } finally { await directory.close(); }
  }
}
