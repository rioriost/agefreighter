import { open, readdir, readFile, rename, unlink } from "node:fs/promises";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { RunnerRecord } from "../core/runner";
import { preparePrivateDirectory } from "./privateDirectory";

export class RunnerLockedError extends Error {}

/** Separate atomic records avoid globalState lost updates between VS Code windows. */
export class RunnerStore {
  private preparing?: Promise<void>;
  constructor(private readonly root: string) {}
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
  async exclusive<T>(id: string, action: () => Promise<T>): Promise<T> {
    await this.prepare();
    const lockPath = this.path(id, ".lock");
    let lock;
    try { lock = await open(lockPath, "wx", 0o600); }
    catch { throw new RunnerLockedError("Another window may own this runner operation. Refresh status; a retained crash lock requires operator review, not automatic replay."); }
    try { return await action(); }
    finally { await lock.close(); await unlink(lockPath); }
  }
}
