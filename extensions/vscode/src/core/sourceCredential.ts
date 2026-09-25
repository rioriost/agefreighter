import { createHash } from "node:crypto";
import { RunnerRecord } from "./runner";

export interface SecretVault {
  get(key: string): Thenable<string | undefined>;
  store(key: string, value: string): Thenable<void>;
  delete(key: string): Thenable<void>;
}
export type SourceConnection = { host: string; port: number; database: string; username: string };
/** One encrypted entry per workflow; a changed endpoint/identity/CA replaces,
 * never reuses, the old credential. Mappings and boot IDs are not credentials. */
export function credentialBinding(r: RunnerRecord, c: SourceConnection): string {
  if (!["neo4j", "postgresql"].includes(r.input.source.type) || !c.host || !c.database || !c.username || !Number.isInteger(c.port) || c.port < 1 || c.port > 65535) throw new Error("Review the source connection first.");
  // A failure invalidates credentials saved BEFORE it. Explicitly preparing a
  // new credential AFTER reviewing that failure must work while compute is off.
  // Moving the same failure to history is not another failure or connection.
  const failures = [...new Set([
    ...r.assessmentHistory ?? [], ...r.migrationHistory?.map(x => x.migration) ?? [],
    r.assessment, r.postgresCatalog, r.migration
  ].flatMap(x => x && ["failed", "interrupted"].includes(x.phase) ? [x.operation] : []))].sort();
  return createHash("sha256").update(JSON.stringify([r.id, r.input.source, c.host.toLowerCase(), c.port, c.database, c.username, r.sourceCA?.sha256 ?? "system-ca", "verified-tls", failures])).digest("hex");
}
export const credentialKey = (workflow: string) => `runner-source-session/${workflow}`;
export async function savedSourceCredential(vault: SecretVault, r: RunnerRecord, c: SourceConnection, now = Date.now()): Promise<string | undefined> {
  const raw = await vault.get(credentialKey(r.id));
  if (!raw) return undefined;
  try {
    const value = JSON.parse(raw);
    if (value.binding === credentialBinding(r, c) && Number.isFinite(value.expiresAt) && value.expiresAt > now && value.expiresAt <= now + 8 * 3600000 && (!r.target || Date.parse(r.target.input.deadline) > now) && typeof value.password === "string" && value.password.length) return value.password;
  } catch { /* Discard malformed/expired entries without exposing their contents. */ }
  await vault.delete(credentialKey(r.id));
  return undefined;
}
export async function rememberSourceCredential(vault: SecretVault, r: RunnerRecord, c: SourceConnection, password: string, now = Date.now()): Promise<void> {
  const expiresAt = Math.min(now + 8 * 3600000, r.target ? Date.parse(r.target.input.deadline) : Infinity);
  if (!password || !Number.isFinite(expiresAt) || expiresAt <= now) throw new Error("Source credential session has expired.");
  await vault.store(credentialKey(r.id), JSON.stringify({ binding: credentialBinding(r, c), expiresAt, password }));
}
