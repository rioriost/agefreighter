import { createHash, randomUUID } from "node:crypto";
import { isIP } from "node:net";
import { object, RunnerRecord } from "./runner";
import { RunnerControl } from "./runnerLifecycle";
import { assertIdleHealth, dispatchGuest, reconcileGuest, GuestReadiness } from "./runnerGuest";
import { reportManifest, verifyReportBytes } from "./runnerBlob";
import { adoptPostgresRecommendations, recommendPostgresMappings } from "./postgresRecommendations";
import { buildSourceDraft, SourceMapping } from "./runnerSource";

export interface CatalogConfiguration {
  schemaVersion: 1; host: string; port: number; database: string; username: string; schemas: string[]; sourceCASHA256: string;
}
export interface PostgresCatalog {
  operation: string; action: "postgres-catalog"; configuration: CatalogConfiguration;
  configurationSHA256: string; bindingSHA256: string; bootId: string;
  artifactSHA256: string; version: string; commit: string;
  readiness: GuestReadiness;
  phase: "submitted" | "accepted" | "running" | "finished" | "failed" | "interrupted";
  reportSHA256?: string; reportBytes?: number;
}
const hash = (x: unknown) => createHash("sha256").update(JSON.stringify(x)).digest("hex");
const identifier = /^[A-Za-z_][A-Za-z0-9_]{0,62}$/;
const sha = /^[a-f0-9]{64}$/;
export function catalogBinding(record: RunnerRecord): string {
  return hash({ vm: record.vmId, input: record.input, artifact: record.artifact, ca: record.sourceCA?.sha256 ?? "" });
}

/** Same field order and sorted ASCII schema scope as the guest canonical JSON. */
export function catalogConfiguration(record: RunnerRecord, raw: unknown, schemas: unknown): CatalogConfiguration {
  if (record.input.source.type !== "postgresql") throw new Error("Catalog discovery requires a PostgreSQL source.");
  const form = object(raw);
  const host = typeof form.host === "string" ? form.host.trim() : "";
  if (!host || host.length > 253 || !isIP(host) && host.split(".").some(x => !/^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$/.test(x))) throw new Error("Enter a catalog hostname or IP address, not a URL.");
  const port = Number(form.port);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error("Catalog port must be 1–65535.");
  const database = typeof form.database === "string" ? form.database.trim() : "", username = typeof form.username === "string" ? form.username.trim() : "";
  if (!identifier.test(database) || !identifier.test(username)) throw new Error("Catalog database and username require ASCII identifiers of at most 63 characters; other names require manual mappings.");
  if (!Array.isArray(schemas) || schemas.length < 1 || schemas.length > 16 || schemas.some(x => typeof x !== "string" || !identifier.test(x) || x.startsWith("pg_") || x === "information_schema") || new Set(schemas).size !== schemas.length) throw new Error("Choose 1–16 unique, explicit non-system schemas.");
  const sourceCASHA256 = record.sourceCA?.sha256 ?? "";
  if (sourceCASHA256 && !sha.test(sourceCASHA256)) throw new Error("Review the source CA checksum.");
  return { schemaVersion: 1, host, port, database, username, schemas: [...schemas].sort(), sourceCASHA256 };
}

export function assertCatalogCurrent(record: RunnerRecord): PostgresCatalog {
  const c = record.postgresCatalog;
  if (!c || record.input.source.type !== "postgresql" || c.bindingSHA256 !== catalogBinding(record) || c.configurationSHA256 !== hash(c.configuration) ||
    record.artifact.sha256 !== c.artifactSHA256 || record.artifact.version !== c.version || record.guestReady?.commit !== c.commit ||
    c.readiness?.bootId !== c.bootId || c.readiness.archiveSha256 !== c.artifactSHA256 || c.readiness.cliVersion !== c.version || c.readiness.commit !== c.commit || !c.readiness.capabilities?.includes("postgresql-catalog-v1")) throw new Error("Catalog source or runner identity changed; retain its evidence, do not reuse it.");
  return c;
}

/** One explicit pre-assessment operation per workflow. Failed/unknown work is
 * retained; a new workflow is required, never automatic replay or replacement. */
export async function startCatalog(control: RunnerControl, record: RunnerRecord, configuration: CatalogConfiguration, secrets: Record<string, string>): Promise<RunnerRecord> {
  if (record.postgresCatalog || record.assessment || record.target || record.migration) throw new Error("Catalog discovery requires a workflow without a retained catalog, assessment or target.");
  if (JSON.stringify(configuration) !== JSON.stringify(catalogConfiguration(record, configuration, configuration.schemas))) throw new Error("Catalog review changed.");
  assertIdleHealth(record);
  const ready = record.guestReady!;
  if (!ready.capabilities?.includes("postgresql-catalog-v1") || record.artifact.development && ready.commit !== record.artifact.development.commit) throw new Error("The installed Linux artifact does not support this reviewed catalog operation.");
  const operation = randomUUID();
  const postgresCatalog: PostgresCatalog = { operation, action: "postgres-catalog", configuration: structuredClone(configuration), configurationSHA256: hash(configuration), bindingSHA256: catalogBinding(record), bootId: ready.bootId, artifactSHA256: ready.archiveSha256, version: ready.cliVersion, commit: ready.commit, readiness: structuredClone(ready), phase: "submitted" };
  return dispatchGuest(control, { ...record, postgresCatalog }, { version: 1, workflow: record.id, operation, action: "postgres-catalog", configuration, secrets });
}

export async function refreshCatalog(control: RunnerControl, record: RunnerRecord): Promise<RunnerRecord> {
  const catalog = assertCatalogCurrent(record);
  const pending = record.guestCommand && ["submitted", "unknown"].includes(record.guestCommand.phase);
  if (!pending) return dispatchGuest(control, record, { version: 1, workflow: record.id, operation: catalog.operation, action: "status" });
  if (record.guestCommand!.operation !== catalog.operation || !["postgres-catalog", "status"].includes(record.guestCommand!.action)) throw new Error("Reconcile the other retained guest control first.");
  const checked = await reconcileGuest(control, record);
  if (!checked.result) return checked.record;
  const value = object(checked.result);
  if (value.action !== "postgres-catalog" || value.bootId !== catalog.bootId || value.configSha256 !== catalog.configurationSHA256 || !["accepted", "running", "finished", "failed", "interrupted"].includes(String(value.phase))) throw new Error("Catalog operation, boot or reviewed configuration mismatch.");
  if (["finished", "failed", "interrupted"].includes(catalog.phase) && value.phase !== catalog.phase) throw new Error("Terminal catalog state changed.");
  const next = { ...checked.record, postgresCatalog: { ...catalog, phase: value.phase as PostgresCatalog["phase"] } };
  if (value.phase === "finished" && (value.exitCode !== 0 || !value.reportSha256 || !value.reportBytes)) throw new Error("Catalog completion requires a successful exit and sealed report.");
  if (value.reportSha256 !== undefined || value.reportBytes !== undefined) {
    if (typeof value.reportSha256 !== "string" || typeof value.reportBytes !== "number") throw new Error("Invalid catalog report manifest.");
    const m = reportManifest({ operation: catalog.operation, sha256: value.reportSha256, bytes: value.reportBytes });
    if (catalog.reportSHA256 && (catalog.reportSHA256 !== m.sha256 || catalog.reportBytes !== m.bytes)) throw new Error("Catalog report seal changed.");
    next.postgresCatalog.reportSHA256 = m.sha256; next.postgresCatalog.reportBytes = m.bytes;
  } else if (catalog.reportSHA256) throw new Error("Catalog report seal disappeared.");
  await control.persist(next); return next;
}

export function catalogRecommendations(record: RunnerRecord, text: string) {
  const c = assertCatalogCurrent(record);
  if (c.phase !== "finished") throw new Error("A successful catalog report is required.");
  if (!c.reportSHA256 || !c.reportBytes) throw new Error("Catalog report manifest is missing.");
  const manifest = reportManifest({ operation: c.operation, sha256: c.reportSHA256, bytes: c.reportBytes });
  verifyReportBytes(Buffer.from(text), manifest);
  if (JSON.stringify(object(JSON.parse(text)).schemas) !== JSON.stringify(c.configuration.schemas)) throw new Error("Catalog schema scope changed.");
  return recommendPostgresMappings(text);
}

export function adoptCatalog(record: RunnerRecord, text: string, raw: unknown, schemas: unknown, selected: unknown): RunnerRecord {
  if (record.assessment || record.target || record.migration) throw new Error("Mappings cannot change after assessment or target review; use a new workflow.");
  const c = assertCatalogCurrent(record), form = object(raw);
  if (hash(catalogConfiguration(record, form, schemas)) !== c.configurationSHA256) throw new Error("Connection or schema scope differs from this catalog; nothing was adopted.");
  if (!Array.isArray(selected) || selected.length > 64 || selected.some(x => typeof x !== "string") || !Array.isArray(form.mappings) || form.mappings.length > 64) throw new Error("Select bounded recommendations and mappings explicitly.");
  const mappings = adoptPostgresRecommendations(form.mappings as SourceMapping[], catalogRecommendations(record, text), selected);
  const sourceDraft = buildSourceDraft(record.input.source, { ...form, mappings }, record.id, record.sourceFiles, record.sourceCA);
  return { ...record, sourceDraft };
}
