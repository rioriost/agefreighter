import { createHash, randomUUID } from "node:crypto";
import { object, RunnerRecord, runnerNames } from "./runner";
import { RunnerControl } from "./runnerLifecycle";
import { ReportManifest, verifyReportBytes } from "./runnerBlob";
import { assertIdleHealth, guestDispatchScript, guestReadinessScript } from "./runnerGuest";
import { ReadinessReceipt, readinessArchive, readinessReceiptReferenced, sealReadinessReceipt } from "./runnerReceipts";

export interface ReceiptRemoval {
  commandId: string;
  receiptSHA256: string;
  archive: ReportManifest;
  accountBinding: string;
  submittedAt: string;
  phase: "submitted" | "unknown" | "absent";
}
export interface RemovalPlan {
  workflow: string;
  commandId: string;
  receiptSHA256: string;
  stateSHA256: string;
  observationSHA256: string;
  accountBinding: string;
  expiresAt: number;
  text: string;
  manifest: ReportManifest;
}
export interface RemovalIO {
  control: RunnerControl;
  /** Refresh same-account authorization and workspace trust; never cache a token. */
  guard(binding: string): Promise<void>;
  retain(manifest: ReportManifest, text: string): Promise<void>;
  read(manifest: ReportManifest): Promise<string>;
  remove(commandId: string, binding: string): Promise<void>;
}
const digest = (value: unknown) => createHash("sha256").update(JSON.stringify(value)).digest("hex");
const api = "2024-07-01";
const terminal = new Set(["finished", "failed", "interrupted", "pass", "imported", "exported", "uploaded", "verified"]);

function eligible(r: RunnerRecord, commandId: string): ReadinessReceipt {
  if (r.phase !== "provisioned" || r.vmId !== runnerNames(r.id, r.input).vmId) throw new Error("Runner identity is not eligible for receipt removal.");
  const receipt = r.readinessReceipts?.find(x => x.command.id === commandId);
  if (!receipt) throw new Error("No sealed readiness receipt; legacy commands cannot be adopted.");
  readinessArchive(r, receipt);
  if (r.readinessRemovals?.some(x => x.commandId === commandId)) throw new Error("A removal intent already exists. Reconcile it; never replay removal.");
  if (readinessReceiptReferenced(r, receipt)) throw new Error("Readiness is still referenced; retain this command.");
  const operations = [r.guestCommand, r.assessment, r.postgresCatalog, r.migration, r.upgrade, r.resize, r.targetDiagnostic,
    r.p1Qualification, r.p1Diagnostic, r.targetRestart, r.target?.configurationRepair,
    ...r.reportTransfers ?? [], ...r.csvTransfers ?? [], ...r.assessmentHistory ?? [], ...r.upgradeHistory ?? [],
    ...r.targetDiagnosticHistory ?? [], ...r.p1QualificationHistory ?? [], ...r.migrationContinuations ?? [],
    ...r.migrationHistory?.flatMap(x => [x.migration, x.diagnostic]) ?? []];
  const deployments = [r.target, r.storageDeployment, r.cosmosAccess];
  if (operations.some(x => x && (!terminal.has(x.phase) || object(x).unknown === true)) ||
      deployments.some(x => x && !["previewed", "ready", "provisioned", "failed"].includes(x.phase)) ||
      r.readinessRemovals?.some(x => x.phase !== "absent")) throw new Error("Reconcile all active or uncertain operations before receipt removal.");
  return receipt;
}

/** A separate current control proves recent idle health; old seals never do. */
function currentReadiness(r: RunnerRecord, receipt: ReadinessReceipt): ReadinessReceipt {
  assertIdleHealth(r);
  const sealed = sealReadinessReceipt(r);
  const current = r.readinessReceipts?.find(x => x.command.id === sealed.command.id && x.sha256 === sealed.sha256);
  if (!current || current.command.id === receipt.command.id || current.readiness.bootId !== receipt.readiness.bootId ||
      Date.parse(current.command.submittedAt) <= Date.parse(receipt.command.submittedAt)) {
    throw new Error("A separate newer, same-boot readiness control with fresh idle health is required. Historical evidence alone cannot authorize removal.");
  }
  readinessArchive(r, current);
  return current;
}

/** GET-only admission never starts compute or dispatches a health probe. */
async function observe(control: RunnerControl, r: RunnerRecord, receipt: ReadinessReceipt) {
  const current = currentReadiness(r, receipt);
  const vm = await control.request(r.input.subscriptionId, `${r.vmId}?api-version=${api}&$expand=instanceView`);
  const v = object(vm.value), p = object(v.properties), tags = object(v.tags), statuses = object(p.instanceView).statuses;
  const power = Array.isArray(statuses) ? statuses.map(x => object(x).code).filter(x => typeof x === "string" && x.startsWith("PowerState/")) : [];
  if (vm.status !== 200 || String(v.id).toLowerCase() !== r.vmId.toLowerCase() || tags.application !== "agefreighter" || tags.workflow !== r.id ||
      tags.purpose !== "discovery-and-migration" || v.location !== r.input.region || JSON.stringify(v.zones) !== JSON.stringify([r.input.zone]) ||
      p.provisioningState !== "Succeeded" || power.length !== 1 || power[0] !== "PowerState/running" ||
      typeof p.vmId !== "string" || !/^[a-f0-9-]{36}$/i.test(p.vmId)) throw new Error("Receipt removal requires the unchanged owned VM already running with fresh idle readiness. This action never starts or stops compute.");
  const command = await observeCommand(control, r, receipt);
  const readinessControl = await observeCommand(control, r, current);
  // Reads can be slow: never turn a stale successful ARM response into freshness.
  currentReadiness(r, receipt);
  return { vm: { id: r.vmId, instanceId: p.vmId, location: v.location, zone: r.input.zone, workflow: r.id, power: power[0],
      configurationSHA256: digest({ tags, identity: v.identity, storage: p.storageProfile, network: p.networkProfile, hardware: p.hardwareProfile, security: p.securityProfile }) },
    command, currentReadiness: { receipt: current, command: readinessControl } };
}

/** Pending/Updating or missing live output remains a blocker for either control. */
async function observeCommand(control: RunnerControl, r: RunnerRecord, receipt: ReadinessReceipt) {
  const response = await control.request(r.input.subscriptionId, `${receipt.command.id}?api-version=${api}&$expand=instanceView`);
  const c = object(response.value), props = object(c.properties), view = object(props.instanceView);
  const start = Date.parse(String(view.startTime)), end = Date.parse(String(view.endTime)), submitted = Date.parse(receipt.command.submittedAt);
  if (response.status !== 200 || String(c.id).toLowerCase() !== receipt.command.id.toLowerCase() || c.location !== r.input.region ||
      props.provisioningState !== "Succeeded" || view.executionState !== "Succeeded" || view.exitCode !== 0 ||
      view.error !== "" && view.error !== undefined || ![guestDispatchScript, guestReadinessScript].includes(String(object(props.source).script)) ||
      Object.keys(object(props.source)).some(k => k !== "script") || props.asyncExecution !== false || props.timeoutInSeconds !== 60 ||
      props.runAsUser || props.runAsPassword || props.outputBlobUri || props.errorBlobUri ||
      props.parameters !== undefined && (!Array.isArray(props.parameters) || props.parameters.length !== 0) ||
      !Number.isFinite(start) || !Number.isFinite(end) || end < start || end > Date.now() || start < submitted - 60_000 || start > submitted + 300_000 ||
      typeof view.output !== "string" || Buffer.byteLength(view.output) >= 4096) throw new Error("Command is absent, changed, pending or not proven successful. Nothing may be removed.");
  let value: Record<string, unknown>;
  try { value = object(JSON.parse(view.output)); }
  catch { throw new Error("Invalid readiness JSON; preserve the command for review."); }
  const fields = ["version", "ready", "os", "architecture", "bootId", "cliVersion", "archiveSha256", "commit", "capabilities", "health"];
  if (Object.keys(value).some(k => !fields.includes(k)) || value.version !== 1 || value.ready !== true || value.os !== "linux" || value.architecture !== "amd64" ||
      value.health !== undefined && Object.keys(object(value.health)).some(k => !["idle", "storageUsedPercent", "swapUsedBytes", "oomEvents"].includes(k))) throw new Error("Unexpected readiness output; preserve it for review.");
  const replay = sealReadinessReceipt({ ...r, artifact: { version: receipt.readiness.cliVersion, sha256: receipt.readiness.archiveSha256, url: "" },
    guestCommand: receipt.command, guestReady: { ...value, checkedAt: receipt.command.submittedAt } as unknown as ReadinessReceipt["readiness"] });
  if (replay.sha256 !== receipt.sha256) throw new Error("Live readiness output no longer matches the sealed receipt.");
  // Full raw responses/parameters are deliberately not archived. The selected
  // evidence and exact validated output hash bind the approved observation.
  return { id: receipt.command.id, location: c.location, scriptSHA256: digest(object(props.source).script), startTime: view.startTime, endTime: view.endTime,
      metadataSHA256: digest({ tags: c.tags, systemData: c.systemData }),
      provisioningState: props.provisioningState, executionState: view.executionState, exitCode: 0, outputSHA256: createHash("sha256").update(view.output).digest("hex") };
}

/** Read-only Azure preview. No archive, intent or deletion until explicit approval. */
export async function previewReceiptRemoval(io: RemovalIO, r: RunnerRecord, commandId: string, accountBinding: string): Promise<RemovalPlan> {
  if (!/^[a-f0-9]{64}$/.test(accountBinding)) throw new Error("Missing account binding.");
  await io.guard(accountBinding);
  const receipt = eligible(r, commandId), observation = await observe(io.control, r, receipt);
  const expiresAt = Math.min(Date.now() + 300_000, Date.parse(observation.currentReadiness.receipt.readiness.checkedAt) + 300_000);
  const text = JSON.stringify({ kind: "agefreighter-readiness-removal-v2", receipt, observation, accountBinding, reviewedAt: new Date().toISOString() });
  return { workflow: r.id, commandId, receiptSHA256: receipt.sha256, stateSHA256: digest(r), observationSHA256: digest(observation), accountBinding, expiresAt, text,
    manifest: { operation: randomUUID(), bytes: Buffer.byteLength(text), sha256: createHash("sha256").update(text).digest("hex") } };
}

/** Caller holds workflow lock. An intent is single-use even when no DELETE ran. */
export async function submitReceiptRemoval(io: RemovalIO, r: RunnerRecord, plan: RemovalPlan, approved: boolean): Promise<RunnerRecord> {
  if (!approved) return r;
  if (r.id !== plan.workflow || digest(r) !== plan.stateSHA256 || !Number.isFinite(plan.expiresAt) || Date.now() > plan.expiresAt || plan.expiresAt > Date.now() + 300_000) throw new Error("Removal review expired or workflow changed. Review again.");
  const receipt = eligible(r, plan.commandId);
  if (receipt.sha256 !== plan.receiptSHA256) throw new Error("Receipt changed after approval.");
  verifyReportBytes(Buffer.from(plan.text), plan.manifest);
  const archive = object(JSON.parse(plan.text));
  if (archive.kind !== "agefreighter-readiness-removal-v2" || JSON.stringify(archive.receipt) !== JSON.stringify(receipt) ||
      digest(archive.observation) !== plan.observationSHA256 || archive.accountBinding !== plan.accountBinding) throw new Error("Removal archive does not match the approved evidence.");
  await io.guard(plan.accountBinding);
  if (digest(await observe(io.control, r, receipt)) !== plan.observationSHA256) throw new Error("Azure evidence changed after approval.");
  await io.retain(plan.manifest, plan.text);
  if (await io.read(plan.manifest) !== plan.text) throw new Error("Durable archive could not be verified.");
  // Re-check immediately before intent, including after slow filesystem work.
  await io.guard(plan.accountBinding);
  if (Date.now() > plan.expiresAt || digest(await observe(io.control, r, receipt)) !== plan.observationSHA256) throw new Error("Removal admission changed during archiving.");
  const intent: ReceiptRemoval = { commandId: plan.commandId, receiptSHA256: receipt.sha256, archive: plan.manifest,
    accountBinding: plan.accountBinding, submittedAt: new Date().toISOString(), phase: "submitted" };
  let next: RunnerRecord = { ...r, readinessRemovals: [...r.readinessRemovals ?? [], intent] };
  await io.control.persist(next);
  try {
    await io.guard(plan.accountBinding);
    if (Date.now() > plan.expiresAt) throw new Error("Removal approval expired before dispatch.");
    await io.remove(plan.commandId, plan.accountBinding);
  } catch {
    next = { ...next, readinessRemovals: next.readinessRemovals!.map(x => x === intent ? { ...x, phase: "unknown" } : x) };
    await io.control.persist(next);
  }
  // HTTP success/202 is not absence. A later explicit GET verifies it.
  return next;
}

/** Lost acknowledgements and restart recovery are GET-only, never DELETE retry. */
export async function reconcileReceiptRemoval(io: RemovalIO, r: RunnerRecord, commandId: string): Promise<RunnerRecord> {
  const intent = r.readinessRemovals?.find(x => x.commandId === commandId);
  const receipt = r.readinessReceipts?.find(x => x.command.id === commandId);
  if (!intent || !receipt || receipt.sha256 !== intent.receiptSHA256) throw new Error("No bound removal intent.");
  readinessArchive(r, receipt);
  await io.guard(intent.accountBinding);
  // Missing archive is a blocker even if ARM would report 404.
  const archive = object(JSON.parse(verifyReportBytes(Buffer.from(await io.read(intent.archive)), intent.archive)));
  if (!["agefreighter-readiness-removal-v1", "agefreighter-readiness-removal-v2"].includes(String(archive.kind)) || JSON.stringify(archive.receipt) !== JSON.stringify(receipt) || archive.accountBinding !== intent.accountBinding) throw new Error("Removal archive binding changed.");
  const result = await io.control.request(r.input.subscriptionId, `${commandId}?api-version=${api}`);
  if (result.status !== 404 && intent.phase !== "absent") return r;
  const next: RunnerRecord = { ...r, readinessRemovals: r.readinessRemovals!.map(x => x === intent ? { ...x, phase: result.status === 404 ? "absent" : "unknown" } : x) };
  await io.control.persist(next);
  return next;
}
