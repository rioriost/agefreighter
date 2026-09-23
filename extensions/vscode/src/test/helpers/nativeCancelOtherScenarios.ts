/** Seven frozen B09 decisions; pure synthetic prerequisites, never cloud
 * evidence or approval. Files below are inert byte fixtures, not executables. */
import { createHash, randomUUID } from "node:crypto";
import { RunnerRecord, previewHash, releaseArtifact, runnerTemplate, sourceWorkflowDraft } from "../../core/runner";
import { ReportManifest } from "../../core/runnerBlob";
import { developmentArtifact } from "../../core/runnerDevelopment";
import { recoveryIdentity } from "../../core/runnerExecution";
import { validateResumeInspection } from "../../core/runnerResume";
import { buildSourceDraft } from "../../core/runnerSource";
import { storageDraft } from "../../core/runnerStorageLifecycle";
import { neo4jTargetEvidence, targetPreview, targetResourceIds } from "../../core/runnerTarget";
import { p1FixtureRoot, p1Root } from "../../core/p1Qualification";
import { sourceForm } from "../sourceFixtures";

export const otherNativeCancelCases = [
  { id: "A01", name: "runner.create", module: "runnerMigration", exported: "registerRunnerMigration" },
  { id: "A02", name: "dev.prepare", module: "developmentRunner", exported: "prepareDevelopmentRunner" },
  { id: "A03", name: "dev.upgrade", module: "developmentRunner", exported: "upgradeDevelopmentRunner" },
  { id: "A17", name: "target.repairPreload", module: "runnerTargetPanel", exported: "reviewRunnerTarget" },
  { id: "A21", name: "execution.start", module: "runnerExecutionPanel", exported: "continueRunnerExecution" },
  { id: "A22", name: "execution.resume", module: "runnerExecutionPanel", exported: "continueRunnerExecution" },
  { id: "A28", name: "p1.failureDiagnosis", module: "p1DiagnosticPanel", exported: "diagnoseP1" }
] as const;
export type OtherNativeCancelCase = typeof otherNativeCancelCases[number];
export interface OtherCancellationFixture {
  record: RunnerRecord; title: string; action: string; responses: Map<string, unknown>; lists: Map<string, unknown[]>;
  reports: { manifest: ReportManifest; text: string }[];
  files: { name: string; text: string }[]; openFile?: string;
  inputValues: string[]; selectionIndexes: number[]; prepareMessages: Record<string, unknown>[]; message?: Record<string, unknown>;
  requiresDevelopmentOptIn: boolean; preModalObservationWrites: number; prerequisiteNote: string;
}
const sha = (text: string) => createHash("sha256").update(text).digest("hex");

export function otherCancellationFixture(scenario: OtherNativeCancelCase, now = Date.now()): OtherCancellationFixture {
  const id = randomUUID(), operation = randomUUID(), boot = randomUUID(), checkedAt = new Date(now - 1000).toISOString();
  const record = sourceWorkflowDraft(id, { subscriptionId: id, resourceGroup: "b09-isolated-synthetic", region: "japaneast", zone: "1", size: "Standard_B2s_v2",
    subnetId: `/subscriptions/${id}/resourceGroups/b09-isolated-synthetic/providers/Microsoft.Network/virtualNetworks/fixture/subnets/runner`,
    source: { type: "neo4j", location: "on-premises" } });
  record.phase = "provisioned";
  record.artifact = releaseArtifact("2.4.0", `${"a".repeat(64)}  agefreighter_v2.4.0_linux_amd64.tar.gz`);
  record.guestReady = { bootId: boot, cliVersion: record.artifact.version, archiveSha256: record.artifact.sha256, commit: "synthetic", checkedAt,
    capabilities: ["neo4j-migration-v1", "explicit-resume-v1", "resume-inspection-v1"], health: { idle: true, storageUsedPercent: 4, swapUsedBytes: 0, oomEvents: 0 } };
  const f: OtherCancellationFixture = { record, title: "", action: "", responses: new Map(), lists: new Map(), reports: [], files: [],
    inputValues: [], selectionIndexes: [0], prepareMessages: [], requiresDevelopmentOptIn: false, preModalObservationWrites: 0,
    prerequisiteNote: "All identities, prior states and evidence are synthetic. No prior cloud action or consent occurred. Positive choices must be denied before controller continuation." };
  const manifest = () => {
    const archive = "synthetic-inert-bytes.tar.gz", text = "INERT NATIVE CANCEL FIXTURE. NOT A TAR ARCHIVE OR EXECUTABLE.\n";
    const raw = { schemaVersion: 1, platform: "linux-amd64", version: `2.4.0-dev.${"b".repeat(12)}`, commit: "b".repeat(40),
      sha256: sha(text), bytes: Buffer.byteLength(text), archive, purpose: "p1-read-only-verifier", fixtureRoot: p1FixtureRoot, canonicalRoot: p1Root };
    f.files = [{ name: archive, text }, { name: "synthetic-manifest.json", text: JSON.stringify(raw) }];
    f.openFile = "synthetic-manifest.json"; f.requiresDevelopmentOptIn = true;
    f.prerequisiteNote += " Local pinned bytes deliberately are not executable or a valid archive; production pre-modal size/hash inspection only, never upload/install.";
    return raw;
  };
  if (scenario.id === "A01") {
    delete record.guestReady;
    record.phase = "previewed"; record.hourlyComputeUSD = 1;
    record.template = runnerTemplate(id, record.input, record.artifact, "ssh-ed25519 AAAA");
    record.previewHash = previewHash(record.template, record.input, record.hourlyComputeUSD);
    record.expiresAt = new Date(now + 900_000).toISOString();
    f.title = `Create the reviewed Linux discovery/migration VM ${record.vmId}?`;
    f.prepareMessages = [{ action: "restore" }];
    f.message = { action: "deploy", workflow: id, hash: record.previewHash, networkApproved: true, costApproved: true };
    f.prerequisiteNote += " Restore this synthetic sealed preview through the real listener; private command registry only, no new VS Code command registration.";
    return f;
  }
  if (scenario.id === "A02" || scenario.id === "A03") {
    const raw = manifest();
    if (scenario.id === "A02") {
      record.phase = "draft"; delete record.guestReady;
      record.storageDeployment = { ...storageDraft(record, randomUUID()), phase: "ready" };
      f.title = "Prepare this unpublished executable for an isolated qualification runner?";
    } else {
      record.artifact = developmentArtifact(record, { ...raw, commit: "c".repeat(40), version: `2.4.0-dev.${"c".repeat(12)}`, sha256: "d".repeat(64) });
      Object.assign(record.guestReady!, { cliVersion: record.artifact.version, archiveSha256: record.artifact.sha256, commit: record.artifact.development!.commit });
      f.title = "Upgrade this idle Linux runner, preserving its data and previous installation?";
    }
    return f;
  }
  record.sourceDraft = buildSourceDraft(record.input.source, { ...sourceForm, host: "source.invalid" }, id);
  const text = JSON.stringify({ schemaVersion: 1, command: "inventory", agefreighterVersion: record.artifact.version, outcome: "pass", errors: [], incompleteChecks: [],
    checks: [{ id: "source-counts", status: "pass" }], sections: [{ title: "Source inventory", fields: [
      { name: "vertices", value: "2", status: "pass" }, { name: "edges", value: "1", status: "pass" },
      { name: "totalRows", value: "3", status: "pass" }, { name: "countMethod", value: "neo4j-transactional-count-store", status: "pass" }
    ] }] });
  const report = { operation, bytes: Buffer.byteLength(text), sha256: sha(text) };
  f.reports.push({ manifest: report, text });
  record.assessment = { operation, action: "inventory", phase: "finished", bootId: boot,
    configurationSHA256: sha(JSON.stringify(record.sourceDraft.configuration)), reportSHA256: report.sha256, reportBytes: report.bytes };
  record.reportTransfers = [{ ...report, blob: "synthetic-retained-report", phase: "imported" }];
  const evidence = neo4jTargetEvidence(record, text);
  record.target = targetPreview(record, { serverName: "b09-synthetic-target", subnetCIDR: "10.0.2.0/24", postgresSKU: "Standard_D4ds_v5", postgresTier: "GeneralPurpose",
    storageGiB: 128, loaderSize: "Standard_D4s_v5", hourlyUSD: 2, additionalReserveUSD: 50, budgetUSD: 800, deadline: new Date(now + 3600_000).toISOString() }, evidence);
  record.target.phase = "provisioned";
  const p = record.target, preload = `${p.serverId}/configurations/shared_preload_libraries`;
  f.responses.set(p.serverId, { location: record.input.region, tags: { workflow: id, application: "agefreighter", purpose: "migration-target" }, sku: { name: p.input.postgresSKU },
    properties: { state: "Ready", availabilityZone: "1", version: "18", storage: { storageSizeGB: 128 }, network: { publicNetworkAccess: "Disabled", delegatedSubnetResourceId: p.subnetId, privateDnsZoneArmResourceId: p.dnsId } } });
  f.responses.set(preload, { properties: { value: "pg_stat_statements,age", isConfigPendingRestart: false } });
  f.responses.set(record.vmId, { tags: { workflow: id }, properties: { provisioningState: "Succeeded", hardwareProfile: { vmSize: p.input.loaderSize }, instanceView: { statuses: [{ code: "PowerState/running" }] } } });
  if (scenario.id === "A17") {
    p.phase = "failed"; f.preModalObservationWrites = 1;
    f.title = "Repair only the failed AGE preload setting?";
    f.responses.set(p.deploymentId, { properties: { provisioningState: "Failed" } });
    f.responses.set(`${p.serverId}/databases/agefreighter`, {});
    f.responses.set(`${p.serverId}/configurations/azure.extensions`, { properties: { value: "AGE" } });
    f.responses.set(preload, { properties: { value: "pg_cron,pg_stat_statements", defaultValue: "pg_cron,pg_stat_statements", source: "system-default", isConfigPendingRestart: false } });
    f.lists.set(`${p.deploymentId}/operations`, targetResourceIds(p).map(id => ({ properties: { targetResource: { id },
      provisioningState: id === preload ? "Failed" : "Succeeded", ...(id === preload ? { statusMessage: { error: { code: "ServerIsBusy" } } } : {}) } })));
    f.prerequisiteNote += " Real controller reconciles and persists unchanged failed target before modal. Assert exact status-only persistence (only updatedAt may change), retain initial and modal snapshots separately, deny writes after modal.";
    return f;
  }
  record.resize = { phase: "finished", size: p.input.loaderSize, previousSize: record.input.size, preservedSHA256: "e".repeat(64), startedAt: checkedAt };
  if (scenario.id === "A21") {
    f.action = "Start new neo4j migration and counts verification"; f.title = "Start this new neo4j migration on the Linux runner?";
    return f;
  }
  record.migration = { operation: randomUUID(), jobId: randomUUID(), phase: "interrupted", startedAt: checkedAt, bootId: boot,
    artifactSHA256: record.artifact.sha256, cliVersion: record.artifact.version, evidence, guestConfigurationSHA256: "f".repeat(64), recoveryIdentitySHA256: recoveryIdentity(record) };
  record.guestReady!.health!.idle = false;
  if (scenario.id === "A22") {
    f.action = "Explicitly resume the retained job and counts verification"; f.title = "Explicitly resume this retained Linux migration?";
    record.resumeInspection = validateResumeInspection(record, { version: 1, workflow: id, operation: record.migration.operation, jobId: record.migration.jobId,
      bootId: boot, configSha256: record.migration.guestConfigurationSHA256, fingerprint: "d".repeat(64), generationId: "1", committedRows: "1",
      checkpointAt: checkedAt, checkedAt, outcome: "review-required", canResume: false, reasons: ["Synthetic explicit-review fixture; not resume approval"] });
    return f;
  }
  manifest(); f.title = "Run the approved read-only P1 failure diagnosis?";
  record.migration.phase = "finished"; record.migration.verification = { outcome: "pass", summary: "Synthetic prerequisite only; no real counts verified" };
  record.p1Qualification = { operation: randomUUID(), commandId: `${record.vmId}/runCommands/synthetic-failure`, jobId: record.migration.jobId,
    artifact: structuredClone(record.artifact), startedAt: checkedAt, phase: "failed" };
  return f;
}
