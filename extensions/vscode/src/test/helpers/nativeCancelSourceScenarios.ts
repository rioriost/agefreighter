/** Pure synthetic prerequisites for ten frozen B09 source-panel decisions.
 * No VS Code, Azure, filesystem, credentials or approval calls occur here.
 * These fixtures never establish installed/native/signed-in qualification. */
import { createHash, randomUUID } from "node:crypto";
import { RunnerRecord, sourceWorkflowDraft } from "../../core/runner";
import { ReportManifest } from "../../core/runnerBlob";
import { catalogBinding, catalogConfiguration, catalogRecommendations } from "../../core/runnerCatalog";
import { reportStorageNames } from "../../core/runnerReportStorage";
import { storageDraft } from "../../core/runnerStorageLifecycle";
import { catalogForm, catalogText } from "../catalogFixtures";
import { sourceForm } from "../sourceFixtures";

export const sourceNativeCancelCases = [
  { id: "A04", name: "catalogRead", title: "Read PostgreSQL schema metadata on this Linux runner?" },
  { id: "A05", name: "catalogTransfer", title: "Transfer this sealed PostgreSQL catalog?" },
  { id: "A06", name: "adoptMappings", title: "Add these selected PostgreSQL mappings?" },
  { id: "A07", name: "retainFailure", title: "Retain failed source assessment and prepare a fresh attempt?" },
  { id: "A08", name: "cosmosGrant", title: "Grant this Linux runner read-only Cosmos data access?" },
  { id: "A10", name: "csvImport", title: "Download and verify this CSV on the Linux runner?" },
  { id: "A12", name: "retainRejectedExport", title: "Retain the rejected report export and prepare a fresh transfer?" },
  { id: "A13", name: "assessmentTransfer", title: "Transfer and verify this assessment report?" },
  { id: "A14", name: "sampledRead", title: "Run a sampled profile from the Linux runner?" },
  { id: "A15", name: "completeInventory", title: "Run a complete mapped-record inventory (up to 100 million rows) from the Linux runner?" }
] as const;
export type SourceNativeCancelCase = typeof sourceNativeCancelCases[number];

export interface SourceCancellationFixture {
  module: "runnerSourcePanel";
  exported: "openRunnerSource";
  record: RunnerRecord;
  responses: Map<string, unknown>;
  inputValues: string[];
  selectionIndexes: number[];
  reports: { manifest: ReportManifest; text: string }[];
  /** Drive the real listener before taking the immutable cancellation baseline.
   * Only A14/A15 use a local `review` message. Never inject reviewedHash. */
  prepareMessages: Record<string, unknown>[];
  message: Record<string, unknown>;
  baselineAfterPrepare: boolean;
  requiresServices: boolean;
  prerequisiteNote: string;
}
const sha = (text: string) => createHash("sha256").update(text).digest("hex");

export function sourceCancellationFixture(scenario: SourceNativeCancelCase, now = Date.now()): SourceCancellationFixture {
  const id = randomUUID(), operation = randomUUID(), boot = randomUUID();
  const record = sourceWorkflowDraft(id, { subscriptionId: id, resourceGroup: "b09-isolated-synthetic", region: "japaneast", zone: "1", size: "Standard_B2s_v2",
    subnetId: `/subscriptions/${id}/resourceGroups/b09-isolated-synthetic/providers/Microsoft.Network/virtualNetworks/fixture/subnets/runner`,
    source: { type: "postgresql", location: "on-premises" } });
  record.phase = "provisioned";
  record.artifact = { version: "2.4.0", sha256: "a".repeat(64), url: "https://example.invalid/synthetic-not-an-artifact" };
  const checkedAt = new Date(now - 10_000).toISOString();
  record.guestReady = { bootId: boot, cliVersion: "2.4.0", archiveSha256: record.artifact.sha256, commit: "synthetic-fixture", checkedAt,
    capabilities: ["postgresql-catalog-v1", "postgresql-inventory-v1", "postgresql-native-floats-v1"],
    health: { idle: true, storageUsedPercent: 4, swapUsedBytes: 0, oomEvents: 0 } };
  const readinessOperation = randomUUID();
  record.guestCommand = { id: `${record.vmId}/runCommands/af-${readinessOperation}`, operation: readinessOperation, action: "ready", phase: "finished", submittedAt: checkedAt };
  const fixture: SourceCancellationFixture = { module: "runnerSourcePanel", exported: "openRunnerSource", record,
    responses: new Map(), inputValues: [], selectionIndexes: [], reports: [], prepareMessages: [], message: {}, baselineAfterPrepare: false,
    requiresServices: false, prerequisiteNote: "Entire workflow, identities, readiness and prior outcomes are synthetic cancellation prerequisites; no cloud operation or prior approval occurred." };
  const form = { ...structuredClone(catalogForm), host: "source.invalid" };
  const manifest = (text: string): ReportManifest => ({ operation, sha256: sha(text), bytes: Buffer.byteLength(text) });
  const prepareStorage = () => { record.storageDeployment = { ...storageDraft(record, randomUUID()), phase: "ready" }; };
  if (scenario.id === "A04") {
    fixture.message = { action: "catalogStart", form, schemas: ["public"] };
  } else if (scenario.id === "A05" || scenario.id === "A06") {
    const configuration = catalogConfiguration(record, form, ["public"]), report = manifest(catalogText), ready = record.guestReady!;
    record.postgresCatalog = { operation, action: "postgres-catalog", phase: "finished", configuration,
      configurationSHA256: sha(JSON.stringify(configuration)), bindingSHA256: catalogBinding(record), bootId: boot,
      artifactSHA256: ready.archiveSha256, version: ready.cliVersion, commit: ready.commit, readiness: structuredClone(ready),
      reportSHA256: report.sha256, reportBytes: report.bytes };
    fixture.reports.push({ manifest: report, text: catalogText });
    if (scenario.id === "A05") {
      prepareStorage(); fixture.requiresServices = true; fixture.message = { action: "catalogReport" };
      // Deliberately no existing transfer: it would skip this approval surface.
    } else {
      const names = reportStorageNames(record);
      record.reportTransfers = [{ ...report, phase: "imported", blob: `${names.origin}/${names.container}/reports/${operation}.json` }];
      const selected = catalogRecommendations(record, catalogText).proposals[0]!.id;
      fixture.message = { action: "catalogAdopt", form, schemas: ["public"], selected: [selected] };
    }
  } else if (scenario.id === "A07") {
    record.assessment = { operation, action: "inventory", phase: "failed", bootId: randomUUID(), configurationSHA256: "c".repeat(64) };
    fixture.message = { action: "retainFailure" };
  } else if (scenario.id === "A08") {
    const scope = `/subscriptions/${id}/resourceGroups/b09-isolated-synthetic/providers/Microsoft.DocumentDB/databaseAccounts/synthetic`;
    record.input.source = { type: "cosmos-nosql", location: "azure", resourceId: scope };
    record.cosmosAccess = { phase: "previewed", principalId: randomUUID(), scope, assignmentId: `${scope}/sqlRoleAssignments/${operation}`,
      roleDefinitionId: `${scope}/sqlRoleDefinitions/00000000-0000-0000-0000-000000000001` };
    fixture.message = { action: "cosmosAccess" };
    fixture.prerequisiteNote += " The retained grant is a synthetic preview, never a submitted or ready assignment.";
  } else if (scenario.id === "A10") {
    const file = randomUUID(), contents = "id,name\n1,synthetic\n";
    record.input.source = { type: "csv", location: "local" };
    record.sourceFiles = [{ id: file, name: "synthetic.csv", path: "/synthetic-not-a-real-file/synthetic.csv" }];
    record.csvTransfers = [{ file, bytes: Buffer.byteLength(contents), sha256: sha(contents), phase: "uploaded" }];
    fixture.requiresServices = true; fixture.message = { action: "importCSV" };
  } else if (scenario.id === "A12" || scenario.id === "A13") {
    const text = JSON.stringify({ synthetic: true, scope: "native-cancel-prerequisite-only", command: "inventory", operation });
    const report = manifest(text), names = reportStorageNames(record);
    record.assessment = { operation, action: "inventory", phase: "finished", bootId: boot, configurationSHA256: "c".repeat(64), reportSHA256: report.sha256, reportBytes: report.bytes };
    fixture.reports.push({ manifest: report, text }); fixture.requiresServices = true;
    if (scenario.id === "A12") {
      record.guestCommand = { id: `${record.vmId}/runCommands/af-${randomUUID()}`, operation, action: "export-report", phase: "unknown",
        submittedAt: new Date(now - 1_260_000).toISOString(), failure: "Azure returned HTTP 409; reconcile before retrying." };
      record.reportTransfers = [{ ...report, blob: `${names.origin}/${names.container}/reports/${operation}.json`, phase: "unknown" }];
      fixture.message = { action: "retainRejectedExport" };
    } else {
      prepareStorage(); fixture.message = { action: "report" };
      // No transfer is retained, so the production handler must request consent.
    }
  } else {
    fixture.prepareMessages = [{ action: "review", form: { ...structuredClone(sourceForm), host: "source.invalid", port: 5432 } }];
    fixture.message = { action: "assess", method: scenario.id === "A14" ? "profile" : "inventory" };
    fixture.baselineAfterPrepare = true;
    fixture.prerequisiteNote += " Before baseline, run the genuine local review message in the disposable store to bind this panel's source review; no modal approval is supplied.";
  }
  return fixture;
}
