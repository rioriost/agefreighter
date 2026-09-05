import * as vscode from "vscode";
import { createHash, randomUUID } from "node:crypto";
import { basename } from "node:path";
import { stat } from "node:fs/promises";
import { object, RunnerRecord } from "./core/runner";
import { RunnerControl } from "./core/runnerLifecycle";
import { RunnerStore } from "./guided/runnerStore";
import { buildSourceDraft, sourceSecrets } from "./core/runnerSource";
import { assessmentActive, refreshAssessment, startAssessment } from "./core/runnerAssessment";
import { runnerSourceHTML } from "./core/runnerSourceView";
import { refreshStorage, storageDraft, submitStorage } from "./core/runnerStorageLifecycle";
import { reportStorageNames, verifyReportStorage, verifyTransferStorage } from "./core/runnerReportStorage";
import { importReport, refreshReportExport, startReportExport } from "./core/runnerReport";
import { escapeHTML } from "./core/report";
import { CSVManifest, inspectCSV } from "./guided/csvTransfer";
import { csvAssessmentReady, refreshCSVImport, startCSVImport } from "./core/runnerCSV";
import { csvFilesInFolder } from "./guided/csvSelection";

export interface RunnerSourceServices {
  storagePrincipal(subscription: string): Promise<string>;
  reportCapability(record: RunnerRecord, operation: string, permission: "r" | "c"): Promise<string>;
  csvCapability(record: RunnerRecord, manifest: CSVManifest): Promise<string>;
  uploadCSV(record: RunnerRecord, path: string, manifest: CSVManifest, progress: (bytes: number) => void): Promise<void>;
}

const hash = (value: unknown) => createHash("sha256").update(JSON.stringify(value)).digest("hex");

export function openRunnerSource(context: vscode.ExtensionContext, control: RunnerControl, store: RunnerStore, workflow: string, services?: RunnerSourceServices): void {
  const panel = vscode.window.createWebviewPanel("agefreighter.runnerSource", "AGEFreighter source assessment", vscode.ViewColumn.One,
    { enableScripts: true, retainContextWhenHidden: true, localResourceRoots: [] });
  panel.webview.html = runnerSourceHTML();
  let busy = false, disposed = false, reviewedHash: string | undefined;
  const post = (value: unknown) => disposed ? Promise.resolve(false) : panel.webview.postMessage(value);
  const initialize = (record: RunnerRecord) => post({ kind: "init", type: record.input.source.type, location: record.input.source.location,
    files: record.sourceFiles?.map(({ id, name }) => ({ id, name })), form: record.sourceDraft?.form, assessment: record.assessment,
    storage: record.storageDeployment ? `${record.storageDeployment.phase}${record.storageDeployment.networkAccess ? ` — public network: ${record.storageDeployment.networkAccess} (provisioning is not transfer readiness)` : ""}` : undefined,
    transferEnabled: !!services, csvTransfers: record.csvTransfers, transfer: record.reportTransfers?.find(item => item.operation === record.assessment?.operation)?.phase,
    canStart: record.phase === "provisioned" && !!record.guestReady && Date.now() - Date.parse(record.guestReady.checkedAt) <= 300000 });
  const listener = panel.webview.onDidReceiveMessage(async raw => {
    if (busy) return;
    busy = true; await post({ kind: "busy", value: true });
    try {
      const message = object(raw);
      switch (message.action) {
        case "ready": await initialize(await store.read(workflow)); break;
        case "uploadCSV": {
          if (!services || !vscode.workspace.isTrusted) throw new Error("Trusted Azure account access is required for CSV transfer.");
          const record = await store.read(workflow);
          if (record.input.source.type !== "csv" || !record.sourceFiles?.length || record.storageDeployment?.phase !== "ready" || assessmentActive(record)) throw new Error("Select CSV files and prepare transfer storage before upload.");
          const manifests = await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: "Hashing selected CSV files", cancellable: false }, async () => Promise.all(record.sourceFiles!.map(file => inspectCSV(file.id, file.path))));
          if (manifests.reduce((n, m) => n + m.bytes, 0) > 10 * 1024 ** 3) throw new Error("The guided CSV transfer limit is 10 GiB per workflow.");
          const confirmed = await vscode.window.showWarningMessage("Upload the selected CSV files to your dedicated Azure transfer storage?", { modal: true,
            detail: `${manifests.length} files, ${manifests.reduce((n, m) => n + m.bytes, 0)} bytes\n${reportStorageNames(record).origin}\nFull source contents leave this computer. Authenticated HTTPS only; storage and request charges apply. No source profiling or migration starts. Changed files fail verification; existing blobs are not overwritten.` }, "Upload reviewed CSV files");
          if (confirmed !== "Upload reviewed CSV files" || disposed) break;
          const next = await store.exclusive(workflow, async () => {
            let current = await store.read(workflow);
            if (assessmentActive(current) || JSON.stringify(current.sourceFiles) !== JSON.stringify(record.sourceFiles)) throw new Error("The selected source changed; review it again.");
            await verifyTransferStorage(control, current);
            for (const manifest of manifests) {
              const previous = current.csvTransfers?.find(item => item.file === manifest.file);
              if (previous && (previous.sha256 !== manifest.sha256 || previous.bytes !== manifest.bytes)) throw new Error("A previously reviewed CSV changed. Preserve the existing workflow and select a new workflow for changed data.");
              if (previous && previous.phase !== "prepared") continue;
              if (!previous) { current = { ...current, csvTransfers: [...current.csvTransfers ?? [], { ...manifest, phase: "prepared" }] }; await control.persist(current); }
              const file = current.sourceFiles!.find(item => item.id === manifest.file)!;
              await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: `Uploading ${file.name}`, cancellable: false }, async progress => {
                await services.uploadCSV(current, file.path, manifest, bytes => progress.report({ message: `${Math.round(100 * bytes / manifest.bytes)}%` }));
              });
              current = { ...current, csvTransfers: current.csvTransfers!.map(item => item.file === manifest.file ? { ...item, phase: "uploaded" } : item) }; await control.persist(current);
            }
            return current;
          });
          reviewedHash = undefined; await initialize(next); break;
        }
        case "importCSV": {
          if (!services || !vscode.workspace.isTrusted) throw new Error("Trusted Azure account access is required for CSV import.");
          const record = await store.read(workflow);
          if (record.input.source.type !== "csv" || assessmentActive(record)) throw new Error("CSV import is not available for this workflow.");
          if (record.csvTransfers?.some(item => ["submitted", "unknown"].includes(item.phase))) {
            await initialize(await store.exclusive(workflow, async () => refreshCSVImport(control, await store.read(workflow)))); break;
          }
          const manifest = record.csvTransfers?.find(item => item.phase === "uploaded");
          if (!manifest) throw new Error("No uploaded CSV is awaiting import. Failed/interrupted imports require evidence review; they are never replayed automatically.");
          const confirmed = await vscode.window.showWarningMessage("Download and verify this CSV on the Linux runner?", { modal: true,
            detail: `${record.sourceFiles?.find(item => item.id === manifest.file)?.name}\n${manifest.bytes} bytes; SHA-256 ${manifest.sha256}\n${record.vmId}\nFull hashing and the 80% disk gate are required. No source assessment or migration starts. Check fresh guest readiness first.` }, "Import and seal CSV");
          if (confirmed !== "Import and seal CSV" || disposed) break;
          const next = await store.exclusive(workflow, async () => { const current = await store.read(workflow); return startCSVImport(control, current, manifest, await services.csvCapability(current, manifest)); });
          reviewedHash = undefined; await initialize(next); break;
        }
        case "storage": {
          if (!services || !vscode.workspace.isTrusted) throw new Error("Trusted Azure account access is required for transfer storage.");
          const record = await store.read(workflow);
          if (record.storageDeployment && record.storageDeployment.phase !== "previewed") {
            await initialize(await store.exclusive(workflow, async () => refreshStorage(control, await store.read(workflow)))); break;
          }
          const principal = await services.storagePrincipal(record.input.subscriptionId), names = reportStorageNames(record);
          const confirmed = await vscode.window.showWarningMessage("Create dedicated transfer storage and grant your Azure user data access?", { modal: true,
            detail: `Account: ${names.id}\nRegion: ${record.input.region}\nUser object ID: ${principal}\nGrant: Storage Blob Data Contributor on this NEW account only.\nStandard LRS storage and request/egress charges apply. The HTTPS endpoint is network-public but anonymous access and shared keys are disabled. Source servers are not exposed. This is not a private-endpoint deployment. Evidence is retained until separately deleted.` }, "Create storage and scoped role");
          if (confirmed !== "Create storage and scoped role" || disposed) break;
          const next = await store.exclusive(workflow, async () => {
            const current = await store.read(workflow);
            if (current.storageDeployment && current.storageDeployment.phase !== "previewed") throw new Error("Storage was already submitted. Refresh its status.");
            const draft = { ...current, storageDeployment: storageDraft(current, principal) };
            await control.persist(draft); return submitStorage(control, draft);
          });
          await initialize(next); break;
        }
        case "report": {
          if (!services || !vscode.workspace.isTrusted) throw new Error("Trusted Azure account access is required for report transfer.");
          const record = await store.read(workflow), assessment = record.assessment;
          if (!assessment?.reportSHA256 || !assessment.reportBytes) throw new Error("Refresh the terminal assessment's report manifest first.");
          const manifest = { operation: assessment.operation, sha256: assessment.reportSHA256, bytes: assessment.reportBytes };
          const existing = record.reportTransfers?.find(item => item.operation === assessment.operation);
          if (existing?.phase !== "imported") {
            const names = reportStorageNames(record);
            if (record.storageDeployment?.phase !== "ready") throw new Error("Prepare transfer storage and refresh it to Ready first.");
            const confirmed = await vscode.window.showWarningMessage("Transfer and verify this assessment report?", { modal: true,
              detail: `${assessment.action}: ${assessment.operation}\n${manifest.bytes} bytes; SHA-256 ${manifest.sha256}\nDestination: ${names.origin}/${names.container}\nSource metadata/sample values may be sensitive. The report is retained privately on this computer and is not sent to an AI model. This action does not repeat source discovery or start migration.` }, "Transfer verified report");
            if (confirmed !== "Transfer verified report" || disposed) break;
            const next = await store.exclusive(workflow, async () => {
              let current = await store.read(workflow);
              if (current.assessment?.operation !== assessment.operation || current.assessment.reportSHA256 !== manifest.sha256) throw new Error("The assessment changed; review it again.");
              await verifyReportStorage(control, current, `${names.origin}/${names.container}/reports/${assessment.operation}.json`);
              if (!current.reportTransfers?.some(item => item.operation === assessment.operation)) {
                return startReportExport(control, current, await services.reportCapability(current, assessment.operation, "c"));
              }
              if (current.guestCommand?.action === "export-report" && current.guestCommand.operation === assessment.operation) current = await refreshReportExport(control, current);
              return importReport(control, current, assessment.operation, await services.reportCapability(current, assessment.operation, "r"), (id, m, text) => store.retainReport(id, m, text));
            });
            await initialize(next);
            if (next.reportTransfers?.find(item => item.operation === assessment.operation)?.phase !== "imported") break;
          }
          const text = await store.readReport(workflow, manifest);
          const view = vscode.window.createWebviewPanel("agefreighter.verifiedSourceReport", "Verified AGEFreighter source report", vscode.ViewColumn.Beside, { enableScripts: false, localResourceRoots: [] });
          view.webview.html = `<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'"></head><body><h1>Hash-verified source report</h1><p>Not a migration or sizing approval.</p><pre>${escapeHTML(text)}</pre></body></html>`;
          break;
        }
        case "folder":
        case "files": {
          const record = await store.read(workflow);
          if (record.input.source.type !== "csv" || assessmentActive(record)) throw new Error("CSV selection is unavailable for this workflow.");
          const folder = message.action === "folder";
          let picked = await vscode.window.showOpenDialog(folder
            ? { canSelectMany: false, canSelectFiles: false, canSelectFolders: true, openLabel: "Select CSV folder (no upload)" }
            : { canSelectMany: true, canSelectFiles: true, canSelectFolders: false, filters: { CSV: ["csv"] }, openLabel: "Select files for source mappings (no upload)" });
          if (!picked) break;
          if (folder) {
            if (picked[0]?.scheme !== "file") throw new Error("Select a local CSV folder.");
            picked = (await csvFilesInFolder(picked[0].fsPath)).map(path => vscode.Uri.file(path));
          }
          if (picked.length > 64 || picked.some(uri => uri.scheme !== "file")) throw new Error("Select at most 64 local CSV files.");
          const files = await Promise.all(picked.map(async uri => {
            if (!(await stat(uri.fsPath)).isFile()) throw new Error("Select regular CSV files.");
            const previous = record.sourceFiles?.find(file => file.path === uri.fsPath);
            return { id: previous?.id ?? randomUUID(), name: basename(uri.fsPath), path: uri.fsPath };
          }));
          const next = await store.exclusive(workflow, async () => {
            const current = await store.read(workflow);
            if (assessmentActive(current)) throw new Error("The source already has a retained operation.");
            const merged = [...current.sourceFiles ?? []];
            for (const file of files) if (!merged.some(existing => existing.path === file.path)) merged.push(file);
            if (merged.length > 64) throw new Error("A workflow supports at most 64 selected CSV files.");
            const next = { ...current, sourceFiles: merged };
            await store.write(next); return next;
          });
          reviewedHash = undefined; await initialize(next); break;
        }
        case "review": {
          const next = await store.exclusive(workflow, async () => {
            const current = await store.read(workflow);
            if (assessmentActive(current)) throw new Error("Retain the existing assessment configuration; it cannot be replaced here.");
            const sourceDraft = buildSourceDraft(current.input.source, message.form, workflow, current.sourceFiles);
            const next = { ...current, sourceDraft };
            if (current.input.source.type === "csv" && csvAssessmentReady(next)) {
              next.sourceDraft = { ...sourceDraft, canAssess: true, warnings: [...sourceDraft.warnings.filter(w => !w.includes("upload")), "All mapped CSV files have guest full-hash seals. Sample profiling is not a complete inventory or migration qualification."] };
            }
            await store.write(next); return next;
          });
          reviewedHash = hash(next.sourceDraft);
          await post({ kind: "review", draft: next.sourceDraft }); break;
        }
        case "assess": {
          if (!vscode.workspace.isTrusted) throw new Error("Trust the workspace before approving source reads.");
          if (message.method !== "profile" && message.method !== "inventory") throw new Error("Unsupported assessment method.");
          const record = await store.read(workflow);
          if (!record.sourceDraft || !reviewedHash || hash(record.sourceDraft) !== reviewedHash) throw new Error("Review current source settings in this window first.");
          if (record.phase !== "provisioned" || !record.guestReady || !Number.isFinite(Date.parse(record.guestReady.checkedAt)) || Date.now() - Date.parse(record.guestReady.checkedAt) > 300000) throw new Error("Provision the runner and check fresh Linux guest readiness before approving source reads.");
          if (!record.sourceDraft.canAssess || assessmentActive(record)) throw new Error("This source cannot start a new assessment here.");
          const confirmed = await vscode.window.showWarningMessage(`Run ${message.method === "profile" ? "a sampled profile" : "an exact Neo4j count inventory"} from the Linux runner?`,
            { modal: true, detail: `${record.input.source.type} / ${record.sourceDraft.form.host} / ${record.sourceDraft.form.database}\nRunner: ${record.vmId}\n${record.sourceDraft.warnings.join("\n")}\nGuest limits: 30 minutes, 4 GiB, no swap. Keep the source unchanged. Closing VS Code will not stop the operation.` }, "Approve source reads");
          if (confirmed !== "Approve source reads" || disposed) break;
          let password: string | undefined;
          if (["neo4j", "postgresql"].includes(record.input.source.type)) {
            password = await vscode.window.showInputBox({ title: "Read-only source password", prompt: "Sent through the protected guest channel; not saved with settings or sent to an AI model.", password: true, ignoreFocusOut: true });
            if (password === undefined || disposed) break;
          }
          try {
            const secrets = sourceSecrets(record.input.source.type, record.sourceDraft.form, password);
            const next = await store.exclusive(workflow, async () => {
              const current = await store.read(workflow);
              if (hash(current.sourceDraft) !== reviewedHash) throw new Error("Source settings changed in another window; review them again.");
              return startAssessment(control, current, message.method as "profile" | "inventory", secrets);
            });
            await post({ kind: "assessment", assessment: next.assessment });
          } finally { password = undefined; }
          break;
        }
        case "refresh": {
          const next = await store.exclusive(workflow, async () => refreshAssessment(control, await store.read(workflow)));
          await post({ kind: "assessment", assessment: next.assessment }); break;
        }
        default: throw new Error("Unsupported source form action.");
      }
    } catch (error) {
      await post({ kind: "error", text: error instanceof Error ? error.message : "Source assessment could not be completed. No automatic replay was attempted." });
    } finally { busy = false; await post({ kind: "busy", value: false }); }
  });
  panel.onDidDispose(() => { disposed = true; listener.dispose(); }, undefined, context.subscriptions);
}
