import * as vscode from "vscode";
import { createHash, randomUUID } from "node:crypto";
import { basename } from "node:path";
import { readFile, stat } from "node:fs/promises";
import { object, RunnerRecord } from "./core/runner";
import { RunnerControl } from "./core/runnerLifecycle";
import { RunnerStore } from "./guided/runnerStore";
import { buildSourceDraft, inspectSourceCA, sourceSecrets } from "./core/runnerSource";
import { assessmentActive, ensureAssessmentReadiness, refreshAssessment, startAssessment, retainFailedAssessment } from "./core/runnerAssessment";
import { runnerSourceHTML } from "./core/runnerSourceView";
import { refreshStorage, storageDraft, submitStorage } from "./core/runnerStorageLifecycle";
import { reportStorageNames, verifyTransferStorage } from "./core/runnerReportStorage";
import { canRetainRejectedReportExport, retainRejectedReportExport } from "./core/runnerReport";
import { escapeHTML } from "./core/report";
import { CSVManifest, CSVTransferCancelledError, inspectCSV } from "./guided/csvTransfer";
import { csvAssessmentReady, refreshCSVImport, startCSVImport } from "./core/runnerCSV";
import { csvFilesInFolder } from "./guided/csvSelection";
import { previewCosmosAccess, refreshCosmosAccess, submitCosmosAccess } from "./core/runnerCosmosAccess";
import { adoptCatalog, assertCatalogCurrent, catalogBinding, catalogConfiguration, catalogRecommendations, refreshCatalog, startCatalog } from "./core/runnerCatalog";
import { sourceCredential, forgetSourceCredential, invalidateStaleSourceCredential } from "./sourceCredentialPanel";
import { watchRetainedOperation } from "./runnerWatch";
import { transferApprovedReport } from "./runnerReportFlow";

export interface RunnerSourceServices {
  storagePrincipal(subscription: string): Promise<string>;
  reportCapability(record: RunnerRecord, operation: string, permission: "r" | "c"): Promise<string>;
  csvCapability(record: RunnerRecord, manifest: CSVManifest): Promise<string>;
  uploadCSV(record: RunnerRecord, path: string, manifest: CSVManifest, progress: (bytes: number) => void, signal?: AbortSignal): Promise<void>;
}

const hash = (value: unknown) => createHash("sha256").update(JSON.stringify(value ?? null)).digest("hex");

export function openRunnerSource(context: vscode.ExtensionContext, control: RunnerControl, store: RunnerStore, workflow: string, services?: RunnerSourceServices): void {
  const panel = vscode.window.createWebviewPanel("agefreighter.runnerSource", "AGEFreighter source assessment", vscode.ViewColumn.One,
    { enableScripts: true, retainContextWhenHidden: true, localResourceRoots: [] });
  panel.webview.html = runnerSourceHTML();
  let busy = false, disposed = false, reviewedHash: string | undefined;
  const post = (value: unknown) => disposed ? Promise.resolve(false) : panel.webview.postMessage(value);
  const postCatalog = async (record: RunnerRecord) => {
    const c = record.postgresCatalog;
    const transfer = record.reportTransfers?.find(x => x.operation === c?.operation);
    const recommendations = c?.phase === "finished" && c.reportSHA256 && c.reportBytes && transfer?.phase === "imported"
      ? catalogRecommendations(record, await store.readReport(workflow, { operation: c.operation, sha256: c.reportSHA256, bytes: c.reportBytes })) : undefined;
    await post({ kind: "catalog", catalog: c, transfer: transfer?.phase, recommendations,
      available: record.phase === "provisioned" && record.guestReady?.capabilities?.includes("postgresql-catalog-v1") === true,
      frozen: !!record.assessment || !!record.target || !!record.migration });
  };
  const initialize = async (record: RunnerRecord) => { await post({ kind: "init", type: record.input.source.type, location: record.input.source.location,
    files: record.sourceFiles?.map(({ id, name }) => ({ id, name })), form: record.sourceDraft?.form ?? (record.postgresCatalog ? { name: "graph-migration", namespace: "migration", ...record.postgresCatalog.configuration, mappings: [] } : undefined), assessment: record.assessment,
    sourceCA: record.sourceCA ? { name: record.sourceCA.name, bytes: record.sourceCA.bytes, sha256: record.sourceCA.sha256 } : undefined,
    cosmosAccess: record.cosmosAccess?.phase,
    storage: record.storageDeployment ? `${record.storageDeployment.phase}${record.storageDeployment.networkAccess ? ` — public network: ${record.storageDeployment.networkAccess} (provisioning is not transfer readiness)` : ""}` : undefined,
    transferEnabled: !!services, csvTransfers: record.csvTransfers, transfer: record.reportTransfers?.find(item => item.operation === record.assessment?.operation)?.phase,
    rejectedExportReview: canRetainRejectedReportExport(record),
    inventoryReady: record.guestReady?.capabilities?.includes(`${record.input.source.type}-inventory-v1`) === true,
    canStart: record.phase === "provisioned" && !!record.guestReady }); await postCatalog(record); };
  const listener = panel.webview.onDidReceiveMessage(async raw => {
    if (busy) return;
    let watch: "assessment" | "postgresCatalog" | undefined;
    busy = true; await post({ kind: "busy", value: true });
    try {
      const message = object(raw);
      switch (message.action) {
        case "ready": {
          const r = await store.read(workflow); await initialize(r);
          watch = r.assessment ? "assessment" : r.postgresCatalog ? "postgresCatalog" : undefined; break;
        }
        case "forgetCredential": await forgetSourceCredential(context, workflow); await post({ kind: "error", text: "Saved source credential removed. The next approved operation will ask for a password." }); break;
        case "catalogStart": {
          if (!vscode.workspace.isTrusted) throw new Error("Trust this workspace before catalog discovery.");
          const record = await store.read(workflow);
          if (record.postgresCatalog || record.assessment || record.target || record.migration) throw new Error("Retain existing operations; catalog discovery requires a fresh pre-assessment workflow.");
          if (!record.guestReady?.capabilities?.includes("postgresql-catalog-v1")) throw new Error("Install a reviewed catalog-capable Linux artifact before discovery.");
          const configuration = catalogConfiguration(record, message.form, message.schemas), binding = catalogBinding(record);
          const confirmed = await vscode.window.showWarningMessage("Read PostgreSQL schema metadata on this Linux runner?", { modal: true,
            detail: `${configuration.host}:${configuration.port}/${configuration.database} as ${configuration.username}\nSchemas: ${configuration.schemas.join(", ")}\nRunner: ${record.vmId}\nRead-only catalog transaction: 2 minutes, 64 tables, 128 columns and 64 key constraints per table; 4 MiB output. No row values, exact counts, target writes or migration. TLS verification is required. This is a single retained operation, never an automatic retry.` }, "Approve catalog read");
          if (confirmed !== "Approve catalog read" || disposed) break;
          let password = await sourceCredential(context, record, configuration, () => disposed);
          if (password === undefined || disposed) break;
          try {
            let pem: string | undefined;
            if (record.sourceCA) {
              const data = await readFile(record.sourceCA.path), checked = inspectSourceCA(record.sourceCA.path, record.sourceCA.name, data);
              if (checked.sha256 !== configuration.sourceCASHA256 || checked.bytes !== record.sourceCA.bytes) throw new Error("Source CA changed; review it again.");
              pem = data.toString("utf8");
            }
            const secrets = sourceSecrets("postgresql", configuration, password, pem);
            const next = await store.exclusive(workflow, async () => {
              if (!vscode.workspace.isTrusted || disposed) throw new Error("Catalog approval cancelled.");
              let current = await store.read(workflow);
              if (catalogBinding(current) !== binding || current.guestReady?.bootId !== record.guestReady?.bootId || hash(current.sourceDraft) !== hash(record.sourceDraft)) throw new Error("The reviewed source or runner changed in another window.");
              current = await ensureAssessmentReadiness(control, current, () => disposed || !vscode.workspace.isTrusted);
              if (!vscode.workspace.isTrusted || disposed) throw new Error("Catalog approval cancelled.");
              return startCatalog(control, current, configuration, secrets);
            });
            reviewedHash = undefined; await postCatalog(next); watch = "postgresCatalog";
          } finally { password = undefined; }
          break;
        }
        case "catalogRefresh": {
          if (!vscode.workspace.isTrusted) throw new Error("Trust this workspace before refreshing catalog evidence.");
          await postCatalog(await store.exclusive(workflow, async () => refreshCatalog(control, await store.read(workflow)))); watch = "postgresCatalog"; break;
        }
        case "catalogReport": {
          if (!services || !vscode.workspace.isTrusted) throw new Error("Trusted Azure access is required for catalog transfer.");
          const record = await store.read(workflow), c = assertCatalogCurrent(record);
          if (c.phase !== "finished" || !c.reportSHA256 || !c.reportBytes) throw new Error("Reconcile a successful catalog and its sealed manifest first.");
          if (record.reportTransfers?.find(x => x.operation === c.operation)?.phase === "imported") { await postCatalog(record); break; }
          if (record.storageDeployment?.phase !== "ready") throw new Error("Prepare transfer storage first.");
          const names = reportStorageNames(record);
          const confirmed = record.reportTransfers?.some(x => x.operation === c.operation) ? "Transfer catalog report" : await vscode.window.showWarningMessage("Transfer this sealed PostgreSQL catalog?", { modal: true,
            detail: `Operation ${c.operation}\n${c.reportBytes} bytes; SHA-256 ${c.reportSHA256}\nDestination: ${names.origin}/${names.container}\nSchema/table/key names may be sensitive. Stored privately on this computer; no AI upload, source re-read or mapping adoption.` }, "Transfer catalog report");
          if (confirmed !== "Transfer catalog report" || disposed) break;
          const next = await transferApprovedReport(control, store, workflow, { operation: c.operation, sha256: c.reportSHA256, bytes: c.reportBytes },
            (...args) => services.reportCapability(...args), () => disposed || !vscode.workspace.isTrusted, (r, text) => { catalogRecommendations(r, text); });
          await postCatalog(next); break;
        }
        case "catalogAdopt": {
          if (!vscode.workspace.isTrusted) throw new Error("Trust this workspace before adopting source mappings.");
          const record = await store.read(workflow), c = assertCatalogCurrent(record);
          if (!c.reportSHA256 || !c.reportBytes || record.reportTransfers?.find(x => x.operation === c.operation)?.phase !== "imported") throw new Error("Import the sealed catalog first.");
          const text = await store.readReport(workflow, { operation: c.operation, sha256: c.reportSHA256, bytes: c.reportBytes });
          const proposed = adoptCatalog(record, text, message.form, message.schemas, message.selected);
          const confirmed = await vscode.window.showWarningMessage("Add these selected PostgreSQL mappings?", { modal: true,
            detail: `Catalog ${c.operation}\nSelected: ${(message.selected as string[]).join(", ")}\n${proposed.sourceDraft!.form.mappings.length} total mappings. Existing manual mappings are preserved, never overwritten. Only identity properties are proposed; review other properties and relationship direction. Review source settings and run a new complete inventory before sizing.` }, "Adopt selected mappings");
          if (confirmed !== "Adopt selected mappings" || disposed) break;
          const next = await store.exclusive(workflow, async () => {
            if (!vscode.workspace.isTrusted || disposed) throw new Error("Mapping adoption cancelled.");
            const current = await store.read(workflow);
            if (hash(current.postgresCatalog) !== hash(c) || hash(current.sourceDraft) !== hash(record.sourceDraft)) throw new Error("Mappings or catalog changed in another window; nothing was overwritten.");
            const updated = adoptCatalog(current, text, message.form, message.schemas, message.selected);
            await store.write(updated); return updated;
          });
          reviewedHash = undefined; await post({ kind: "catalogAdopted", form: next.sourceDraft!.form, original: message.form }); break;
        }
        case "retainFailure": {
          if (!vscode.workspace.isTrusted) throw new Error("Trust this workspace before reconciling source operations.");
          const current = await store.read(workflow), operation = current.assessment?.operation ?? "";
          retainFailedAssessment(current, operation);
          const choice = await vscode.window.showWarningMessage("Retain failed source assessment and prepare a fresh attempt?", { modal: true, detail: `Operation ${operation}. Confirm you reviewed its guest failure evidence and the changes for a fresh attempt. Previous boot: ${current.assessment!.bootId}. Verified current boot: ${current.guestReady!.bootId}. Current runner: ${current.artifact.version}. The failed operation and files remain retained. This action performs no source reads, credential changes, or target writes. Review the source again and separately approve a new operation; the old operation is never resumed.` }, "Retain failure and review source");
          if (choice !== "Retain failure and review source" || disposed) break;
          const next = await store.exclusive(workflow, async () => {
            if (!vscode.workspace.isTrusted) throw new Error("Trust this workspace before reconciling source operations.");
            const latest = await store.read(workflow);
            if (latest.vmId !== current.vmId || latest.guestReady?.bootId !== current.guestReady?.bootId || latest.artifact.sha256 !== current.artifact.sha256 || latest.artifact.version !== current.artifact.version) throw new Error("The reviewed runner boot or installation changed; review the retained failure again.");
            const updated = retainFailedAssessment(latest, operation);
            await store.write(updated); return updated;
          });
          reviewedHash = undefined; await initialize(next); break;
        }
        case "cosmosAccess": {
          if (!vscode.workspace.isTrusted) throw new Error("Trust this workspace before reviewing or granting Cosmos data access.");
          let record = await store.read(workflow);
          if (record.input.source.type !== "cosmos-nosql" || assessmentActive(record)) throw new Error("Cosmos read access is unavailable for this workflow.");
          if (!record.cosmosAccess) record = await store.exclusive(workflow, async () => previewCosmosAccess(control, await store.read(workflow)));
          if (record.cosmosAccess?.phase === "previewed") {
            const binding = (r: RunnerRecord) => hash({ subscription: r.input.subscriptionId, vm: r.vmId, source: r.input.source, access: r.cosmosAccess });
            const reviewedGrant = binding(record);
            const confirmed = await vscode.window.showWarningMessage("Grant this Linux runner read-only Cosmos data access?", { modal: true,
              detail: `Principal: ${record.cosmosAccess.principalId}\nScope: ${record.cosmosAccess.scope}\nRole: Cosmos DB Built-in Data Reader\nOnly this new assignment is created. It does not expose the account, grant writes, use keys, start assessment or migrate data. An uncertain PUT is reconciled by GET and never replayed.` }, "Grant Data Reader");
            if (confirmed !== "Grant Data Reader" || disposed) { await initialize(record); break; }
            record = await store.exclusive(workflow, async () => {
              if (!vscode.workspace.isTrusted) throw new Error("Trust this workspace before granting Cosmos data access.");
              const current = await store.read(workflow);
              if (binding(current) !== reviewedGrant || assessmentActive(current)) throw new Error("The reviewed Cosmos grant or source operation changed; review it again. No grant was submitted.");
              return submitCosmosAccess(control, current);
            });
          } else record = await store.exclusive(workflow, async () => refreshCosmosAccess(control, await store.read(workflow)));
          await initialize(record); break;
        }
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
              await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: `Uploading ${file.name}`, cancellable: true }, async (progress, token) => {
                const abort = new AbortController();
                const cancellation = token.onCancellationRequested(() => abort.abort());
                if (token.isCancellationRequested) abort.abort();
                try {
                  await services.uploadCSV(current, file.path, manifest, bytes => progress.report({ message: `${Math.round(100 * bytes / manifest.bytes)}%` }), abort.signal);
                  // Cancellation at an acknowledged-commit boundary still leaves
                  // this file prepared, for an explicit HEAD/hash reconciliation.
                  if (abort.signal.aborted) throw new CSVTransferCancelledError();
                } finally { cancellation.dispose(); }
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
            detail: `${record.sourceFiles?.find(item => item.id === manifest.file)?.name}\n${manifest.bytes} bytes; SHA-256 ${manifest.sha256}\n${record.vmId}\n${manifest.rejectedImports?.length ? "A prior request was rejected by the guest decoder before execution; its evidence is retained. This approves a new corrected request.\n" : ""}Full hashing and the 80% disk gate are required. No source assessment or migration starts. Check fresh guest readiness first.` }, "Import and seal CSV");
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
        case "retainRejectedExport": {
          if (!services || !vscode.workspace.isTrusted) throw new Error("Trusted Azure access is required for export reconciliation.");
          const record = await store.read(workflow);
          if (!canRetainRejectedReportExport(record)) throw new Error("No eligible rejected report export.");
          const operation = record.assessment!.operation, commandId = record.guestCommand!.id;
          const confirmed = await vscode.window.showWarningMessage("Retain the rejected report export and prepare a fresh transfer?", { modal: true,
            detail: `Operation ${operation}\nReport ${record.assessment!.reportBytes} bytes; SHA-256 ${record.assessment!.reportSHA256}\nRequires an HTTP 409 rejection older than twenty minutes (original capability expired), a deallocated runner, absent ARM command and absent exact report blob. The old command and transfer are retained locally. This does not start a VM, export, source read or migration. Start the runner, verify fresh idle readiness and separately approve the transfer afterward.` }, "Retain rejected export");
          if (confirmed !== "Retain rejected export" || disposed) break;
          const next = await store.exclusive(workflow, async () => {
            if (!vscode.workspace.isTrusted || disposed) throw new Error("Trusted active source panel is required for export reconciliation.");
            const current = await store.read(workflow);
            if (current.guestCommand?.id !== commandId || current.assessment?.operation !== operation || hash(current) !== hash(record)) throw new Error("Export changed; review it again.");
            return retainRejectedReportExport(control, current, await services.reportCapability(current, operation, "r"));
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
            const confirmed = existing ? "Transfer verified report" : await vscode.window.showWarningMessage("Transfer and verify this assessment report?", { modal: true,
              detail: `${assessment.action}: ${assessment.operation}\n${manifest.bytes} bytes; SHA-256 ${manifest.sha256}\nDestination: ${names.origin}/${names.container}\nSource metadata/sample values may be sensitive. The report is retained privately on this computer and is not sent to an AI model. This action does not repeat source discovery or start migration.` }, "Transfer verified report");
            if (confirmed !== "Transfer verified report" || disposed) break;
            const next = await transferApprovedReport(control, store, workflow, manifest, (...args) => services.reportCapability(...args), () => disposed || !vscode.workspace.isTrusted);
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
        case "sourceCA": {
          const record = await store.read(workflow);
          if (!["neo4j", "postgresql"].includes(record.input.source.type) || assessmentActive(record)) throw new Error("Custom source CA selection is unavailable for this workflow.");
          const picked = await vscode.window.showOpenDialog({ canSelectMany: false, canSelectFiles: true, canSelectFolders: false, filters: { "PEM certificates": ["pem", "crt", "cer"] }, openLabel: "Select source CA bundle (no upload yet)" });
          if (!picked) break;
          const uri = picked[0]!;
          if (uri.scheme !== "file") throw new Error("Select a local source CA bundle.");
          const sourceCA = inspectSourceCA(uri.fsPath, basename(uri.fsPath), await readFile(uri.fsPath));
          const next = await store.exclusive(workflow, async () => {
            const current = await store.read(workflow);
            if (assessmentActive(current)) throw new Error("The source already has a retained operation.");
            const updated: RunnerRecord = { ...current, sourceCA };
            // Rebind persisted settings, but require a new in-window review.
            if (current.sourceDraft) updated.sourceDraft = buildSourceDraft(current.input.source,
              current.sourceDraft.form, workflow, current.sourceFiles, sourceCA);
            await store.write(updated); return updated;
          });
          reviewedHash = undefined;
          await post({ kind: "sourceCA", sourceCA: { name: next.sourceCA!.name,
            bytes: next.sourceCA!.bytes, sha256: next.sourceCA!.sha256 } });
          break;
        }
        case "review": {
          const next = await store.exclusive(workflow, async () => {
            const current = await store.read(workflow);
            if (assessmentActive(current) || current.target || current.migration) throw new Error("Retain the existing assessment configuration; it cannot be replaced here.");
            const sourceDraft = buildSourceDraft(current.input.source, message.form, workflow, current.sourceFiles, current.sourceCA);
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
          if (record.phase !== "provisioned" || !record.guestReady) throw new Error("Provision the runner and check Linux guest readiness before approving source reads.");
          if (!record.sourceDraft.canAssess || assessmentActive(record)) throw new Error("This source cannot start a new assessment here.");
          const confirmed = await vscode.window.showWarningMessage(`Run ${message.method === "profile" ? "a sampled profile" : record.input.source.type === "csv" ? "a complete CSV inventory (all mapped rows and before/after file hashes; up to 64 files / 10 GiB / 100 million rows)" : record.input.source.type === "neo4j" ? "an exact Neo4j count inventory" : "a complete mapped-record inventory (up to 100 million rows)"} from the Linux runner?`,
            { modal: true, detail: `${record.input.source.type} / ${record.sourceDraft.form.host} / ${record.sourceDraft.form.database}\nRunner: ${record.vmId}\n${record.sourceDraft.warnings.join("\n")}\nGuest limits: 30 minutes, 4 GiB, no swap. Keep the source unchanged. Closing VS Code will not stop the operation.` }, "Approve source reads");
          if (confirmed !== "Approve source reads" || disposed) break;
          if (!vscode.workspace.isTrusted) throw new Error("Trust the workspace before approving source reads.");
          let password: string | undefined;
          if (["neo4j", "postgresql"].includes(record.input.source.type)) {
            password = await sourceCredential(context, record, record.sourceDraft.form, () => disposed);
            if (password === undefined || disposed) break;
          }
          try {
            let sourceCAPEM: string | undefined;
            if (record.sourceCA) {
              const data = await readFile(record.sourceCA.path), checked = inspectSourceCA(record.sourceCA.path, record.sourceCA.name, data);
              if (checked.bytes !== record.sourceCA.bytes || checked.sha256 !== record.sourceCA.sha256 || record.sourceDraft.sourceCASHA256 !== checked.sha256) throw new Error("The selected source CA changed; select and review it again.");
              sourceCAPEM = data.toString("utf8");
            }
            const secrets = sourceSecrets(record.input.source.type, record.sourceDraft.form, password, sourceCAPEM);
            const next = await store.exclusive(workflow, async () => {
              const cancelled = () => disposed || !vscode.workspace.isTrusted;
              if (cancelled()) throw new Error("Source assessment cancelled or workspace trust changed; no source read was submitted.");
              let current = await store.read(workflow);
              if (hash(current.sourceDraft) !== reviewedHash) throw new Error("Source settings changed in another window; review them again.");
              current = await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: "Checking Linux readiness before source reads", cancellable: false },
                () => ensureAssessmentReadiness(control, current, cancelled));
              if (cancelled()) throw new Error("Source assessment cancelled or workspace trust changed; no source read was submitted.");
              return startAssessment(control, current, message.method as "profile" | "inventory", secrets, cancelled);
            });
            await post({ kind: "assessment", assessment: next.assessment }); watch = "assessment";
          } finally { password = undefined; }
          break;
        }
        case "refresh": {
          const next = await store.exclusive(workflow, async () => refreshAssessment(control, await store.read(workflow)));
          await post({ kind: "assessment", assessment: next.assessment }); watch = "assessment"; break;
        }
        default: throw new Error("Unsupported source form action.");
      }
    } catch (error) {
      await post({ kind: "error", text: error instanceof Error ? error.message : "Source assessment could not be completed. No automatic replay was attempted." });
    } finally { busy = false; await post({ kind: "busy", value: false }); }
    if (watch && !disposed) {
      const kind = watch;
      void watchRetainedOperation(control, store, workflow, kind, () => disposed, async r => {
        if (["failed", "interrupted"].includes(r[kind]?.phase ?? "")) await invalidateStaleSourceCredential(context, await store.read(workflow));
        if (kind === "postgresCatalog") await postCatalog(r); else await post({ kind: "assessment", assessment: r.assessment });
      }).catch(async () => { await post({ kind: "error", text: "Automatic status watch stopped. Retained work was not cancelled or replayed; use Refresh to reconcile." }); });
    }
  });
  panel.onDidDispose(() => { disposed = true; listener.dispose(); }, undefined, context.subscriptions);
}
