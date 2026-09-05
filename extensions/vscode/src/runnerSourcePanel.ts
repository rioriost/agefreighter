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

const hash = (value: unknown) => createHash("sha256").update(JSON.stringify(value)).digest("hex");

export function openRunnerSource(context: vscode.ExtensionContext, control: RunnerControl, store: RunnerStore, workflow: string): void {
  const panel = vscode.window.createWebviewPanel("agefreighter.runnerSource", "AGEFreighter source assessment", vscode.ViewColumn.One,
    { enableScripts: true, retainContextWhenHidden: true, localResourceRoots: [] });
  panel.webview.html = runnerSourceHTML();
  let busy = false, disposed = false, reviewedHash: string | undefined;
  const post = (value: unknown) => disposed ? Promise.resolve(false) : panel.webview.postMessage(value);
  const initialize = (record: RunnerRecord) => post({ kind: "init", type: record.input.source.type, location: record.input.source.location,
    files: record.sourceFiles?.map(({ id, name }) => ({ id, name })), form: record.sourceDraft?.form, assessment: record.assessment,
    canStart: record.phase === "provisioned" && !!record.guestReady && Date.now() - Date.parse(record.guestReady.checkedAt) <= 300000 });
  const listener = panel.webview.onDidReceiveMessage(async raw => {
    if (busy) return;
    busy = true; await post({ kind: "busy", value: true });
    try {
      const message = object(raw);
      switch (message.action) {
        case "ready": await initialize(await store.read(workflow)); break;
        case "files": {
          const record = await store.read(workflow);
          if (record.input.source.type !== "csv" || assessmentActive(record)) throw new Error("CSV selection is unavailable for this workflow.");
          const picked = await vscode.window.showOpenDialog({ canSelectMany: true, canSelectFiles: true, canSelectFolders: false, filters: { CSV: ["csv"] }, openLabel: "Select files for source mappings (no upload)" });
          if (!picked) break;
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
            const next = { ...current, sourceDraft }; await store.write(next); return next;
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
