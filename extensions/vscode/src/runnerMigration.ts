import { runnerHTML } from "./core/runnerView";
import * as vscode from "vscode";
import { generateKeyPairSync, randomUUID } from "node:crypto";
import { AzureSession } from "./guided/azure";
import { object, parseRunnerInput, previewHash, releaseArtifact, RunnerRecord, runnerNames, runnerTemplate, sourceWorkflowDraft } from "./core/runner";
import { preflightRunner, refreshRunner, RunnerControl, submitRunner, whatIfRunner } from "./core/runnerLifecycle";
import { RunnerLockedError, RunnerStore } from "./guided/runnerStore";
import { basename, join } from "node:path";
import { assertPlacementSelection, placementCatalog } from "./core/runnerPlacement";
import { dispatchGuest, reconcileGuest } from "./core/runnerGuest";
import { openRunnerSource } from "./runnerSourcePanel";


/** Guided execution has no dependency on the local process runner or workspace. */
export function registerRunnerMigration(context: vscode.ExtensionContext): void {
  const azure = new AzureSession();
  let panel: vscode.WebviewPanel | undefined;
  let current: RunnerRecord | undefined;
  let busy = false;
  let pendingCSV: { id: string; name: string; path: string }[] = [];
  const store = new RunnerStore(join(context.globalStorageUri.fsPath, "runner-v2"));
  const catalog = async (subscription: string) => {
    const [groups, regions] = await Promise.all([
      azure.runnerList(subscription, `/subscriptions/${subscription}/resourcegroups?api-version=2021-04-01`),
      azure.locations(subscription)
    ]);
    return placementCatalog(groups, regions);
  };
  const control: RunnerControl = {
    request: (...args) => azure.runnerRequest(...args),
    list: (...args) => azure.runnerList(...args),
    sleep: ms => new Promise(resolve => setTimeout(resolve, ms)),
    persist: async record => {
      await store.write(record);
      current = record;
    }
  };
  const post = (value: unknown) => panel?.webview.postMessage(value);
  const display = (record: RunnerRecord) => post({ kind: "record", record: {
    id: record.id, phase: record.phase, input: record.input, vmId: record.vmId,
    deploymentId: record.deploymentId, version: record.artifact.version, sha256: record.artifact.sha256,
    hourlyComputeUSD: record.hourlyComputeUSD, expiresAt: record.expiresAt, updatedAt: record.updatedAt,
    previewHash: record.previewHash, guestCommand: record.guestCommand, guestReady: record.guestReady
  } });
  context.subscriptions.push(azure, vscode.commands.registerCommand("agefreighter.newGuidedMigration", () => {
    if (panel) { panel.reveal(); return; }
    panel = vscode.window.createWebviewPanel("agefreighter.runnerMigration", "New AGEFreighter migration", vscode.ViewColumn.One,
      { enableScripts: true, retainContextWhenHidden: true, localResourceRoots: [] });
    panel.webview.html = runnerHTML(panel.webview.cspSource);
    panel.onDidDispose(() => { panel = undefined; });
    panel.webview.onDidReceiveMessage(async raw => {
      if (busy) return;
      busy = true;
      await post({ kind: "busy", value: true });
      try {
        const message = object(raw);
        switch (message.action) {
          case "ready":
          case "accounts":
            await post({ kind: "subscriptions", values: await azure.subscriptions() });
            if (current) await display(current);
            break;
          case "groups": {
            const subscription = selection(message.subscription);
            await post({ kind: "groups", subscription, values: (await azure.runnerList(subscription,
              `/subscriptions/${subscription}/resourcegroups?api-version=2021-04-01`)).map(item => ({ name: object(item).name })) });
            break;
          }
          case "placementOptions": {
            const subscription = selection(message.subscription);
            if (message.scope !== "runner" && message.scope !== "both") throw new Error("Invalid placement-list scope.");
            await post({ kind: "placementOptions", subscription, scope: message.scope, catalog: await catalog(subscription) });
            break;
          }
          case "sources": {
            const subscription = selection(message.subscription);
            const group = selection(message.group);
            const type = message.type;
            const allowed = type === "cosmos-nosql" ? ["microsoft.documentdb/databaseaccounts"] : type === "postgresql"
              ? ["microsoft.compute/virtualmachines", "microsoft.dbforpostgresql/flexibleservers"] : ["microsoft.compute/virtualmachines"];
            const values = (await azure.runnerList(subscription, `/subscriptions/${subscription}/resourceGroups/${group}/resources?api-version=2021-04-01`))
              .map(object).filter(r => allowed.includes(String(r.type).toLowerCase())).map(r => ({ id: r.id, name: r.name, region: r.location,
                zone: Array.isArray(r.zones) && r.zones.length === 1 ? r.zones[0] : "", type: r.type }));
            await post({ kind: "sources", subscription, group, type, values });
            break;
          }
          case "csv": {
            const selected = await vscode.window.showOpenDialog({ canSelectMany: true, canSelectFiles: true, canSelectFolders: false,
              openLabel: "Select local CSV files (no upload)", filters: { CSV: ["csv"] } });
            if (selected) {
              if (selected.length > 64 || selected.some(uri => uri.scheme !== "file")) throw new Error("Select at most 64 local CSV files.");
              pendingCSV = selected.map(uri => ({ id: randomUUID(), name: basename(uri.fsPath), path: uri.fsPath }));
              await post({ kind: "csv", files: pendingCSV.map(file => file.name) });
            }
            break;
          }
          case "restore": {
            const picked = await vscode.window.showQuickPick((await store.list()).map(record => ({ label: `${record.input.source.type} — ${record.phase}`,
              description: record.id, record })), { placeHolder: "Reconnect to a retained runner workflow (no replay)" });
            if (picked) { current = picked.record; await display(current); }
            break;
          }
          case "preview": {
            const input = parseRunnerInput(message.input);
            assertPlacementSelection(input, await catalog(input.subscriptionId));
            const draft = typeof message.draftId === "string" ? await store.read(message.draftId) : undefined;
            if (draft && (draft.phase !== "draft" || JSON.stringify(draft.input.source) !== JSON.stringify(input.source))) throw new Error("Source selection changed. Reopen the matching draft or start a separate source configuration.");
            const id = draft?.id ?? randomUUID();
            // Matching released software is a mandatory prerequisite. Never install a
            // stale version or execute mutable repository source on a customer VM.
            const version = String(context.extension.packageJSON.version);
            if (!/^2\.4\.\d+(?:-[a-z0-9.]+)?$/.test(version)) throw new Error("Runner release version is invalid.");
            const response = await fetch(`https://github.com/rioriost/agefreighter/releases/download/v${version}/checksums.txt`, { signal: AbortSignal.timeout(30_000) });
            if (!response.ok) throw new Error(`The matching AGEFreighter ${version} Linux release/checksums are not available. No Azure deployment was submitted.`);
            const checksums = await response.text();
            if (checksums.length > 1024 * 1024) throw new Error("Release checksum metadata is too large.");
            const artifact = releaseArtifact(version, checksums);
            await preflightRunner(control, input);
            const rates = (await azure.retailRates(input.region, [input.size])).filter(r => r.serviceName === "Virtual Machines");
            const ratesNow = rates.filter(r => Date.parse(r.effectiveStartDate) <= Date.now());
            if (ratesNow.length !== 1 || !Number.isFinite(ratesNow[0]!.hourlyUSD) || ratesNow[0]!.hourlyUSD <= 0) throw new Error("A unique current Linux compute price is unavailable. Deployment is blocked.");
            const hourlyComputeUSD = ratesNow[0]!.hourlyUSD;
            const template = runnerTemplate(id, input, artifact, bootstrapPublicKey());
            const record: RunnerRecord = { schemaVersion: 2, id, phase: "previewed", input, artifact, ...runnerNames(id, input), template,
              previewHash: previewHash(template, input, hourlyComputeUSD), expiresAt: new Date(Date.now() + 15 * 60_000).toISOString(),
              updatedAt: new Date().toISOString(), hourlyComputeUSD };
            if (!draft && input.source.type === "csv") record.sourceFiles = pendingCSV;
            await whatIfRunner(control, record);
            const reviewed = await store.exclusive(id, async () => {
              if (draft) {
                const latest = await store.read(id);
                if (latest.phase !== "draft" || JSON.stringify(latest.input.source) !== JSON.stringify(input.source)) throw new Error("This draft changed in another window. Review it again.");
                record.sourceDraft = latest.sourceDraft; record.sourceFiles = latest.sourceFiles;
              }
              await control.persist(record); return record;
            });
            await display(reviewed);
            break;
          }
          case "deploy": {
            if (!vscode.workspace.isTrusted) throw new Error("Trust this VS Code workspace before approving Azure deployment.");
            if (!current || message.hash !== current.previewHash || message.networkApproved !== true || message.costApproved !== true) throw new Error("Review a fresh preview, network prerequisites and additional charges first.");
            const confirmed = await vscode.window.showWarningMessage(
              `Create the reviewed Linux discovery/migration VM ${current.vmId}?`,
              { modal: true, detail: `${current.input.region} / zone ${current.input.zone}; ${current.input.size}; compute estimate USD ${current.hourlyComputeUSD}/hour. Disk, network, NAT and other charges are additional. Resources remain until separately stopped/deleted. No source firewall, role assignment, target database or migration is created. Remote assessment is not available in this preview build.` }, "Create reviewed runner");
            if (confirmed !== "Create reviewed runner") break;
            const workflowId = current.id;
            current = await store.exclusive(workflowId, async () => {
              // A different extension window may have submitted while the modal
              // was open. Re-read the durable record under its exclusive lock.
              return submitRunner(control, await store.read(workflowId));
            });
            await display(current);
            break;
          }
          case "refresh":
            if (!current) throw new Error("Select a retained workflow first.");
            try { current = await store.exclusive(current.id, async () => refreshRunner(control, await store.read(current!.id))); }
            catch (error) {
              if (!(error instanceof RunnerLockedError)) throw error;
              // Even a retained crash lock must not prevent read-only diagnosis.
              current = await refreshRunner({ ...control, persist: async () => {} }, await store.read(current.id));
            }
            await display(current);
            break;
          case "guestReady": {
            if (!vscode.workspace.isTrusted) throw new Error("Trust this workspace before executing guest controls.");
            if (!current) throw new Error("Select a provisioned workflow first.");
            const id = current.id;
            current = await store.exclusive(id, async () => dispatchGuest(control, await store.read(id), { version: 1, workflow: id, operation: randomUUID(), action: "ready" }));
            await display(current);
            break;
          }
          case "configureSource": {
            if (!current || message.workflow !== current.id) {
              const input = parseRunnerInput(message.input);
              const draft = sourceWorkflowDraft(randomUUID(), input);
              if (input.source.type === "csv") draft.sourceFiles = pendingCSV;
              await control.persist(draft);
            }
            await display(current!);
            openRunnerSource(context, control, store, current!.id);
            break;
          }
          case "guestRefresh": {
            if (!current) throw new Error("Select a retained workflow first.");
            const id = current.id;
            current = await store.exclusive(id, async () => (await reconcileGuest(control, await store.read(id))).record);
            await display(current);
            break;
          }
          default: throw new Error("Unsupported guided migration operation.");
        }
      } catch (error) {
        await post({ kind: "error", text: error instanceof Error ? error.message : "The operation could not be completed. No automatic retry was made." });
      } finally {
        busy = false;
        await post({ kind: "busy", value: false });
      }
    }, undefined, context.subscriptions);
  }));
}

function selection(value: unknown): string {
  if (typeof value !== "string" || !/^[\w().-]{1,90}$/.test(value)) throw new Error("Select a subscription and resource group.");
  return value;
}

function bootstrapPublicKey(): string {
  // No inbound SSH exists. The unused private key is not persisted; control and
  // recovery use Azure VM agent permissions, not an extension-managed SSH key.
  const publicKey = generateKeyPairSync("ed25519").publicKey.export({ type: "spki", format: "der" }).subarray(-32);
  const parts = [Buffer.from("ssh-ed25519"), publicKey].flatMap(part => {
    const size = Buffer.alloc(4); size.writeUInt32BE(part.length); return [size, part];
  });
  return `ssh-ed25519 ${Buffer.concat(parts).toString("base64")}`;
}
