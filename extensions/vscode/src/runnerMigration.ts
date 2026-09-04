import { runnerHTML } from "./core/runnerView";
import * as vscode from "vscode";
import { generateKeyPairSync, randomUUID } from "node:crypto";
import { AzureSession } from "./guided/azure";
import { object, parseRunnerInput, previewHash, releaseArtifact, RunnerRecord, runnerNames, runnerTemplate } from "./core/runner";
import { preflightRunner, refreshRunner, RunnerControl, submitRunner, whatIfRunner } from "./core/runnerLifecycle";
import { RunnerLockedError, RunnerStore } from "./guided/runnerStore";
import { join } from "node:path";


/** Guided execution has no dependency on the local process runner or workspace. */
export function registerRunnerMigration(context: vscode.ExtensionContext): void {
  const azure = new AzureSession();
  let panel: vscode.WebviewPanel | undefined;
  let current: RunnerRecord | undefined;
  let busy = false;
  const store = new RunnerStore(join(context.globalStorageUri.fsPath, "runner-v2"));
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
    previewHash: record.previewHash
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
            if (selected) await post({ kind: "csv", files: selected.map(uri => uri.fsPath) });
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
            const id = randomUUID();
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
            await whatIfRunner(control, record);
            await control.persist(record);
            await display(record);
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
