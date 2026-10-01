import { runnerHTML } from "./core/runnerView";
import * as vscode from "vscode";
import { createHash, generateKeyPairSync, randomUUID } from "node:crypto";
import { AzureSession } from "./guided/azure";
import { assertFreshPreview, assertPreviewableDraft, object, parseRunnerInput, previewHash, releaseArtifact, retainDraftSetup, RunnerRecord, runnerNames, runnerTemplate, sourceWorkflowDraft, runnerReleaseVersion } from "./core/runner";
import { preflightRunner, refreshRunner, RunnerControl, submitRunner, whatIfRunner } from "./core/runnerLifecycle";
import { RunnerLockedError, RunnerStore } from "./guided/runnerStore";
import { basename, join } from "node:path";
import { assertPlacementSelection, discoverComputeSubnets, placementCatalog } from "./core/runnerPlacement";
import { dispatchGuest, reconcileGuest } from "./core/runnerGuest";
import { openRunnerSource } from "./runnerSourcePanel";
import { developmentEnabled, prepareDevelopmentRunner, upgradeDevelopmentRunner } from "./developmentRunner";
import { reviewRunnerTarget, TargetReviewFeedback } from "./runnerTargetPanel";
import { continueRunnerExecution } from "./runnerExecutionPanel";
import { requirePanelWorkflow } from "./core/runnerPanelBinding";
import { archiveRunnerReadiness } from "./runnerReceiptsPanel";
import { manageReadinessRemoval } from "./runnerReceiptRemovalPanel";
import { reviewRunnerCrashLock } from "./runnerLockRecoveryPanel";
import { sourceCredential, forgetSourceCredential } from "./sourceCredentialPanel";
import { watchRunnerState, watchTargetState } from "./runnerWatch";
import { targetPending, targetStatusMessage } from "./core/runnerTarget";
import { sourceReportSummary } from "./core/runnerSourceReport";
import { executionSummary, parseExecutionAction } from "./core/runnerExecutionActions";


/** Guided execution has no dependency on the local process runner or workspace. */
export function registerRunnerMigration(context: vscode.ExtensionContext, output: vscode.LogOutputChannel): void {
  const azure = new AzureSession();
  let panel: vscode.WebviewPanel | undefined;
  const store = new RunnerStore(join(context.globalStorageUri.fsPath, "runner-v2"));
  context.subscriptions.push(vscode.commands.registerCommand("agefreighter.reviewRunnerCrashLock", async () => {
    try { await reviewRunnerCrashLock(store); }
    catch { await vscode.window.showErrorMessage("Interrupted runner lock could not be recovered. Evidence was preserved and no Azure operation was replayed."); }
  }));
  context.subscriptions.push(vscode.commands.registerCommand("agefreighter.sourceCredential", async () => {
    try {
      if(!vscode.workspace.isTrusted)throw new Error("Trust this workspace before managing source credentials.");
      const selected=await vscode.window.showQuickPick((await store.list()).filter(r=>["neo4j","postgresql"].includes(r.input.source.type)&&r.sourceDraft).map(r=>({label:r.id,description:`${r.input.source.type}: ${r.sourceDraft!.form.host} / ${r.sourceDraft!.form.username}`,id:r.id})),{placeHolder:"Prepare or forget a source credential — no Azure resources will be started"});
      if(!selected)return;
      const r=await store.read(selected.id);
      const action=await vscode.window.showQuickPick(["Prepare / reuse credential","Forget saved credential","Replace saved credential"],{placeHolder:"Encrypted, workflow-scoped source credential"});
      if(!action)return;
      if(action!=="Prepare / reuse credential")await forgetSourceCredential(context,r.id);
      if(action!=="Forget saved credential")await sourceCredential(context,r,r.sourceDraft!.form,undefined,true);
    } catch { await vscode.window.showErrorMessage("Source credential could not be prepared. No Azure operation was performed."); }
  }));
  context.subscriptions.push(vscode.commands.registerCommand("agefreighter.archiveRunnerReadiness", async () => {
    try { await archiveRunnerReadiness(store); }
    catch (error) { await vscode.window.showErrorMessage(error instanceof Error ? error.message : "Readiness evidence could not be archived. Nothing was removed."); }
  }));
  const catalog = async (subscription: string) => {
    const [groups, regions] = await Promise.all([
      azure.runnerList(subscription, `/subscriptions/${subscription}/resourcegroups?api-version=2021-04-01`),
      azure.locations(subscription)
    ]);
    return placementCatalog(groups, regions);
  };
  const sharedControl: RunnerControl = {
    request: (...args) => azure.runnerRequest(...args),
    list: (...args) => azure.runnerList(...args),
    sleep: ms => new Promise(resolve => setTimeout(resolve, ms)),
    persist: async record => {
      await store.write(record);
    }
  };
  context.subscriptions.push(vscode.commands.registerCommand("agefreighter.manageReadinessRemoval", async () => {
    try { await manageReadinessRemoval(sharedControl, store, azure); }
    catch (error) { await vscode.window.showErrorMessage(error instanceof Error ? error.message : "Readiness control removal needs review. No automatic retry was made."); }
  }));
  context.subscriptions.push(vscode.commands.registerCommand("agefreighter.prepareDevelopmentRunner", async () => {
    try { await azure.subscriptions(); await prepareDevelopmentRunner(sharedControl, store, azure, message => output.info(message)); }
    catch (error) {
      output.error("Development artifact preparation failed", error);
      await vscode.window.showErrorMessage(error instanceof Error ? error.message : "Development artifact preparation failed.");
    }
  }));
  context.subscriptions.push(vscode.commands.registerCommand("agefreighter.upgradeDevelopmentRunner", async () => {
    try { await azure.subscriptions(); await upgradeDevelopmentRunner(sharedControl, store, azure); }
    catch (error) { await vscode.window.showErrorMessage(error instanceof Error ? error.message : "Runner upgrade requires evidence review."); }
  }));
  const reviewTarget = async (workflow?: string, feedback?: TargetReviewFeedback) => {
    try { await azure.subscriptions(); await reviewRunnerTarget(context,sharedControl,store,azure,workflow,feedback); }
    catch (error) { output.error("Private target review failed", error); throw error; }
  };
  context.subscriptions.push(vscode.commands.registerCommand("agefreighter.reviewRunnerTarget", async (workflow?: string) => {
    try { await reviewTarget(workflow); }
    catch(error){ void vscode.window.showErrorMessage(error instanceof Error?error.message:"Target review could not complete. No automatic retry was made."); }
  }));
  context.subscriptions.push(vscode.commands.registerCommand("agefreighter.continueRunnerExecution", async () => {
    try { await azure.subscriptions(); await continueRunnerExecution(context,sharedControl,store,azure); }
    catch(error){ await vscode.window.showErrorMessage(error instanceof Error?error.message:"Execution requires evidence review; no automatic retry was made."); }
  }));
  context.subscriptions.push(azure, vscode.commands.registerCommand("agefreighter.newGuidedMigration", () => {
    if (panel) { panel.reveal(); return; }
    // Each new panel starts empty. An old panel's pending operation may still
    // persist evidence, but must never select a workflow in a later panel.
    let current: RunnerRecord | undefined;
    let busy = false;
    let disposed = false;
    let watching: number | undefined, monitorGeneration = 0;
    let pendingCSV: { id: string; name: string; path: string }[] = [];
    const control: RunnerControl = { ...sharedControl, persist: async record => {
      await sharedControl.persist(record);
      current = record;
    } };
    panel = vscode.window.createWebviewPanel("agefreighter.runnerMigration", "New AGEFreighter migration", vscode.ViewColumn.One,
      { enableScripts: true, retainContextWhenHidden: true, localResourceRoots: [] });
    const owner = panel;
    const post = (value: unknown) => disposed ? undefined : owner.webview.postMessage(value);
    const display = async (record: RunnerRecord) => {
      const a = record.assessment;
      const summary = a?.reportSHA256 && a.reportBytes && record.reportTransfers?.some(t => t.operation === a.operation && t.phase === "imported")
        ? sourceReportSummary(record, await store.readReport(record.id, { operation: a.operation, sha256: a.reportSHA256, bytes: a.reportBytes })) : undefined;
      return post({ kind: "record", record: {
      id: record.id, phase: record.phase, input: record.input, vmId: record.vmId,
      deploymentId: record.deploymentId, version: record.artifact.version, sha256: record.artifact.sha256,
      hourlyComputeUSD: record.hourlyComputeUSD, expiresAt: record.expiresAt, updatedAt: record.updatedAt,
      previewHash: record.previewHash, guestCommand: record.guestCommand, guestReady: record.guestReady,
      assessment: record.assessment, sourceReport: summary, targetPhase: record.target?.phase, migrationPhase: record.migration?.phase,
      targetMessage: record.target ? targetStatusMessage(record) : undefined,
      execution: executionSummary(record)
    } }); };
    owner.webview.html = runnerHTML(owner.webview.cspSource);
    owner.onDidDispose(() => { disposed = true; if (panel === owner) panel = undefined; });
    owner.onDidChangeViewState(async event => {
      if (!event.webviewPanel.visible || busy || !current || disposed) return;
      const workflow = current.id;
      try {
        const record = await store.read(workflow);
        if (!disposed && !busy && current?.id === workflow) { current = record; await display(record); }
      } catch (error) {
        await post({ kind: "error", text: error instanceof Error ? error.message : "The saved workflow could not be refreshed." });
      }
    }, undefined, context.subscriptions);
    owner.webview.onDidReceiveMessage(async raw => {
      if (object(raw).action === "selectionChanged" && !busy) {
        monitorGeneration++; current = undefined;
        await post({ kind: "progress", active: false, text: "Selection changed. Review the new source and runner placement; retained cloud work was not cancelled." }); return;
      }
      if (object(raw).action === "stopWatch") {
        monitorGeneration++; await post({ kind: "progress", active: false, text: "Automatic refresh stopped. Cloud resources and operations are unchanged." }); return;
      }
      if (busy || disposed) return;
      busy = true;
      let actionSucceeded = false;
      await post({ kind: "busy", value: true });
      try {
        const message = object(raw);
        if (["deploy", "refresh", "guestReady", "guestRefresh", "reviewTarget", "continueExecution", "executionAction"].includes(String(message.action))) {
          requirePanelWorkflow(current, message.workflow);
        }
        switch (message.action) {
          case "ready":
          case "accounts":
            await post({ kind: "subscriptions", values: await azure.subscriptions() });
            if (current) {
              await post({ kind: "restoreInput", input: current.input, files: current.sourceFiles?.map(file => file.name) ?? [] });
              await display(current);
            }
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
          case "selectSubnet": {
            const subscription = selection(message.subscription), region = selection(message.region);
            const subnets = await discoverComputeSubnets(control, subscription, region, typeof message.sourceId === "string" ? message.sourceId : undefined);
            if (!subnets.length) throw new Error("No available non-delegated compute subnets were found in this region. Create or request a suitable subnet, then choose again.");
            const selected = await vscode.window.showQuickPick(subnets.map(subnet => ({
              label: `${subnet.vnet} / ${subnet.name}`,
              description: `${subnet.prefixes.join(", ")} — ${subnet.resourceGroup}`,
              detail: subnet.containsSource ? "Contains the source VM. Source-subnet firewall rules may block another VM here; review before selecting."
                : subnet.attachedInterfaces ? `${subnet.attachedInterfaces} attached IP configurations. Review source connectivity.`
                : "No attached IP configurations. Suitable for review as a dedicated runner subnet; connectivity is not yet verified.",
              subnet
            })), { placeHolder: "Choose the runner subnet — not the future PostgreSQL delegated subnet", matchOnDescription: true, matchOnDetail: true });
            if (selected && !disposed) await post({ kind: "subnet", subscription, region, sourceId: message.sourceId, subnet: selected.subnet });
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
            if (picked) {
              monitorGeneration++;
              current = picked.record;
              await post({ kind: "restoreInput", input: current.input, files: current.sourceFiles?.map(file => file.name) ?? [] });
              await display(current);
            }
            break;
          }
          case "preview": {
            const input = parseRunnerInput(message.input);
            assertPlacementSelection(input, await catalog(input.subscriptionId));
            const draft = typeof message.draftId === "string" ? await store.read(message.draftId) : undefined;
            if (draft) assertPreviewableDraft(draft, input);
            const id = draft?.id ?? randomUUID();
            // UI patches retain the reviewed Linux release/protocol. Never fall
            // back to an arbitrary available release or mutable repository source.
            const version = runnerReleaseVersion;
            // Surface read-only placement failures even when a matching release
            // is unavailable. Artifact validation still precedes pricing,
            // what-if, persistence and every approved deployment.
            await preflightRunner(control, input);
            let artifact;
            if (draft?.developmentUpload?.phase === "ready") {
              if (!developmentEnabled() || !vscode.workspace.isTrusted || JSON.stringify(draft.input) !== JSON.stringify(input)) throw new Error("Development artifact opt-in or approved placement changed.");
              artifact = draft.developmentUpload.artifact;
            } else {
            const response = await fetch(`https://github.com/rioriost/agefreighter/releases/download/v${version}/checksums.txt`, { signal: AbortSignal.timeout(30_000) });
            if (!response.ok) throw new Error(`The matching AGEFreighter ${version} Linux release/checksums are not available. No Azure deployment was submitted.`);
            const checksums = await response.text();
            if (checksums.length > 1024 * 1024) throw new Error("Release checksum metadata is too large.");
            artifact = releaseArtifact(version, checksums);
            }
            const rates = (await azure.retailRates(input.region, [input.size])).filter(r => r.serviceName === "Virtual Machines");
            const ratesNow = rates.filter(r => Date.parse(r.effectiveStartDate) <= Date.now());
            if (ratesNow.length !== 1 || !Number.isFinite(ratesNow[0]!.hourlyUSD) || ratesNow[0]!.hourlyUSD <= 0) throw new Error("A unique current Linux compute price is unavailable. Deployment is blocked.");
            const hourlyComputeUSD = ratesNow[0]!.hourlyUSD;
            const template = runnerTemplate(id, input, artifact, bootstrapPublicKey());
            let record: RunnerRecord = { schemaVersion: 2, id, phase: "previewed", input, artifact, ...runnerNames(id, input), template,
              previewHash: previewHash(template, input, hourlyComputeUSD), expiresAt: new Date(Date.now() + 15 * 60_000).toISOString(),
              updatedAt: new Date().toISOString(), hourlyComputeUSD };
            if (!draft && input.source.type === "csv") record.sourceFiles = pendingCSV;
            await whatIfRunner(control, record);
            const reviewed = await store.exclusive(id, async () => {
              if (draft) {
                const latest = await store.read(id);
                if (JSON.stringify(latest) !== JSON.stringify(draft)) throw new Error("This draft changed during preview. Reconnect and review it again.");
                record = retainDraftSetup(record, latest);
              }
              await control.persist(record); return record;
            });
            await display(reviewed);
            break;
          }
          case "deploy": {
            if (!vscode.workspace.isTrusted) throw new Error("Trust this VS Code workspace before approving Azure deployment.");
            if (!current || message.hash !== current.previewHash || message.networkApproved !== true || message.costApproved !== true) throw new Error("Review a fresh preview, network prerequisites and additional charges first.");
            if (current.artifact.development && !developmentEnabled()) throw new Error("User-level development artifact opt-in was removed.");
            assertFreshPreview(current);
            // Renewal can preserve previewHash. Bind the entire original review,
            // including its expiry/revision, before yielding to native consent.
            const review = { workflow: current.id, hash: current.previewHash, expiresAt: current.expiresAt, revision: current.updatedAt,
              contentSHA256: createHash("sha256").update(JSON.stringify(current)).digest("hex") };
            const confirmed = await vscode.window.showWarningMessage(
              `Create the reviewed Linux discovery/migration VM ${current.vmId}?`,
              { modal: true, detail: `${current.input.region} / zone ${current.input.zone}; ${current.input.size}; compute estimate USD ${current.hourlyComputeUSD}/hour. Disk, network, NAT and other charges are additional. Resources remain until separately stopped/deleted. No source firewall, target database or migration is created. ${current.artifact.development ? `TEST build ${current.artifact.development.commit}; SHA-256 ${current.artifact.sha256}. Grants this VM identity Blob Reader only on this workflow container.` : "No role assignment is created."} Source assessment requires a separate approval after guest readiness.` }, "Create reviewed runner");
            if (confirmed !== "Create reviewed runner") break;
            current = await store.exclusive(review.workflow, async () => {
              const latest = await store.read(review.workflow);
              const assertApproved = () => {
                if (disposed || !vscode.workspace.isTrusted) throw new Error("The approving panel closed or workspace trust changed. Review deployment again.");
                if (latest.artifact.development && !developmentEnabled()) throw new Error("User-level development artifact opt-in was removed.");
                if (Date.now() >= Date.parse(review.expiresAt) || latest.id !== review.workflow || latest.previewHash !== review.hash ||
                    latest.expiresAt !== review.expiresAt || latest.updatedAt !== review.revision ||
                    createHash("sha256").update(JSON.stringify(latest)).digest("hex") !== review.contentSHA256) {
                  throw new Error("The approved preview changed or expired. Reconnect and review deployment again.");
                }
                assertFreshPreview(latest);
              };
              assertApproved();
              // Read-only preflight may await network I/O. Recheck consent at
              // the intent and request boundaries, not just after the modal.
              return submitRunner({ ...control,
                persist: async record => { if (record.phase === "deployment-submitted") assertApproved(); await control.persist(record); },
                request: (...args) => { if (args[2] && args[2] !== "GET") assertApproved(); return control.request(...args); }
              }, latest);
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
            if (current.guestCommand?.phase === "unknown") await post({ kind: "error", text: current.guestCommand.failure ?? "Guest submission was not confirmed; refresh status before retrying." });
            break;
          }
          case "configureSource": {
            if (current && message.workflow === current.id && JSON.stringify(parseRunnerInput(message.input)) !== JSON.stringify(current.input)) {
              throw new Error("Source or placement changed. Review a new draft or reconnect to the saved workflow.");
            }
            if (!current || message.workflow !== current.id) {
              const input = parseRunnerInput(message.input);
              const draft = sourceWorkflowDraft(randomUUID(), input);
              if (input.source.type === "csv") draft.sourceFiles = pendingCSV;
              await control.persist(draft);
            }
            await display(current!);
            openRunnerSource(context, control, store, current!.id, azure, reviewTarget);
            break;
          }
          case "guestRefresh": {
            if (!current) throw new Error("Select a retained workflow first.");
            const id = current.id;
            current = await store.exclusive(id, async () => (await reconcileGuest(control, await store.read(id))).record);
            await display(current);
            break;
          }
          case "reviewTarget": {
            if(!current)throw new Error("Select a retained migration workflow first.");
            const id=current.id, generation=monitorGeneration;
            const stopped=()=>disposed||generation!==monitorGeneration||!vscode.workspace.isTrusted;
            try{
              await reviewRunnerTarget(context,control,store,azure,id,{
                cancelled:stopped,
                progress:async(record,text,active)=>{
                  if(disposed||generation!==monitorGeneration)return;
                  current=record;await display(record);await post({kind:"progress",text,active});
                }
              });
            }finally{
              if(!disposed && current?.id===id){current=await store.read(id);await display(current);}
            }
            break;
          }
          case "executionAction":
          case "continueExecution": {
            if(!current)throw new Error("Select a retained migration workflow first.");
            const step=message.action==="executionAction"?parseExecutionAction(message.step):undefined;
            const id=current.id, generation=monitorGeneration;
            try{
              await continueRunnerExecution(context,control,store,azure,id,step,async(record,progress)=>{
                if(disposed||generation!==monitorGeneration)return;
                current=record;await display(record);
                if(progress)await post({kind:"progress",...progress});
              },()=>disposed||generation!==monitorGeneration);
            }finally{
              if(!disposed && current?.id===id){current=await store.read(id);await display(current);}
            }
            break;
          }
          default: throw new Error("Unsupported guided migration operation.");
        }
        actionSucceeded = true;
      } catch (error) {
        await post({ kind: "error", text: error instanceof Error ? error.message : "The operation could not be completed. No automatic retry was made." });
      } finally {
        busy = false;
        await post({ kind: "busy", value: false });
      }
      const monitorTarget = !!current && targetPending(current) && ["restore", "refresh", "ready", "accounts"].includes(String(object(raw).action));
      if (actionSucceeded && !disposed && watching !== monitorGeneration && current && (monitorTarget || ["deployment-submitted", "unknown"].includes(current.phase) ||
          current.guestCommand?.action === "ready" && ["submitted", "unknown"].includes(current.guestCommand.phase))) {
        const workflow = current.id, generation = monitorGeneration;
        watching = generation;
        const attached = () => !disposed && current?.id === workflow && generation === monitorGeneration;
        const stopped = () => !attached() || !vscode.workspace.isTrusted;
        await post({ kind: "progress", active: true, text: monitorTarget ? targetStatusMessage(current) : "Automatically refreshing Azure deployment / Linux readiness. No operation will be resubmitted." });
        const watch = monitorTarget ? watchTargetState : watchRunnerState;
        void watch(sharedControl, store, workflow, stopped, async record => {
          if (stopped()) return;
          current = record; await display(record);
          await post({ kind: "progress", active: monitorTarget ? targetPending(record) : true, text: monitorTarget ? targetStatusMessage(record) : `Last checked ${new Date().toLocaleTimeString()}: runner ${record.phase}; ${record.guestCommand?.action ?? "deployment"} ${record.guestCommand?.phase ?? ""}.` });
        }).then(async () => {
          if (attached()) await post({ kind: "progress", active: false, text: !vscode.workspace.isTrusted
            ? "Workspace trust revoked. Monitoring stopped; Azure resources and operations are unchanged."
            : monitorTarget
            ? targetStatusMessage(current!)+(targetPending(current!) ? " Monitoring stopped or reached its time limit. Use Review / reconcile private target to resume; Azure work was not cancelled." : "")
            : "Automatic refresh finished or reached its time limit. Review the current step below; Refresh remains available." });
        }).catch(async error => {
          if (attached()) await post({ kind: "error", text: error instanceof Error ? error.message : "Automatic refresh stopped. Use Refresh to reconcile retained evidence." });
        }).finally(() => { if (watching === generation) watching = undefined; });
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
