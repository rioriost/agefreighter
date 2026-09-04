import { randomBytes, randomUUID } from "node:crypto";
import * as vscode from "vscode";
import {
  buildNeo4jDraftYAML,
  combineCapacityAndInventory,
  createGuidedState,
  extractCapacityEvidence,
  extractInventoryEvidence,
  GuidedState,
  Neo4jDraftInput,
  normalizeNeo4jInput
} from "./core/guided";
import { runJSON } from "./core/process";
import { redactText } from "./core/security";
import { AzureSession } from "./guided/azure";
import { GuidedStorage } from "./guided/storage";
import { JobsProvider } from "./jobsView";

interface GuidedMessage {
  type: string;
  source?: unknown;
  password?: unknown;
  placement?: unknown;
}

export function registerGuidedMigration(
  context: vscode.ExtensionContext,
  jobs: JobsProvider,
  output: vscode.OutputChannel
): void {
  const azure = new AzureSession();
  context.subscriptions.push(
    azure,
    vscode.commands.registerCommand("agefreighter.newGuidedMigration", async () => {
      try {
        if (!vscode.workspace.isTrusted) {
          void vscode.window.showWarningMessage("Trust this workspace before creating a guided migration.");
          return;
        }
        const workspace = await chooseWorkspace();
        if (!workspace) {
          return;
        }
        openGuidedPanel(context, workspace, azure, jobs, output);
      } catch (error) {
        showGuidedError(output, error);
      }
    })
  );
}

function openGuidedPanel(
  context: vscode.ExtensionContext,
  workspace: vscode.WorkspaceFolder,
  azure: AzureSession,
  jobs: JobsProvider,
  output: vscode.OutputChannel
): void {
  const panel = vscode.window.createWebviewPanel(
    "agefreighter.guidedMigration",
    "New AGEFreighter migration",
    vscode.ViewColumn.Active,
    { enableScripts: true, retainContextWhenHidden: true }
  );
  panel.webview.html = guidedHTML(panel.webview);
  const storage = new GuidedStorage(context, workspace);
  let busy = false;
  panel.webview.onDidReceiveMessage(async (raw: unknown) => {
    const message = parseMessage(raw);
    if (!message || busy) {
      return;
    }
    try {
      switch (message.type) {
        case "ready":
          await sendSubscriptions(panel, azure);
          break;
        case "profile":
          busy = true;
          await panel.webview.postMessage({ type: "busy", value: true, message: "Connecting and profiling the source…" });
          await profileSource(panel, workspace, storage, azure, jobs, output, message);
          break;
      }
    } catch (error) {
      const detail = safeError(error);
      output.appendLine(`${new Date().toISOString()} guided migration failed: ${detail}`);
      await panel.webview.postMessage({ type: "error", message: detail });
    } finally {
      if (message.type === "profile") {
        busy = false;
        await panel.webview.postMessage({ type: "busy", value: false });
      }
    }
  });
}

async function profileSource(
  panel: vscode.WebviewPanel,
  workspace: vscode.WorkspaceFolder,
  storage: GuidedStorage,
  azure: AzureSession,
  jobs: JobsProvider,
  output: vscode.OutputChannel,
  message: GuidedMessage
): Promise<void> {
  const input = normalizeNeo4jInput(parseNeo4jInput(message.source));
  if (typeof message.password !== "string" || message.password.length === 0) {
    throw new Error("Enter the Neo4j password.");
  }
  const placementInput = parsePlacement(message.placement);
  const id = randomUUID();
  const state = createGuidedState(id, input, placementInput.kind);
  state.source.subscriptionId = placementInput.subscriptionId;
  if (placementInput.kind === "azure") {
    if (!placementInput.subscriptionId || !placementInput.resourceId) {
      throw new Error("Select a subscription and enter the source Azure resource ID.");
    }
    const placement = await azure.placement(placementInput.subscriptionId, placementInput.resourceId);
    state.source.resourceId = placement.resourceId;
    state.source.resolvedLocation = placement.location;
    state.source.resolvedZone = placement.zone;
    state.source.placementConfidence = "verified";
  } else {
    if (!placementInput.declaredLocation) {
      throw new Error("Enter the physical source location so Azure regions can be recommended.");
    }
    state.source.declaredLocation = placementInput.declaredLocation;
    state.source.placementConfidence = "declared";
  }
  const sourceSecretPath = await storage.writeSecret(id, "source-password", message.password);
  const draft = buildNeo4jDraftYAML(input, sourceSecretPath, storage.targetSecretPath(id));
  const draftURI = await storage.writeText(id, "draft.yaml", draft);
  state.jobPath = draftURI.fsPath;
  await storage.writeState(state);

  const options = processOptions(workspace.uri);
  const binary = vscode.workspace.getConfiguration("agefreighter", workspace.uri)
    .get<string>("binaryPath", "agefreighter").trim();
  await runJSON(binary, ["validate", "--format", "json", draftURI.fsPath], options);
  const inventory = await runJSON(binary, ["inventory", "--format", "json", draftURI.fsPath], options);
  const profile = await runJSON(binary, ["profile", "--format", "json", draftURI.fsPath], options);
  const outcome = profileOutcome(profile);
  const inventoryURI = await storage.writeText(id, "source-inventory.json", `${JSON.stringify(inventory, null, 2)}\n`);
  const profileURI = await storage.writeText(id, "source-profile.json", `${JSON.stringify(profile, null, 2)}\n`);
  const nextState: GuidedState = {
    ...state,
    revision: state.revision + 1,
    phase: "profiled",
    updatedAt: new Date().toISOString(),
    profile: {
      outcome,
      evidencePath: profileURI.fsPath,
      inventoryEvidencePath: inventoryURI.fsPath,
      generatedAt: profileGeneratedAt(profile)
    }
  };
  await storage.writeState(nextState);
  jobs.refresh();
  const capacity = combineCapacityAndInventory(
    extractCapacityEvidence(profile),
    extractInventoryEvidence(inventory)
  );
  output.appendLine(`${new Date().toISOString()} guided source profile completed for workflow ${id}`);
  await panel.webview.postMessage({
    type: "profileComplete",
    workflowId: id,
    outcome,
    placement: {
      location: nextState.source.resolvedLocation ?? nextState.source.declaredLocation,
      zone: nextState.source.resolvedZone,
      confidence: nextState.source.placementConfidence
    },
    capacity: {
      method: capacity.method,
      targetRows: capacity.targetRows?.toString(),
      targetRowsLowerBound: capacity.targetRowsLowerBound,
      recommendedStorageLow: capacity.recommendedStorageLow?.toString(),
      recommendedStorageHigh: capacity.recommendedStorageHigh?.toString(),
      deployable: capacity.deployable,
      reason: capacity.reason
    },
    inventory: {
      vertices: extractInventoryEvidence(inventory).vertices.toString(),
      edges: extractInventoryEvidence(inventory).edges.toString()
    },
    evidencePath: profileURI.fsPath
  });
}

async function sendSubscriptions(panel: vscode.WebviewPanel, azure: AzureSession): Promise<void> {
  try {
    const subscriptions = await azure.subscriptions();
    await panel.webview.postMessage({ type: "subscriptions", subscriptions });
  } catch (error) {
    await panel.webview.postMessage({
      type: "azureSignedOut",
      message: safeError(error)
    });
  }
}

async function chooseWorkspace(): Promise<vscode.WorkspaceFolder | undefined> {
  const folders = vscode.workspace.workspaceFolders ?? [];
  if (folders.length === 0) {
    void vscode.window.showInformationMessage("Open a local folder before creating a guided migration.");
    return undefined;
  }
  if (folders.length === 1) {
    return folders[0];
  }
  const selected = await vscode.window.showQuickPick(
    folders.map((folder) => ({ label: folder.name, description: folder.uri.fsPath, folder })),
    { placeHolder: "Select the workspace for migration evidence" }
  );
  return selected?.folder;
}

function parseMessage(value: unknown): GuidedMessage | undefined {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }
  const message = value as Record<string, unknown>;
  return typeof message.type === "string" ? {
    type: message.type,
    source: message.source,
    password: message.password,
    placement: message.placement
  } : undefined;
}

function parseNeo4jInput(value: unknown): Neo4jDraftInput {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("The source form is invalid.");
  }
  const item = value as Record<string, unknown>;
  return {
    name: stringField(item, "name"),
    host: stringField(item, "host"),
    port: numberField(item, "port"),
    encrypted: item.encrypted === true,
    database: stringField(item, "database"),
    sourceId: stringField(item, "sourceId"),
    namespace: stringField(item, "namespace"),
    username: stringField(item, "username"),
    vertexKeyProperty: stringField(item, "vertexKeyProperty"),
    edgeKeyProperty: stringField(item, "edgeKeyProperty")
  };
}

function parsePlacement(value: unknown): {
  kind: "azure" | "on-premises";
  subscriptionId?: string;
  resourceId?: string;
  declaredLocation?: string;
} {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("The source placement is invalid.");
  }
  const item = value as Record<string, unknown>;
  if (item.kind !== "azure" && item.kind !== "on-premises") {
    throw new Error("Select whether the source is in Azure or on-premises.");
  }
  return {
    kind: item.kind,
    subscriptionId: optionalString(item.subscriptionId),
    resourceId: optionalString(item.resourceId),
    declaredLocation: optionalString(item.declaredLocation)
  };
}

function stringField(value: Record<string, unknown>, name: string): string {
  if (typeof value[name] !== "string") {
    throw new Error(`The ${name} field is required.`);
  }
  return value[name];
}

function numberField(value: Record<string, unknown>, name: string): number {
  if (typeof value[name] !== "number") {
    throw new Error(`The ${name} field must be a number.`);
  }
  return value[name];
}

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function profileOutcome(value: unknown): "pass" | "fail" | "incomplete" {
  const outcome = value !== null && typeof value === "object" ? (value as Record<string, unknown>).outcome : undefined;
  return outcome === "pass" || outcome === "fail" ? outcome : "incomplete";
}

function profileGeneratedAt(value: unknown): string | undefined {
  const generatedAt = value !== null && typeof value === "object"
    ? (value as Record<string, unknown>).generatedAt
    : undefined;
  return typeof generatedAt === "string" ? generatedAt : undefined;
}

function processOptions(scope: vscode.Uri) {
  const configuration = vscode.workspace.getConfiguration("agefreighter", scope);
  const seconds = configuration.get<number>("readTimeoutSeconds", 120);
  const bytes = configuration.get<number>("maxOutputBytes", 4 * 1024 * 1024);
  return {
    cwd: vscode.workspace.getWorkspaceFolder(scope)?.uri.fsPath ?? process.cwd(),
    timeoutMs: Math.max(5, Math.min(1800, seconds)) * 1000,
    maxOutputBytes: Math.max(64 * 1024, Math.min(16 * 1024 * 1024, bytes)),
    env: { ...process.env }
  };
}

function safeError(error: unknown): string {
  return redactText(error instanceof Error ? error.message : String(error));
}

function showGuidedError(output: vscode.OutputChannel, error: unknown): void {
  const detail = safeError(error);
  output.appendLine(`${new Date().toISOString()} guided migration failed: ${detail}`);
  void vscode.window.showErrorMessage(`AGEFreighter guided migration failed: ${detail}`);
}

function guidedHTML(webview: vscode.Webview): string {
  const nonce = randomBytes(18).toString("base64");
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
  <title>New AGEFreighter migration</title>
  <style>
    :root { color-scheme: light dark; }
    body { max-width: 980px; margin: 0 auto; padding: 28px 30px 64px; color: var(--vscode-foreground); background: var(--vscode-editor-background); font-family: var(--vscode-font-family); }
    h1 { font-size: 25px; margin: 0 0 6px; }
    h2 { font-size: 16px; margin: 0 0 14px; }
    .lead, .muted { color: var(--vscode-descriptionForeground); }
    .steps { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; margin: 24px 0; }
    .step { padding: 8px 10px; border-bottom: 2px solid var(--vscode-panel-border); color: var(--vscode-descriptionForeground); font-size: 12px; }
    .step.active { border-color: var(--vscode-focusBorder); color: var(--vscode-foreground); }
    .card { border: 1px solid var(--vscode-panel-border); border-radius: 8px; padding: 20px; margin: 14px 0; background: var(--vscode-sideBar-background); }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px 16px; }
    .wide { grid-column: 1 / -1; }
    label { display: flex; flex-direction: column; gap: 5px; font-size: 12px; color: var(--vscode-descriptionForeground); }
    input, select { box-sizing: border-box; width: 100%; min-height: 30px; padding: 5px 7px; color: var(--vscode-input-foreground); background: var(--vscode-input-background); border: 1px solid var(--vscode-input-border, transparent); }
    input:focus, select:focus { outline: 1px solid var(--vscode-focusBorder); }
    .choice { display: flex; flex-direction: row; align-items: center; gap: 8px; color: var(--vscode-foreground); }
    .choice input { width: auto; min-height: auto; }
    .actions { display: flex; align-items: center; gap: 10px; margin-top: 18px; }
    button { border: 0; border-radius: 2px; padding: 7px 14px; color: var(--vscode-button-foreground); background: var(--vscode-button-background); cursor: pointer; }
    button:hover { background: var(--vscode-button-hoverBackground); }
    button.secondary { color: var(--vscode-button-secondaryForeground); background: var(--vscode-button-secondaryBackground); }
    button:disabled { opacity: .55; cursor: default; }
    .status { margin-top: 14px; padding: 10px 12px; border-left: 3px solid var(--vscode-focusBorder); background: var(--vscode-textBlockQuote-background); }
    .error { border-left-color: var(--vscode-errorForeground); }
    .success { border-left-color: var(--vscode-testing-iconPassed); }
    dl { display: grid; grid-template-columns: 220px 1fr; gap: 7px 14px; margin: 0; }
    dt { color: var(--vscode-descriptionForeground); }
    dd { margin: 0; overflow-wrap: anywhere; }
    [hidden] { display: none !important; }
    @media (max-width: 700px) { .grid { grid-template-columns: 1fr; } .wide { grid-column: auto; } .steps { grid-template-columns: 1fr; } dl { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <h1>New guided migration</h1>
  <p class="lead">Connect a source first. AGEFreighter will preserve uncertainty, recommend Azure capacity, and ask before any deployment or migration.</p>
  <div class="steps" aria-label="Migration steps">
    <div class="step active">1 · Source</div><div class="step">2 · Azure plan</div><div class="step">3 · Deploy</div><div class="step">4 · Migrate</div><div class="step">5 · Verify</div>
  </div>

  <section class="card">
    <h2>Azure account</h2>
    <div id="azureStatus" class="status">Checking the Azure account signed into VS Code…</div>
    <p class="muted">Prerequisite: sign in to Azure through VS Code before opening this workflow. AGEFreighter does not start a second sign-in session.</p>
  </section>

  <form id="sourceForm">
    <section class="card">
      <h2>Neo4j source</h2>
      <p class="muted">2.4.0 starts with the discovery-based Neo4j path. The password is placed directly in VS Code SecretStorage and is not saved with this form.</p>
      <div class="grid">
        <label>Migration name<input id="name" value="neo4j-migration" autocomplete="off" required></label>
        <label>Target subscription<select id="subscription" required><option value="">Loading…</option></select></label>
        <label>Host<input id="host" placeholder="neo4j.internal" autocomplete="off" required></label>
        <label>Port<input id="port" type="number" min="1" max="65535" value="7687" required></label>
        <label>Database<input id="database" value="neo4j" autocomplete="off" required></label>
        <label>Source ID<input id="sourceId" value="neo4j-primary" autocomplete="off" required></label>
        <label>Namespace<input id="namespace" value="migration" autocomplete="off" required></label>
        <label>Username<input id="username" value="neo4j" autocomplete="username" required></label>
        <label>Password<input id="password" type="password" autocomplete="current-password" required></label>
        <label>Transport<select id="encrypted"><option value="true">TLS (neo4j+s)</option><option value="false">Private network / no TLS (neo4j)</option></select></label>
        <label>Vertex identity property<input id="vertexKeyProperty" value="id" autocomplete="off" required></label>
        <label>Relationship identity property<input id="edgeKeyProperty" value="id" autocomplete="off" required></label>
      </div>
    </section>

    <section class="card">
      <h2>Where is the source?</h2>
      <div class="actions">
        <label class="choice"><input type="radio" name="placement" value="azure" checked> Azure resource</label>
        <label class="choice"><input type="radio" name="placement" value="on-premises"> On-premises or another cloud</label>
      </div>
      <div id="azurePlacement" class="grid" style="margin-top:14px">
        <label class="wide">Source ARM resource ID<input id="resourceId" placeholder="/subscriptions/.../resourceGroups/.../providers/Microsoft.Compute/virtualMachines/..." autocomplete="off"></label>
        <p class="wide muted">Selecting the actual ARM resource is required to verify its Azure region and logical availability zone. A hostname alone is not enough.</p>
      </div>
      <div id="onPremPlacement" class="grid" style="margin-top:14px" hidden>
        <label class="wide">Physical source location<input id="declaredLocation" placeholder="Tokyo, Japan" autocomplete="off"></label>
        <p class="wide muted">This becomes an input to the nearest-region recommendation. You will confirm the final region before deployment.</p>
      </div>
    </section>

    <div class="actions">
      <button id="profile" type="submit">Connect and profile source</button>
      <span id="working" class="muted"></span>
    </div>
  </form>

  <section id="result" class="card" hidden>
    <h2>Source profile and sizing input</h2>
    <dl id="facts"></dl>
    <div id="gate" class="status"></div>
    <p class="muted">The next milestone will use these facts to check current region, zone, SKU, quota, AGE availability, and price data before showing a deployable Azure proposal.</p>
  </section>

  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const byId = (id) => document.getElementById(id);
    const form = byId('sourceForm');
    const password = byId('password');
    const profileButton = byId('profile');
    const working = byId('working');
    const azureStatus = byId('azureStatus');
    const subscription = byId('subscription');

    function selectedPlacement() {
      return document.querySelector('input[name="placement"]:checked').value;
    }
    document.querySelectorAll('input[name="placement"]').forEach((input) => input.addEventListener('change', () => {
      const azure = selectedPlacement() === 'azure';
      byId('azurePlacement').hidden = !azure;
      byId('onPremPlacement').hidden = azure;
    }));
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      const kind = selectedPlacement();
      vscode.postMessage({
        type: 'profile',
        source: {
          name: byId('name').value,
          host: byId('host').value,
          port: Number(byId('port').value),
          encrypted: byId('encrypted').value === 'true',
          database: byId('database').value,
          sourceId: byId('sourceId').value,
          namespace: byId('namespace').value,
          username: byId('username').value,
          vertexKeyProperty: byId('vertexKeyProperty').value,
          edgeKeyProperty: byId('edgeKeyProperty').value
        },
        password: password.value,
        placement: {
          kind,
          subscriptionId: subscription.value,
          resourceId: byId('resourceId').value,
          declaredLocation: byId('declaredLocation').value
        }
      });
      password.value = '';
    });
    window.addEventListener('message', (event) => {
      const message = event.data;
      if (!message || typeof message.type !== 'string') return;
      if (message.type === 'subscriptions') {
        subscription.replaceChildren();
        for (const item of message.subscriptions) {
          const option = document.createElement('option');
          option.value = item.id;
          option.textContent = item.name + ' — ' + item.accountLabel;
          subscription.appendChild(option);
        }
        azureStatus.textContent = message.subscriptions.length
          ? 'Signed in. Select the Azure subscription that will own the migration resources.'
          : 'Signed in, but no selected Azure subscriptions are available.';
        azureStatus.className = 'status success';
      } else if (message.type === 'azureSignedOut') {
        azureStatus.textContent = message.message;
        azureStatus.className = 'status error';
      } else if (message.type === 'busy') {
        profileButton.disabled = message.value;
        working.textContent = message.value ? message.message : '';
      } else if (message.type === 'error') {
        const gate = byId('gate');
        byId('result').hidden = false;
        gate.textContent = message.message;
        gate.className = 'status error';
      } else if (message.type === 'profileComplete') {
        const capacity = message.capacity;
        const pairs = [
          ['Profile outcome', message.outcome],
          ['Source placement', (message.placement.location || 'unknown') + (message.placement.zone ? ' / zone ' + message.placement.zone : '')],
          ['Placement confidence', message.placement.confidence],
          ['Exact source nodes', message.inventory.vertices],
          ['Exact source relationships', message.inventory.edges],
          ['Estimated target rows', (capacity.targetRowsLowerBound ? 'at least ' : '') + (capacity.targetRows || 'unavailable')],
          ['Recommended storage evidence', formatBytes(capacity.recommendedStorageLow) + ' – ' + formatBytes(capacity.recommendedStorageHigh)],
          ['Evidence', message.evidencePath]
        ];
        const facts = byId('facts');
        facts.replaceChildren();
        for (const [key, value] of pairs) {
          const dt = document.createElement('dt'); dt.textContent = key;
          const dd = document.createElement('dd'); dd.textContent = value;
          facts.append(dt, dd);
        }
        const gate = byId('gate');
        gate.textContent = capacity.deployable
          ? 'Complete inventory evidence is available for the Azure proposal.'
          : capacity.reason;
        gate.className = 'status ' + (capacity.deployable ? 'success' : 'error');
        byId('result').hidden = false;
        byId('result').scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
    function formatBytes(value) {
      if (!value) return 'unavailable';
      const bytes = Number(value);
      if (!Number.isFinite(bytes)) return value + ' bytes';
      const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
      let n = bytes, i = 0;
      while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
      return n.toFixed(i === 0 ? 0 : 1) + ' ' + units[i];
    }
    vscode.postMessage({ type: 'ready' });
  </script>
</body>
</html>`;
}
