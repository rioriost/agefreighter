import * as path from "node:path";
import * as vscode from "vscode";
import {
  buildReadArguments,
  buildTerminalArguments,
  isConnectedReadOperation,
  ReadOperation,
  requiresJobId,
  TerminalOperation,
  validateJobId
} from "./core/operations";
import { CLIError, runJSON, runText } from "./core/process";
import { reportHTML } from "./core/report";
import { redactText } from "./core/security";
import { JobItem, JobsProvider } from "./jobsView";

export interface RunEvidenceOptions {
  confirmConnection?: boolean;
  showPanel?: boolean;
}

export class ExtensionController {
  public constructor(
    private readonly jobs: JobsProvider,
    private readonly output: vscode.OutputChannel
  ) {}

  public register(context: vscode.ExtensionContext): void {
    const command = (name: string, callback: (...args: unknown[]) => unknown): void => {
      context.subscriptions.push(vscode.commands.registerCommand(name, callback));
    };
    command("agefreighter.refreshJobs", () => this.jobs.refresh());
    command("agefreighter.selectBinary", () => this.selectBinary());
    command("agefreighter.checkVersion", () => this.checkVersion());
    for (const operation of ["validate", "plan", "profile", "doctor", "status", "report", "optimize"] as const) {
      command(`agefreighter.${operation}`, (argument?: unknown) => this.runReadCommand(operation, argument));
    }
    for (const operation of ["load", "resume", "verify", "cleanup"] as const) {
      command(`agefreighter.${operation}`, (argument?: unknown) => this.runTerminalCommand(operation, argument));
    }
    command("agefreighter.openDocumentation", () => vscode.env.openExternal(
      vscode.Uri.parse("https://github.com/rioriost/agefreighter/blob/main/docs/reference/vscode-extension.md")
    ));
  }

  public async pickJob(argument?: unknown): Promise<vscode.Uri | undefined> {
    if (argument instanceof vscode.Uri) {
      return argument;
    }
    if (argument instanceof JobItem) {
      return argument.uri;
    }
    const jobs = await this.jobs.getJobs();
    if (jobs.length === 0) {
      void vscode.window.showInformationMessage(
        "No AGEFreighter LoadJob files were found in this workspace."
      );
      return undefined;
    }
    const selected = await vscode.window.showQuickPick(
      jobs.map((job) => ({ label: job.label!.toString(), description: job.uri.fsPath, uri: job.uri })),
      { placeHolder: "Select an AGEFreighter migration job" }
    );
    return selected?.uri;
  }

  public async promptJobId(): Promise<string | undefined> {
    const value = await vscode.window.showInputBox({
      title: "AGEFreighter durable job ID",
      prompt: "Enter the UUID returned by load",
      ignoreFocusOut: true,
      validateInput: (input) => {
        try {
          validateJobId(input);
          return undefined;
        } catch (error) {
          return error instanceof Error ? error.message : String(error);
        }
      }
    });
    return value?.trim();
  }

  public async runEvidence(
    operation: ReadOperation,
    uri: vscode.Uri,
    jobId?: string,
    options: RunEvidenceOptions = {}
  ): Promise<unknown | undefined> {
    if (!this.ensureTrusted()) {
      return undefined;
    }
    if (options.confirmConnection && isConnectedReadOperation(operation)) {
      const answer = await vscode.window.showWarningMessage(
        `${operation} connects to the source or target configured by ${path.basename(uri.fsPath)}. Continue?`,
        { modal: true },
        "Continue"
      );
      if (answer !== "Continue") {
        return undefined;
      }
    }
    const args = buildReadArguments(operation, uri.fsPath, jobId);
    const abort = new AbortController();
    const value = await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: `AGEFreighter: ${operation}`,
        cancellable: true
      },
      async (_progress, token) => {
        token.onCancellationRequested(() => abort.abort());
        return await runJSON(this.binary(uri), args, {
          cwd: path.dirname(uri.fsPath),
          timeoutMs: this.timeout(uri),
          maxOutputBytes: this.maxOutput(uri),
          env: { ...process.env },
          signal: abort.signal
        });
      }
    );
    this.output.appendLine(`${new Date().toISOString()} ${operation} completed for ${uri.fsPath}`);
    if (options.showPanel !== false) {
      this.showReport(`${operation}: ${path.basename(uri.fsPath)}`, value);
    }
    return value;
  }

  public async resolveToolJob(inputPath: string): Promise<vscode.Uri> {
    if (!vscode.workspace.isTrusted) {
      throw new Error("The workspace must be trusted before AGEFreighter can run.");
    }
    if (!path.isAbsolute(inputPath)) {
      throw new Error("The language-model tool requires an absolute job path.");
    }
    const candidate = vscode.Uri.file(path.resolve(inputPath));
    const folder = vscode.workspace.getWorkspaceFolder(candidate);
    if (!folder) {
      throw new Error("The job path must be inside the trusted workspace.");
    }
    const jobs = await this.jobs.getJobs();
    const matched = jobs.find((job) => samePath(job.uri.fsPath, candidate.fsPath));
    if (!matched) {
      throw new Error("The path is not a discovered AGEFreighter LoadJob file.");
    }
    return matched.uri;
  }

  private async runReadCommand(operation: ReadOperation, argument?: unknown): Promise<void> {
    try {
      const uri = await this.pickJob(argument);
      if (!uri) {
        return;
      }
      const jobId = requiresJobId(operation) ? await this.promptJobId() : undefined;
      if (requiresJobId(operation) && !jobId) {
        return;
      }
      await this.runEvidence(operation, uri, jobId, { confirmConnection: true, showPanel: true });
    } catch (error) {
      this.showError(operation, error);
    }
  }

  private async runTerminalCommand(operation: TerminalOperation, argument?: unknown): Promise<void> {
    try {
      if (!this.ensureTrusted()) {
        return;
      }
      const uri = await this.pickJob(argument);
      if (!uri) {
        return;
      }
      const jobId = requiresJobId(operation) ? await this.promptJobId() : undefined;
      if (requiresJobId(operation) && !jobId) {
        return;
      }
      const warning = operation === "cleanup"
        ? "This permanently removes the retained replacement backup."
        : operation === "verify"
          ? "This can perform a substantial target scan."
          : "This changes or continues migration state in the configured target.";
      const answer = await vscode.window.showWarningMessage(
        `${warning}\n\nJob: ${uri.fsPath}`,
        { modal: true },
        operation === "cleanup" ? "Remove backup" : "Run in terminal"
      );
      if (!answer) {
        return;
      }
      const args = buildTerminalArguments(operation, uri.fsPath, jobId);
      const terminal = vscode.window.createTerminal({
        name: `AGEFreighter ${operation}`,
        cwd: path.dirname(uri.fsPath),
        shellPath: this.binary(uri),
        shellArgs: args,
        env: { ...process.env },
        isTransient: false
      });
      terminal.show(true);
      this.output.appendLine(`${new Date().toISOString()} ${operation} opened in a terminal for ${uri.fsPath}`);
    } catch (error) {
      this.showError(operation, error);
    }
  }

  private async selectBinary(): Promise<void> {
    const selected = await vscode.window.showOpenDialog({
      title: "Select the AGEFreighter CLI binary",
      canSelectFiles: true,
      canSelectFolders: false,
      canSelectMany: false,
      openLabel: "Use AGEFreighter binary"
    });
    const uri = selected?.[0];
    if (!uri) {
      return;
    }
    await vscode.workspace.getConfiguration("agefreighter").update(
      "binaryPath",
      uri.fsPath,
      vscode.ConfigurationTarget.Workspace
    );
    await this.checkVersion();
  }

  private async checkVersion(): Promise<void> {
    if (!this.ensureTrusted()) {
      return;
    }
    const folder = vscode.workspace.workspaceFolders?.[0];
    const cwd = folder?.uri.fsPath ?? process.cwd();
    try {
      const result = await runText(this.binary(folder?.uri), ["version"], {
        cwd,
        timeoutMs: 10_000,
        maxOutputBytes: 64 * 1024,
        env: { ...process.env }
      });
      const version = result.stdout.trim();
      const match = /agefreighter\s+(\d+)\.(\d+)\.(\d+)/.exec(version);
      if (!match || Number(match[1]) < 2 || (Number(match[1]) === 2 && Number(match[2]) < 3)) {
        void vscode.window.showWarningMessage(
          `AGEFreighter 2.3.0 or newer is required. Detected: ${version || "unknown"}`
        );
        return;
      }
      void vscode.window.showInformationMessage(version);
    } catch (error) {
      this.showError("version", error);
    }
  }

  private showReport(title: string, value: unknown): void {
    const panel = vscode.window.createWebviewPanel(
      "agefreighter.report",
      title,
      vscode.ViewColumn.Active,
      { enableScripts: false, retainContextWhenHidden: false }
    );
    panel.webview.html = reportHTML(title, value);
  }

  private showError(operation: string, error: unknown): void {
    const detail = error instanceof CLIError && error.stderr
      ? `${error.message} ${redactText(error.stderr)}`
      : error instanceof Error ? error.message : String(error);
    this.output.appendLine(`${new Date().toISOString()} ${operation} failed: ${detail}`);
    this.output.show(true);
    void vscode.window.showErrorMessage(`AGEFreighter ${operation} failed: ${detail}`);
  }

  private ensureTrusted(): boolean {
    if (vscode.workspace.isTrusted) {
      return true;
    }
    void vscode.window.showWarningMessage(
      "Trust this workspace before running the AGEFreighter CLI."
    );
    return false;
  }

  private binary(scope?: vscode.Uri): string {
    return vscode.workspace.getConfiguration("agefreighter", scope).get<string>(
      "binaryPath",
      "agefreighter"
    ).trim();
  }

  private timeout(scope?: vscode.Uri): number {
    const seconds = vscode.workspace.getConfiguration("agefreighter", scope).get<number>(
      "readTimeoutSeconds",
      120
    );
    return Math.max(5, Math.min(1800, seconds)) * 1000;
  }

  private maxOutput(scope?: vscode.Uri): number {
    const bytes = vscode.workspace.getConfiguration("agefreighter", scope).get<number>(
      "maxOutputBytes",
      4 * 1024 * 1024
    );
    return Math.max(64 * 1024, Math.min(16 * 1024 * 1024, bytes));
  }
}

function samePath(left: string, right: string): boolean {
  return process.platform === "win32"
    ? left.toLocaleLowerCase() === right.toLocaleLowerCase()
    : left === right;
}
