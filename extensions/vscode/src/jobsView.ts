import * as path from "node:path";
import * as vscode from "vscode";
import { looksLikeAgefreighterJob } from "./core/jobs";

const exclude = "**/{.git,.hg,.svn,node_modules,vendor,dist,bin,.coverage,production-simulation/work,production-simulation/results/raw}/**";

interface OperationDefinition {
  label: string;
  command: string;
  icon: string;
}

const operations: readonly OperationDefinition[] = [
  { label: "Validate", command: "agefreighter.validate", icon: "pass" },
  { label: "Static plan", command: "agefreighter.plan", icon: "list-tree" },
  { label: "Profile source", command: "agefreighter.profile", icon: "pulse" },
  { label: "Diagnose target", command: "agefreighter.doctor", icon: "stethoscope" },
  { label: "Start migration", command: "agefreighter.load", icon: "play" },
  { label: "Resume migration", command: "agefreighter.resume", icon: "debug-continue" },
  { label: "Job status", command: "agefreighter.status", icon: "history" },
  { label: "Verify", command: "agefreighter.verify", icon: "verified" },
  { label: "Migration report", command: "agefreighter.report", icon: "preview" },
  { label: "Optimization advice", command: "agefreighter.optimize", icon: "lightbulb" }
];

export class JobItem extends vscode.TreeItem {
  public constructor(public readonly uri: vscode.Uri) {
    super(relativeDisplayPath(uri), vscode.TreeItemCollapsibleState.Collapsed);
    this.contextValue = "agefreighter.job";
    this.resourceUri = uri;
    this.iconPath = new vscode.ThemeIcon("type-hierarchy-sub");
    this.tooltip = uri.fsPath;
    this.command = {
      command: "vscode.open",
      title: "Open AGEFreighter job",
      arguments: [uri]
    };
  }
}

class OperationItem extends vscode.TreeItem {
  public constructor(definition: OperationDefinition, job: JobItem) {
    super(definition.label, vscode.TreeItemCollapsibleState.None);
    this.contextValue = "agefreighter.operation";
    this.iconPath = new vscode.ThemeIcon(definition.icon);
    this.command = {
      command: definition.command,
      title: definition.label,
      arguments: [job.uri]
    };
  }
}

function relativeDisplayPath(uri: vscode.Uri): string {
  const folder = vscode.workspace.getWorkspaceFolder(uri);
  if (!folder) {
    return path.basename(uri.fsPath);
  }
  return path.relative(folder.uri.fsPath, uri.fsPath);
}

export class JobsProvider implements vscode.TreeDataProvider<vscode.TreeItem>, vscode.Disposable {
  private readonly changed = new vscode.EventEmitter<vscode.TreeItem | undefined>();
  private readonly watcher: vscode.FileSystemWatcher;
  private jobs: JobItem[] = [];

  public readonly onDidChangeTreeData = this.changed.event;

  public constructor() {
    this.watcher = vscode.workspace.createFileSystemWatcher("**/*.{yaml,yml,json}");
    this.watcher.onDidCreate(() => this.refresh());
    this.watcher.onDidChange(() => this.refresh());
    this.watcher.onDidDelete(() => this.refresh());
  }

  public dispose(): void {
    this.watcher.dispose();
    this.changed.dispose();
  }

  public refresh(): void {
    this.jobs = [];
    this.changed.fire(undefined);
  }

  public getTreeItem(element: vscode.TreeItem): vscode.TreeItem {
    return element;
  }

  public async getChildren(element?: vscode.TreeItem): Promise<vscode.TreeItem[]> {
    if (element instanceof JobItem) {
      return operations.map((operation) => new OperationItem(operation, element));
    }
    return await this.getJobs();
  }

  public async getJobs(): Promise<JobItem[]> {
    if (this.jobs.length > 0) {
      return this.jobs;
    }
    const candidates = await vscode.workspace.findFiles(
      "**/*.{yaml,yml,json}",
      exclude,
      2000
    );
    const jobs: JobItem[] = [];
    for (const uri of candidates) {
      try {
        const data = await vscode.workspace.fs.readFile(uri);
        if (looksLikeAgefreighterJob(data)) {
          jobs.push(new JobItem(uri));
        }
      } catch {
        // A concurrently removed or unreadable file is ignored until refresh.
      }
    }
    jobs.sort((left, right) => left.label!.toString().localeCompare(right.label!.toString()));
    this.jobs = jobs;
    return jobs;
  }
}
