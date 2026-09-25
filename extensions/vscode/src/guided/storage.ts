import * as fs from "node:fs/promises";
import * as path from "node:path";
import * as vscode from "vscode";
import { assertPersistableState, GuidedState } from "../core/guided";

export class GuidedStorage {
  public constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly workspace: vscode.WorkspaceFolder
  ) {}

  public workspaceDirectory(id: string): vscode.Uri {
    return vscode.Uri.joinPath(this.workspace.uri, ".agefreighter", "guided", id);
  }

  public async writeSecret(id: string, name: "source-password" | "target-dsn", value: string): Promise<string> {
    if (!value || Buffer.byteLength(value, "utf8") > 1024 * 1024 || /\0/u.test(value)) {
      throw new Error("The secret must be non-empty, under 1 MiB, and contain no NUL byte.");
    }
    const key = `agefreighter.guided.${id}.${name}`;
    await this.context.secrets.store(key, value);
    const directory = path.join(this.context.globalStorageUri.fsPath, "guided-secrets", id);
    await fs.mkdir(directory, { recursive: true, mode: 0o700 });
    if (process.platform !== "win32") {
      await fs.chmod(directory, 0o700);
    }
    const secretPath = path.join(directory, name);
    await fs.writeFile(secretPath, value, { encoding: "utf8", mode: 0o600 });
    if (process.platform !== "win32") {
      await fs.chmod(secretPath, 0o600);
    }
    return secretPath;
  }

  public targetSecretPath(id: string): string {
    return path.join(this.context.globalStorageUri.fsPath, "guided-secrets", id, "target-dsn");
  }

  public async writeText(id: string, name: string, value: string): Promise<vscode.Uri> {
    if (!/^[a-z0-9][a-z0-9.-]{0,127}$/i.test(name)) {
      throw new Error("Invalid guided artifact name.");
    }
    const directory = this.workspaceDirectory(id);
    await vscode.workspace.fs.createDirectory(directory);
    const uri = vscode.Uri.joinPath(directory, name);
    await vscode.workspace.fs.writeFile(uri, Buffer.from(value, "utf8"));
    return uri;
  }

  public async writeState(state: GuidedState): Promise<vscode.Uri> {
    const safe = assertPersistableState(state);
    return await this.writeText(state.id, "state.json", `${JSON.stringify(safe, null, 2)}\n`);
  }
}
