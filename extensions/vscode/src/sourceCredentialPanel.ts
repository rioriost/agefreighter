import * as vscode from "vscode";
import { RunnerRecord } from "./core/runner";
import { credentialKey, rememberSourceCredential, savedSourceCredential, SourceConnection } from "./core/sourceCredential";

/** Native UI only: no password or secret reference is posted to a webview. */
export async function sourceCredential(context: vscode.ExtensionContext, record: RunnerRecord, connection: SourceConnection, cancelled: () => boolean = () => false, prepareOnly = false): Promise<string | undefined> {
  if (!vscode.workspace.isTrusted || cancelled()) return undefined;
  const stored = await savedSourceCredential(context.secrets, record, connection);
  if (!vscode.workspace.isTrusted || cancelled()) return undefined;
  if (stored !== undefined) return stored;
  const password = await vscode.window.showInputBox({ title: "Read-only source password", password: true, ignoreFocusOut: true,
    prompt: `${record.input.source.type}: ${connection.host}:${connection.port}/${connection.database} as ${connection.username}. Protected guest transport only; never sent to AI.` });
  if (password === undefined || cancelled() || !vscode.workspace.isTrusted) return undefined;
  const choice = await vscode.window.showQuickPick(prepareOnly ? ["Remember for this workflow (up to 8 hours)"] : ["Use once", "Remember for this workflow (up to 8 hours)"], { ignoreFocusOut: true,
    placeHolder: "Optional encrypted SecretStorage reuse across catalog, inventory and migration. Changed connections require a new password." });
  if (!choice || cancelled() || !vscode.workspace.isTrusted) return undefined;
  if (choice.startsWith("Remember")) await rememberSourceCredential(context.secrets, record, connection, password);
  return password;
}
export async function forgetSourceCredential(context: vscode.ExtensionContext, workflow: string): Promise<void> {
  await context.secrets.delete(credentialKey(workflow));
}

/** Reconciliation may observe the same failure more than once. Discard only
 * credentials predating it, not explicit post-failure preparation. No prompt. */
export async function invalidateStaleSourceCredential(context: vscode.ExtensionContext, record: RunnerRecord): Promise<void> {
  if (!record.sourceDraft) { await forgetSourceCredential(context, record.id); return; }
  await savedSourceCredential(context.secrets, record, record.sourceDraft.form);
}
