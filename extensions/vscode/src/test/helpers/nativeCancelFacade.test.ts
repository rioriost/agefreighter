import assert from "node:assert/strict";
import test from "node:test";
import { Script } from "node:vm";
import { ModuleKind, transpileModule } from "typescript";
import { lazyNativeFacade } from "./nativeCancelFacade";

test("compiled namespace import stays lazy and retains window/workspace without proposed APIs", () => {
  let proposedReads = 0;
  const actualWindow = Object.freeze(Object.defineProperty({ showWarningMessage: () => "real-native-marker" }, "linkPresentation", {
    enumerable: true, get: () => { proposedReads++; throw Error("Unapproved proposed API"); }
  }));
  const namespace = Object.freeze(Object.defineProperty({ window: actualWindow, workspace: { isTrusted: true } }, "proposal", {
    enumerable: true, get: () => { proposedReads++; throw Error("Unapproved namespace getter"); }
  }));
  const windowFacade = lazyNativeFacade(actualWindow, {}), facade = lazyNativeFacade(namespace, { window: windowFacade });
  const { outputText } = transpileModule('import * as vscode from "vscode"; export const result = [vscode.workspace.isTrusted, vscode.window.showWarningMessage()];',
    { compilerOptions: { module: ModuleKind.CommonJS, esModuleInterop: true } });
  assert.match(outputText, /__importStar/);
  const module = { exports: {} as { result: unknown[] } };
  new Script(outputText).runInNewContext({ module, exports: module.exports, require: (name: string) => { assert.equal(name, "vscode"); return facade; } });
  assert.deepEqual(Array.from(module.exports.result), [true, "real-native-marker"]);
  assert.equal(proposedReads, 0);
});
