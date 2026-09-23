/** Normal disposable host only, never a release command. */
import assert from "node:assert/strict";
import { readFile, lstat, realpath, mkdir } from "node:fs/promises";
import { join } from "node:path";
import type * as VSCode from "vscode";
import { AzureSession } from "../../guided/azure";
import { RunnerStore } from "../../guided/runnerStore";
import { RunnerRecord } from "../../core/runner";
import { registerRunnerMigration } from "../../runnerMigration";
import { validateLostResponseSeed } from "./deploymentLostResponseSeed";
import { lazyNativeFacade } from "./nativeCancelFacade";
import { lostResponseNativeFacade } from "./deploymentLostResponseNativeFacade";
import { LostResponseScope, LostResponseStages } from "./deploymentLostResponseStages";
import { retainLostResponseFile, lostResponseHash } from "./deploymentLostResponse";
declare global { var __afLostResponse: { vscode: typeof VSCode; azure: object }; }
const actual = module.require("vscode") as typeof VSCode;
export async function activate(context: VSCode.ExtensionContext) {
  const root = context.extensionPath;
  assert.match(root,/^\/private\/tmp\/af-deployment-lost-response-[a-zA-Z0-9]{6}$/);
  assert.equal(await realpath(root),root); assert.equal((await lstat(root)).mode & 0o777,0o700);
  assert.notEqual(context.extensionMode,actual.ExtensionMode.Test);
  assert.equal(actual.workspace.isTrusted,true); assert.equal(actual.workspace.workspaceFolders?.length ?? 0,0);
  const scope = JSON.parse(await readFile(join(root,"scope.json"),"utf8")) as LostResponseScope;
  assert.ok(Date.now() < Date.parse(scope.expiresAt));
  await retainLostResponseFile(root,"activation.json",{pid:process.pid,at:new Date().toISOString(),scopeSHA256:lostResponseHash(scope),nativeQualification:false});
  const store = new RunnerStore(join(root,"runner-v2")); const initial=JSON.parse(await readFile(join(root,"initial-record.json"),"utf8")) as RunnerRecord;validateLostResponseSeed(initial,scope);await store.write(initial);
  await mkdir(join(root,"ledger"),{mode:0o700});
  const stages = new LostResponseStages(join(root,"ledger"),scope,()=>store.read(scope.workflow),new AzureSession());
  const registry = new Map<string,(...args:unknown[])=>unknown>(); let event = 0;
  const retain = async (kind:string,details:object={}) => retainLostResponseFile(root,`native-${String(++event).padStart(4,"0")}.json`,{kind,...details,at:new Date().toISOString()});
  const window = lazyNativeFacade(actual.window, {
    createWebviewPanel: (...args: Parameters<typeof actual.window.createWebviewPanel>) => {
      assert.ok(["agefreighter.runnerMigration","agefreighter.runnerSource"].includes(args[0]));
      const panel = actual.window.createWebviewPanel(args[0],"B09 NEGATIVE FIXTURE — "+args[1],args[2],args[3]);
      const webview = lostResponseNativeFacade(panel.webview, { onDidReceiveMessage: (listener,thisArgs,disposables) => panel.webview.onDidReceiveMessage(async raw => {
        const action = String(raw?.action);
        const allowed = args[0] === "agefreighter.runnerMigration" ? ["ready","accounts","groups","placementOptions","restore","preview","deploy","refresh","configureSource","guestReady","guestRefresh"] : ["ready","storage"];
        assert.ok(allowed.includes(action),"Action outside source-free B09 scope");
        if (action === "configureSource") { assert.equal(raw.workflow,scope.workflow); assert.deepEqual(raw.input,scope.input); }
        if (action === "preview") { assert.equal(raw.draftId,scope.workflow); assert.deepEqual(raw.input,scope.input); }
        await retain("panel-entry",{panel:args[0],action}); await listener.call(thisArgs,raw);
        const current = await store.read(scope.workflow);
        await retain("panel-return",{panel:args[0],action,record:current,recordSHA256:lostResponseHash(current),liveCreditRequiresIndependentReview:true});
      },undefined,disposables) });
      return lostResponseNativeFacade(panel,{webview});
    },
    showWarningMessage: (async (title:string,options:VSCode.MessageOptions,...items:string[]) => {
      const permitted = ["Create dedicated transfer storage and grant your Azure user data access?","Prepare this unpublished executable for an isolated qualification runner?",`Create the reviewed Linux discovery/migration VM ${(await store.read(scope.workflow)).vmId}?`];
      assert.ok(permitted.includes(title)); assert.equal(options.modal,true); await retain("modal-open",{title,items});
      const choice = await actual.window.showWarningMessage(title,{...options,detail:scope.annotation+"\n\n"+(options.detail??"")},...items);
      await retain("modal-return",{title,choice:choice??null}); if(choice === "Create reviewed runner") await stages.approvePreview(); return choice;
    }) as typeof actual.window.showWarningMessage,
    showOpenDialog: (async (options:VSCode.OpenDialogOptions) => {
      assert.equal(options.openLabel,"Review Linux development archive manifest"); const selected = await actual.window.showOpenDialog(options);
      if(selected) assert.ok(selected.length===1 && selected[0]!.fsPath===scope.manifestPath,"Select only the pinned negative fixture manifest"); return selected;
    }) as typeof actual.window.showOpenDialog
  });
  globalThis.__afLostResponse = { vscode:lazyNativeFacade(actual,{window,commands:lazyNativeFacade(actual.commands,{
    registerCommand:(name,callback)=>{registry.set(name,callback);return new actual.Disposable(()=>registry.delete(name));}
  })}), azure:new Proxy({}, {get:(_target,key)=>(...args:unknown[])=>stages.invoke(String(key),args)}) };
  const productionContext = lazyNativeFacade(context,{subscriptions:[] as VSCode.Disposable[],globalStorageUri:actual.Uri.file(root),extension:{packageJSON:{version:"2.4.0"}},
    secrets:new Proxy({},{get:()=>()=>{throw Error("No source credentials in B09 fixture");}})} as unknown as Partial<VSCode.ExtensionContext>);
  registerRunnerMigration(productionContext,{info:()=>{},error:()=>{}} as unknown as VSCode.LogOutputChannel); context.subscriptions.push(...productionContext.subscriptions);
  for(const [command,target] of [["open","agefreighter.newGuidedMigration"],["artifact","agefreighter.prepareDevelopmentRunner"]]) context.subscriptions.push(actual.commands.registerCommand(`agefreighterB09.${command}`,()=>registry.get(target!)!()));
  const note=await actual.workspace.openTextDocument({language:"markdown",content:`# B09 disposable negative fixture\n\n${scope.annotation}\n\nWorkflow ${scope.workflow}. Action window ends ${scope.expiresAt}. No automatic stop/delete: operator monitor must enforce approved cost/runtime bounds.\n\nRun B09: Open trial. Reconnect to the only draft, use source settings only for transfer storage. Run B09: Prepare pinned negative artifact. Reconnect and preview the same draft, then use normal native approvals. The sole runner deployment response is withheld after acceptance. Preserve unknown before Refresh. One source-free readiness control is available. No source dispatch; no automatic PASS. Authenticate interactively. Single-use host: a restart requires independent external GET reconciliation.\n\nPrivate evidence ${root}.`});
  await actual.window.showTextDocument(note,{preview:false});
}
