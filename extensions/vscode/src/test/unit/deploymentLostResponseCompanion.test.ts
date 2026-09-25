import assert from "node:assert/strict";
import { mkdtemp,readFile,writeFile,rm,stat } from "node:fs/promises";
import { join,resolve } from "node:path";
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { runInNewContext } from "node:vm";
import test from "node:test";
import { prepareLostResponseCompanion } from "../helpers/deploymentLostResponseCompanion";
import { otherCancellationFixture,otherNativeCancelCases } from "../helpers/nativeCancelOtherScenarios";
test("build and inert compiled activation retain only draft, native command hooks and lazy context; no authentication/network or live credit",{skip:process.platform!=="darwin"?"The native companion is explicitly scoped to macOS /private/tmp":false},async()=>{
  const root=await mkdtemp("/private/tmp/af-deployment-lost-response-"),artifactRoot=await mkdtemp("/private/tmp/af-lost-response-artifact-");
  try{
    const record=otherCancellationFixture(otherNativeCancelCases[0]).record;record.input.source={type:"csv",location:"local"};
    const bytes=Buffer.from("INERT UNIT BYTES. NOT EXECUTABLE."),sha=createHash("sha256").update(bytes).digest("hex"),archivePath=join(artifactRoot,"negative.tar.gz"),manifestPath=join(artifactRoot,"manifest.json");
    await writeFile(archivePath,bytes);await writeFile(manifestPath,JSON.stringify({schemaVersion:1,platform:"linux-amd64",commit:"b".repeat(40),version:`2.4.0-dev.${"b".repeat(12)}`,sha256:sha,bytes:bytes.length,archive:"negative.tar.gz"}));
    const args=await prepareLostResponseCompanion(root,resolve(__dirname,"../../.."),{workflow:record.id,input:record.input,artifactSHA256:sha,artifactBytes:bytes.length,archivePath,manifestPath,annotation:"UNIT-ONLY negative fixture missing agefreighter-tools; no cloud or native evidence.",expiresAt:new Date(Date.now()+3600000).toISOString()});
    assert.ok(!args.includes("--disable-extensions"));
    const pkg=JSON.parse(await readFile(join(root,"package.json"),"utf8"));assert.deepEqual(pkg.activationEvents,["onCommand:agefreighterB09.open","onCommand:agefreighterB09.artifact"]);
    const registry=new Map<string,unknown>();let accountReads=0,getterReads=0,fetches=0;
    class Disposable{constructor(readonly dispose:()=>void=()=>{}){}}
    class EventEmitter{event=()=>new Disposable();dispose(){}fire(){}}
    class NativeWebview {
      #html=""; #messages:Record<string,unknown>[]=[]; #listener?: (raw:unknown)=>Promise<void>;
      get html(){return this.#html;} set html(value:string){this.#html=value;}
      get cspSource(){return "synthetic-native-csp";}
      async postMessage(value:Record<string,unknown>){this.#messages.push(value);return true;}
      onDidReceiveMessage(listener:(raw:unknown)=>Promise<void>){this.#listener=listener;return new Disposable();}
      async receive(value:unknown){assert.ok(this.#listener);await this.#listener(value);}
      get messages(){return [...this.#messages];}
    }
    class NativePanel {
      #webview=new NativeWebview(); #disposeListener?:()=>void;
      get webview(){return this.#webview;}
      onDidDispose(listener:()=>void){this.#disposeListener=listener;return new Disposable();}
      dispose(){this.#disposeListener?.();}
    }
    let panel:NativePanel|undefined;
    const fake={Disposable,EventEmitter,ExtensionMode:{Test:3},Uri:{file:(fsPath:string)=>({fsPath})},ViewColumn:{One:1},ProgressLocation:{Notification:15},
      authentication:{getAccounts:async()=>{accountReads++;return[];}},env:{},
      workspace:{isTrusted:true,workspaceFolders:[],getConfiguration:()=>({get:()=>undefined,inspect:()=>({globalValue:true})}),openTextDocument:async(v:unknown)=>v},
      window:{showTextDocument:async()=>{},createWebviewPanel:()=>{panel=new NativePanel();return panel;}},commands:{registerCommand:(name:string,callback:unknown)=>{registry.set(name,callback);return new Disposable();}},l10n:{t:(s:string)=>s}};
    const req=createRequire(__filename),compiled={exports:{} as {activate:(c:unknown)=>Promise<void>}},custom=(name:string)=>name==="vscode"?fake:req(name);
    runInNewContext(await readFile(join(root,"runtime.cjs"),"utf8"),{module:Object.assign(compiled,{require:custom}),exports:compiled.exports,require:custom,process,Buffer,URL,URLSearchParams,setTimeout,clearTimeout,setInterval,clearInterval,console,AbortController,AbortSignal,TextEncoder,TextDecoder,structuredClone,fetch:()=>{fetches++;throw Error("No network in unit smoke");}});
    const context={extensionPath:root,extensionMode:2,subscriptions:[],globalStorageUri:{fsPath:"must-not-use-existing-store"}};
    Object.defineProperty(context,"extensionRuntime",{enumerable:true,get:()=>{getterReads++;throw Error("Unrelated proposed getter");}});
    await compiled.exports.activate(context);
    assert.deepEqual([...registry.keys()],["agefreighterB09.open","agefreighterB09.artifact"]);assert.equal(getterReads,0);assert.equal(fetches,0);assert.equal(accountReads,1);
    await (registry.get("agefreighterB09.open") as ()=>Promise<void>)();assert.ok(panel);
    assert.match(panel.webview.html,/Approve &amp; deploy|Approve & deploy/);
    await panel.webview.receive({action:"ready"});
    assert.ok(panel.webview.messages.some(m=>m.kind==="busy"&&m.value===true));
    assert.ok(panel.webview.messages.some(m=>m.kind==="busy"&&m.value===false));
    assert.equal(fetches,0);assert.equal(getterReads,0);panel.dispose();
    const retained=JSON.parse(await readFile(join(root,"runner-v2",record.id+".json"),"utf8"));assert.equal(retained.phase,"draft");assert.equal(retained.storageDeployment,undefined);assert.equal(retained.guestReady,undefined);
    assert.equal((await stat(join(root,"runner-v2",record.id+".json"))).mode&0o777,0o600);
    await assert.rejects(compiled.exports.activate(context),/EEXIST/);
    await assert.rejects(prepareLostResponseCompanion(root,resolve(__dirname,"../../.."),JSON.parse(await readFile(join(root,"scope.json"),"utf8"))),/EEXIST/);
  }finally{await rm(root,{recursive:true,force:true});await rm(artifactRoot,{recursive:true,force:true});}
});
