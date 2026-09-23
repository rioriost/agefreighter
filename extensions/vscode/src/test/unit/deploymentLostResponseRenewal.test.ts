import assert from "node:assert/strict";
import test from "node:test";
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { runInNewContext } from "node:vm";
import { existsSync } from "node:fs";
import { mkdtemp,writeFile,readFile,mkdir,rm } from "node:fs/promises";
import { join,resolve } from "node:path";
import { validateLostResponseRenewal,LostResponseSetupRenewal } from "../helpers/deploymentLostResponseRenewal";
import { LostResponseScope,LostResponseStages } from "../helpers/deploymentLostResponseStages";
import { validateLostResponseSeed } from "../helpers/deploymentLostResponseSeed";
import { prepareLostResponseCompanion } from "../helpers/deploymentLostResponseCompanion";
import { lostResponseHash } from "../helpers/deploymentLostResponse";
import { sourceWorkflowDraft } from "../../core/runner";
import { storageDraft } from "../../core/runnerStorageLifecycle";
import { otherCancellationFixture,otherNativeCancelCases } from "../helpers/nativeCancelOtherScenarios";
import type { AzureSession } from "../../guided/azure";
function fixture(){
  const input=otherCancellationFixture(otherNativeCancelCases[0]).record.input;input.source={type:"csv",location:"local"};
  const previous:LostResponseScope={workflow:input.subscriptionId,input,artifactSHA256:"a".repeat(64),artifactBytes:1,manifestPath:"/inert/manifest.json",archivePath:"/inert/negative.tar.gz",annotation:"negative fixture without agefreighter-tools",expiresAt:new Date(Date.now()-3600000).toISOString()};
  const current={...previous,expiresAt:new Date(Date.now()+3600000).toISOString()},renewal:LostResponseSetupRenewal={schemaVersion:1,kind:"explicit-ready-storage-setup-renewal",previousExpiresAt:previous.expiresAt,expiresAt:current.expiresAt};
  const record=sourceWorkflowDraft(previous.workflow,input);record.storageDeployment={...storageDraft(record,record.id),phase:"ready",networkAccess:"Disabled"};
  const d=record.storageDeployment,claim={path:`${d.id}?api-version=2022-09-01`,bodySHA256:lostResponseHash({properties:{mode:"Incremental",template:d.template}}),roleId:d.roleId};
  return {previous,current,renewal,record,claim};
}
test("renewal accepts only explicit later bounded setup expiry with all other scope fields unchanged",()=>{
  const f=fixture();assert.doesNotThrow(()=>validateLostResponseRenewal(f.renewal,f.previous,f.current));
  for(const current of [{...f.current,expiresAt:f.previous.expiresAt},{...f.current,expiresAt:new Date(Date.now()+3*3600000).toISOString()},{...f.current,artifactSHA256:"b".repeat(64)},{...f.current,input:{...f.current.input,size:"Standard_D4s_v5"}}])assert.throws(()=>validateLostResponseRenewal({...f.renewal,expiresAt:current.expiresAt},f.previous,current));
  assert.throws(()=>validateLostResponseRenewal({...f.renewal,schemaVersion:2} as unknown as LostResponseSetupRenewal,f.previous,f.current));
  assert.throws(()=>validateLostResponseRenewal(f.renewal,{...f.previous,expiresAt:f.current.expiresAt},f.current));
});
test("ready seed preserves old storage state only with exact canonical PUT claim; stale/other effects cannot revive",()=>{
  const f=fixture();assert.doesNotThrow(()=>validateLostResponseSeed(f.record,f.current,f.claim));assert.throws(()=>validateLostResponseSeed(f.record,f.current));
  for(const claim of [{...f.claim,path:f.claim.path+"other"},{...f.claim,bodySHA256:"0".repeat(64)},{...f.claim,roleId:f.claim.roleId+"other"}])assert.throws(()=>validateLostResponseSeed(f.record,f.current,claim));
  for(const phase of ["submitted","unknown","failed","previewed"] as const)assert.throws(()=>validateLostResponseSeed({...f.record,storageDeployment:{...f.record.storageDeployment!,phase}},f.current,f.claim));
  for(const name of ["guestCommand","guestReady","developmentUpload","sourceDraft","migration"])assert.throws(()=>validateLostResponseSeed({...f.record,[name]:{}},f.current,f.claim));
});
test("compiled renewal restores exact storage claim before any API and blocks even a forged local storage resubmit",async()=>{
  const prior=await mkdtemp("/private/tmp/af-deployment-lost-response-"),root=await mkdtemp("/private/tmp/af-deployment-lost-response-"),assets=await mkdtemp("/private/tmp/af-renewal-assets-");
  try{
    const f=fixture(),bytes=Buffer.from("INERT RENEWAL UNIT FIXTURE"),sha=createHash("sha256").update(bytes).digest("hex");
    Object.assign(f.previous,{artifactSHA256:sha,artifactBytes:bytes.length,manifestPath:join(assets,"manifest.json"),archivePath:join(assets,"negative.tar.gz")});f.current={...f.previous,expiresAt:f.current.expiresAt};
    await writeFile(f.previous.archivePath,bytes);await writeFile(f.previous.manifestPath,JSON.stringify({schemaVersion:1,platform:"linux-amd64",commit:"b".repeat(40),version:`2.4.0-dev.${"b".repeat(12)}`,sha256:sha,bytes:bytes.length,archive:"negative.tar.gz"}));
    await mkdir(join(prior,"runner-v2"),{mode:0o700});await mkdir(join(prior,"ledger"),{mode:0o700});
    const recordBytes=JSON.stringify(f.record),claimBytes=JSON.stringify(f.claim,null,2)+"\n";
    await writeFile(join(prior,"scope.json"),JSON.stringify(f.previous,null,2)+"\n",{mode:0o600});await writeFile(join(prior,"runner-v2",f.record.id+".json"),recordBytes,{mode:0o600});await writeFile(join(prior,"ledger/storage-put-intent.json"),claimBytes,{mode:0o600});
    await prepareLostResponseCompanion(root,resolve(__dirname,"../../.."),f.current,prior,f.renewal);
    assert.equal(await readFile(join(root,"initial-record.json"),"utf8"),recordBytes);assert.equal(await readFile(join(root,"preserved-storage-put-intent.json"),"utf8"),claimBytes);
    const preparation=JSON.parse(await readFile(join(root,"preparation.json"),"utf8"));assert.equal(preparation.storagePutIntentSHA256,createHash("sha256").update(claimBytes).digest("hex"));
    let apiReads=0,fetches=0;class Disposable{constructor(readonly dispose=()=>{}){}}class EventEmitter{event=()=>new Disposable();dispose(){}fire(){}}
    const fake={Disposable,EventEmitter,ExtensionMode:{Test:3},Uri:{file:(fsPath:string)=>({fsPath})},ViewColumn:{One:1},ProgressLocation:{Notification:15},
      authentication:{getAccounts:async()=>{assert.ok(existsSync(join(root,"ledger/storage-put-intent.json")));apiReads++;return[];}},env:{},
      workspace:{isTrusted:true,workspaceFolders:[],getConfiguration:()=>({get:()=>undefined,inspect:()=>({globalValue:true})}),openTextDocument:async(v:unknown)=>v},window:{showTextDocument:async()=>{}},commands:{registerCommand:()=>new Disposable()},l10n:{t:(s:string)=>s}};
    const req=createRequire(__filename),compiled={exports:{} as {activate:(c:unknown)=>Promise<void>}},custom=(name:string)=>name==="vscode"?fake:req(name);
    runInNewContext(await readFile(join(root,"runtime.cjs"),"utf8"),{module:Object.assign(compiled,{require:custom}),exports:compiled.exports,require:custom,process,Buffer,URL,URLSearchParams,setTimeout,clearTimeout,setInterval,clearInterval,console,AbortController,AbortSignal,TextEncoder,TextDecoder,structuredClone,fetch:()=>{fetches++;throw Error("No live network");}});
    await compiled.exports.activate({extensionPath:root,extensionMode:2,subscriptions:[],globalStorageUri:{fsPath:"unused"}});
    assert.equal(apiReads,1);assert.equal(fetches,0);assert.equal(await readFile(join(root,"ledger/storage-put-intent.json"),"utf8"),claimBytes);assert.equal(await readFile(join(root,"runner-v2",f.record.id+".json"),"utf8"),recordBytes);
    let transport=0;const submitted={...f.record,storageDeployment:{...f.record.storageDeployment!,phase:"submitted" as const}};
    const stage=new LostResponseStages(join(root,"ledger"),f.current,async()=>submitted,{runnerRequest:async()=>{transport++;return {status:201,value:{}};}} as unknown as AzureSession);
    await assert.rejects(stage.request(f.current.input.subscriptionId,f.claim.path,"PUT",{properties:{mode:"Incremental",template:submitted.storageDeployment.template}}),/EEXIST/);assert.equal(transport,0);
    assert.equal(await readFile(join(prior,"runner-v2",f.record.id+".json"),"utf8"),recordBytes);assert.equal(await readFile(join(prior,"ledger/storage-put-intent.json"),"utf8"),claimBytes);
  }finally{await rm(root,{recursive:true,force:true});await rm(prior,{recursive:true,force:true});await rm(assets,{recursive:true,force:true});}
});
