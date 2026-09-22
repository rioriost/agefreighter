import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { Script } from "node:vm";
import { transformSync } from "esbuild";
import * as credentials from "../../core/sourceCredential";
import * as drafts from "../../core/targetDraft";
import { boundedWatch } from "../../core/boundedWatch";
import { authorizeResize, resizeAuthorized } from "../../core/resizeAuthorization";
import { catalogFixture, catalogForm } from "../catalogFixtures";
import { RunnerRecord } from "../../core/runner";

function load(file:string,modules:Record<string,unknown>){
  const output={exports:{} as any},native=createRequire(__filename);
  const code=transformSync(readFileSync(join(__dirname,"../../",file),"utf8"),{loader:"ts",format:"cjs"}).code;
  new Script(code).runInNewContext({module:output,exports:output.exports,Error,Buffer,Date,require:(name:string)=>name in modules?modules[name]:name.startsWith("node:")?native(name):(()=>{throw Error("Unexpected dependency "+name);})()});
  return output.exports;
}
test("native credential adapter asks once then reuses only opt-in SecretStorage and supports forgetting",async()=>{
  const data=new Map<string,string>(),r=catalogFixture().record;let prompts=0,choices=0;
  const context={secrets:{get:async(k:string)=>data.get(k),store:async(k:string,v:string)=>{data.set(k,v);},delete:async(k:string)=>{data.delete(k);}}};
  const adapter=load("sourceCredentialPanel.ts",{"./core/sourceCredential":credentials,vscode:{workspace:{isTrusted:true},window:{showInputBox:async()=>{prompts++;return "SECRET";},showQuickPick:async()=>{choices++;return "Remember for this workflow (up to 8 hours)";}}}});
  assert.equal(await adapter.sourceCredential(context,r,catalogForm),"SECRET");
  assert.equal(await adapter.sourceCredential(context,structuredClone(r),catalogForm),"SECRET");
  assert.equal(prompts,1);assert.equal(choices,1);
  await adapter.forgetSourceCredential(context,r.id);assert.equal(data.size,0);
  assert.equal(await adapter.sourceCredential(context,r,catalogForm,()=>true),undefined);assert.equal(prompts,1);
});
for(const action of ["Use once",undefined])test(`credential choice ${action??"cancel"} never stores a password`,async()=>{
  const r=catalogFixture().record;let stores=0;
  const context={secrets:{get:async()=>undefined,store:async()=>{stores++;},delete:async()=>{}}};
  const adapter=load("sourceCredentialPanel.ts",{"./core/sourceCredential":credentials,vscode:{workspace:{isTrusted:true},window:{showInputBox:async()=>"SECRET",showQuickPick:async()=>action}}});
  assert.equal(await adapter.sourceCredential(context,r,catalogForm),action?"SECRET":undefined);assert.equal(stores,0);
});

for(const kind of ["assessment","postgresCatalog","migration"] as const)test(`${kind} watcher reconciles same operation to terminal without starting work`,async()=>{
  let r=catalogFixture().record,calls=0,locks=0;
  Object.assign(r,{[kind]:{operation:"exact",phase:"running"}});
  const refresh=async(_c:unknown,current:RunnerRecord)=>{assert.equal(current[kind]?.operation,"exact");calls++;r={...current,[kind]:{...current[kind],phase:calls===2?"finished":"running"}};return r;};
  const adapter=load("runnerWatch.ts",{vscode:{workspace:{isTrusted:true},ProgressLocation:{Notification:1},window:{withProgress:async(_o:unknown,fn:any)=>fn({}, {isCancellationRequested:false})}},"./core/boundedWatch":{boundedWatch},"./core/runnerAssessment":{refreshAssessment:refresh},"./core/runnerCatalog":{refreshCatalog:refresh},"./core/runnerExecution":{refreshMigration:refresh}});
  const store={read:async()=>r,exclusive:async(_id:string,fn:any)=>{assert.equal(locks,0);locks++;try{return await fn();}finally{locks--;}}};
  await adapter.watchRetainedOperation({sleep:async()=>{assert.equal(locks,0);}},store,r.id,kind);
  assert.equal(calls,2);
  await adapter.watchRetainedOperation({sleep:async()=>{}},store,r.id,kind);assert.equal(calls,2);
});
test("report flow submits exactly once, imports after polling, then reopens without requests",async()=>{
  let r=catalogFixture().record,starts=0,refreshes=0,imports=0,locks=0;
  const manifest={operation:"report",sha256:"a".repeat(64),bytes:3};
  r.assessment={operation:"report",action:"inventory",phase:"finished",bootId:"boot",configurationSHA256:"b".repeat(64),reportSHA256:manifest.sha256,reportBytes:3};
  const adapter=load("runnerReportFlow.ts",{"./core/boundedWatch":{boundedWatch},"./core/runnerReport":{
    startReportExport:async(_c:unknown,current:RunnerRecord)=>{starts++;r={...current,reportTransfers:[{...manifest,blob:"approved",phase:"submitted"}],guestCommand:{operation:"report",action:"export-report",phase:"submitted",id:"cmd",submittedAt:new Date().toISOString()}};return r;},
    refreshReportExport:async(_c:unknown,current:RunnerRecord)=>{refreshes++;r={...current,reportTransfers:[{...manifest,blob:"approved",phase:"exported"}]};return r;},
    importReport:async(_c:unknown,current:RunnerRecord)=>{imports++;r={...current,reportTransfers:[{...manifest,blob:"approved",phase:"imported"}]};return r;}
  }});
  const store={read:async()=>r,exclusive:async(_id:string,fn:any)=>{locks++;try{return await fn();}finally{locks--;}}};
  const control={sleep:async()=>{assert.equal(locks,0);}};
  await adapter.transferApprovedReport(control,store,r.id,manifest,async()=>"protected");
  assert.equal(starts,1);assert.equal(refreshes,1);assert.equal(imports,1);
  await adapter.transferApprovedReport(control,store,r.id,manifest,async()=>{throw Error("No capability expected");});assert.equal(starts,1);assert.equal(imports,1);
  await assert.rejects(()=>adapter.transferApprovedReport(control,store,r.id,{...manifest,sha256:"b".repeat(64)},async()=>""),/changed/);
});

test("target inputs and chosen folder survive stopped-VM preflight failure and offline plan needs no guest",async()=>{
  let r=catalogFixture().record;delete r.guestReady;
  r.sourceDraft={configuration:{source:{type:"postgresql"}},form:catalogForm as any,warnings:[],canAssess:true};
  r.assessment={operation:"inventory",action:"inventory",phase:"finished",bootId:"boot",configurationSHA256:"a".repeat(64),reportSHA256:"b".repeat(64),reportBytes:3};
  const input={serverName:"test-server",subnetCIDR:"10.0.24.0/24",postgresSKU:"Standard_E8ds_v5",postgresTier:"MemoryOptimized" as const,storageGiB:128,loaderSize:"Standard_D4s_v5",deadline:new Date(Date.now()+3600000).toISOString(),hourlyUSD:1,budgetUSD:800,additionalReserveUSD:700};
  r=drafts.retainTargetDraft(r,drafts.targetDraftBinding(r),input,"/approved-folder");
  let deploy=false,ready=0,files=0,submits=0;
  const evidence={rows:"5600000",storageHighBytes:"1",labels:{},reportSHA256:"b".repeat(64)};
  const adapter=load("runnerTargetPanel.ts",{vscode:{workspace:{isTrusted:true},window:{showQuickPick:async()=>"Reuse saved target inputs",showWarningMessage:async()=>deploy?"Save plan and approve target deployment":"Save plan only",showInformationMessage:async()=>{}}},
    "node:fs/promises":{open:async()=>{files++;return{writeFile:async()=>{},sync:async()=>{},close:async()=>{}};}},
    "./core/targetDraft":drafts,"./core/runnerAssessment":{ensureAssessmentReadiness:async()=>{ready++;throw Error("VM stopped");}},
    "./core/runnerTargetPreflight":{targetComputeRate:()=>1,preflightTarget:async()=>{throw Error("Unexpected live preflight");}},
    "./core/runnerTarget":{sourceTargetEvidence:()=>evidence,targetPreview:(_r:unknown,i:unknown)=>({input:i,hash:"hash",serverId:"server",phase:"previewed",subnetId:"subnet"}),submitTarget:async()=>{submits++;}}
  });
  const store={read:async()=>r,write:async(next:RunnerRecord)=>{r=next;},readReport:async()=>"abc",exclusive:async(_id:string,fn:any)=>fn()};
  const control={persist:store.write},azure={retailRates:async()=>[]};
  // First deployment fails before writing a plan or creating a target; draft survives.
  deploy=true;await assert.rejects(()=>adapter.reviewRunnerTarget({},control,store,azure,r.id),/VM stopped/);
  assert.equal(r.targetDraft?.folder,"/approved-folder");assert.equal(r.target,undefined);assert.equal(submits,0);assert.equal(files,0);
  // Save-only uses the same inputs without requesting live readiness.
  deploy=false;
  await adapter.reviewRunnerTarget({},control,store,azure,r.id);
  assert.equal(ready,1);assert.equal(files,2);assert.equal((r as RunnerRecord).target?.phase,"previewed");assert.equal(submits,0);
});
test("one resize grant is exact-scope, deadline-limited and invalidated by plan changes",()=>{
  const r=catalogFixture().record,now=Date.now();
  r.target={phase:"provisioned",hash:"hash",serverId:"server",input:{deadline:new Date(now+3600000).toISOString(),hourlyUSD:1,budgetUSD:800,additionalReserveUSD:700}} as any;
  const next=authorizeResize(r,now);assert.equal(resizeAuthorized(next,now),true);assert.equal(resizeAuthorized(next,now+20*60000),false);
  next.target!.input.loaderSize="Standard_D16s_v5";assert.equal(resizeAuthorized(next,now),false);
});
