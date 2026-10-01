import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { Script } from "node:vm";
import { transformSync } from "esbuild";
import { RunnerRecord } from "../../core/runner";
import { RunnerControl } from "../../core/runnerLifecycle";
import * as target from "../../core/runnerTarget";
import * as drafts from "../../core/targetDraft";
import * as executionActions from "../../core/runnerExecutionActions";
import { catalogFixture, catalogForm } from "../catalogFixtures";

function load<T>(file:string,modules:Record<string,unknown>,clock:unknown=Date):T {
  const output={exports:{}},native=createRequire(__filename);
  const code=transformSync(readFileSync(join(__dirname,"../../",file),"utf8"),{loader:"ts",format:"cjs"}).code;
  new Script(code).runInNewContext({module:output,exports:output.exports,Error,Buffer,Date:clock,
    require:(name:string)=>name in modules?modules[name]:name.startsWith("node:")?native(name):(()=>{throw Error("Unexpected dependency "+name);})()});
  return output.exports as T;
}
type Item={label:string;prefix?:string;hours?:number;value?:string;detail?:string};
type Options={placeHolder?:string;title?:string};
type NativeController=(context:unknown,control:RunnerControl,store:unknown,azure:unknown,workflow:string)=>Promise<void>;
function fixture(){
  const record=catalogFixture().record,workspace={isTrusted:true},token={isCancellationRequested:false};
  record.input.subnetId=`/subscriptions/${record.input.subscriptionId}/resourceGroups/trial/providers/Microsoft.Network/virtualNetworks/demo/subnets/runner`;
  const vnet={properties:{addressSpace:{addressPrefixes:["10.0.0.0/16"]},subnets:[{properties:{addressPrefix:"10.0.1.0/24"}}]}};
  const calls:{items:Item[];options:Options}[]=[],requests:string[]=[];
  let select:(items:Item[],options:Options)=>Item|undefined=items=>items[0];
  const window={
    withProgress:async(_options:unknown,run:(progress:unknown,token:{isCancellationRequested:boolean})=>Promise<unknown>)=>run({},token),
    showQuickPick:async(items:Item[],options:Options)=>{calls.push({items,options});return select(items,options);}
  };
  const vscode={workspace,window,ProgressLocation:{Notification:1}};
  const control:RunnerControl={
    request:async(sub,path,method="GET")=>{
      assert.equal(sub,record.input.subscriptionId);assert.equal(method,"GET");
      assert.equal(path,record.input.subnetId.replace(/\/subnets\/[^/]+$/i,"")+"?api-version=2024-05-01");
      requests.push(path);return {status:200,value:vnet};
    },
    persist:async()=>{throw Error("No cloud intent expected");},
    list:async()=>{throw Error("No other network read expected");},
    sleep:async()=>{throw Error("No polling expected");}
  };
  return {record,workspace,token,vnet,control,vscode,calls,requests,setSelect:(fn:typeof select)=>{select=fn;},
    adapter:load<typeof import("../../runnerTargetInputs")>("runnerTargetInputs.ts",{vscode,"./core/runnerTarget":target})};
}
test("subnet picker reads exact VNet and preserves a saved free CIDR, even outside its suggested /28 list",async()=>{
  const f=fixture();
  assert.equal(await f.adapter.pickTargetSubnet(f.control,f.record,"10.0.24.0/24",true),"10.0.24.0/24");
  assert.equal(f.requests.length,1);assert.equal(f.calls.length,0);
  assert.equal(await f.adapter.pickTargetSubnet(f.control,f.record,"10.0.24.0/24"),"10.0.24.0/24");
  assert.equal(f.calls[0]!.items[0]!.label,"10.0.24.0/24");
});
test("saved overlapping CIDR triggers an explained replacement choice, never silent reuse",async()=>{
  const f=fixture();
  assert.equal(await f.adapter.pickTargetSubnet(f.control,f.record,"10.0.1.0/24",true),"10.0.0.0/28");
  assert.match(f.calls[0]!.options.placeHolder!,/cannot be reused.*overlap/);
  assert.ok(f.calls[0]!.items.every(item=>item.prefix!=="10.0.1.0/24"));
  for(const item of f.calls[0]!.items)target.validateTargetSubnet(item.prefix!,f.vnet);
});
test("subnet selection cancel, discovery cancel and trust loss never choose a range",async()=>{
  const f=fixture();f.setSelect(()=>undefined);
  assert.equal(await f.adapter.pickTargetSubnet(f.control,f.record),undefined);
  f.token.isCancellationRequested=true;
  assert.equal(await f.adapter.pickTargetSubnet(f.control,f.record),undefined);
  assert.equal(f.calls.length,1);
  f.token.isCancellationRequested=false;f.setSelect(items=>{f.workspace.isTrusted=false;return items[0];});
  assert.equal(await f.adapter.pickTargetSubnet(f.control,f.record),undefined);
});
test("network read failure, incomplete evidence and exhausted VNet give actionable errors",async()=>{
  const f=fixture();
  await assert.rejects(()=>f.adapter.pickTargetSubnet({...f.control,request:async()=>({status:403,value:{}})},f.record),/read permissions/);
  await assert.rejects(()=>f.adapter.pickTargetSubnet({...f.control,request:async()=>({status:200,value:{properties:{addressSpace:{},subnets:[]}}})},f.record),/prefixes/);
  f.vnet.properties.addressSpace.addressPrefixes=["10.0.1.0/24"];
  await assert.rejects(()=>f.adapter.pickTargetSubnet(f.control,f.record),/No free IPv4.*network owner/);
  assert.equal(f.calls.length,0);
});
test("all six durations compute UTC at selection time, not when the picker opens",async()=>{
  const f=fixture(),opened=Date.parse("2026-09-29T08:02:53.453Z");
  let now=opened;
  class Clock extends Date {static override now(){return now;}}
  const adapter=load<typeof import("../../runnerTargetInputs")>("runnerTargetInputs.ts",{vscode:f.vscode,"./core/runnerTarget":target},Clock);
  for(const [label,hours] of [["1 hour",1],["6 hours",6],["12 hours",12],["1 day",24],["2 days",48],["3 days",72]] as const){
    f.setSelect(items=>{now+=300000;return items.find(item=>item.label===label);});
    assert.equal(await adapter.pickTargetDeadline(),new Date(now+hours*3600000).toISOString());
    assert.equal(f.calls.at(-1)!.items.length,6);
    assert.match(f.calls.at(-1)!.options.placeHolder!,/does not schedule shutdown/);
  }
});
test("reused deadline never extends silently; expired/invalid values require selection and permit cancel",async()=>{
  const f=fixture(),previous=new Date(Date.now()+3600000).toISOString();
  assert.equal(await f.adapter.pickTargetDeadline(previous,true),previous);assert.equal(f.calls.length,0);
  f.setSelect(()=>undefined);
  for(const invalid of ["bad","2020-01-01T00:00:00Z",new Date(Date.now()+97*3600000).toISOString()]){
    assert.equal(await f.adapter.pickTargetDeadline(invalid,true),undefined);
    assert.match(f.calls.at(-1)!.options.placeHolder!,/no automatic renewal/);
  }
  f.setSelect(items=>{f.workspace.isTrusted=false;return items[0];});
  assert.equal(await f.adapter.pickTargetDeadline(),undefined);
});

test("target review uses free-range and duration pickers, GiB-only sizing, and retains inputs without deployment on cancel",async()=>{
  const f=fixture();let r=f.record;
  r.sourceDraft={configuration:{source:{type:"postgresql"}},form:catalogForm,warnings:[],canAssess:true};
  r.assessment={operation:"inventory",action:"inventory",phase:"finished",bootId:"boot",configurationSHA256:"a".repeat(64),reportSHA256:"b".repeat(64),reportBytes:3};
  const prompts:string[]=[],picks:{items:Item[];options:Options}[]=[];
  let modal="";
  const window={
    ...f.vscode.window,
    showInputBox:async(options:{prompt:string;value:string})=>{prompts.push(options.prompt);return options.value;},
    showQuickPick:async(items:Item[],options:Options)=>{picks.push({items,options});return items.find(item=>item.hours===24)??items[0];},
    showWarningMessage:async(_title:string,options:{detail:string})=>{modal=options.detail;return undefined;}
  };
  const vscode={...f.vscode,window};
  const inputs=load<typeof import("../../runnerTargetInputs")>("runnerTargetInputs.ts",{vscode,"./core/runnerTarget":target});
  const panel=load<{reviewRunnerTarget:NativeController}>("runnerTargetPanel.ts",{
    vscode,"./runnerWatch":{},"./runnerTargetInputs":inputs,"./core/targetDraft":drafts,"./core/runnerAssessment":{},
    "./core/runnerTargetPreflight":{targetComputeRate:()=>1},
    "./core/runnerTarget":{...target,sourceTargetEvidence:()=>({operation:"inventory",reportSHA256:"b".repeat(64),configurationSHA256:"a".repeat(64),artifactSHA256:r.artifact.sha256,sourceType:"postgresql",rows:"350000",labels:{},storageHighBytes:"5734400000"})}
  });
  const store={read:async()=>r,write:async(next:RunnerRecord)=>{r=next;},readReport:async()=>"abc",exclusive:async<T>(_id:string,run:()=>Promise<T>)=>run()};
  const before=Date.now();
  await panel.reviewRunnerTarget({},f.control,store,{retailRates:async()=>[]},r.id);
  assert.equal(r.target,undefined);assert.equal(r.targetDraft!.input.subnetCIDR,"10.0.0.0/28");
  assert.equal(r.targetDraft!.input.storageGiB,128);
  assert.ok(Date.parse(r.targetDraft!.input.deadline!)>=before+24*3600000);
  assert.ok(Date.parse(r.targetDraft!.input.deadline!)<=Date.now()+24*3600000);
  assert.ok(!prompts.some(p=>/CIDR|deadline|ISO 8601/.test(p)));
  const storage=picks.find(p=>p.options.placeHolder?.startsWith("Target storage:"))!;
  assert.match(storage.options.placeHolder!,/5.35 GiB.*6.68 GiB/);
  assert.doesNotMatch(storage.options.placeHolder!,/bytes/);
  assert.ok(storage.items.every(item=>/^\d+ GiB$/.test(item.label)));
  assert.match(modal,/128 GiB/);assert.ok(modal.includes(r.targetDraft!.input.deadline!));
  assert.equal(f.requests.length,1);
});

test("cost renewal uses duration choices and preserves the original authorization when approval is cancelled",async()=>{
  const f=fixture(),originalDeadline=new Date(Date.now()+3600000).toISOString();
  f.record.target=target.targetPreview(f.record,{
    deadline:originalDeadline,budgetUSD:100,additionalReserveUSD:50,hourlyUSD:1,
    loaderSize:"Standard_D4s_v5",postgresSKU:"Standard_D4ds_v5",postgresTier:"GeneralPurpose",
    serverName:"test-target",subnetCIDR:"10.0.2.0/28",storageGiB:128
  },{operation:"inventory",reportSHA256:"b".repeat(64),configurationSHA256:"a".repeat(64),
    artifactSHA256:f.record.artifact.sha256,sourceType:"postgresql",rows:"350000",labels:{},storageHighBytes:"5734400000"});
  f.record.target.phase="provisioned";
  const initial=JSON.stringify(f.record),prompts:string[]=[],action="Review a new cost authorization (no Azure mutation)";
  let modal="";
  const window={
    ...f.vscode.window,
    showQuickPick:async(items:(Item|string)[])=>items.includes(action)?action:items.find(item=>typeof item!=="string"&&item.hours===6),
    showInputBox:async(options:{prompt:string;value:string})=>{prompts.push(options.prompt);return options.value;},
    showWarningMessage:async(_title:string,options:{detail:string})=>{modal=options.detail;return undefined;}
  };
  const vscode={...f.vscode,window},modules:Record<string,unknown>={vscode,
    "./runnerTargetInputs":load<typeof import("../../runnerTargetInputs")>("runnerTargetInputs.ts",{vscode,"./core/runnerTarget":target}),
    "./core/runnerTarget":target,"./core/runnerTargetPreflight":{targetComputeRate:()=>1},
    "./core/runnerExecutionActions":executionActions,"./core/runnerGuest":{}};
  for(const name of ["./core/runnerResize","./core/runnerExecution","./core/report","./core/runnerDiagnostic","./p1QualificationPanel","./p1DiagnosticPanel","./core/runnerSource","./core/runnerAssessment","./core/runnerResume","./migrationVerificationPanel","./sourceCredentialPanel","./runnerWatch","./runnerReportFlow","./core/resizeAuthorization","./core/boundedWatch"])modules[name]={};
  const panel=load<{continueRunnerExecution:NativeController}>("runnerExecutionPanel.ts",modules);
  const before=Date.now();
  await panel.continueRunnerExecution({},f.control,{read:async()=>f.record},{retailRates:async()=>[]},f.record.id);
  assert.equal(JSON.stringify(f.record),initial);assert.equal(prompts.length,2);
  assert.ok(!prompts.some(p=>/deadline|ISO/.test(p)));
  const deadline=/New deadline ([^;]+);/.exec(modal)![1]!;
  assert.ok(Date.parse(deadline)>=before+6*3600000 && Date.parse(deadline)<=Date.now()+6*3600000);
  assert.match(modal,/No Azure operation/);
});
