import assert from "node:assert/strict";
import test from "node:test";
import {readFileSync} from "node:fs";
import {createRequire} from "node:module";
import {join} from "node:path";
import {Script} from "node:vm";
import {transformSync} from "esbuild";
import {object, sourceWorkflowDraft} from "../../core/runner";
import * as assessment from "../../core/runnerAssessment";

const id="11111111-1111-4111-8111-111111111111",boot="22222222-2222-4222-8222-222222222222";
const code=transformSync(readFileSync(join(__dirname,"../../runnerSourcePanel.ts"),"utf8"),{loader:"ts",format:"cjs"}).code;
function fixture(){
  let record=sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"trial",region:"japaneast",zone:"1",size:"Standard_B2s_v2",subnetId:"unused",source:{type:"neo4j",location:"on-premises"}});
  record.phase="provisioned";
  record.artifact={version:"2.4.0",sha256:"a".repeat(64),url:"https://example.invalid/a"};
  record.assessment={operation:id,action:"inventory",phase:"failed",bootId:id,configurationSHA256:"c".repeat(64)};
  const time=new Date().toISOString();
  record.guestCommand={id:"ready-command",operation:boot,action:"ready",phase:"finished",submittedAt:time};
  record.guestReady={bootId:boot,cliVersion:"2.4.0",commit:"commit",archiveSha256:"a".repeat(64),checkedAt:time,health:{idle:true,storageUsedPercent:4,swapUsedBytes:0,oomEvents:0}};
  let receive=async(_m:unknown)=>{},onConfirm=()=>{},dispose=()=>{},answer:string|undefined="Retain failure and review source";
  const workspace={isTrusted:true},messages:Record<string,any>[]=[],saved:typeof record[]=[],details:string[]=[];
  const store={read:async()=>structuredClone(record),write:async(r:typeof record)=>{record=structuredClone(r);saved.push(structuredClone(r));},exclusive:async(_id:string,fn:()=>unknown)=>fn()};
  const modules:Record<string,unknown>={
    vscode:{workspace,ViewColumn:{One:1},window:{
      createWebviewPanel:()=>({onDidDispose:(fn:()=>void)=>{dispose=fn;},webview:{html:"",postMessage:async(m:Record<string,any>)=>{messages.push(m);},onDidReceiveMessage:(fn:typeof receive)=>{receive=fn;return{dispose:()=>{}};}}}),
      showWarningMessage:async(_title:string,options:{detail:string})=>{details.push(options.detail);onConfirm();return answer;}
    }},"./core/runner":{object},"./core/runnerAssessment":assessment,"./core/runnerSourceView":{runnerSourceHTML:()=>"test"}
  };
  const output={exports:{openRunnerSource:(_a:unknown,_b:unknown,_c:unknown,_d:string)=>{}}},native=createRequire(__filename);
  new Script(code).runInNewContext({module:output,exports:output.exports,Error,require:(n:string)=>n in modules?modules[n]:n.startsWith("node:")?native(n):{}});
  output.exports.openRunnerSource({subscriptions:[]},{},store,id); // No Azure/guest dispatcher available.
  return{record:()=>record,workspace,messages,saved,details,run:()=>receive({action:"retainFailure"}),duringConfirm:(f:()=>void)=>{onConfirm=f;},cancel:()=>{answer=undefined;},dispose:()=>dispose()};
}
test("source panel explicitly retains terminal evidence across reboot without any dispatch",async()=>{
  const f=fixture(),before=structuredClone(f.record().assessment);
  await f.run();
  assert.equal(f.saved.length,1);assert.equal(f.record().assessment,undefined);
  assert.deepEqual(f.record().assessmentHistory,[before]);
  assert.ok(f.details[0]?.includes(`Previous boot: ${id}`));
  assert.ok(f.details[0]?.includes(`Verified current boot: ${boot}`));
  await f.run();assert.equal(f.saved.length,1);
});
for(const change of ["boot","artifact","vm","trust","busy","cancel","dispose"] as const)test(`retained-failure confirmation rejects ${change} without archiving`,async()=>{
  const f=fixture();
  f.duringConfirm(()=>{
    if(change==="boot")f.record().guestReady!.bootId="33333333-3333-4333-8333-333333333333";
    if(change==="artifact")f.record().artifact.sha256="b".repeat(64);
    if(change==="vm")f.record().vmId+="-changed";
    if(change==="trust")f.workspace.isTrusted=false;
    if(change==="busy")f.record().guestReady!.health!.idle=false;
    if(change==="cancel")f.cancel();
    if(change==="dispose")f.dispose();
  });
  await f.run();assert.equal(f.saved.length,0);assert.equal(f.record().assessment?.phase,"failed");
  if(change!=="cancel"&&change!=="dispose")assert.ok(f.messages.some(m=>m.kind==="error"));
});
