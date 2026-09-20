import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { Script } from "node:vm";
import { transformSync } from "esbuild";
import { object, sourceWorkflowDraft } from "../../core/runner";
import { canRetainRejectedReportExport } from "../../core/runnerReport";

const id="11111111-1111-4111-8111-111111111111";
const code=transformSync(readFileSync(join(__dirname,"../../runnerSourcePanel.ts"),"utf8"),{loader:"ts",format:"cjs"}).code;
function fixture(){
  const record=sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"trial",region:"japaneast",zone:"1",size:"Standard_B2s_v2",subnetId:"unused",source:{type:"postgresql",location:"azure"}});
  record.phase="provisioned";
  record.assessment={operation:id,action:"inventory",phase:"finished",bootId:id,configurationSHA256:"b".repeat(64),reportSHA256:"a".repeat(64),reportBytes:2947};
  record.guestCommand={id:`${record.vmId}/runCommands/af-${id}`,operation:id,action:"export-report",phase:"unknown",submittedAt:new Date(Date.now()-1260000).toISOString(),failure:"Azure returned HTTP 409; reconcile before retrying."};
  record.reportTransfers=[{operation:id,sha256:"a".repeat(64),bytes:2947,blob:"owned destination",phase:"unknown"}];
  let receive=async(_m:unknown)=>{},onConfirm=()=>{},dispose=()=>{},answer:string|undefined="Retain rejected export",calls=0,capabilities=0;
  const workspace={isTrusted:true},messages:any[]=[],details:string[]=[];
  const store={read:async()=>structuredClone(record),exclusive:async(_id:string,fn:()=>unknown)=>fn()};
  const modules:Record<string,unknown>={
    vscode:{workspace,ViewColumn:{One:1},window:{
      createWebviewPanel:()=>({onDidDispose:(fn:()=>void)=>{dispose=fn;},webview:{html:"",postMessage:async(m:unknown)=>{messages.push(m);},onDidReceiveMessage:(fn:typeof receive)=>{receive=fn;return{dispose:()=>{}};}}}),
      showWarningMessage:async(_title:string,options:{detail:string})=>{details.push(options.detail);onConfirm();return answer;}
    }},
    "./core/runner":{object},"./core/runnerSourceView":{runnerSourceHTML:()=>"test"},
    "./core/runnerReport":{canRetainRejectedReportExport,retainRejectedReportExport:async(_c:unknown,r:typeof record,cap:string)=>{assert.equal(cap,"read-only-capability");calls++;return r;}}
  };
  const output={exports:{openRunnerSource:(_a:unknown,_b:unknown,_c:unknown,_d:string,_e:unknown)=>{}}},native=createRequire(__filename);
  new Script(code).runInNewContext({module:output,exports:output.exports,Error,require:(n:string)=>n in modules?modules[n]:n.startsWith("node:")?native(n):{}});
  output.exports.openRunnerSource({subscriptions:[]},{},store,id,{reportCapability:async(_r:unknown,op:string,permission:string)=>{assert.equal(op,id);assert.equal(permission,"r");capabilities++;return "read-only-capability";}});
  return{record,workspace,messages,details,run:()=>receive({action:"retainRejectedExport"}),duringConfirm:(f:()=>void)=>{onConfirm=f;},cancel:()=>{answer=undefined;},dispose:()=>dispose(),calls:()=>calls,capabilities:()=>capabilities};
}
test("rejected export native approval requests only a read capability and never dispatches a worker",async()=>{
  const f=fixture();await f.run();assert.equal(f.calls(),1);assert.equal(f.capabilities(),1);
  assert.match(f.details[0]!,/twenty minutes/);assert.match(f.details[0]!,/does not start a VM/);
  assert.ok(!JSON.stringify(f.messages).includes("read-only-capability"));
});
for(const change of ["trust","cancel","dispose","seal","vm","command","transfer"] as const)test(`rejected export native approval refuses ${change} before capability or state change`,async()=>{
  const f=fixture();f.duringConfirm(()=>{
    if(change==="trust")f.workspace.isTrusted=false;
    if(change==="cancel")f.cancel();
    if(change==="dispose")f.dispose();
    if(change==="seal")f.record.assessment!.reportSHA256="c".repeat(64);
    if(change==="vm")f.record.vmId+="other";
    if(change==="command")f.record.guestCommand!.id+="other";
    if(change==="transfer")f.record.reportTransfers![0]!.blob="other";
  });await f.run();assert.equal(f.calls(),0);assert.equal(f.capabilities(),0);
});
