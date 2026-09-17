import assert from "node:assert/strict";
import test from "node:test";
import {readFileSync} from "node:fs";
import {createRequire} from "node:module";
import {join} from "node:path";
import {Script} from "node:vm";
import {transformSync} from "esbuild";
import {object, sourceWorkflowDraft} from "../../core/runner";
import * as cosmosAccess from "../../core/runnerCosmosAccess";

// Production source-panel handler and Cosmos controller, inert environment only.
// Does not edit the operator's store or exercise an installed Extension Host.
const id = "11111111-1111-4111-8111-111111111111", principal = "22222222-2222-4222-8222-222222222222";
const account = `/subscriptions/${id}/resourceGroups/source/providers/Microsoft.DocumentDB/databaseAccounts/p1source`;
const code = transformSync(readFileSync(join(__dirname,"../../runnerSourcePanel.ts"),"utf8"),{loader:"ts",format:"cjs"}).code;
function fixture() {
  let record = sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"migration",region:"japaneast",zone:"1",size:"Standard_B2s_v2",subnetId:"unused",source:{type:"cosmos-nosql",location:"azure",resourceId:account}});
  record.phase = "provisioned";
  record.cosmosAccess = {phase:"previewed",principalId:principal,scope:account,assignmentId:`${account}/sqlRoleAssignments/${id}`,roleDefinitionId:`${account}/sqlRoleDefinitions/00000000-0000-0000-0000-000000000001`};
  let receive = async (_m:unknown)=>{}, dispose = ()=>{}, onConfirm = ()=>{}, answer:string|undefined = "Grant Data Reader", confirmations = 0;
  const workspace = {isTrusted:true}, methods:string[] = [], snapshots:typeof record[] = [], messages:Record<string,any>[] = [];
  const store = {read:async()=>structuredClone(record),write:async(r:typeof record)=>{record=structuredClone(r);snapshots.push(structuredClone(r));},exclusive:async(_id:string,fn:()=>unknown)=>fn()};
  const control = {persist:store.write,request:async(_subscription:string,path:string,method="GET")=>{
    methods.push(method);
    if(path===`${record.vmId}?api-version=2024-07-01`)return {status:200,value:{identity:{type:"SystemAssigned",principalId:principal},tags:{workflow:id,application:"agefreighter"}}};
    return {status:method==="PUT"?201:404,value:{}};
  }};
  const modules:Record<string,unknown> = {
    vscode:{workspace,ViewColumn:{One:1},window:{
      createWebviewPanel:()=>({onDidDispose:(fn:()=>void)=>{dispose=fn;},webview:{html:"",postMessage:async(m:Record<string,any>)=>{messages.push(m);},onDidReceiveMessage:(fn:typeof receive)=>{receive=fn;return {dispose:()=>{}};}}}),
      showWarningMessage:async()=>{confirmations++;onConfirm();return answer;}
    }},
    "./core/runner":{object},"./core/runnerAssessment":{assessmentActive:()=>false},
    "./core/runnerSourceView":{runnerSourceHTML:()=>"test"},"./core/runnerCosmosAccess":cosmosAccess
  };
  const output={exports:{openRunnerSource:(_context:unknown,_control:unknown,_store:unknown,_id:string)=>{}}},native=createRequire(__filename);
  new Script(code).runInNewContext({module:output,exports:output.exports,Error,require:(name:string)=>name in modules?modules[name]:name.startsWith("node:")?native(name):{}});
  output.exports.openRunnerSource({subscriptions:[]},control,store,id);
  return {run:()=>receive({action:"cosmosAccess"}),record:()=>record,workspace,methods,snapshots,messages,
    confirmations:()=>confirmations,cancel:()=>{answer=undefined;},duringConfirm:(fn:()=>void)=>{onConfirm=fn;},dispose:()=>dispose()};
}

test("untrusted source panel cannot preview or grant Cosmos access",async()=>{
  const f=fixture();f.workspace.isTrusted=false;f.record().cosmosAccess=undefined;
  await f.run();
  assert.deepEqual(f.methods,[]);assert.equal(f.snapshots.length,0);assert.equal(f.confirmations(),0);
  assert.ok(f.messages.some(m=>m.kind==="error"&&/Trust/.test(m.text)));
});

test("Cosmos approval rechecks trust after the native dialog",async()=>{
  const f=fixture();f.duringConfirm(()=>{f.workspace.isTrusted=false;});await f.run();
  assert.equal(f.confirmations(),1);assert.deepEqual(f.methods,[]);assert.equal(f.snapshots.length,0);
  assert.ok(f.messages.some(m=>m.kind==="error"&&/Trust/.test(m.text)));
});

for(const action of ["cancel","dispose"] as const)test(`Cosmos ${action} leaves the reviewed grant untouched`,async()=>{
  const f=fixture(),before=structuredClone(f.record());
  if(action==="cancel")f.cancel();else f.duringConfirm(f.dispose);
  await f.run();assert.deepEqual(f.record(),before);assert.deepEqual(f.methods,[]);assert.equal(f.snapshots.length,0);
});

for(const change of ["principal","scope","vm"] as const)test(`Cosmos approval rejects concurrently changed ${change} instead of granting unreviewed access`,async()=>{
  const f=fixture();f.duringConfirm(()=>{
    if(change==="principal")f.record().cosmosAccess!.principalId="33333333-3333-4333-8333-333333333333";
    if(change==="vm")f.record().vmId+="-other";
    if(change==="scope"){
      f.record().input.source.resourceId=account+"-other";
      const a=f.record().cosmosAccess!;a.scope=account+"-other";a.assignmentId=a.assignmentId.replace(account,a.scope);a.roleDefinitionId=a.roleDefinitionId.replace(account,a.scope);
    }
  });
  await f.run();assert.deepEqual(f.methods,[]);assert.equal(f.snapshots.length,0);
  assert.ok(f.messages.some(m=>m.kind==="error"&&/changed/.test(m.text)));
});

test("unchanged explicitly approved Cosmos grant submits exactly once",async()=>{
  const f=fixture();await f.run();
  assert.equal(f.confirmations(),1);assert.equal(f.methods.filter(m=>m==="PUT").length,1);
  assert.equal(f.record().cosmosAccess?.phase,"submitted");assert.equal(f.snapshots.length,1);
  await f.run();assert.equal(f.confirmations(),1);assert.equal(f.methods.filter(m=>m==="PUT").length,1);
});
