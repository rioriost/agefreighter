import assert from "node:assert/strict";
import test from "node:test";
import {sourceWorkflowDraft} from "../../core/runner";
import {diagnoseTarget,targetDiagnosticScript,archiveEmptyTargetFailure,needsTargetDiagnosis} from "../../core/runnerDiagnostic";
import {RunnerControl} from "../../core/runnerLifecycle";
const id="11111111-1111-4111-8111-111111111111";
test("only expired terminal diagnostics request a new approved read",()=>{
  const r=sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"test",region:"japaneast",zone:"1",subnetId:"subnet",size:"Standard_D4s_v5",source:{type:"csv",location:"local"}});
  const now=Date.now();assert.equal(needsTargetDiagnosis(r,now),true);
  for(const phase of ["submitted","unknown","finished","failed"] as const){
    r.targetDiagnostic={operation:id,commandId:"retained",phase,submittedAt:new Date(now-900001).toISOString()};
    assert.equal(needsTargetDiagnosis(r,now),phase==="finished"||phase==="failed");
    for(const timestamp of [new Date(now).toISOString(),"invalid",new Date(now+1).toISOString()]){
      r.targetDiagnostic.submittedAt=timestamp;assert.equal(needsTargetDiagnosis(r,now),false);
    }
  }
});
test("fresh diagnostic preserves expired evidence and persists a new intent before PUT",async()=>{
  const r=sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"test",region:"japaneast",zone:"1",subnetId:"subnet",size:"Standard_D4s_v5",source:{type:"csv",location:"local"}});
  r.migration={phase:"failed",jobId:id} as any;
  r.artifact={version:"dev",sha256:"a".repeat(64),url:"https://example.invalid"};
  r.guestReady={checkedAt:new Date().toISOString(),bootId:id,cliVersion:"dev",archiveSha256:"a".repeat(64),commit:"b".repeat(40),health:{idle:true,storageUsedPercent:7,swapUsedBytes:0,oomEvents:0}};
  r.target={phase:"provisioned",serverId:"target",input:{serverName:"afpg-test",deadline:new Date(Date.now()+3600000).toISOString(),budgetUSD:800,additionalReserveUSD:600,hourlyUSD:1.448}} as any;
  const previous={operation:id,commandId:r.vmId+"/runCommands/af-"+id,phase:"finished" as const,submittedAt:new Date(Date.now()-900001).toISOString(),result:{retained:true}};
  r.targetDiagnostic=previous;let persisted:any;let puts=0;
  const c:RunnerControl={list:async()=>[],sleep:async()=>{},persist:async value=>{persisted=value;},request:async(_s,path,method="GET")=>{
    if(method==="PUT"){puts++;assert.equal(persisted.targetDiagnosticHistory[0],previous);assert.equal(persisted.targetDiagnostic.phase,"submitted");throw Error("lost response");}
    if(path.startsWith("target?"))return {status:200,value:{tags:{workflow:id,application:"agefreighter",purpose:"migration-target"},properties:{state:"Ready"}}};
    if(persisted?.targetDiagnostic?.phase==="unknown")return {status:200,value:{properties:{instanceView:{executionState:"Running"}}}};
    return {status:404,value:{}};
  }};
  await assert.rejects(diagnoseTarget(c,r),/Credentials/);assert.equal(puts,0);
  const next=await diagnoseTarget(c,r,"x".repeat(32));assert.equal(puts,1);assert.equal(next.targetDiagnostic?.phase,"unknown");assert.equal(next.targetDiagnosticHistory?.[0],previous);assert.equal(r.targetDiagnostic,previous);
  assert.notEqual(next.targetDiagnostic?.operation,id);assert.equal(needsTargetDiagnosis(next),false);
  assert.equal(await diagnoseTarget(c,next),next);assert.equal(puts,1);
});
test("empty-target reconciliation preserves failed evidence and refuses existing or uncertain jobs",async()=>{
  const r=sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"test",region:"japaneast",zone:"1",subnetId:"subnet",size:"Standard_D4s_v5",source:{type:"csv",location:"local"}});
  r.migration={phase:"failed",jobId:id} as any;
  r.artifact={version:"dev",sha256:"a".repeat(64),url:"https://example.invalid"};
  r.guestReady={checkedAt:new Date().toISOString(),bootId:id,cliVersion:"dev",archiveSha256:"a".repeat(64),commit:"b".repeat(40),health:{idle:true,storageUsedPercent:7,swapUsedBytes:0,oomEvents:0}};
  r.targetDiagnostic={operation:id,commandId:"retained",phase:"finished",submittedAt:new Date().toISOString(),result:{jobId:id,exitCode:0,summary:{errors:[],checks:[{id:"metadata-schema",status:"unavailable",detail:"installed=0 supported=21 pending=0; doctor does not migrate"},{id:"target-graph",status:"pass",detail:"target graph is absent and create mode may create it"}]}}};
  const c:RunnerControl={list:async()=>{throw Error("unexpected");},request:async()=>{throw Error("unexpected");},sleep:async()=>{},persist:async()=>{}};
  const next=await archiveEmptyTargetFailure(c,r);
  assert.equal(next.migration,undefined);assert.equal(next.migrationHistory?.[0]?.migration,r.migration);assert.equal(next.migrationHistory?.[0]?.diagnostic,r.targetDiagnostic);
  for(const patch of [{phase:"running"},{fingerprint:"a".repeat(64)},{reportSHA256:"b".repeat(64)}])await assert.rejects(archiveEmptyTargetFailure(c,{...r,migration:{...r.migration!,...patch} as any}));
  await assert.rejects(archiveEmptyTargetFailure(c,{...r,targetDiagnostic:{...r.targetDiagnostic!,submittedAt:"2020-01-01T00:00:00Z"}}));
  await assert.rejects(archiveEmptyTargetFailure(c,{...r,targetDiagnostic:{...r.targetDiagnostic!,result:{jobId:id,exitCode:0,summary:{errors:[],checks:[]}}}}));
});
test("diagnosis is terminal-only, GET-reconciled and never replays a lost submission",async()=>{
  const r=sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"test",region:"japaneast",zone:"1",subnetId:"subnet",size:"Standard_D4s_v5",source:{type:"csv",location:"local"}});
  const requests:string[]=[];
  const c:RunnerControl={list:async()=>[],sleep:async()=>{},persist:async()=>{},request:async(_s,_p,method="GET")=>{requests.push(method);return {status:200,value:{properties:{instanceView:{executionState:"Running"}}}};}};
  await assert.rejects(diagnoseTarget(c,r),/terminal/);assert.deepEqual(requests,[]);
  r.target={phase:"provisioned"} as any;r.migration={phase:"running",jobId:id} as any;
  await assert.rejects(diagnoseTarget(c,r),/terminal/);
  r.migration!.phase="failed";r.targetDiagnostic={operation:id,commandId:r.vmId+"/runCommands/af-"+id,phase:"unknown",submittedAt:new Date().toISOString()};
  assert.equal(await diagnoseTarget(c,r),r);assert.deepEqual(requests,["GET"]);
  assert.ok(targetDiagnosticScript.includes("os.environ.pop('AF_DIAGNOSTIC')"));
  assert.ok(targetDiagnosticScript.includes("'doctor','--target'"));
  assert.ok(!targetDiagnosticScript.includes("'--persist'"));
  assert.ok(!targetDiagnosticScript.includes("'load'"));
});
