import assert from "node:assert/strict";
import test from "node:test";
import {createHash} from "node:crypto";
import {sourceWorkflowDraft} from "../../core/runner";
import {RunnerControl} from "../../core/runnerLifecycle";
import {inspectResume,validateResumeInspection,resumeAdmission,resumeMigration} from "../../core/runnerResume";
import {recoveryIdentity,refreshMigration} from "../../core/runnerExecution";
const id="11111111-1111-4111-8111-111111111111";
function fixture(){
  const r=sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"test",region:"japaneast",zone:"1",subnetId:"subnet",size:"Standard_D4s_v5",source:{type:"csv",location:"local"}});
  r.phase="provisioned";r.artifact={version:"dev",sha256:"a".repeat(64),url:"https://example.invalid"};
  r.target={phase:"provisioned",serverId:"target",input:{serverName:"afpg-test",deadline:new Date(Date.now()+3600000).toISOString(),budgetUSD:800,additionalReserveUSD:400,hourlyUSD:0.736}} as any;
  r.migration={operation:id,jobId:id,phase:"interrupted",bootId:id,startedAt:new Date().toISOString(),artifactSHA256:r.artifact.sha256,cliVersion:"dev",guestConfigurationSHA256:"b".repeat(64),evidence:{rows:"5600000"} as any};
  r.guestReady={bootId:id,cliVersion:"dev",archiveSha256:r.artifact.sha256,commit:"a".repeat(40),checkedAt:new Date().toISOString(),capabilities:["resume-inspection-v1"]};
  const receipt={version:1,workflow:id,operation:id,jobId:id,bootId:id,configSha256:"b".repeat(64),fingerprint:"c".repeat(64),generationId:"9007199254740993",committedRows:"1400000",checkpointAt:new Date(Date.now()-1000).toISOString(),checkedAt:new Date().toISOString(),outcome:"review-required",canResume:false,reasons:["Explicit review required"]};
  return {r,receipt};
}
test("recovery inspection preserves 64-bit identity and never authorizes resume or PASS",()=>{
  const {r,receipt}=fixture(),v=validateResumeInspection(r,receipt);
  assert.equal(v.generationId,"9007199254740993");assert.equal(v.canResume,false);
  assert.equal(r.migration?.verification,undefined);
});

function resumable(){
  const f=fixture(),r=f.r;
  r.target!.input.loaderSize="Standard_D4s_v5";
  const vm={location:"japaneast",zones:["1"],identity:{type:"SystemAssigned",principalId:id,tenantId:id},tags:{application:"agefreighter",workflow:id,purpose:"discovery-and-migration"},properties:{provisioningState:"Succeeded",hardwareProfile:{vmSize:"Standard_D4s_v5"},instanceView:{statuses:[{code:"PowerState/running"}]},networkProfile:{networkInterfaces:[{id:"nic"}]},storageProfile:{osDisk:{managedDisk:{id:"disk"}},dataDisks:[]}}};
  r.resize={phase:"finished",preservedSHA256:createHash("sha256").update(JSON.stringify({disk:"disk",nic:"nic",principal:id,tenant:id,location:"japaneast",zones:["1"]})).digest("hex")} as any;
  r.guestReady!.capabilities!.push("explicit-resume-v1");r.guestReady!.health={idle:false,storageUsedPercent:6,swapUsedBytes:0,oomEvents:0};
  r.migration!.recoveryIdentitySHA256=recoveryIdentity(r);r.resumeInspection=validateResumeInspection(r,f.receipt);
  return {...f,vm};
}
test("resume admission binds the original build/source/target and rejects old unbound jobs",()=>{
  const {r}=resumable();assert.equal(resumeAdmission(r).jobId,id);
  for(const change of [(r:any)=>delete r.migration.recoveryIdentitySHA256,(r:any)=>r.artifact.sha256="d".repeat(64),(r:any)=>r.vmId+="changed",(r:any)=>r.target.input.serverName="another",(r:any)=>r.migration.phase="running",(r:any)=>r.resumeInspection.checkpointAt="2020-01-01T00:00:00Z",(r:any)=>r.guestReady.health.oomEvents=1]){const c=structuredClone(r);change(c);assert.throws(()=>resumeAdmission(c));}
});
test("explicit resume seals a new operation but preserves original job and evidence on lost reply",async()=>{
  const {r,vm}=resumable();const old=r.migration!,saved:any[]=[],methods:string[]=[];let payload:any;
  const control:RunnerControl={persist:async v=>{saved.push(structuredClone(v));},sleep:async()=>{},list:async()=>[],request:async(_s,path,method="GET",body?:unknown)=>{
    methods.push(method);
    if(method==="PUT"){
      assert.equal(saved.length,1);payload=JSON.parse(Buffer.from((body as any).properties.protectedParameters[0].value,"base64").toString());throw new Error("lost");
    }
    if(path.startsWith(r.vmId+"?"))return {status:200,value:vm};
    if(path.startsWith("target?"))return {status:200,value:{tags:{workflow:id,application:"agefreighter"},properties:{state:"Ready",version:"18",network:{publicNetworkAccess:"Disabled"}}}};
    return {status:404,value:undefined};
  }};
  const next=await resumeMigration(control,r,"p".repeat(32));
  assert.equal(next.migration?.jobId,old.jobId);assert.notEqual(next.migration?.operation,old.operation);
  assert.deepEqual(next.migrationContinuations,[old]);assert.equal(next.migration?.verification,undefined);assert.equal(next.resumeInspection,undefined);
  assert.equal(payload.action,"resume-migration");assert.equal(payload.configuration,undefined);assert.equal(payload.resume.jobId,id);assert.equal(payload.resume.generationId,"9007199254740993");
  assert.equal(methods.filter(x=>x==="PUT").length,1);assert.equal(next.guestCommand?.phase,"unknown");
  await assert.rejects(resumeMigration(control,next,"p".repeat(32)));assert.equal(methods.filter(x=>x==="PUT").length,1);
  assert(!JSON.stringify(saved).includes("p".repeat(32)));
  control.request=async(_s,_p,method="GET")=>{assert.equal(method,"GET");return {status:200,value:{properties:{instanceView:{executionState:"Succeeded",exitCode:0,output:JSON.stringify({version:1,workflow:id,operation:next.migration!.operation,jobId:id,action:"resume-migration",bootId:id,configSha256:old.guestConfigurationSHA256,phase:"running",resume:next.migration!.resume})}}}};};
  const checked=await refreshMigration(control,next);assert.equal(checked.migration?.phase,"running");assert.equal(checked.migration?.jobId,id);assert.equal(checked.migration?.verification,undefined);
});
test("recovery inspection rejects mismatched, stale, truncated and false-success evidence",()=>{
  const {r,receipt}=fixture();
  for(const patch of [{jobId:"foreign"},{operation:"foreign"},{bootId:"foreign"},{configSha256:"d".repeat(64)},{generationId:"9223372036854775808"},{generationId:1},{committedRows:1},{committedRows:"5600001"},{committedRows:"-1"},{fingerprint:"bad"},{checkedAt:"2020-01-01T00:00:00Z"},{checkpointAt:"2099-01-01T00:00:00Z"},{outcome:"pass"},{canResume:true},{reasons:[]},{reasons:[{}]}])assert.throws(()=>validateResumeInspection(r,{...receipt,...patch}));
  assert.throws(()=>validateResumeInspection(r,{}));
});
test("unsupported runner and nonterminal job block inspection without ARM submission",async()=>{
  const {r}=fixture();let calls=0;
  const control:RunnerControl={persist:async()=>{},sleep:async()=>{},list:async()=>[],request:async()=>{calls++;throw new Error("unexpected");}};
  r.guestReady!.capabilities=[];await assert.rejects(inspectResume(control,r,"a".repeat(32)),/reviewed runner/);
  r.migration!.phase="running";await assert.rejects(inspectResume(control,r,"a".repeat(32)),/failed/);
  assert.equal(calls,0);
});
test("lost inspection response reconciles GET-only and retains evidence without credentials or a new job",async()=>{
  const {r,receipt}=fixture();r.guestCommand={id:r.vmId+"/runCommands/af-"+id,operation:id,action:"inspect-resume",phase:"unknown",submittedAt:new Date().toISOString()};
  const methods:string[]=[],persisted:unknown[]=[];
  const control:RunnerControl={persist:async r=>{persisted.push(r);},sleep:async()=>{},list:async()=>{throw new Error("unexpected list");},request:async(_s,_p,method="GET")=>{methods.push(method);return {status:200,value:{properties:{instanceView:{executionState:"Succeeded",exitCode:0,output:JSON.stringify(receipt)}}}};}};
  const checked=await inspectResume(control,r);
  assert.deepEqual(methods,["GET"]);assert.equal(checked.record.migration,r.migration);
  assert.equal(checked.record.resumeInspection?.canResume,false);assert.equal(checked.record.migration?.verification,undefined);
  assert(!JSON.stringify(persisted).includes("AGEFREIGHTER_TARGET_DSN"));
});
test("inspection persists one intent and sends only target credentials in protected parameters",async()=>{
  const {r}=fixture();const saved:unknown[]=[],methods:string[]=[];let payload:any;
  const password="q".repeat(32);
  const control:RunnerControl={persist:async r=>{saved.push(JSON.parse(JSON.stringify(r)));},sleep:async()=>{},list:async()=>[],request:async(_s,path,method="GET",body?:unknown)=>{
    methods.push(method);
    if(method==="PUT"){
      assert.equal(saved.length,1);
      const b=body as any;assert.equal(b.properties.parameters,undefined);
      payload=JSON.parse(Buffer.from(b.properties.protectedParameters[0].value,"base64").toString());
      throw new Error("unconfirmed response with sensitive SDK data");
    }
    if(path.startsWith("target?"))return {status:200,value:{tags:{workflow:id,application:"agefreighter"},properties:{state:"Ready",network:{publicNetworkAccess:"Disabled"}}}};
    return {status:404,value:undefined};
  }};
  const checked=await inspectResume(control,r,password);
  assert.equal(methods.filter(x=>x==="PUT").length,1);
  assert.equal(checked.record.guestCommand?.phase,"unknown");
  assert.equal(payload.action,"inspect-resume");assert.equal(payload.operation,r.migration!.operation);assert.equal(payload.expectedBootId,id);
  assert.equal(payload.configuration,undefined);assert.deepEqual(Object.keys(payload.secrets),["AGEFREIGHTER_TARGET_DSN"]);
  assert(!JSON.stringify(saved).includes(password));assert(!JSON.stringify(saved).includes("sensitive SDK data"));
  assert.equal(checked.record.migration,r.migration);
});
