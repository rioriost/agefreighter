import assert from "node:assert/strict";
import test from "node:test";
import {sourceWorkflowDraft} from "../../core/runner";
import {RunnerControl} from "../../core/runnerLifecycle";
import {inspectResume,validateResumeInspection} from "../../core/runnerResume";
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
