import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp,readFile,readdir,rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describeLostResponsePoll,validateLostResponsePoll } from "../helpers/deploymentLostResponsePoll";
import { LostResponseStages } from "../helpers/deploymentLostResponseStages";
import { otherCancellationFixture,otherNativeCancelCases } from "../helpers/nativeCancelOtherScenarios";
import { sourceWorkflowDraft } from "../../core/runner";
import { storageDraft } from "../../core/runnerStorageLifecycle";
import type { AzureSession } from "../../guided/azure";
const sub="11111111-1111-4111-8111-111111111111",region="japaneast";
function signed(subscription=sub){return `https://management.azure.com/subscriptions/${subscription}/operationresults/${"A".repeat(326)}?api-version=2022-09-01&t=20260923T143500Z&c=PUBLIC_CERT%2BBASE64&s=DO_NOT_RETAIN_SIGNATURE%2BVALUE&h=DO_NOT_RETAIN_HASH`;}
test("observed subscription result shape allows one exact signed URI without retaining signed values",()=>{
  assert.doesNotThrow(()=>validateLostResponsePoll(signed(),sub,region));
  assert.doesNotThrow(()=>validateLostResponsePoll(signed().replace("/subscriptions/","/SUBSCRIPTIONS/").replace("/operationresults/","/operationResults/"),sub,region));
  const receipt=JSON.stringify(describeLostResponsePoll(signed(),sub,region));
  assert.ok(receipt.includes('"pathKind":"subscription-operation-results"'));
  for(const text of ["DO_NOT_RETAIN","PUBLIC_CERT","20260923T143500Z","A".repeat(30)])assert.ok(!receipt.includes(text));
});
for(const [name,change] of [
  ["foreign host",(p:string)=>p.replace("management.azure.com","example.invalid")],
  ["foreign subscription",(p:string)=>p.replace(sub,"22222222-2222-4222-8222-222222222222")],
  ["duplicate query",(p:string)=>p+"&s=other"],
  ["unknown query",(p:string)=>p+"&sig=other"],
  ["wrong version",(p:string)=>p.replace("2022-09-01","2025-04-01")],
  ["path traversal",(p:string)=>p.replace("operationresults/","operationresults/../")],
  ["userinfo",(p:string)=>p.replace("https://","https://user:secret@")],
  ["fragment",(p:string)=>p+"#secret"],
  ["oversize opaque",(p:string)=>p.replace("A".repeat(326),"A".repeat(1025))]
] as const)test(`signed poll rejects ${name} with fixed redacted errors`,()=>{assert.throws(()=>validateLostResponsePoll(change(signed()),sub,region),{message:"Unapproved what-if polling URL; sanitized shape receipt retained"});});
test("stage learns poll only from approved what-if, redacts GET/header receipts, and enforces shared chain budget",{skip:process.platform==="win32"?"The test companion requires POSIX directory fsync":false},async()=>{
  const root=await mkdtemp(join(tmpdir(),"af-poll-unit-"));
  try{
    const f=otherCancellationFixture(otherNativeCancelCases[0]),r=sourceWorkflowDraft(f.record.id,f.record.input);r.input.source={type:"csv",location:"local"};r.storageDeployment=storageDraft(r,sub);
    const poll=signed(r.input.subscriptionId),second=poll.replace("A".repeat(326),"B".repeat(326));let next=poll,calls=0;
    const actual={runnerRequest:async()=>{calls++;return {status:202,value:{},poll:next};}} as unknown as AzureSession;
    const stages=new LostResponseStages(root,{workflow:r.id,input:r.input,artifactSHA256:"a".repeat(64),artifactBytes:1,manifestPath:"/inert",archivePath:"/inert",annotation:"unit",expiresAt:new Date(Date.now()+3600000).toISOString()},async()=>r,actual);
    await assert.rejects(stages.request(r.input.subscriptionId,poll));assert.equal(calls,0);
    await stages.request(r.input.subscriptionId,`${r.storageDeployment.id}/whatIf?api-version=2022-09-01`,"POST",{properties:{mode:"Incremental",template:r.storageDeployment.template,whatIfSettings:{resultFormat:"ResourceIdOnly"}}});
    next=second;await stages.request(r.input.subscriptionId,poll);for(let n=0;n<29;n++)await stages.request(r.input.subscriptionId,second);
    await assert.rejects(stages.request(r.input.subscriptionId,second),/budget expired/);assert.equal(calls,31);
    const receipts=(await Promise.all((await readdir(root)).map(n=>readFile(join(root,n),"utf8")))).join();
    assert.ok(!receipts.includes("DO_NOT_RETAIN"));assert.ok(!receipts.includes("PUBLIC_CERT"));assert.ok(!receipts.includes("A".repeat(30)));assert.ok(receipts.includes("whatIf-poll-header"));
  }finally{await rm(root,{recursive:true,force:true});}
});
