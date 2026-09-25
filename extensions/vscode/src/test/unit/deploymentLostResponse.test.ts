import assert from "node:assert/strict";
import { mkdtemp, readFile, readdir, realpath, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test, {describe} from "node:test";
import { lostResponseControl, lostResponseHash, LostResponsePlan, retainLostResponseFile } from "../helpers/deploymentLostResponse";
import { otherCancellationFixture, otherNativeCancelCases } from "../helpers/nativeCancelOtherScenarios";
import { RunnerControl, refreshRunner } from "../../core/runnerLifecycle";
async function fixture() {
  const root=await realpath(await mkdtemp(join(tmpdir(),"lost-response-unit-"))), record=otherCancellationFixture(otherNativeCancelCases[0]).record;
  const plan:LostResponsePlan={schemaVersion:1,workflow:record.id,subscription:record.input.subscriptionId,deploymentId:record.deploymentId,
    templateSHA256:lostResponseHash(record.template),previewSHA256:lostResponseHash(record),expiresAt:record.expiresAt,
    annotation:"Synthetic unit test only; no cloud acceptance or native interaction.",permits:[{kind:"request",path:`${record.deploymentId}?api-version=2022-09-01`,method:"GET",maxCalls:1}]};
  const calls:string[]=[],actual:RunnerControl={persist:async r=>{calls.push(`persist:${r.phase}`);},list:async()=>[],sleep:async()=>{},request:async(_s,_p,m="GET")=>{calls.push(m);return {status:201,value:{properties:{provisioningState:"Succeeded"},secret:"MUST-NOT-BE-RETAINED"}};}};
  const body={properties:{mode:"Incremental",template:structuredClone(record.template)}};
  return {root,record,plan,calls,actual,body,control:await lostResponseControl(root,plan,record,actual)};
}
// This disposable adapter requires POSIX private directory modes and fsync.
describe("POSIX lost-response durable adapter", {skip: process.platform === "win32"}, () => {
test("durable claim then sanitized receipt, one PUT despite new adapter and GET-only reconcile",async()=>{
  const f=await fixture();try{
    await f.control.persist({...f.record,phase:"deployment-submitted"});
    await assert.rejects(f.control.request(f.plan.subscription,`${f.plan.deploymentId}?api-version=2022-09-01`,"PUT",f.body),/withheld/);
    assert.deepEqual(f.calls,["persist:deployment-submitted","PUT"]);
    const receipt=JSON.parse(await readFile(join(f.root,"azure-acceptance.json"),"utf8"));assert.equal(receipt.status,201);assert.equal(receipt.realServiceQualifiedByThisFile,false);
    const resumed=await lostResponseControl(f.root,f.plan,f.record,f.actual);await resumed.persist({...f.record,phase:"deployment-submitted"});
    await assert.rejects(resumed.request(f.plan.subscription,`${f.plan.deploymentId}?api-version=2022-09-01`,"PUT",f.body),/EEXIST/);
    const refreshed=await refreshRunner(resumed,{...f.record,phase:"unknown"});assert.equal(refreshed.phase,"provisioned");
    assert.equal(f.calls.filter(c=>c==="PUT").length,1);
    for(const file of await readdir(f.root))assert.ok(!(await readFile(join(f.root,file),"utf8")).includes("MUST-NOT-BE-RETAINED"));
  }finally{await rm(f.root,{recursive:true,force:true});}
});
for(const kind of ["no-intent","wrong-path","changed-template","PATCH","exhausted-GET"] as const)test(`denies ${kind} without unauthorized transport`,async()=>{
  const f=await fixture();try{
    if(kind==="exhausted-GET"){
      await f.control.request(f.plan.subscription,`${f.plan.deploymentId}?api-version=2022-09-01`);
      await assert.rejects(f.control.request(f.plan.subscription,`${f.plan.deploymentId}?api-version=2022-09-01`),/exhausted/);assert.deepEqual(f.calls,["GET"]);return;
    }
    if(kind!=="no-intent")await f.control.persist({...f.record,phase:"deployment-submitted"});
    if(kind==="changed-template"){f.record.template.changed=true;f.body.properties.template=f.record.template;}
    await assert.rejects(f.control.request(f.plan.subscription,`${f.plan.deploymentId}${kind==="wrong-path"?"-other":""}?api-version=2022-09-01`,kind==="PATCH"?"PATCH":"PUT",f.body));
    assert.equal(f.calls.filter(c=>c==="PUT"||c==="PATCH").length,0);
  }finally{await rm(f.root,{recursive:true,force:true});}
});
test("snapshot incoming body before asynchronous intent publication",async()=>{
  const f=await fixture();try{
    let observed:unknown;f.actual.request=async(_s,_p,_m,body)=>{observed=body;return {status:201,value:{}};};
    const c=await lostResponseControl(f.root,f.plan,f.record,f.actual);await c.persist({...f.record,phase:"deployment-submitted"});
    const pending=c.request(f.plan.subscription,`${f.plan.deploymentId}?api-version=2022-09-01`,"PUT",f.body);f.body.properties.mode="Complete";
    await assert.rejects(pending,/withheld/);assert.equal((observed as typeof f.body).properties.mode,"Incremental");
  }finally{await rm(f.root,{recursive:true,force:true});}
});

});

test("Windows cannot turn an undurable companion claim into a transport attempt", {skip: process.platform !== "win32"}, async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), "lost-response-unsupported-")));
  try {
    const record = otherCancellationFixture(otherNativeCancelCases[0]).record;
    const plan: LostResponsePlan = {schemaVersion: 1, workflow: record.id, subscription: record.input.subscriptionId,
      deploymentId: record.deploymentId, templateSHA256: lostResponseHash(record.template), previewSHA256: lostResponseHash(record),
      expiresAt: record.expiresAt, annotation: "Synthetic unsupported-host test only; no cloud or native interaction.", permits: []};
    const calls: string[] = [], actual: RunnerControl = {persist: async () => {calls.push("persist");}, list: async () => {calls.push("LIST"); return [];},
      sleep: async () => {}, request: async () => {calls.push("transport"); throw Error("Unexpected transport");}};
    const mode = (await stat(root)).mode & 0o777;
    if (mode !== 0o700) {
      await assert.rejects(lostResponseControl(root, plan, record, actual), {name: "AssertionError", actual: mode, expected: 0o700});
      assert.deepEqual(await readdir(root), []); assert.deepEqual(calls, []);
    } else {
      // If this Windows filesystem represents private modes, directory fsync
      // must still succeed before an explicit PUT could reach the adapter.
      const control = await lostResponseControl(root, plan, record, actual);
      await control.persist({...record, phase: "deployment-submitted"});
      await assert.rejects(control.request(plan.subscription, `${plan.deploymentId}?api-version=2022-09-01`, "PUT",
        {properties: {mode: "Incremental", template: record.template}}), {code: "EPERM", syscall: "fsync"});
      assert.deepEqual(calls, ["persist"]);
    }
    await assert.rejects(retainLostResponseFile(root, "unsupported-claim.json", {synthetic: true}), {code: "EPERM", syscall: "fsync"});
    const retained = await readFile(join(root, "unsupported-claim.json"));
    assert.deepEqual(JSON.parse(retained.toString()), {synthetic: true});
    await assert.rejects(retainLostResponseFile(root, "unsupported-claim.json", {synthetic: false}), {code: "EEXIST"});
    assert.deepEqual(await readFile(join(root, "unsupported-claim.json")), retained);
    assert.ok(!calls.includes("transport") && !calls.includes("LIST"));
  } finally {await rm(root, {recursive: true, force: true});}
});
