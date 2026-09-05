import assert from "node:assert/strict";
import test from "node:test";
import { startCSVImport, refreshCSVImport } from "../../core/runnerCSV";
import { RunnerRecord } from "../../core/runner";
import { RunnerControl } from "../../core/runnerLifecycle";
import { reportStorageNames } from "../../core/runnerReportStorage";

const id = "11111111-1111-4111-8111-111111111111", file = "22222222-2222-4222-8222-222222222222";
function fixture() {
  const record: RunnerRecord = { schemaVersion: 2, id, phase: "provisioned", input: {
    subscriptionId: id, resourceGroup: "test", region: "japaneast", zone: "1", subnetId: "subnet", size: "Standard_B2s_v2", source: {type: "csv", location: "local"}
  }, artifact: {version: "2.4.0", sha256: "a".repeat(64), url: "https://example.invalid"}, vmId: `/subscriptions/${id}/resourceGroups/test/providers/Microsoft.Compute/virtualMachines/runner`, deploymentId: "deployment", template: {}, previewHash: "", expiresAt: "", updatedAt: "", hourlyComputeUSD: .109,
    sourceFiles: [{id: file, name: "test.csv", path: "/test.csv"}], csvTransfers: [{file, bytes: 5, sha256: "b".repeat(64), phase: "uploaded"}],
    guestReady: {bootId: id, cliVersion: "2.4.0", archiveSha256: "a".repeat(64), commit: "commit", checkedAt: new Date().toISOString()}
  };
  const events: string[] = [], bodies: any[] = [];
  let view: unknown;
  const names = reportStorageNames(record);
  const control: RunnerControl = { sleep: async()=>{}, list: async()=>[], persist: async()=>{events.push("persist");}, request: async(_sub, path, method="GET", body)=>{
    events.push(method);
    if (method === "PUT") {bodies.push(body); return {status: 201, value: {}};}
    if (path === `${names.id}?api-version=2023-05-01`) return {status: 200, value: {id: names.id, location: "japaneast", tags: {application: "agefreighter", workflow: id, purpose: "artifact-transfer"}, properties: {provisioningState: "Succeeded", supportsHttpsTrafficOnly: true, allowBlobPublicAccess: false, allowSharedKeyAccess: false, minimumTlsVersion: "TLS1_2", primaryEndpoints: {blob: `${names.origin}/`}}}};
    if (path === `${names.containerId}?api-version=2023-05-01`) return {status: 200, value: {id: names.containerId, properties: {publicAccess: "None"}}};
    return view === undefined || !path.includes("$expand=instanceView") ? {status: 404, value: {}} : {status: 200, value: {properties: {instanceView: view}}};
  }};
  const now = Date.now(), q = new URLSearchParams({sv:"2023-11-03",spr:"https",sr:"b",sp:"r",sig:"test-only",skoid:id,sktid:file,sks:"b",skv:"2023-11-03",
    st:new Date(now-60000).toISOString(),se:new Date(now+600000).toISOString(),skt:new Date(now-3600000).toISOString(),ske:new Date(now+3600000).toISOString()});
  const capability = `${names.origin}/${names.container}/uploads/${file}/${"b".repeat(64)}.csv?${q}`;
  return {record, control, events, bodies, capability, result:(value:unknown)=>{view=value;}};
}

test("CSV import wire DTO excludes controller phase, operation and rejection history", async()=>{
  const f=fixture(), transfer=f.record.csvTransfers![0]!;
  transfer.rejectedImports=[{operation:id,commandId:"retained",reason:"protocol-rejected-before-execution"}];
  const next=await startCSVImport(f.control,f.record,transfer,f.capability);
  const payload=JSON.parse(Buffer.from(f.bodies[0].properties.protectedParameters[0].value,"base64").toString());
  assert.deepEqual(Object.keys(payload.import).sort(),["bytes","file","sha256","url"]);
  assert.equal(payload.expectedBootId,id);
  assert.equal(next.csvTransfers![0]!.rejectedImports?.length,1);
  assert.equal(next.csvTransfers![0]!.phase,"submitted");
});

test("proven pre-execution decoder rejection retains evidence and only enables a new approved attempt", async()=>{
  const f=fixture();
  const r=await startCSVImport(f.control,f.record,f.record.csvTransfers![0]!,f.capability);
  r.guestCommand!.phase="failed";
  f.result({executionState:"Failed",exitCode:1,output:"",error:"invalid runner request\n"});
  const before=f.events.length, next=await refreshCSVImport(f.control,r);
  assert.deepEqual(f.events.slice(before),["GET","persist"]);
  assert.equal(next.csvTransfers![0]!.phase,"uploaded");
  assert.equal(next.csvTransfers![0]!.operation,undefined);
  assert.equal(next.csvTransfers![0]!.rejectedImports![0]!.commandId,r.guestCommand!.id);
  assert.equal(next.guestCommand!.phase,"failed");
});

test("unknown or post-execution failure never makes an import replayable", async()=>{
  for (const error of ["CSV start acknowledgement is uncertain\n","invalid runner request\nextra"]) {
    const f=fixture(), r=await startCSVImport(f.control,f.record,f.record.csvTransfers![0]!,f.capability);
    r.guestCommand!.phase="failed";
    // Refresh issues only a status query, never a new import, if the exact
    // pre-execution rejection is absent.
    f.result({executionState:"Failed",exitCode:1,output:"",error});
    const next=await refreshCSVImport(f.control,r);
    assert.equal(next.csvTransfers![0]!.phase,"submitted");
    assert.equal(next.csvTransfers![0]!.rejectedImports,undefined);
    assert.equal(f.bodies.length,2);
    assert.equal(JSON.parse(Buffer.from(f.bodies[1].properties.protectedParameters[0].value,"base64").toString()).action,"status");
  }
});
