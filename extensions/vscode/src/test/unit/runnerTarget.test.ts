import assert from "node:assert/strict";
import test from "node:test";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { sourceWorkflowDraft, RunnerRecord } from "../../core/runner";
import { csvTargetEvidence, mappedNetworkTargetEvidence, neo4jTargetEvidence, targetPreview, assertTargetFresh, validateTargetSubnet, targetBudget, renewTargetAuthorization, submitTarget, refreshTarget, targetResourceIds, TargetInput } from "../../core/runnerTarget";
import { RunnerControl } from "../../core/runnerLifecycle";
const id="11111111-1111-4111-8111-111111111111",op="22222222-2222-4222-8222-222222222222",file="33333333-3333-4333-8333-333333333333";
const sha=(v:string)=>createHash("sha256").update(v).digest("hex");
function fixture(){
  const r=sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"test",region:"japaneast",zone:"1",subnetId:`/subscriptions/${id}/resourceGroups/test/providers/Microsoft.Network/virtualNetworks/net/subnets/runner`,size:"Standard_B2s_v2",source:{type:"csv",location:"local"}});
  r.phase="provisioned";r.artifact={version:"dev",sha256:"a".repeat(64),url:"https://example.invalid"};
  const doc=JSON.parse(readFileSync("../../production-simulation/vscode-e2e/evidence/csv-inventory-local-20260906.json","utf8"));
  const fields=doc.sections.find((x:any)=>x.title==="Mapped record counts").fields as {name:string}[];
  const rows=(kind:string)=>fields.filter(x=>x.name.startsWith(kind+":")).map(x=>({label:x.name.slice(kind.length+1),path:`/var/lib/agefreighter/workflows/${id}/uploads/${file}.csv`}));
  const configuration={source:{type:"csv",csv:{vertices:rows("vertex"),edges:rows("edge")}}};
  r.sourceDraft={configuration,canAssess:true,warnings:[],form:{} as any};
  r.csvTransfers=[{file,sha256:"b".repeat(64),bytes:10,phase:"verified"}];
  const text=JSON.stringify(doc), h=sha(text);
  r.assessment={operation:op,action:"inventory",phase:"finished",configurationSHA256:sha(JSON.stringify(configuration)),bootId:file,reportSHA256:h,reportBytes:Buffer.byteLength(text)};
  r.reportTransfers=[{operation:op,sha256:h,bytes:Buffer.byteLength(text),blob:"retained",phase:"imported"}];
  const input:TargetInput={serverName:"csv-test",subnetCIDR:"10.0.2.0/24",postgresSKU:"Standard_D4ds_v5",postgresTier:"GeneralPurpose",storageGiB:128,loaderSize:"Standard_D4s_v5",hourlyUSD:2,additionalReserveUSD:50,budgetUSD:800,deadline:new Date(Date.now()+86400000).toISOString()};
  return {r,doc,text,input};
}
test("target evidence consumes the real complete CSV report with all 18 mapped labels",()=>{
  const {r,text}=fixture(), e=csvTargetEvidence(r,text);
  assert.equal(e.rows,"5600000");assert.equal(Object.keys(e.labels).length,18);assert.equal(e.labels["v.Carrier"],1000);assert.equal(e.labels["e.CONTAINS"],1000000);
  assert.throws(()=>csvTargetEvidence(r,text+" "),/Import/);
  r.sourceDraft!.configuration.extra="changed";assert.throws(()=>csvTargetEvidence(r,text),/Import/);
});
test("Neo4j target evidence uses exact count-store totals and a conservative storage bound",()=>{
  const r=sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"test",region:"japaneast",zone:"1",subnetId:`/subscriptions/${id}/resourceGroups/test/providers/Microsoft.Network/virtualNetworks/net/subnets/runner`,size:"Standard_B2s_v2",source:{type:"neo4j",location:"azure",resourceId:`/subscriptions/${id}/resourceGroups/test/providers/Microsoft.Compute/virtualMachines/source`}});
  r.phase="provisioned";r.artifact={version:"dev",sha256:"a".repeat(64),url:"https://example.invalid"};
  const configuration={source:{type:"neo4j",neo4j:{uri:"neo4j+s://source.internal:7687",database:"neo4j"}}};
  r.sourceDraft={configuration,canAssess:true,warnings:[],form:{} as any};
  const doc={schemaVersion:1,command:"inventory",agefreighterVersion:"dev",outcome:"pass",errors:[],incompleteChecks:[],checks:[{id:"source-counts",status:"pass"}],sections:[{title:"Source inventory",fields:[{name:"vertices",value:"1600000",status:"pass"},{name:"edges",value:"4000000",status:"pass"},{name:"totalRows",value:"5600000",status:"pass"},{name:"countMethod",value:"neo4j-transactional-count-store",status:"pass"}]}]};
  const text=JSON.stringify(doc),h=sha(text);
  r.assessment={operation:op,action:"inventory",phase:"finished",configurationSHA256:sha(JSON.stringify(configuration)),bootId:file,reportSHA256:h,reportBytes:Buffer.byteLength(text)};
  r.reportTransfers=[{operation:op,sha256:h,bytes:Buffer.byteLength(text),blob:"retained",phase:"imported"}];
  const evidence=neo4jTargetEvidence(r,text);
  assert.equal(evidence.sourceType,"neo4j");assert.equal(evidence.rows,"5600000");assert.equal(evidence.vertices,"1600000");assert.equal(evidence.edges,"4000000");
  assert.equal(evidence.storageHighBytes,(5600000n*16384n).toString());assert.deepEqual(Object.keys(evidence.labels),[]);
  assert.throws(()=>neo4jTargetEvidence(r,text+" "),/Import/);
});
test("PostgreSQL and Cosmos target evidence requires a complete mapped stream for every approved label",()=>{
  for(const type of ["postgresql","cosmos-nosql"] as const){
    const r=sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"test",region:"japaneast",zone:"1",subnetId:`/subscriptions/${id}/resourceGroups/test/providers/Microsoft.Network/virtualNetworks/net/subnets/runner`,size:"Standard_B2s_v2",source:{type,location:type==="cosmos-nosql"?"azure":"on-premises",...(type==="cosmos-nosql"?{resourceId:`/subscriptions/${id}/resourceGroups/test/providers/Microsoft.DocumentDB/databaseAccounts/source`}:{})}});
    r.phase="provisioned";r.artifact={version:"dev",sha256:"a".repeat(64),url:"https://example.invalid"};
    const configuration={source:{type}};
    r.sourceDraft={configuration,canAssess:true,warnings:[],form:{mappings:[{kind:"vertex",label:"Person"},{kind:"edge",label:"KNOWS"}]} as any};
    const method=type==="postgresql"?"postgresql-repeatable-read-complete-stream":"cosmos-nosql-complete-stream";
    const doc={schemaVersion:1,command:"inventory",agefreighterVersion:"dev",outcome:"pass",errors:[],incompleteChecks:[],checks:[{id:"source-counts",status:"pass"},{id:"read-only",status:"pass"}],sections:[
      {title:"Source inventory",fields:[{name:"vertices",value:"2",status:"pass"},{name:"edges",value:"1",status:"pass"},{name:"totalRows",value:"3",status:"pass"},{name:"countMethod",value:method,status:"pass"}]},
      {title:"Mapped record counts",fields:[{name:"edge:KNOWS",value:"1",status:"pass"},{name:"vertex:Person",value:"2",status:"pass"}]},
      {title:"Capacity indicators",fields:[{name:"estimatedTargetRows",value:"3",status:"pass"},{name:"method",value:"complete-stream-range",status:"pass"},{name:"recommendedStorageBytesRange",value:"1024..4096",status:"pass"}]}
    ]};
    const text=JSON.stringify(doc),h=sha(text);
    r.assessment={operation:op,action:"inventory",phase:"finished",configurationSHA256:sha(JSON.stringify(configuration)),bootId:file,reportSHA256:h,reportBytes:Buffer.byteLength(text)};
    r.reportTransfers=[{operation:op,sha256:h,bytes:Buffer.byteLength(text),blob:"retained",phase:"imported"}];
    const evidence=mappedNetworkTargetEvidence(r,text);
    assert.equal(evidence.sourceType,type);assert.equal(evidence.rows,"3");assert.deepEqual({...evidence.labels},{"e.KNOWS":1,"v.Person":2});
    r.sourceDraft.form.mappings[1]!.label="OTHER";
    assert.throws(()=>mappedNetworkTargetEvidence(r,text),/approved inventory/);
  }
});
test("incomplete, wrong version, duplicate and wrong-source reports cannot size a target",()=>{
  for(const mutate of [(d:any)=>{d.outcome="incomplete";},(d:any)=>{d.agefreighterVersion="other";},(d:any)=>{d.command="profile";},(d:any)=>{d.sections.find((s:any)=>s.title==="Mapped record counts").fields.push(d.sections.find((s:any)=>s.title==="Mapped record counts").fields[0]);}]){
    const {r,doc}=fixture();mutate(doc);const text=JSON.stringify(doc);r.assessment!.reportSHA256=sha(text);r.assessment!.reportBytes=Buffer.byteLength(text);r.reportTransfers![0]!.sha256=sha(text);r.reportTransfers![0]!.bytes=Buffer.byteLength(text);
    assert.throws(()=>csvTargetEvidence(r,text));
  }
});
test("target plan is private, same region/zone/group, secure-string-only and capacity bounded",()=>{
  const {r,text,input}=fixture(),e=csvTargetEvidence(r,text),p=targetPreview(r,input,e);
  r.target=p;assert.equal(assertTargetFresh(r),p);
  const resources=p.template.resources as any[],server=resources.find(x=>x.type==="Microsoft.DBforPostgreSQL/flexibleServers");
  assert.equal(server.location,r.input.region);assert.equal(server.properties.availabilityZone,r.input.zone);assert.equal(server.properties.network.publicNetworkAccess,"Disabled");
  assert.equal(server.properties.administratorLoginPassword,"[parameters('administratorPassword')]");assert.equal((p.template.parameters as any).administratorPassword.type,"secureString");
  assert.equal(resources.some(x=>/firewall|publicIP|peering/i.test(x.type)),false);
  assert.equal(targetResourceIds(p).length,7);assert.ok(p.serverId.includes(`/resourceGroups/${r.input.resourceGroup}/`));
  r.sourceDraft!.configuration.changed=true;assert.throws(()=>assertTargetFresh(r),/changed/);
});
test("network, deadline, price and storage gates reject unsafe proposals",()=>{
  const vnet={properties:{addressSpace:{addressPrefixes:["10.0.0.0/16"]},subnets:[{properties:{addressPrefix:"10.0.1.0/24"}}]}};
  validateTargetSubnet("10.0.2.0/24",vnet);
  for(const prefix of ["10.0.1.0/24","10.0.0.0/16","10.0.2.1/24","10.1.0.0/24","256.0.0.0/24"])assert.throws(()=>validateTargetSubnet(prefix,vnet));
  const {r,text,input}=fixture();for(const patch of [{hourlyUSD:NaN},{budgetUSD:1},{deadline:"bad"},{deadline:"2020-01-01T00:00:00Z"},{additionalReserveUSD:-1}])assert.throws(()=>targetBudget({...input,...patch}));
  const e=csvTargetEvidence(r,text);assert.throws(()=>targetPreview(r,input,{...e,storageHighBytes:String(200*1024**3)}),/storage/);
});
test("a renewed authorization preserves target identity and records the expired cost window",()=>{
  const {r,text,input}=fixture(),now=Date.parse("2026-09-12T07:00:00Z");
  r.target=targetPreview(r,input,csvTargetEvidence(r,text));
  r.target.phase="provisioned";
  r.target.input.deadline="2026-09-09T08:55:00Z";
  const next=renewTargetAuthorization(r,{deadline:"2026-09-16T06:59:00Z",budgetUSD:800,additionalReserveUSD:200,hourlyUSD:2},now);
  assert.equal(next.target?.serverId,r.target.serverId);
  assert.equal(next.target?.input.deadline,"2026-09-16T06:59:00Z");
  assert.deepEqual(next.costAuthorizations?.[0]?.previous,{deadline:"2026-09-09T08:55:00Z",budgetUSD:800,additionalReserveUSD:50,hourlyUSD:2});
  assert.equal(next.costAuthorizations?.[0]?.authorizedAt,"2026-09-12T07:00:00.000Z");
  assert.throws(()=>renewTargetAuthorization({...r,migration:{} as any},{deadline:"2026-09-16T06:59:00Z",budgetUSD:800,additionalReserveUSD:200,hourlyUSD:2},now));
});
test("target deployment persists once, carries secrets only in ARM secure parameters, and reconciles by GET",async()=>{
  const {r,text,input}=fixture();r.target=targetPreview(r,input,csvTargetEvidence(r,text));const events:string[]=[],saved:RunnerRecord[]=[];let completed=false;
  const password="Q9!".repeat(12);
  const control:RunnerControl={list:async()=>[],sleep:async()=>{},persist:async x=>{events.push("persist");saved.push(structuredClone(x));},request:async(_s,path,method="GET",body)=>{
    events.push(method);
    if(method==="POST")return {status:200,value:{status:"Succeeded",properties:{changes:targetResourceIds(r.target!).map(resourceId=>({resourceId,changeType:"Create"}))}}};
    if(method==="PUT"){assert.equal((body as any).properties.parameters.administratorPassword.value,password);throw new Error("lost acknowledgement");}
    return completed && path.includes("/deployments/")?{status:200,value:{properties:{provisioningState:"Succeeded"}}}:{status:404,value:{}};
  }};
  const next=await submitTarget(control,r,password,async()=>{events.push("preflight");});assert.equal(next.target?.phase,"unknown");assert.equal(events.filter(x=>x==="PUT").length,1);
  assert.ok(events.indexOf("persist")<events.indexOf("PUT"));assert.ok(!JSON.stringify(saved).includes(password));
  await assert.rejects(submitTarget(control,next,password,async()=>{}));
  const before=events.length;await refreshTarget(control,next);assert.deepEqual(events.slice(before),["GET","persist"]);
  completed=true;assert.equal((await refreshTarget(control,next)).target?.phase,"provisioned");
});
