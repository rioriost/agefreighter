import assert from "node:assert/strict";
import test from "node:test";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { sourceWorkflowDraft, RunnerRecord } from "../../core/runner";
import { csvTargetEvidence, mappedNetworkTargetEvidence, neo4jTargetEvidence, targetPreview, assertTargetFresh, validateTargetSubnet, targetBudget, renewTargetAuthorization, submitTarget, refreshTarget, targetResourceIds, TargetInput, repairBusyTargetPreload } from "../../core/runnerTarget";
import { RunnerControl } from "../../core/runnerLifecycle";
import { buildSourceDraft } from "../../core/runnerSource";
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
function gremlinFixture(){
  const {r,input}=fixture();r.input.source={type:"cosmos-nosql",location:"azure"};
  r.sourceDraft=buildSourceDraft(r.input.source,{name:"gremlin-test",namespace:"migration",host:"source.documents.azure.com",database:"p1",cosmosFormat:"gremlin",container:"graph",partitionKey:"partitionKey",gremlinPropertyTypes:"score=float64,distance_km=float64"},id);
  const doc=JSON.parse(readFileSync("../../production-simulation/vscode-e2e/evidence/gremlin-inventory-20260918.json","utf8"));
  r.artifact.version=doc.agefreighterVersion;
  const seal=()=>{const text=JSON.stringify(doc),h=sha(text);r.assessment={operation:op,action:"inventory",phase:"finished",bootId:file,configurationSHA256:sha(JSON.stringify(r.sourceDraft!.configuration)),reportSHA256:h,reportBytes:Buffer.byteLength(text)};r.reportTransfers=[{operation:op,sha256:h,bytes:Buffer.byteLength(text),blob:"retained",phase:"imported"}];return text;};
  return {r,input,doc,seal};
}
test("real complete Gremlin inventory sizes a private target without synthetic manual mappings",()=>{
  const {r,input,doc,seal}=gremlinFixture(),text=seal(),before=JSON.stringify(r),e=mappedNetworkTargetEvidence(r,text);
  assert.equal(e.rows,"5600000");assert.equal(e.vertices,"1600000");assert.equal(e.edges,"4000000");
  assert.equal(Object.keys(e.labels).length,18);assert.equal(e.labels["v.Supplier"],40000);assert.equal(e.labels["e.SUPPLIES"],400000);
  assert.equal(e.storageHighBytes,"13444452070");assert.equal(e.sourceType,"cosmos-nosql");
  assert.equal(JSON.stringify(r),before);assert.deepEqual(r.sourceDraft!.form.mappings,[]);
  const p=targetPreview(r,input,e);r.target=p;assert.equal(assertTargetFresh(r),p);
  assert.equal(p.evidence.reportSHA256,sha(text));assert.equal(doc.errors.length,0);
});
test("Gremlin inventory cannot bypass complete-report, label-count or capacity gates",()=>{
  for(const mutate of [
    (d:any)=>{d.outcome="incomplete";},(d:any)=>{d.command="profile";},(d:any)=>{d.agefreighterVersion="other";},
    (d:any)=>{d.incompleteChecks=["source-counts"];},(d:any)=>{d.errors=["failed"];},
    (d:any)=>{d.checks[1].status="fail";},
    (d:any)=>{d.sections.find((s:any)=>s.title==="Capacity indicators").fields.find((f:any)=>f.name==="estimatedTargetRows").value="1";},
    (d:any)=>{d.sections.find((s:any)=>s.title==="Mapped record counts").fields.pop();},
    (d:any)=>{const f=d.sections.find((s:any)=>s.title==="Mapped record counts").fields;f.push(f[0]);},
    (d:any)=>{d.sections.find((s:any)=>s.title==="Mapped record counts").fields[0].value="9007199254740992";},
    (d:any)=>{d.sections.find((s:any)=>s.title==="Mapped record counts").fields[0].name="edge:bad-label";},
    (d:any)=>{d.sections.find((s:any)=>s.title==="Mapped record counts").fields[0].status="unknown";},
  ]){const {r,doc,seal}=gremlinFixture();mutate(doc);assert.throws(()=>mappedNetworkTargetEvidence(r,seal()));}
});
test("Gremlin auto-discovery is admitted only for the matching bounded source configuration",()=>{
  for(const mutate of [
    (r:RunnerRecord)=>{r.sourceDraft!.form.cosmosFormat="explicit";},
    (r:RunnerRecord)=>{r.sourceDraft!.form.container="different";},
    (r:RunnerRecord)=>{r.sourceDraft!.form.partitionKey="other";},
    (r:RunnerRecord)=>{r.sourceDraft!.form.mappings=[{kind:"vertex",label:"Injected"}] as any;},
    (r:RunnerRecord)=>{(r.sourceDraft!.configuration.source as any).cosmos.gremlin.enabled=false;},
    (r:RunnerRecord)=>{(r.sourceDraft!.configuration.source as any).cosmos.gremlin.maxLabels=8;},
    (r:RunnerRecord)=>{(r.sourceDraft!.configuration.source as any).cosmos.vertices=[];},
    (r:RunnerRecord)=>{(r.sourceDraft!.configuration.source as any).type="postgresql";},
  ]){const {r,seal}=gremlinFixture();mutate(r);assert.throws(()=>mappedNetworkTargetEvidence(r,seal()),/Gremlin/);}
});
test("Gremlin report bytes, configuration and imported operation stay bound",()=>{
  for(const mutate of [
    (r:RunnerRecord)=>{r.reportTransfers![0]!.phase="submitted";},
    (r:RunnerRecord)=>{r.reportTransfers![0]!.operation=file;},
    (r:RunnerRecord)=>{r.assessment!.configurationSHA256="b".repeat(64);},
    (r:RunnerRecord)=>{(r.sourceDraft!.configuration.source as any).cosmos.gremlin.container="changed";},
  ]){const {r,seal}=gremlinFixture(),text=seal();mutate(r);assert.throws(()=>mappedNetworkTargetEvidence(r,text),/Import/);}
  const {r,seal}=gremlinFixture(),text=seal();assert.throws(()=>mappedNetworkTargetEvidence(r,text+" "),/Import/);
});
test("Gremlin catalog admission remains bounded even when extra zero-count labels preserve totals",()=>{
  const {r,doc,seal}=gremlinFixture(),fields=doc.sections.find((s:any)=>s.title==="Mapped record counts").fields;
  (r.sourceDraft!.configuration.source as any).cosmos.gremlin.maxLabels=9;
  fields.push({name:"edge:EXTRA",status:"pass",value:"0"});
  assert.throws(()=>mappedNetworkTargetEvidence(r,seal()),/discovery bounds/);
  for(let i=0;i<256;i++)fields.push({name:`edge:EXTRA_${i}`,status:"pass",value:"0"});
  assert.throws(()=>mappedNetworkTargetEvidence(r,seal()),/approved inventory/);
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

test("target children serialize database, AGE allow-list and preload writes",()=>{
  const {r,text,input}=fixture(),p=targetPreview(r,input,csvTargetEvidence(r,text)),resources=p.template.resources as any[];
  const bySuffix=(suffix:string)=>resources.find(x=>x.name===`${input.serverName}/${suffix}`);
  assert.deepEqual(bySuffix("agefreighter").dependsOn,[p.serverId]);
  assert.deepEqual(bySuffix("azure.extensions").dependsOn,[`${p.serverId}/databases/agefreighter`]);
  assert.deepEqual(bySuffix("shared_preload_libraries").dependsOn,[`${p.serverId}/configurations/azure.extensions`]);
});

function crossGroupFixture(){
  const f=fixture();f.r.input.subnetId=f.r.input.subnetId.replace("/resourceGroups/test/","/resourceGroups/network-only/");
  f.r.target=targetPreview(f.r,f.input,csvTargetEvidence(f.r,f.text));return f;
}

test("independent network group scopes only a new subnet with a retained child identity",()=>{
  const {r}=crossGroupFixture(),p=r.target!,n=p.networkDeployment!,resources=p.template.resources as any[];
  assert.equal(n.resourceGroup,"network-only");assert.equal((n.template.resources as any[]).length,1);
  assert.equal((n.template.resources as any[])[0].type,"Microsoft.Network/virtualNetworks/subnets");
  assert.equal(resources[0].resourceGroup,"network-only");assert.equal(resources[0].properties.mode,"Incremental");
  assert.equal(resources[0].properties.expressionEvaluationOptions.scope,"inner");
  assert.equal(resources[0].properties.template,n.template);
  assert.deepEqual(resources[3].dependsOn,[n.deploymentId,`${p.dnsId}/virtualNetworkLinks/runner`]);
  assert.ok(p.serverId.includes("/resourceGroups/test/"));assert.ok(p.dnsId.includes("/resourceGroups/test/"));
  assert.ok(p.subnetId.startsWith(n.vnetId+"/subnets/"));assert.ok(n.deploymentId.includes("/resourceGroups/network-only/"));
  assert.equal(JSON.stringify(n).includes("administratorPassword"),false);assert.equal(assertTargetFresh(r),p);
  for(const patch of [{region:"japanwest"},{zone:"2"},{resourceGroup:"other"},{subnetId:r.input.subnetId.replace("/net/","/other/")}])
    assert.throws(()=>assertTargetFresh({...r,input:{...r.input,...patch}}),/changed/);
  const f=fixture();f.r.input.subnetId=f.r.input.subnetId.replace(id,op);
  assert.throws(()=>targetPreview(f.r,f.input,csvTargetEvidence(f.r,f.text)),/runner subscription/);
});

function crossGroupControl(r:RunnerRecord){
  const p=r.target!,n=p.networkDeployment!,calls:{path:string;method:string;body?:unknown}[]=[],saved:RunnerRecord[]=[];
  const changes=(ids:string[])=>({status:"Succeeded",properties:{changes:ids.map(resourceId=>({resourceId,changeType:"Create"}))}});
  const state={terminal:false,loseAck:false,childStatus:"Succeeded",existingId:"",wrapper:true,
    networkWhatIf:changes([p.subnetId]),parentWhatIf:changes(targetResourceIds(p)),childOperations:[{properties:{targetResource:{id:p.subnetId},provisioningState:"Succeeded"}}],
    parentOperations:[...targetResourceIds(p).filter(x=>x!==p.subnetId),n.deploymentId].map(id=>({properties:{targetResource:{id},provisioningState:"Succeeded"}}))};
  const control:RunnerControl={sleep:async()=>{},persist:async x=>{calls.push({path:"persist",method:"persist"});saved.push(structuredClone(x));},
    request:async(_s,path,method="GET",body)=>{
      calls.push({path,method,body});
      if(method==="POST"){
        if(path.startsWith(n.deploymentId+"/whatIf")){
          assert.deepEqual((body as any).properties.parameters,{});return {status:200,value:state.networkWhatIf};
        }
        const value=structuredClone(state.parentWhatIf);
        if(state.wrapper)value.properties.changes.push({resourceId:n.deploymentId,changeType:"Create"});
        return {status:200,value};
      }
      if(method==="PUT"){
        assert.equal(path,`${p.deploymentId}?api-version=2022-09-01`);
        if(state.loseAck)throw new Error("lost acknowledgement");return {status:202,value:{}};
      }
      if(state.existingId && path.startsWith(state.existingId+"?"))return {status:200,value:{}};
      if(state.terminal && path.startsWith(p.deploymentId+"?"))return {status:200,value:{properties:{provisioningState:"Succeeded"}}};
      if(state.terminal && path.startsWith(n.deploymentId+"?"))return {status:200,value:{properties:{provisioningState:state.childStatus}}};
      return {status:404,value:{}};
    },list:async(_s,path)=>{calls.push({path,method:"LIST"});
      if(path.startsWith(p.deploymentId+"/operations"))return state.parentOperations;
      if(path.startsWith(n.deploymentId+"/operations"))return state.childOperations;
      return [];
    }};
  return {control,calls,state,saved};
}

test("cross-group deployment previews both scopes before a single persisted parent PUT and GET-only reconciliation",async()=>{
  for(const wrapper of [false,true]){
    const {r}=crossGroupFixture(),f=crossGroupControl(r);f.state.wrapper=wrapper;f.state.loseAck=true;
    const next=await submitTarget(f.control,r,"Q9!".repeat(12),async()=>{});
    assert.equal(next.target?.phase,"unknown");
    assert.equal(f.calls.filter(x=>x.method==="POST").length,2);assert.equal(f.calls.filter(x=>x.method==="PUT").length,1);
    assert.ok(f.calls.findIndex(x=>x.method==="persist")<f.calls.findIndex(x=>x.method==="PUT"));
    assert.ok(f.calls.some(x=>x.path.includes("/resourceGroups/network-only/resources?")));
    assert.ok(f.calls.some(x=>x.path.includes("/resourceGroups/test/resources?")));
    assert.ok(!JSON.stringify(f.saved).includes("Q9!"));
    await assert.rejects(submitTarget(f.control,next,"Q9!".repeat(12),async()=>{}));
    f.state.terminal=true;f.calls.length=0;
    assert.equal((await refreshTarget(f.control,next)).target?.phase,"provisioned");
    assert.ok(f.calls.every(x=>["GET","LIST","persist"].includes(x.method)));
  }
});

test("cross-group deployment fails closed on occupied identities, missing leaves or unsafe what-if changes",async()=>{
  for(const mutate of [
    (f:ReturnType<typeof crossGroupControl>,r:RunnerRecord)=>{f.state.existingId=r.target!.networkDeployment!.deploymentId;},
    (f:ReturnType<typeof crossGroupControl>,r:RunnerRecord)=>{f.state.existingId=r.target!.subnetId;},
    (f:ReturnType<typeof crossGroupControl>)=>{f.state.networkWhatIf.properties.changes[0]!.changeType="Modify";},
    (f:ReturnType<typeof crossGroupControl>)=>{f.state.parentWhatIf.properties.changes.shift();},
    (f:ReturnType<typeof crossGroupControl>)=>{f.state.parentWhatIf.properties.changes[0]!.changeType="Ignore";},
    (f:ReturnType<typeof crossGroupControl>)=>{f.state.networkWhatIf.properties.changes.push({resourceId:"/foreign",changeType:"Create"});},
    (f:ReturnType<typeof crossGroupControl>)=>{f.state.parentWhatIf.properties.changes.push({resourceId:"/foreign",changeType:"Delete"});}
  ]){
    const {r}=crossGroupFixture(),f=crossGroupControl(r);mutate(f,r);
    await assert.rejects(submitTarget(f.control,r,"Q9!".repeat(12),async()=>{}));
    assert.ok(!f.calls.some(x=>x.method==="PUT" || x.method==="persist"));
  }
});

test("parent success does not hide missing, failed, duplicated or foreign nested operations",async()=>{
  for(const mutate of [
    (f:ReturnType<typeof crossGroupControl>)=>{f.state.childStatus="Running";},
    (f:ReturnType<typeof crossGroupControl>)=>{f.state.childOperations=[];},
    (f:ReturnType<typeof crossGroupControl>)=>{f.state.childOperations[0]!.properties.targetResource.id="/foreign";},
    (f:ReturnType<typeof crossGroupControl>)=>{f.state.childOperations[0]!.properties.provisioningState="Failed";},
    (f:ReturnType<typeof crossGroupControl>)=>{f.state.parentOperations.pop();},
    (f:ReturnType<typeof crossGroupControl>)=>{f.state.parentOperations[0]=f.state.parentOperations[1]!;}
  ]){
    const {r}=crossGroupFixture();r.target!.phase="submitted";const f=crossGroupControl(r);f.state.terminal=true;mutate(f);
    assert.equal((await refreshTarget(f.control,r)).target?.phase,"unknown");
    assert.ok(f.calls.every(x=>["GET","LIST","persist"].includes(x.method)));
  }
});

function repairFixture(crossGroup=false){
  const {r,text,input}=fixture();
  if(crossGroup)r.input.subnetId=r.input.subnetId.replace("/resourceGroups/test/","/resourceGroups/network-only/");
  r.target=targetPreview(r,input,csvTargetEvidence(r,text));r.target.phase="failed";
  r.guestReady={bootId:file,cliVersion:r.artifact.version,archiveSha256:r.artifact.sha256,commit:"a".repeat(40),checkedAt:new Date().toISOString(),health:{idle:true,storageUsedPercent:4,swapUsedBytes:0,oomEvents:0}};
  const p=r.target,preloadId=`${p.serverId}/configurations/shared_preload_libraries`;
  const operations=targetResourceIds(p).map(id=>({properties:{targetResource:{id:crossGroup && id===p.subnetId?p.networkDeployment!.deploymentId:id},provisioningState:id===preloadId?"Failed":"Succeeded",statusMessage:id===preloadId?{error:{code:"ServerIsBusy"}}:undefined}}));
  const childOperations=[{properties:{targetResource:{id:p.subnetId},provisioningState:"Succeeded"}}];
  const server={location:r.input.region,tags:{application:"agefreighter",workflow:r.id,purpose:"migration-target"},sku:{name:p.input.postgresSKU},properties:{state:"Ready",version:"18",availabilityZone:r.input.zone,storage:{storageSizeGB:128},network:{publicNetworkAccess:"Disabled",delegatedSubnetResourceId:p.subnetId,privateDnsZoneArmResourceId:p.dnsId}}};
  const config={value:"pg_cron,pg_stat_statements",defaultValue:"pg_cron,pg_stat_statements",source:"system-default",isConfigPendingRestart:false};
  const events:string[]=[],saved:RunnerRecord[]=[];let loseAck=false;
  const control:RunnerControl={sleep:async()=>{},persist:async r=>{events.push("persist");saved.push(structuredClone(r));},list:async(_s,path)=>p.networkDeployment && path.startsWith(p.networkDeployment.deploymentId+"/operations")?childOperations:operations,request:async(_sub,path,method="GET",body)=>{
    events.push(method);
    if(method!=="GET"){
      assert.equal(method,"PUT");assert.equal(path,`${preloadId}?api-version=2024-08-01`);
      assert.deepEqual(body,{properties:{value:"pg_stat_statements,age",source:"user-override"}});
      if(loseAck)throw new Error("private failure text must not be persisted");return {status:200,value:{}};
    }
    if(path.startsWith(p.deploymentId+"?"))return {status:200,value:{properties:{provisioningState:"Failed"}}};
    if(p.networkDeployment && path.startsWith(p.networkDeployment.deploymentId+"?"))return {status:200,value:{properties:{provisioningState:"Succeeded"}}};
    if(path.startsWith(p.serverId+"?"))return {status:200,value:server};
    if(path.includes("/databases/"))return {status:200,value:{}};
    if(path.includes("/configurations/azure.extensions"))return {status:200,value:{properties:{value:"AGE"}}};
    if(path.startsWith(preloadId+"?"))return {status:200,value:{properties:config}};
    throw new Error("Unexpected read: "+path);
  }};
  return {r,control,events,saved,server,config,operations,childOperations,loseAck:()=>{loseAck=true;}};
}

test("cross-group preload repair audits both deployments and never repairs an unproven network",async()=>{
  const f=repairFixture(true);await repairBusyTargetPreload(f.control,f.r);
  const next=await repairBusyTargetPreload(f.control,f.r,true);
  assert.equal(next.target?.configurationRepair?.phase,"submitted");assert.equal(f.events.filter(x=>x==="PUT").length,1);
  for(const mutate of [
    (f:ReturnType<typeof repairFixture>)=>{f.childOperations.length=0;},
    (f:ReturnType<typeof repairFixture>)=>{f.childOperations[0]!.properties.targetResource.id="/foreign";},
    (f:ReturnType<typeof repairFixture>)=>{f.childOperations[0]!.properties.provisioningState="Failed";}
  ]){
    const bad=repairFixture(true);mutate(bad);
    await assert.rejects(repairBusyTargetPreload(bad.control,bad.r,true));assert.ok(!bad.events.includes("PUT"));
  }
});

test("preload repair is explicit, single-write, persists original failure and never replays deployment",async()=>{
  const f=repairFixture();await repairBusyTargetPreload(f.control,f.r);assert.equal(f.events.some(x=>x!=="GET"),false);
  f.events.length=0;const next=await repairBusyTargetPreload(f.control,f.r,true);
  assert.equal(next.target?.phase,"failed");assert.equal(next.target?.configurationRepair?.phase,"submitted");
  assert.equal(next.target?.configurationRepair?.originalDeploymentState,"Failed");assert.ok(f.events.indexOf("persist")<f.events.indexOf("PUT"));
  assert.equal(f.events.filter(x=>x==="PUT").length,1);assert.equal(next.target?.template,f.r.target?.template);
  f.events.length=0;await repairBusyTargetPreload(f.control,next,true);assert.ok(!f.events.includes("PUT"));
  f.config.value="pg_stat_statements,age";f.config.isConfigPendingRestart=true;
  const done=await refreshTarget(f.control,next);assert.equal(done.target?.phase,"provisioned");assert.equal(done.target?.configurationRepair?.phase,"finished");
  assert.equal((await refreshTarget(f.control,{...done,resize:{phase:"finished"} as any})).target?.phase,"provisioned");
});

test("lost preload repair acknowledgement is retained and reconciled without another write",async()=>{
  const f=repairFixture();f.loseAck();const next=await repairBusyTargetPreload(f.control,f.r,true);
  assert.equal(next.target?.configurationRepair?.phase,"unknown");assert.ok(!JSON.stringify(f.saved).includes("private failure"));
  f.events.length=0;await refreshTarget(f.control,next);assert.ok(!f.events.includes("PUT"));
  f.config.value="pg_stat_statements,age";assert.equal((await refreshTarget(f.control,next)).target?.phase,"provisioned");
});

test("repair refuses other failures, partial coverage and unexpected resources",async()=>{
  for(const mutate of [(f:ReturnType<typeof repairFixture>)=>{f.operations.at(-1)!.properties.statusMessage!.error.code="Forbidden";},
    (f:ReturnType<typeof repairFixture>)=>{f.operations.pop();},
    (f:ReturnType<typeof repairFixture>)=>{f.operations[0]!.properties.provisioningState="Failed";},
    (f:ReturnType<typeof repairFixture>)=>{f.operations[0]!.properties.targetResource.id="/foreign";}]){
    const f=repairFixture();mutate(f);await assert.rejects(repairBusyTargetPreload(f.control,f.r,true));assert.ok(!f.events.includes("PUT"));
  }
});

test("repair refuses changed target security, ownership, settings, plan or unsafe health",async()=>{
  for(const mutate of [(f:ReturnType<typeof repairFixture>)=>{f.server.properties.network.publicNetworkAccess="Enabled";},
    (f:ReturnType<typeof repairFixture>)=>{f.server.tags.workflow=file;},
    (f:ReturnType<typeof repairFixture>)=>{f.server.properties.state="Updating";},
    (f:ReturnType<typeof repairFixture>)=>{f.config.value="operator_custom_library";},
    (f:ReturnType<typeof repairFixture>)=>{f.config.source="user-override";},
    (f:ReturnType<typeof repairFixture>)=>{f.r.target!.input.storageGiB=256;},
    (f:ReturnType<typeof repairFixture>)=>{f.r.guestReady!.health!.oomEvents=1;},
    (f:ReturnType<typeof repairFixture>)=>{f.r.guestReady!.checkedAt="2020-01-01T00:00:00Z";},
    (f:ReturnType<typeof repairFixture>)=>{f.r.sourceDraft!.configuration.changed=true;},
    (f:ReturnType<typeof repairFixture>)=>{f.r.migration={} as any;}]){
    const f=repairFixture();mutate(f);await assert.rejects(repairBusyTargetPreload(f.control,f.r,true));assert.ok(!f.events.includes("PUT"));
  }
});
