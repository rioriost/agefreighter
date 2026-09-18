import { createHash } from "node:crypto";
import { RunnerRecord, sourceWorkflowDraft } from "../core/runner";
import { RunnerControl } from "../core/runnerLifecycle";
import { sourceForm, workflow } from "./sourceFixtures";
export const catalogForm = { ...sourceForm, port: 5432, mappings: [] };
export const catalogText = JSON.stringify({schemaVersion:1,command:"postgres-catalog",complete:true,schemas:["public"],tables:[{
  schema:"public",name:"people",kind:"r",readable:true,rls:false,inheritance:false,
  columns:[{name:"id",typeSchema:"pg_catalog",type:"int8",notNull:true,deterministic:true}],
  constraints:[{name:"people_pkey",kind:"p",columns:["id"],referencedSchema:"",referencedTable:"",referencedColumns:[],validated:true,enforced:true,deferrable:false,temporal:false,standardEquality:true}]
}]});
export const catalogSHA = createHash("sha256").update(catalogText).digest("hex");
export function catalogFixture() {
  let record: RunnerRecord = sourceWorkflowDraft(workflow,{subscriptionId:workflow,resourceGroup:"trial",region:"japaneast",zone:"1",size:"Standard_B2s_v2",subnetId:"unused",source:{type:"postgresql",location:"on-premises"}});
  record.phase="provisioned"; record.artifact={version:"2.4.0",sha256:"a".repeat(64),url:"https://example.invalid/fixed"};
  record.guestReady={bootId:workflow,commit:"commit",cliVersion:"2.4.0",archiveSha256:"a".repeat(64),checkedAt:new Date().toISOString(),capabilities:["postgresql-catalog-v1","postgresql-native-floats-v1","postgresql-inventory-v1"],health:{idle:true,storageUsedPercent:4,swapUsedBytes:0,oomEvents:0}};
  const saved:RunnerRecord[]=[],requests:{method:string;path:string;body?:unknown}[]=[];
  let response:unknown, fail=false;
  const control:RunnerControl={sleep:async()=>{},list:async()=>[],persist:async r=>{record=structuredClone(r);saved.push(structuredClone(r));},request:async(_subscription,path,method="GET",body)=>{
    requests.push({path,method,body});
    if(method==="PUT"){if(fail)throw new Error("uncertain protected request");return{status:201,value:{}};}
    return response===undefined||!path.includes("$expand=instanceView")?{status:404,value:{}}:{status:200,value:{properties:{instanceView:{executionState:"Succeeded",exitCode:0,output:JSON.stringify(response)}}}};
  }};
  return {get record(){return record;},set record(r:RunnerRecord){record=r;},saved,requests,control,set:(r:unknown)=>{response=r;},fail:()=>{fail=true;}};
}
