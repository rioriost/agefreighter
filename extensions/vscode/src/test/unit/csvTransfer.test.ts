import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { inspectCSV, uploadCSV, validateCSVManifest } from "../../guided/csvTransfer";
import { sourceWorkflowDraft } from "../../core/runner";
import { csvAssessmentReady } from "../../core/runnerCSV";
import { buildSourceDraft } from "../../core/runnerSource";

const id="11111111-1111-4111-8111-111111111111", file="22222222-2222-4222-8222-222222222222";
const record=()=>sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"test",region:"japaneast",zone:"1",subnetId:"subnet",size:"Standard_B2s_v2",source:{type:"csv",location:"local"}});
const credential={getToken:async()=>({token:"private-token",expiresOnTimestamp:Date.now()+3600000})};
test("CSV upload hashes all bytes, stages bounded blocks and conditionally commits exact content",async()=>{
 const dir=await mkdtemp(join(tmpdir(),"af-csv-"));try{
  const path=join(dir,"input.csv"),data=Buffer.alloc(9*1024*1024,65);await writeFile(path,data);
  const manifest=await inspectCSV(file,path);assert.equal(manifest.sha256,createHash("sha256").update(data).digest("hex"));
  const calls:{url:string;method:string}[]=[],blocks:Buffer[]=[];
  await uploadCSV(record(),path,manifest,credential,async(url,init)=>{
   calls.push({url:String(url),method:String(init?.method)});assert.equal(init?.redirect,"error");assert.ok(init?.signal);
   assert.equal(new Headers(init?.headers).get("authorization"),"Bearer private-token");
   assert.ok(Math.abs(Date.now()-Date.parse(new Headers(init?.headers).get("x-ms-date")!)) < 5000);
   if(init?.method==="HEAD")return new Response(null,{status:404});
   if(String(url).includes("comp=block&"))blocks.push(Buffer.from(init?.body as Uint8Array));
   else {assert.equal(new Headers(init?.headers).get("if-none-match"),"*");assert.ok(String(init?.body).includes("<Latest>"));}
   return new Response(null,{status:201});
  });
  assert.equal(calls.length,4);assert.deepEqual(Buffer.concat(blocks),data);assert.ok(blocks.every(b=>b.length<=8*1024*1024));
 }finally{await rm(dir,{recursive:true,force:true});}
});
test("CSV changed after approval never commits and a matching existing blob reconciles without PUT",async()=>{
 const dir=await mkdtemp(join(tmpdir(),"af-csv-"));try{
  const path=join(dir,"input.csv");await writeFile(path,"id\n1\n");const manifest=await inspectCSV(file,path);await writeFile(path,"id\n2\n");let commits=0;
  const fake:typeof fetch=async(url,init)=>{if(String(url).includes("blocklist"))commits++;return new Response(null,{status:init?.method==="HEAD"?404:201});};
  await assert.rejects(uploadCSV(record(),path,manifest,credential,fake),/changed/);assert.equal(commits,0);
  await writeFile(path,"id\n1\n");let calls=0;
  await uploadCSV(record(),path,manifest,credential,async(_url,init)=>{calls++;assert.equal(init?.method,"HEAD");return new Response(null,{headers:{"content-length":String(manifest.bytes),"x-ms-meta-sha256":manifest.sha256}});});
  assert.equal(calls,1);
 }finally{await rm(dir,{recursive:true,force:true});}
});
test("CSV failures do not leak credentials or silently retry; manifest bounds fail closed",async()=>{
 const dir=await mkdtemp(join(tmpdir(),"af-csv-"));try{
  const path=join(dir,"input.csv");await writeFile(path,"id\n1\n");const manifest=await inspectCSV(file,path);let calls=0;
  await assert.rejects(uploadCSV(record(),path,manifest,credential,async()=>{calls++;throw new Error("private-token");}),e=>e instanceof Error&&!e.message.includes("private-token"));assert.equal(calls,1);
  for(const bytes of [0,-1,2**31+1,NaN])assert.throws(()=>validateCSVManifest({...manifest,bytes}));
 }finally{await rm(dir,{recursive:true,force:true});}
});
test("CSV assessment requires every mapped path to have an independent verified guest receipt",()=>{
 const r=record();r.sourceFiles=[{id:file,name:"file.csv",path:"local"}];
 r.sourceDraft=buildSourceDraft(r.input.source,{name:"test",namespace:"test",nullValue:"",mappings:[{kind:"vertex",label:"Person",collection:file,identity:"id",properties:""}]},id,r.sourceFiles);
 assert.equal(csvAssessmentReady(r),false);r.csvTransfers=[{file,bytes:5,sha256:"a".repeat(64),phase:"uploaded"}];assert.equal(csvAssessmentReady(r),false);
 r.csvTransfers[0]!.phase="verified";assert.equal(csvAssessmentReady(r),true);
 r.csvTransfers[0]!.file=id;assert.equal(csvAssessmentReady(r),false);
});
