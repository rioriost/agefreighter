import assert from "node:assert/strict";
import test from "node:test";
import { catalogConfiguration, startCatalog, refreshCatalog, catalogRecommendations, adoptCatalog } from "../../core/runnerCatalog";
import { sourceSecrets } from "../../core/runnerSource";
import { assessmentActive, startAssessment } from "../../core/runnerAssessment";
import { assertUpgradeIdle } from "../../core/runnerUpgrade";
import { catalogFixture, catalogForm, catalogText, catalogSHA } from "../catalogFixtures";
import { sourceForm, workflow } from "../sourceFixtures";
import { readinessReceiptReferenced } from "../../core/runnerReceipts";

test("catalog connection review has guest-compatible canonical JSON, explicit bounded scope and no mappings",()=>{
  const f=catalogFixture(),c=catalogConfiguration(f.record,catalogForm,["z","public"]);
  assert.equal(JSON.stringify(c),'{'+'"schemaVersion":1,"host":"source.example.com","port":5432,"database":"source","username":"reader","schemas":["public","z"],"sourceCASHA256":""}');
  for(const form of [{...catalogForm,host:"https://source"},{...catalogForm,port:0},{...catalogForm,database:"bad-name"},{...catalogForm,username:""}]) assert.throws(()=>catalogConfiguration(f.record,form,["public"]));
  for(const scope of [[],["pg_catalog"],["public","public"],["information_schema"],Array(17).fill("public"),["public;SELECT"]])assert.throws(()=>catalogConfiguration(f.record,catalogForm,scope));
});

test("catalog persists exact intent before protected dispatch; lost response never repeats the worker",async()=>{
  const f=catalogFixture();f.fail();
  const c=catalogConfiguration(f.record,catalogForm,["public"]);
  const r=await startCatalog(f.control,f.record,c,sourceSecrets("postgresql",c,"CATALOG-SECRET"));
  assert.equal(r.guestCommand?.phase,"unknown"); assert.equal(f.saved[0]?.postgresCatalog?.operation,r.guestCommand?.operation);
  assert.notEqual(r.postgresCatalog!.configuration,c);
  const historical = structuredClone(r.postgresCatalog!.readiness);
  const later = structuredClone(r);later.guestReady!.checkedAt="2099-01-01T00:00:00Z";
  assert.equal(readinessReceiptReferenced(later,{readiness:historical} as any),true);
  assert.ok(!JSON.stringify(f.saved).includes("CATALOG-SECRET"));
  const body=f.requests.find(x=>x.method==="PUT")!.body as any;
  const request=JSON.parse(Buffer.from(body.properties.protectedParameters[0].value,"base64").toString());
  assert.equal(request.action,"postgres-catalog");assert.deepEqual(request.configuration,c);assert.equal(request.expectedBootId,workflow);
  assert.match(request.secrets.AGEFREIGHTER_SOURCE_DSN,/CATALOG-SECRET/);
  await assert.rejects(startCatalog(f.control,r,c,{}),/retained catalog/);
  await refreshCatalog(f.control,r);assert.equal(f.requests.filter(x=>x.method==="PUT").length,1);
  assert.ok(f.requests.every(x=>x.path.startsWith(r.vmId+"/runCommands/")));
  assert.equal(assessmentActive(r),true);assert.throws(()=>assertUpgradeIdle(r));
  await assert.rejects(startAssessment(f.control,r,"inventory",{}));
});

test("catalog admission refuses old guests, stale/unsafe health, targets and prior work before dispatch",async()=>{
  for(const change of ["capability","stale","busy","disk","swap","oom","target","assessment","migration","source"]){
    const f=catalogFixture(),r=f.record;
    if(change==="capability")r.guestReady!.capabilities=[];
    if(change==="stale")r.guestReady!.checkedAt="2020-01-01T00:00:00Z";
    if(change==="busy")r.guestReady!.health!.idle=false;
    if(change==="disk")r.guestReady!.health!.storageUsedPercent=80;
    if(change==="swap")r.guestReady!.health!.swapUsedBytes=1;
    if(change==="oom")r.guestReady!.health!.oomEvents=1;
    if(change==="target")r.target={} as any;if(change==="migration")r.migration={} as any;if(change==="assessment")r.assessment={} as any;
    const c=catalogConfiguration(r,catalogForm,["public"]);
    if(change==="source")r.input.source.type="neo4j";
    await assert.rejects(startCatalog(f.control,r,c,{}));assert.equal(f.requests.length,0);
  }
});

test("catalog reconciliation requires original operation, boot, canonical request and immutable terminal seal",async()=>{
  const f=catalogFixture(),c=catalogConfiguration(f.record,catalogForm,["public"]);
  const r=await startCatalog(f.control,f.record,c,{}),a=r.postgresCatalog!;
  const result={version:1,workflow,operation:a.operation,action:"postgres-catalog",bootId:a.bootId,configSha256:a.configurationSHA256,phase:"finished",exitCode:0,reportBytes:Buffer.byteLength(catalogText),reportSha256:catalogSHA};
  for(const patch of [{bootId:"other"},{configSha256:"b".repeat(64)},{action:"inventory"},{exitCode:1},{reportBytes:0}]){
    f.set({...result,...patch});await assert.rejects(refreshCatalog(f.control,r));
  }
  f.set(result);const done=await refreshCatalog(f.control,r);assert.equal(done.postgresCatalog!.phase,"finished");assert.equal(assessmentActive(done),false);
  assert.equal(catalogRecommendations(done,catalogText).proposals.length,1);
  assert.throws(()=>catalogRecommendations(done,catalogText+" "),/SHA/);
  const changed=structuredClone(done);changed.input.source.location="other-cloud";assert.throws(()=>catalogRecommendations(changed,catalogText),/identity/);
  const next=await refreshCatalog(f.control,done);f.set({...result,reportSha256:"b".repeat(64)});await assert.rejects(refreshCatalog(f.control,next),/seal changed/);
});

test("explicit adoption preserves manual mappings, refuses connection drift and requires a fresh source review",async()=>{
  const f=catalogFixture(),c=catalogConfiguration(f.record,catalogForm,["public"]);
  const r=await startCatalog(f.control,f.record,c,{}),a=r.postgresCatalog!;
  r.postgresCatalog={...a,phase:"finished",reportSHA256:catalogSHA,reportBytes:Buffer.byteLength(catalogText)};
  const id=catalogRecommendations(r,catalogText).proposals[0]!.id;
  const raw={...catalogForm,mappings:[sourceForm.mappings[0]!]};
  const next=adoptCatalog(r,catalogText,raw,["public"],[id]);
  assert.equal(next.sourceDraft!.form.mappings.length,2);assert.deepEqual(next.sourceDraft!.form.mappings[0],raw.mappings[0]);assert.equal(r.sourceDraft,undefined);assert.equal(next.assessment,undefined);
  for(const form of [{...raw,host:"changed.example"},{...raw,database:"other"},{...raw,mappings:next.sourceDraft!.form.mappings}])assert.throws(()=>adoptCatalog(r,catalogText,form,["public"],[id]));
  assert.throws(()=>adoptCatalog(r,catalogText,raw,["other"],[id]));assert.throws(()=>adoptCatalog(r,catalogText,raw,["public"],[]));
  r.assessment={} as any;assert.throws(()=>adoptCatalog(r,catalogText,raw,["public"],[id]),/after assessment/);
});
