import assert from "node:assert/strict";
import test from "node:test";
import { adoptPostgresRecommendations, recommendPostgresMappings } from "../../core/postgresRecommendations";
import { buildSourceDraft } from "../../core/runnerSource";

const column=(name:string,notNull=true,type="int8")=>({name,notNull,type,typeSchema:"pg_catalog",deterministic:true});
const pk=()=>({name:"pk",kind:"p",columns:["id"],referencedSchema:"",referencedTable:"",referencedColumns:[] as string[],validated:true,enforced:true,deferrable:false,temporal:false,standardEquality:true});
const fk=()=>({...pk(),name:"person_fk",kind:"f",columns:["person_id"],referencedSchema:"public",referencedTable:"person",referencedColumns:["id"]});
function fixture(){return {schemaVersion:1,command:"postgres-catalog",complete:true,schemas:["public"],tables:[
  {schema:"public",name:"person",kind:"r",readable:true,rls:false,inheritance:false,columns:[column("id"),column("name",false,"text")],constraints:[pk()]},
  {schema:"public",name:"orders",kind:"r",readable:true,rls:false,inheritance:false,columns:[column("id"),column("person_id"),column("amount",false,"float8")],constraints:[pk(),fk()]}
]};}
const run=(doc:unknown)=>recommendPostgresMappings(JSON.stringify(doc));

test("catalog proposals preserve typed metadata and generate explicit editable mappings, not row counts",()=>{
  const r=run(fixture());assert.equal(r.proposals.length,3);assert.equal(r.reportSHA256.length,64);
  const e=r.proposals.find(p=>p.mapping.kind==="edge")!.mapping;
  assert.equal(e.collection,"orders");assert.equal(e.startField,"id");assert.equal(e.endField,"person_id");
  const mappings=adoptPostgresRecommendations([],r,r.proposals.map(p=>p.id));
  const draft=buildSourceDraft({type:"postgresql",location:"on-premises"},{name:"pg-migration",namespace:"migration",host:"pg.example.test",port:5432,database:"p1",username:"reader",mappings},"11111111-1111-4111-8111-111111111111");
  const source=(draft.configuration.source as any).postgresql;
  assert.equal(source.vertices.length,2);assert.equal(source.edges.length,1);
  assert.equal(source.edges[0].query,'SELECT "id", "person_id" FROM "public"."orders" ORDER BY "id"');
  assert.ok(r.warnings.some(w=>w.includes("not row-count")));
  assert.ok(r.proposals.every(p=>p.mapping.properties==="id=id"));
  assert.deepEqual(run({...fixture(),tables:fixture().tables.reverse()}).proposals,r.proposals);
});

test("ambiguous identities, inaccessible/RLS/inherited/non-table sources require manual review",()=>{
  for(const mutate of [
    (d:ReturnType<typeof fixture>)=>{d.tables[0]!.readable=false;},
    (d:ReturnType<typeof fixture>)=>{d.tables[0]!.rls=true;},
    (d:ReturnType<typeof fixture>)=>{d.tables[0]!.inheritance=true;},
    (d:ReturnType<typeof fixture>)=>{d.tables[0]!.kind="p";},
    (d:ReturnType<typeof fixture>)=>{d.tables[0]!.name="person with spaces";},
    (d:ReturnType<typeof fixture>)=>{d.tables[0]!.constraints=[];},
    (d:ReturnType<typeof fixture>)=>{d.tables[0]!.constraints[0]!.columns.push("name");},
    (d:ReturnType<typeof fixture>)=>{d.tables[0]!.columns[0]!.notNull=false;},
    (d:ReturnType<typeof fixture>)=>{d.tables[0]!.columns[0]!.type="float8";},
    (d:ReturnType<typeof fixture>)=>{d.tables[0]!.columns[0]!.typeSchema="user_types";},
    (d:ReturnType<typeof fixture>)=>{d.tables[0]!.constraints[0]!.temporal=true;},
    (d:ReturnType<typeof fixture>)=>{d.tables[0]!.constraints[0]!.deferrable=true;},
    (d:ReturnType<typeof fixture>)=>{d.tables[0]!.constraints[0]!.standardEquality=false;},
    (d:ReturnType<typeof fixture>)=>{d.tables[0]!.columns[0]!.deterministic=false;}
  ]){
    const d=fixture();mutate(d);const r=run(d);
    assert.equal(r.proposals.length,1);assert.equal(r.proposals[0]!.mapping.collection,"orders");assert.ok(r.warnings.length>1);
  }
});

test("nullable, mismatched, unvalidated, unenforced, temporal and out-of-scope FKs never become edges",()=>{
  for(const mutate of [
    (d:ReturnType<typeof fixture>)=>{d.tables[1]!.columns[1]!.notNull=false;},
    (d:ReturnType<typeof fixture>)=>{d.tables[1]!.columns[1]!.type="text";},
    (d:ReturnType<typeof fixture>)=>{d.tables[1]!.constraints[1]!.validated=false;},
    (d:ReturnType<typeof fixture>)=>{d.tables[1]!.constraints[1]!.enforced=false;},
    (d:ReturnType<typeof fixture>)=>{d.tables[1]!.constraints[1]!.temporal=true;},
    (d:ReturnType<typeof fixture>)=>{d.tables[1]!.constraints[1]!.deferrable=true;},
    (d:ReturnType<typeof fixture>)=>{d.tables[1]!.constraints[1]!.standardEquality=false;},
    (d:ReturnType<typeof fixture>)=>{d.tables[1]!.columns[1]!.deterministic=false;},
    (d:ReturnType<typeof fixture>)=>{d.tables[1]!.constraints[1]!.referencedSchema="other";},
    (d:ReturnType<typeof fixture>)=>{d.tables[1]!.constraints[1]!.referencedColumns=["name"];},
    (d:ReturnType<typeof fixture>)=>{d.tables[1]!.constraints[1]!.columns.push("id");d.tables[1]!.constraints[1]!.referencedColumns.push("name");}
  ]){
    const d=fixture();mutate(d);const r=run(d);assert.equal(r.proposals.length,2);assert.ok(r.proposals.every(p=>p.mapping.kind==="vertex"));
  }
});

test("partial, malformed, duplicate and oversized catalogs cannot generate proposals",()=>{
  for(const mutate of [
    (d:any)=>{d.complete=false;},(d:any)=>{d.command="inventory";},(d:any)=>{d.schemas=[];},
    (d:any)=>{d.schemas=["public","public"];},(d:any)=>{d.schemas=["pg_catalog"];},
    (d:any)=>{d.tables.push(d.tables[0]);},(d:any)=>{d.tables[0].columns.push(d.tables[0].columns[0]);},
    (d:any)=>{d.tables[0].constraints.push(d.tables[0].constraints[0]);},
    (d:any)=>{delete d.tables[0].rls;},(d:any)=>{delete d.tables[0].constraints[0].enforced;},
    (d:any)=>{d.tables[0].constraints[0].columns=["missing"];},
    (d:any)=>{d.tables[1].constraints[1].referencedColumns=[];},
    (d:any)=>{d.tables[0].schema="foreign";},(d:any)=>{d.tables[0].name="x".repeat(64);},
    (d:any)=>{d.tables=Array(65).fill(d.tables[0]);}
  ]){const d=fixture();mutate(d);assert.throws(()=>run(d));}
  assert.throws(()=>recommendPostgresMappings(" ".repeat(4*1024*1024+1)),/4 MiB/);
});

test("explicit adoption preserves manual mappings and rejects replacement, unknown selections and missing endpoints",()=>{
  const r=run(fixture()),ids=r.proposals.map(p=>p.id),edge=r.proposals.find(p=>p.mapping.kind==="edge")!;
  assert.throws(()=>adoptPostgresRecommendations([],r,[edge.id]),/vertex endpoints/);
  assert.throws(()=>adoptPostgresRecommendations([],r,[]),/explicitly/);
  assert.throws(()=>adoptPostgresRecommendations([],r,["unknown"]),/Unknown/);
  assert.throws(()=>adoptPostgresRecommendations([],r,[ids[0]!,ids[0]!]),/Duplicate/);
  const manual={...r.proposals[0]!.mapping,properties:"amount=amount"},before=structuredClone(manual);
  assert.throws(()=>adoptPostgresRecommendations([manual],r,ids),/nothing was overwritten/);assert.deepEqual(manual,before);
  const rest=adoptPostgresRecommendations([manual],r,ids.slice(1));assert.deepEqual(rest[0],before);assert.notEqual(rest[0],manual);
  assert.throws(()=>adoptPostgresRecommendations([{...manual,identity:"different"}],r,ids.slice(1)),/matching vertex/);
  assert.throws(()=>adoptPostgresRecommendations(Array(64).fill(manual),r,[ids[1]!]),/64 total/);
});

test("same table names in different schemas and long identifiers get stable bounded unique labels",()=>{
  const d=fixture();d.schemas.push("other");d.tables.push({...d.tables[0]!,schema:"other"});
  const r=run(d);assert.equal(new Set(r.proposals.map(p=>p.mapping.label)).size,r.proposals.length);
  d.tables[0]!.name="a".repeat(63);const long=run(d);
  assert.ok(long.proposals.every(p=>/^[A-Za-z_][A-Za-z0-9_]{0,62}$/.test(p.mapping.label)));
});
