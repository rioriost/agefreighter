import { createHash } from "node:crypto";
import { object } from "./runner";
import { SourceMapping } from "./runnerSource";

interface Column { name:string; typeSchema:string; type:string; notNull:boolean; deterministic:boolean }
interface Constraint {
  name:string; kind:"p"|"f"; columns:string[]; referencedSchema:string; referencedTable:string; referencedColumns:string[];
  validated:boolean; enforced:boolean; deferrable:boolean; temporal:boolean; standardEquality:boolean;
}
interface Table { schema:string; name:string; kind:string; readable:boolean; rls:boolean; inheritance:boolean; columns:Column[]; constraints:Constraint[] }
export interface PostgresRecommendation { id:string; mapping:SourceMapping; reason:string }
export interface PostgresRecommendations { reportSHA256:string; proposals:PostgresRecommendation[]; warnings:string[] }
const sha=(s:string)=>createHash("sha256").update(s).digest("hex");
const identifier=/^[A-Za-z_][A-Za-z0-9_]{0,62}$/;
function name(x:unknown,empty=false):string {
  if(typeof x!=="string" || Buffer.byteLength(x)>63 || !empty&&!x || /[\x00-\x1f\x7f]/.test(x))throw new Error("Invalid PostgreSQL catalog name.");
  return x;
}
function flag(x:unknown):boolean {if(typeof x!=="boolean")throw new Error("Missing PostgreSQL catalog safety flag.");return x;}
function list(x:unknown,max:number):unknown[] {if(!Array.isArray(x)||x.length>max)throw new Error("PostgreSQL catalog exceeds its collection bound.");return x;}
function unique(xs:string[]):string[] {if(new Set(xs).size!==xs.length)throw new Error("Duplicate PostgreSQL catalog identity.");return xs;}
const key=(schema:string,table:string)=>`${schema}\0${table}`;
const label=(parts:string[])=>{const stem=parts.join("_");return `${stem.slice(0,50)}_${sha(JSON.stringify(parts)).slice(0,12)}`;};
const safeConstraint=(c:Constraint)=>c.validated&&c.enforced&&!c.deferrable&&!c.temporal&&c.standardEquality;
const identityFamily=(c:Column|undefined)=>!c?.notNull||!c.deterministic||c.typeSchema!=="pg_catalog"?undefined:["int2","int4","int8"].includes(c.type)?"integer":["text","varchar"].includes(c.type)?"text":c.type==="uuid"?"uuid":undefined;

/** Pure metadata interpretation. Caller must independently verify the report's
 * operation/source/boot/artifact binding before presenting adoption. This hash
 * alone is not trust or freshness evidence and this function performs no I/O. */
export function recommendPostgresMappings(reportJSON:string):PostgresRecommendations {
  if(Buffer.byteLength(reportJSON)>4*1024*1024)throw new Error("PostgreSQL catalog report exceeds 4 MiB.");
  const doc=object(JSON.parse(reportJSON));
  if(doc.schemaVersion!==1||doc.command!=="postgres-catalog"||doc.complete!==true)throw new Error("A complete PostgreSQL metadata catalog is required.");
  const schemas=unique(list(doc.schemas,16).map(x=>name(x)));
  if(!schemas.length||schemas.some(x=>!identifier.test(x)||x.startsWith("pg_")||x==="information_schema"))throw new Error("Review explicit non-system schema scope.");
  const tables:Table[]=list(doc.tables,64).map(raw=>{
    const t=object(raw),schema=name(t.schema),table=name(t.name);
    if(!schemas.includes(schema)||typeof t.kind!=="string"||!["r","p","v","m","f"].includes(t.kind))throw new Error("Catalog table is outside the reviewed schema scope.");
    const columns=list(t.columns,128).map(raw=>{const c=object(raw);return {name:name(c.name),typeSchema:name(c.typeSchema),type:name(c.type),notNull:flag(c.notNull),deterministic:flag(c.deterministic)};});
    unique(columns.map(x=>x.name));
    const constraints:Constraint[]=list(t.constraints,64).map(raw=>{
      const c=object(raw);if(c.kind!=="p"&&c.kind!=="f")throw new Error("Unsupported catalog constraint kind.");
      const fields=unique(list(c.columns,128).map(x=>name(x))),refs=unique(list(c.referencedColumns,128).map(x=>name(x)));
      const referencedSchema=name(c.referencedSchema,true),referencedTable=name(c.referencedTable,true);
      if(!fields.length||fields.some(f=>!columns.some(x=>x.name===f))||c.kind==="f"&&(!referencedSchema||!referencedTable||refs.length!==fields.length)||c.kind==="p"&&(referencedSchema||referencedTable||refs.length))throw new Error("Catalog key coverage is incomplete.");
      return {name:name(c.name),kind:c.kind,columns:fields,referencedSchema,referencedTable,referencedColumns:refs,
        validated:flag(c.validated),enforced:flag(c.enforced),deferrable:flag(c.deferrable),temporal:flag(c.temporal),standardEquality:flag(c.standardEquality)};
    });
    unique(constraints.map(c=>c.name));
    return {schema,name:table,kind:t.kind,readable:flag(t.readable),rls:flag(t.rls),inheritance:flag(t.inheritance),columns,constraints};
  });
  unique(tables.map(t=>key(t.schema,t.name)));
  // Stable order independent of SQL collation and transport array ordering.
  tables.sort((a,b)=>key(a.schema,a.name)<key(b.schema,b.name)?-1:1);
  const proposals:PostgresRecommendation[]=[],warnings:string[]=[];
  const vertices=new Map<string,{mapping:SourceMapping;column:Column}>();
  for(const t of tables){
    const pks=t.constraints.filter(c=>c.kind==="p"),pk=pks[0],column=t.columns.find(c=>c.name===pk?.columns[0]);
    if(t.kind!=="r"||!t.readable||t.rls||t.inheritance||!identifier.test(t.schema)||!identifier.test(t.name)||pks.length!==1||!safeConstraint(pk!)||pk!.columns.length!==1||!column||!identifier.test(column.name)||!identityFamily(column)){
      warnings.push(`${t.schema}.${t.name}: manual review required (table access/kind, RLS/inheritance, primary key or identity type).`);continue;
    }
    const mapping:SourceMapping={kind:"vertex",label:label([t.schema,t.name]),schema:t.schema,collection:t.name,identity:column.name,startLabel:"",startField:"",endLabel:"",endField:"",properties:`${column.name}=${column.name}`};
    vertices.set(key(t.schema,t.name),{mapping,column});
    proposals.push({id:`v:${mapping.label}`,mapping,reason:"Single-column non-null primary key. Only the identity property is projected; select other properties explicitly."});
  }
  for(const t of tables){
    const start=vertices.get(key(t.schema,t.name));if(!start)continue;
    for(const fk of [...t.constraints].filter(c=>c.kind==="f").sort((a,b)=>a.name<b.name?-1:1)){
      const end=vertices.get(key(fk.referencedSchema,fk.referencedTable)),column=t.columns.find(c=>c.name===fk.columns[0]);
      if(!end||!safeConstraint(fk)||fk.columns.length!==1||fk.referencedColumns.length!==1||fk.referencedColumns[0]!==end.mapping.identity||!column||!identifier.test(column.name)||!identityFamily(column)||identityFamily(column)!==identityFamily(end.column)){
        warnings.push(`${t.schema}.${t.name} / ${fk.name}: manual relationship review required (nullable/composite/unvalidated key, identity mismatch or unavailable endpoint).`);continue;
      }
      const mapping:SourceMapping={...start.mapping,kind:"edge",label:label([t.schema,t.name,"fk",sha(fk.name).slice(0,12)]),startLabel:start.mapping.label,startField:start.mapping.identity,endLabel:end.mapping.label,endField:column.name};
      proposals.push({id:`e:${mapping.label}`,mapping,reason:`Optional directed relationship from ${t.schema}.${t.name} to ${fk.referencedSchema}.${fk.referencedTable} via ${fk.name}. Business direction and property selection need review.`});
    }
  }
  unique(proposals.map(p=>p.id));unique(proposals.map(p=>p.mapping.label));
  warnings.unshift("Metadata is not row-count, sizing, data-immutability or migration evidence. Keep the source unchanged and run a new complete inventory after mapping review.");
  return {reportSHA256:sha(reportJSON),proposals,warnings};
}

/** Explicit selected IDs only. Never replace/edit existing manual mappings.
 * Adopting an edge requires its matching vertex proposal (or exact prior mapping).
 * A caller must still buildSourceDraft and require ordinary review afterward. */
export function adoptPostgresRecommendations(existing:SourceMapping[], recommendations:PostgresRecommendations, selectedIds:string[]):SourceMapping[]{
  unique(selectedIds);
  if(!selectedIds.length)throw new Error("Select recommendations explicitly.");
  const additions=selectedIds.map(id=>{const p=recommendations.proposals.find(p=>p.id===id);if(!p)throw new Error("Unknown recommendation selection.");return structuredClone(p.mapping);});
  const merged=[...structuredClone(existing),...additions];
  if(merged.length>64)throw new Error("Select at most 64 total mappings.");
  if(new Set(merged.map(m=>m.label)).size!==merged.length)throw new Error("Recommendation conflicts with existing mappings; nothing was overwritten.");
  for(const edge of additions.filter(m=>m.kind==="edge")){
    for(const endpoint of [edge.startLabel,edge.endLabel]){
      const expected=recommendations.proposals.find(p=>p.mapping.kind==="vertex"&&p.mapping.label===endpoint)?.mapping;
      const actual=merged.find(m=>m.kind==="vertex"&&m.label===endpoint);
      if(!expected||!actual||["schema","collection","identity"].some(k=>actual[k as keyof SourceMapping]!==expected[k as keyof SourceMapping]))throw new Error("Select the matching vertex endpoints before adopting this relationship.");
    }
  }
  return merged;
}
