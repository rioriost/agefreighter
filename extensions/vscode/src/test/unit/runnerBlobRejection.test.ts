import assert from "node:assert/strict";
import test from "node:test";
import {createHash} from "node:crypto";
import {downloadReport,ReportManifest,ReportRejectionCategory,ReportRejectionError} from "../../core/runnerBlob";

// Inert HTTP adapters only. These are not signed-in Azure transport outcomes.
const id="11111111-1111-4111-8111-111111111111",secret="private-capability-must-not-escape";
const text='{"value":9223372036854775807,"name":"工場"}';
const seal=(data:Uint8Array):ReportManifest=>({operation:id,bytes:data.byteLength,sha256:createHash("sha256").update(data).digest("hex")});
const manifest=seal(Buffer.from(text));
function capability():string{
  const now=Date.now(),query=new URLSearchParams({sv:"2023-11-03",spr:"https",sr:"b",sp:"r",st:new Date(now-1000).toISOString(),se:new Date(now+600000).toISOString(),sig:secret,skoid:id,sktid:id,skt:new Date(now-2000).toISOString(),ske:new Date(now+700000).toISOString(),sks:"b",skv:"2023-11-03"});
  return `https://af1111111111111111111111.blob.core.windows.net/af-${id}/reports/${id}.json?${query}`;
}
const cases:{name:string;category:ReportRejectionCategory;response:()=>Response;manifest?:ReportManifest}[]=[
  {name:"403 authorization refusal",category:"http-status",response:()=>new Response(secret,{status:403})},
  {name:"redirect response",category:"http-status",response:()=>new Response(null,{status:302,headers:{location:`https://example.invalid/${secret}`}})},
  {name:"missing body",category:"missing-body",response:()=>new Response(null)},
  {name:"encoded body",category:"content-encoding",response:()=>new Response(text,{headers:{"content-encoding":"gzip"}})},
  {name:"declared length mismatch",category:"length",response:()=>new Response(text,{headers:{"content-length":String(manifest.bytes+1)}})},
  {name:"truncated body without content length",category:"length",response:()=>new Response(text.slice(0,-1))},
  {name:"stream longer than guest seal",category:"length",response:()=>new Response(text+" ")},
  {name:"same-length changed bytes",category:"sha256",response:()=>new Response(text.replace("value","other"))},
  {name:"hash-valid malformed JSON",category:"json",response:()=>new Response('{"value":'),manifest:seal(Buffer.from('{"value":'))},
  {name:"hash-valid invalid UTF-8",category:"json",response:()=>new Response(new Uint8Array([0xff])),manifest:seal(new Uint8Array([0xff]))},
  {name:"hash-valid non-object JSON",category:"json",response:()=>new Response("[]"),manifest:seal(Buffer.from("[]"))},
  {name:"stream interruption",category:"transport",response:()=>new Response(new ReadableStream({start(controller){controller.error(new Error(secret));}}))}
];
for(const scenario of cases)test(`report rejection identifies ${scenario.name} without disclosing transport input`,async()=>{
  let calls=0;
  await assert.rejects(downloadReport(capability(),id,scenario.manifest??manifest,async(_url,init)=>{
    calls++;assert.equal(init?.method,"GET");assert.equal(init?.redirect,"error");assert.ok(init?.signal);
    assert.equal(new Headers(init.headers).has("authorization"),false);return scenario.response();
  }),error=>{
    assert.ok(error instanceof ReportRejectionError);assert.equal(error.category,scenario.category);
    assert.equal("cause" in error,false);assert.ok(!String(error.stack).includes(secret));
    assert.ok(!JSON.stringify(error).includes(secret));return true;
  });
  assert.equal(calls,1,"Exactly one fresh GET, with no retry");
});

test("header rejection survives a failing stream cancellation without leaking the cause",async()=>{
  const response=new Response(new ReadableStream({cancel(){throw new Error(secret);}}),{status:403});
  await assert.rejects(downloadReport(capability(),id,manifest,async()=>response),error=>error instanceof ReportRejectionError&&error.category==="http-status"&&!String(error.stack).includes(secret));
});

test("bad manifest or capability rejects before a request, not as an observed Azure fault",async()=>{
  let calls=0;const fetcher:typeof fetch=async()=>{calls++;return new Response(text);};
  await assert.rejects(downloadReport(capability(),id,{...manifest,bytes:0},fetcher),/manifest/);
  await assert.rejects(downloadReport(capability().replace("sp=r","sp=c"),id,manifest,fetcher),/capability/);
  assert.equal(calls,0);
});
