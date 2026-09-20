import assert from "node:assert/strict";
import test from "node:test";
import { Script } from "node:vm";
import { runnerSourceHTML } from "../../core/runnerSourceView";
import { sourceForm, csvFile } from "../sourceFixtures";
import { buildSourceDraft } from "../../core/runnerSource";
import { workflow } from "../sourceFixtures";

class Element {
  type=""; checked=false;
  children: Element[] = []; handlers = new Map<string, (() => void)[]>(); value = ""; textContent = ""; hidden = false; disabled = false; className = "";
  constructor(readonly tag: string) {}
  append(element: Element) { this.children.push(element); }
  replaceChildren() { this.children = []; }
  remove() {}
  addEventListener(event: string, callback: () => void) { this.handlers.set(event, [...this.handlers.get(event) ?? [], callback]); }
  trigger(event: string) { for (const callback of this.handlers.get(event) ?? []) callback(); }
}
function view() {
  const html = runnerSourceHTML(), elements = new Map<string, Element>(), all: Element[] = [], messages: any[] = [];
  const make = (tag: string) => { const element = new Element(tag); all.push(element); return element; };
  for (const match of html.matchAll(/<(\w+)[^>]*\bid="([^"]+)"[^>]*>/g)) {
    const element = make(match[1]!); element.value = /\bvalue="([^"]*)"/.exec(match[0])?.[1] ?? ""; elements.set(match[2]!, element);
  }
  const receivers: ((event: { data: unknown }) => void)[] = [];
  new Script(/<script nonce="[^"]+">([\s\S]+)<\/script>/.exec(html)![1]!).runInNewContext({
    document: { getElementById: (id: string) => elements.get(id), createElement: make, querySelectorAll: (selector: string) => all.filter(e => selector.split(',').includes(e.tag)) },
    window: { addEventListener: (_event: string, callback: (event: {data: unknown}) => void) => { receivers.push(callback); } },
    acquireVsCodeApi: () => ({ postMessage: (value: unknown) => messages.push(JSON.parse(JSON.stringify(value))) })
  });
  return { html, el: (id: string) => elements.get(id)!, all, send: (data: unknown) => receivers.forEach(receive => receive({ data })), messages };
}

test("CSV choices serialize immutable file IDs rather than duplicate display names or list order", () => {
  const files = [csvFile, { id: "33333333-3333-4333-8333-333333333333", name: csvFile.name }];
  const form = { ...sourceForm, mappings: sourceForm.mappings.map((m, i) => ({ ...m, collection: files[i]!.id })) };
  for (const listed of [files, [...files].reverse()]) {
    const v = view();
    v.send({ kind: "init", type: "csv", location: "local", files: listed, form });
    v.send({ kind: "busy", value: false });
    const choices = v.all.filter(e => e.tag === "select" && e.children.some(c => c.value === csvFile.id));
    assert.equal(choices.length, 2);
    assert.deepEqual(choices.map(e => e.value), files.map(f => f.id));
    assert.deepEqual(choices[0]!.children.map(e => e.value), ["", ...listed.map(f => f.id)]);
    v.el("review").trigger("click");
    const sent = v.messages.at(-1).form;
    assert.deepEqual(sent.mappings.map((m: any) => m.collection), files.map(f => f.id));
    const csv = (buildSourceDraft({ type: "csv", location: "local" }, sent, workflow, listed).configuration.source as any).csv;
    assert.equal(csv.vertices[0].path, `/var/lib/agefreighter/workflows/${workflow}/uploads/${files[0]!.id}.csv`);
    assert.equal(csv.edges[0].path, `/var/lib/agefreighter/workflows/${workflow}/uploads/${files[1]!.id}.csv`);
    assert.ok(v.messages.every(m => ["ready", "review"].includes(m.action)));
  }
});

test("CSV typed properties, IDs, endpoints and null choices survive the webview-to-config boundary", () => {
  const declarations = "s=s:string,i=i:int64,f=f:float64,b=b:boolean,sa=sa:string[],ia=ia:int64[],fa=fa:float64[],ba=ba:boolean[]";
  for (const nullValue of ["\\N", "", "NULL"]) {
    const v = view(), form = { ...sourceForm, nullValue, mappings: sourceForm.mappings.map(m => ({ ...m, collection: csvFile.id, properties: declarations })) };
    v.send({ kind: "init", type: "csv", location: "local", files: [csvFile], form });
    v.send({ kind: "busy", value: false });
    assert.equal(v.el("nullValue").value, nullValue);
    v.el("review").trigger("click");
    const sent = v.messages.at(-1).form, draft = buildSourceDraft({ type: "csv", location: "local" }, sent, workflow, [csvFile]);
    const csv = (draft.configuration.source as any).csv;
    assert.deepEqual(csv.defaults, { delimiter: ",", quote: '"', escape: '"', header: true, encoding: "utf-8", nullValue });
    assert.deepEqual({ ...csv.vertices[0].propertyTypes }, { s: "string", i: "int64", f: "float64", b: "boolean", sa: "string[]", ia: "int64[]", fa: "float64[]", ba: "boolean[]" });
    assert.deepEqual(csv.edges[0].propertyTypes, csv.vertices[0].propertyTypes);
    assert.equal(csv.vertices[0].idColumn, "id"); assert.equal(csv.edges[0].externalIdColumn, "id");
    assert.deepEqual(csv.edges[0].start, { label: "Person", field: "from_id" });
    assert.deepEqual(csv.edges[0].end, { label: "Person", field: "to_id" });
    assert.equal(draft.canAssess, false, "settings alone are not guest file verification");
  }
});

test("CSV file and null edits invalidate approval and missing selections fail closed", () => {
  const v = view(), form = { ...sourceForm, mappings: sourceForm.mappings.map(m => ({ ...m, collection: csvFile.id })) };
  v.send({ kind: "init", type: "csv", files: [csvFile], form, canStart: true, inventoryReady: true });
  v.send({ kind: "busy", value: false });
  const choice = v.all.find(e => e.tag === "select" && e.children.some(c => c.value === csvFile.id))!;
  for (const control of [choice, v.el("nullValue")]) {
    v.send({ kind: "review", draft: { canAssess: true, warnings: [], configuration: {} } });
    assert.equal(v.el("inventory").disabled, false);
    control.trigger("change");
    assert.equal(v.el("inventory").disabled, true); assert.equal(v.el("reviewSection").hidden, true);
  }
  choice.value = ""; v.el("review").trigger("click");
  assert.throws(() => buildSourceDraft({ type: "csv", location: "local" }, v.messages.at(-1).form, workflow, [csvFile]));
  assert.throws(() => buildSourceDraft({ type: "csv", location: "local" }, form, workflow, []), /file picker/);
});

test("rejected export has a distinct host-gated action and never auto-replays",()=>{
  const v=view();v.send({kind:"init",type:"postgresql",transferEnabled:true,rejectedExportReview:false});v.send({kind:"busy",value:false});
  assert.equal(v.el("retainRejectedExport").disabled,true);
  v.send({kind:"init",type:"postgresql",transferEnabled:true,rejectedExportReview:true});
  assert.equal(v.el("retainRejectedExport").disabled,false);assert.ok(v.messages.every(m=>m.action==="ready"));
  v.el("retainRejectedExport").trigger("click");assert.deepEqual(v.messages.at(-1),{action:"retainRejectedExport"});
  assert.equal(v.el("retainRejectedExport").disabled,true);
  v.send({kind:"busy",value:false});v.send({kind:"init",type:"postgresql",transferEnabled:false,rejectedExportReview:true});assert.equal(v.el("retainRejectedExport").disabled,true);
});

test("catalog UI gates old guests, preserves unsaved forms during status/import and requires explicit selection",()=>{
  const v=view();v.send({kind:"init",type:"postgresql",canStart:true,transferEnabled:true,form:{...sourceForm,mappings:[]}});v.send({kind:"busy",value:false});
  v.send({kind:"catalog",available:false,frozen:false});assert.equal(v.el("catalogStart").disabled,true);
  v.send({kind:"catalog",available:true,frozen:false});assert.equal(v.el("catalogStart").disabled,false);
  v.el("host").value="unsaved.example";v.el("name").value="unsaved-name";v.el("catalogSchemas").value="public, other";
  v.el("catalogStart").trigger("click");const sent=v.messages.at(-1);assert.equal(sent.action,"catalogStart");assert.equal(sent.form.host,"unsaved.example");assert.deepEqual(sent.schemas,["public","other"]);
  const catalog={operation:"catalog",phase:"running",configuration:{schemas:["public","other"]}};
  v.send({kind:"catalog",catalog,available:true,frozen:false});v.send({kind:"busy",value:false});assert.equal(v.el("review").disabled,true);assert.equal(v.el("host").value,"unsaved.example");
  const proposal={id:"v:example",mapping:{...sourceForm.mappings[0],label:"<untrusted>"},reason:"metadata <script>"};
  v.send({kind:"catalog",catalog:{...catalog,phase:"finished",reportSHA256:"sealed"},available:true,frozen:false,recommendations:{reportSHA256:"sealed",proposals:[proposal],warnings:["Not row counts"]}});
  assert.equal(v.el("host").value,"unsaved.example");assert.equal(v.el("name").value,"unsaved-name");assert.equal(v.el("catalogAdopt").disabled,true);
  const choice=v.all.find(x=>x.type==="checkbox")!;assert.equal(choice.checked,false);choice.checked=true;choice.trigger("change");assert.equal(v.el("catalogAdopt").disabled,false);
  v.el("catalogAdopt").trigger("click");assert.deepEqual(v.messages.at(-1).selected,["v:example"]);assert.equal(v.messages.at(-1).form.host,"unsaved.example");
  assert.ok(v.all.some(x=>x.textContent.includes("metadata <script>")));assert.doesNotMatch(v.html,/innerHTML/);
});

test("catalog adoption invalidates source review and refuses to overwrite a concurrently edited webview",()=>{
  const v=view();v.send({kind:"init",type:"postgresql",canStart:true,form:{...sourceForm,mappings:[]}});v.send({kind:"busy",value:false});
  v.el("review").trigger("click");const original=v.messages.at(-1).form;
  v.send({kind:"review",draft:{canAssess:true,warnings:[],configuration:{}}});v.send({kind:"busy",value:false});
  v.send({kind:"catalogAdopted",original,form:sourceForm});assert.equal(v.el("reviewSection").hidden,true);
  v.el("review").trigger("click");assert.equal(v.messages.at(-1).form.mappings.length,2);
  const before=v.messages.at(-1).form;v.el("name").value="new-unsaved-name";
  v.send({kind:"catalogAdopted",original:before,form:{...sourceForm,mappings:[]}});
  assert.match(v.el("error").textContent,/unsaved edits were preserved/);assert.equal(v.el("name").value,"new-unsaved-name");
});

test("all source form branches render and submit fields without passwords or YAML input", () => {
  for (const type of ["neo4j", "postgresql", "cosmos-nosql", "csv"]) {
    const v = view(); v.send({ kind: "init", type, location: type === "csv" ? "local" : "azure", form: sourceForm, files: [csvFile] }); v.send({ kind: "busy", value: false });
    assert.equal(v.el("neo4j").hidden, type !== "neo4j"); assert.equal(v.el("cosmos").hidden, type !== "cosmos-nosql");
    assert.equal(v.el("hostLabel").hidden, type === "csv"); assert.equal(v.el("mappingSection").hidden, type === "neo4j");
    assert.equal(v.el("sourceTLS").hidden, !["neo4j", "postgresql"].includes(type));
    v.el("review").trigger("click");
    const message = v.messages.at(-1); assert.equal(message.action, "review"); assert.equal(message.form.mappings.length, 2);
    assert.equal(message.form.mappings[1].startField, "from_id"); assert.equal(message.form.password, undefined);
    assert.doesNotMatch(v.html, /type="password"|<textarea|innerHTML/);
    assert.match(v.html,/Only explicitly mapped properties are copied/);
    assert.match(v.html,/Exact counts do not prove that all source properties were preserved/);
  }
});
test("custom source CA uses a host file action and exposes only metadata to the webview", () => {
  const v = view(); v.send({ kind: "init", type: "postgresql", location: "on-premises", sourceCA: { name: "root.pem", bytes: 1234, sha256: "a".repeat(64) } }); v.send({ kind: "busy", value: false });
  assert.match(v.el("sourceCA").textContent, /root\.pem.*1234.*SHA-256/); assert.doesNotMatch(v.el("sourceCA").textContent, /Users|private|path/);
  v.el("sourceCAButton").trigger("click"); assert.deepEqual(v.messages.at(-1), { action: "sourceCA" });
});

test("Gremlin type controls restore, serialize and invalidate previous review",()=>{
  const v=view();v.send({kind:"init",type:"cosmos-nosql",location:"azure",form:{...sourceForm,cosmosFormat:"gremlin",gremlinPropertyTypes:"score=float64"},canStart:true});v.send({kind:"busy",value:false});
  assert.equal(v.el("gremlinPropertyTypes").value,"score=float64");
  assert.equal(v.el("gremlin").hidden,false);
  v.send({kind:"review",draft:{canAssess:true,warnings:[],configuration:{}}});
  v.el("gremlinPropertyTypes").value="score=int64";v.el("gremlinPropertyTypes").trigger("change");
  assert.equal(v.el("assess").disabled,true);assert.equal(v.el("reviewSection").hidden,true);
  v.el("review").trigger("click");assert.equal(v.messages.at(-1).form.gremlinPropertyTypes,"score=int64");
  assert.match(v.html,/IDs retain both partition key and ID/);
});
test("Cosmos view distinguishes ARM role readiness from data-plane access and states fixed authentication", () => {
  const v = view();
  assert.match(v.html, /does not accept Cosmos account keys/);
  assert.match(v.html, /not that data-plane permissions have propagated/);
  assert.match(v.html, /denied assessment is retained for review, not automatically retried/);
});
test("edits invalidate review, CSV cannot assess, and Gremlin toggles mapping controls", () => {
  const v = view(); v.send({ kind: "init", type: "csv", location: "local", form: sourceForm, files: [csvFile] }); v.send({ kind: "busy", value: false });
  v.send({ kind: "review", draft: { canAssess: false, warnings: [], configuration: {} } });
  assert.equal(v.el("assess").disabled, true); assert.equal(v.el("inventory").disabled, true);
  v.send({ kind: "init", type: "cosmos-nosql", location: "azure", form: sourceForm, canStart: true });
  v.el("cosmosFormat").value = "gremlin"; v.el("cosmosFormat").trigger("change");
  assert.equal(v.el("mappingSection").hidden, true); assert.equal(v.el("gremlin").hidden, false);
  v.send({ kind: "review", draft: { canAssess: true, warnings: [], configuration: {} } }); assert.equal(v.el("assess").disabled, false);
  v.el("host").trigger("change"); assert.equal(v.el("assess").disabled, true); assert.equal(v.el("reviewSection").hidden, true);
});
test("CA selection preserves unsaved connection and mapping edits while invalidating approval", () => {
  const v = view();
  v.send({ kind: "init", type: "postgresql", location: "azure", form: sourceForm, canStart: true, inventoryReady: true });
  v.send({ kind: "busy", value: false });
  v.el("database").value = "unsaved_database";
  v.el("addVertex").trigger("click");
  v.send({ kind: "review", draft: { canAssess: true, warnings: [], configuration: {} } });
  assert.equal(v.el("inventory").disabled, false);
  v.send({ kind: "sourceCA", sourceCA: { name: "new.pem", bytes: 1513, sha256: "b".repeat(64) } });
  assert.equal(v.el("database").value, "unsaved_database");
  assert.equal(v.el("inventory").disabled, true);
  assert.equal(v.el("reviewSection").hidden, true);
  assert.match(v.el("sourceCA").textContent, /new\.pem.*1513/);
  v.el("review").trigger("click");
  assert.equal(v.messages.at(-1).form.mappings.length, sourceForm.mappings.length + 1);
});
test("active operations disable starts while retaining status refresh", () => {
  const v = view(); v.send({ kind: "init", type: "neo4j", location: "on-premises", form: sourceForm }); v.send({ kind: "busy", value: false });
  v.send({ kind: "review", draft: { canAssess: true, warnings: [], configuration: {} } });
  v.send({ kind: "assessment", assessment: { operation: "op", phase: "running" } });
  assert.equal(v.el("assess").disabled, true); assert.equal(v.el("inventory").disabled, true); assert.equal(v.el("refresh").disabled, false);
  v.el("refresh").trigger("click"); assert.equal(v.messages.at(-1).action, "refresh");
});

test("failed assessment requires a separate retain action before any new source approval", () => {
  const v = view();
  v.send({ kind: "init", type: "postgresql", canStart: true, inventoryReady: true, assessment: { operation: "old", phase: "failed" } });
  v.send({ kind: "busy", value: false });
  v.send({ kind: "review", draft: { canAssess: true, warnings: [], configuration: {} } });
  assert.equal(v.el("inventory").disabled, true);
  assert.equal(v.el("retainFailure").disabled, false);
  v.el("retainFailure").trigger("click");
  assert.deepEqual(v.messages.at(-1), { action: "retainFailure" });
  v.send({ kind: "init", type: "postgresql", canStart: true, inventoryReady: true });
  v.send({ kind: "busy", value: false });
  assert.equal(v.el("retainFailure").disabled, true);
  assert.equal(v.el("inventory").disabled, true);
});
test("storage and CSV controls use host actions without URLs or credentials in the webview", () => {
  const v = view(); v.send({ kind: "init", type: "csv", location: "local", transferEnabled: true, csvTransfers: [{ file: csvFile.id, phase: "uploaded" }] }); v.send({ kind: "busy", value: false });
  assert.match(v.el("csvStatus").textContent, /uploaded/); assert.equal(v.el("uploadCSV").disabled, false);
  for (const action of ["storage", "uploadCSV", "importCSV"]) { v.el(action).trigger("click"); assert.deepEqual(v.messages.at(-1), { action }); v.send({ kind: "busy", value: false }); }
  v.send({kind:"init",type:"csv",location:"local",transferEnabled:false}); assert.equal(v.el("uploadCSV").disabled,true);
});

test("complete CSV inventory stays disabled on old guests and requires review", () => {
  const v = view();
  v.send({ kind: "init", type: "csv", canStart: true, inventoryReady: false, form: sourceForm });
  v.send({ kind: "busy", value: false });
  v.send({ kind: "review", draft: { canAssess: true, warnings: [], configuration: {} } });
  assert.equal(v.el("inventory").disabled, true);
  v.send({ kind: "init", type: "csv", canStart: true, inventoryReady: true, form: sourceForm });
  assert.equal(v.el("inventory").disabled, true);
  v.send({ kind: "review", draft: { canAssess: true, warnings: [], configuration: {} } });
  assert.equal(v.el("inventory").disabled, false);
  v.el("inventory").trigger("click"); assert.deepEqual(v.messages.at(-1), { action: "assess", method: "inventory" });
});

test("new inventory does not inherit the old sample report's imported label",()=>{
  const v=view();v.send({kind:"init",type:"csv",transfer:"imported",assessment:{operation:"sample",phase:"finished",reportSHA256:"old"}});
  assert.equal(v.el("transferStatus").textContent,"imported");
  v.send({kind:"assessment",assessment:{operation:"inventory",phase:"submitted"}});
  assert.equal(v.el("transferStatus").textContent,"No report transferred");
});

test("successful CSV selection clears the old folder error without clearing retained failures", () => {
  const v = view();
  v.send({ kind: "init", type: "csv", location: "local" });
  v.send({ kind: "error", text: "No regular CSV files were found directly in this folder." });
  v.send({ kind: "busy", value: false });
  assert.match(v.el("error").textContent, /No regular CSV/);
  v.send({ kind: "init", type: "csv", location: "local", files: [csvFile], csvTransfers: [{ file: csvFile.id, phase: "failed" }] });
  assert.equal(v.el("error").textContent, "");
  assert.equal(v.el("files").textContent, csvFile.name);
  assert.match(v.el("csvStatus").textContent, /failed/);
  assert.equal(v.el("inventory").disabled, true);
});
