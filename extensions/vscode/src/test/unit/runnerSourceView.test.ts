import assert from "node:assert/strict";
import test from "node:test";
import { Script } from "node:vm";
import { runnerSourceHTML } from "../../core/runnerSourceView";
import { sourceForm, csvFile } from "../sourceFixtures";

class Element {
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
  return { html, el: (id: string) => elements.get(id)!, send: (data: unknown) => receivers.forEach(receive => receive({ data })), messages };
}

test("all source form branches render and submit fields without passwords or YAML input", () => {
  for (const type of ["neo4j", "postgresql", "cosmos-nosql", "csv"]) {
    const v = view(); v.send({ kind: "init", type, location: type === "csv" ? "local" : "azure", form: sourceForm, files: [csvFile] }); v.send({ kind: "busy", value: false });
    assert.equal(v.el("neo4j").hidden, type !== "neo4j"); assert.equal(v.el("cosmos").hidden, type !== "cosmos-nosql");
    assert.equal(v.el("hostLabel").hidden, type === "csv"); assert.equal(v.el("mappingSection").hidden, type === "neo4j");
    v.el("review").trigger("click");
    const message = v.messages.at(-1); assert.equal(message.action, "review"); assert.equal(message.form.mappings.length, 2);
    assert.equal(message.form.mappings[1].startField, "from_id"); assert.equal(message.form.password, undefined);
    assert.doesNotMatch(v.html, /type="password"|<textarea|innerHTML/);
  }
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
test("active operations disable starts while retaining status refresh", () => {
  const v = view(); v.send({ kind: "init", type: "neo4j", location: "on-premises", form: sourceForm }); v.send({ kind: "busy", value: false });
  v.send({ kind: "review", draft: { canAssess: true, warnings: [], configuration: {} } });
  v.send({ kind: "assessment", assessment: { operation: "op", phase: "running" } });
  assert.equal(v.el("assess").disabled, true); assert.equal(v.el("inventory").disabled, true); assert.equal(v.el("refresh").disabled, false);
  v.el("refresh").trigger("click"); assert.equal(v.messages.at(-1).action, "refresh");
});
test("storage and CSV controls use host actions without URLs or credentials in the webview", () => {
  const v = view(); v.send({ kind: "init", type: "csv", location: "local", transferEnabled: true, csvTransfers: [{ file: csvFile.id, phase: "uploaded" }] }); v.send({ kind: "busy", value: false });
  assert.match(v.el("csvStatus").textContent, /uploaded/); assert.equal(v.el("uploadCSV").disabled, false);
  for (const action of ["storage", "uploadCSV", "importCSV"]) { v.el(action).trigger("click"); assert.deepEqual(v.messages.at(-1), { action }); v.send({ kind: "busy", value: false }); }
  v.send({kind:"init",type:"csv",location:"local",transferEnabled:false}); assert.equal(v.el("uploadCSV").disabled,true);
});
