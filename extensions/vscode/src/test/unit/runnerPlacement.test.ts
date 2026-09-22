import assert from "node:assert/strict";
import test from "node:test";
import { Script } from "node:vm";
import { assertPlacementSelection, placementCatalog } from "../../core/runnerPlacement";
import { RunnerInput } from "../../core/runner";
import { runnerHTML } from "../../core/runnerView";

const subscription = "11111111-1111-4111-8111-111111111111";
const otherSubscription = "22222222-2222-4222-8222-222222222222";
test("placement guidance permits a separately reviewed network resource group", () => {
  const html = runnerHTML("https://webview.example");
  assert.match(html, /existing VNet, which may be in a different resource group/);
  assert.match(html, /review of both resource scopes and approval/);
  assert.doesNotMatch(html, /requires its existing VNet in the migration group/);
});

const catalog = placementCatalog([{ name: "source-rg", location: "westus", properties: { token: "omit" } }, { name: "migration-rg" }], [
  { name: "japaneast", displayName: "Japan East", latitude: 35.68 }, { name: "japanwest", displayName: "Japan West" }, { name: "global", displayName: "Global" }
]);

test("placement catalogs return sorted labels only, without inferring region from RG metadata", () => {
  assert.deepEqual(catalog, { groups: [{ name: "migration-rg" }, { name: "source-rg" }], regions: [{ name: "japaneast", displayName: "Japan East" }, { name: "japanwest", displayName: "Japan West" }] });
  assert.doesNotMatch(JSON.stringify(catalog), /token|latitude|westus/);
});

test("preview rejects a deleted RG or a region outside the subscription catalog", () => {
  const input: RunnerInput = { subscriptionId: subscription, resourceGroup: "migration-rg", region: "japaneast", zone: "1", size: "Standard_B2s_v2", subnetId: "unused", source: { type: "csv", location: "local" } };
  assertPlacementSelection(input, catalog);
  assert.throws(() => assertPlacementSelection({ ...input, resourceGroup: "new-not-created" }, catalog), /existing migration resource group/);
  assert.throws(() => assertPlacementSelection({ ...input, region: "madeupregion" }, catalog), /current subscription list/);
  assert.throws(() => assertPlacementSelection(input, { groups: [], regions: [] }));
});

// Execute the actual nonce-protected view script against a small DOM adapter.
// This tests selection/event behavior rather than just matching HTML strings.
class Element {
  readonly options: Element[] = [];
  readonly handlers = new Map<string, (() => void)[]>();
  textContent = "";
  disabled = false;
  checked = false;
  hidden = false;
  private selected: string | undefined;
  constructor(readonly tag: string, readonly id = "") {}
  get value(): string { return this.selected ?? (this.tag === "select" ? this.options[0]?.value ?? "" : ""); }
  set value(value: string) { this.selected = this.tag !== "select" || this.options.some(option => option.value === value) ? value : ""; }
  replaceChildren(): void { this.options.length = 0; this.selected = undefined; }
  append(element: Element): void { this.options.push(element); }
  addEventListener(event: string, handler: () => void): void { this.handlers.set(event, [...this.handlers.get(event) ?? [], handler]); }
  trigger(event: string): void { for (const handler of this.handlers.get(event) ?? []) handler(); }
}

function view() {
  const html = runnerHTML("https://webview.example");
  const elements = new Map<string, Element>();
  for (const match of html.matchAll(/<(\w+)[^>]*\bid="([^"]+)"[^>]*>/g)) elements.set(match[2]!, new Element(match[1]!, match[2]!));
  for (const match of html.matchAll(/<select id="([^"]+)">([\s\S]*?)<\/select>/g)) {
    for (const option of match[2]!.matchAll(/<option(?: value="([^"]*)")?>([^<]*)<\/option>/g)) {
      const element = new Element("option"); element.value = option[1] ?? option[2]!; element.textContent = option[2]!;
      elements.get(match[1]!)!.append(element);
    }
  }
  const future = new Element("button");
  let listener: ((event: { data: unknown }) => void) | undefined;
  const messages: Record<string, unknown>[] = [];
  const script = /<script nonce="[^"]+">([\s\S]+)<\/script>/.exec(html)![1]!;
  new Script(script).runInNewContext({
    document: { getElementById: (id: string) => elements.get(id), createElement: (tag: string) => new Element(tag),
      querySelectorAll: (selectors: string) => [...elements.values()].filter(el => selectors.split(',').includes(el.tag)), querySelector: () => future },
    window: { addEventListener: (_event: string, fn: typeof listener) => { listener = fn; } },
    acquireVsCodeApi: () => ({ postMessage: (message: unknown) => messages.push(JSON.parse(JSON.stringify(message))) })
  });
  const receive = (data: unknown) => listener!({ data });
  receive({ kind: "subscriptions", values: [{ id: subscription, name: "First", accountLabel: "Account" }, { id: otherSubscription, name: "Other", accountLabel: "Account" }] });
  receive({ kind: "busy", value: false });
  const choose = (id: string, value: string) => { elements.get(id)!.value = value; elements.get(id)!.trigger("change"); };
  const fillCatalog = (scope = "runner", sub = subscription) => { receive({ kind: "placementOptions", scope, subscription: sub, catalog }); receive({ kind: "busy", value: false }); };
  return { el: (id: string) => elements.get(id)!, choose, fillCatalog, receive, messages };
}

test("pending bootstrap shows an explicit check action and never dispatches from a status message", () => {
  const v=view(),before=v.messages.length;
  v.receive({kind:"record",record:{id:subscription,phase:"provisioned",input:{source:{type:"csv"}},guestCommand:{phase:"bootstrap-pending"}}});
  assert.match(v.el("guestStatus").textContent,/bootstrap is still running/);
  assert.match(v.el("guestStatus").textContent,/explicitly check/);
  assert.doesNotMatch(v.el("guestStatus").textContent,/verified at|could not be verified/);
  assert.equal(v.messages.length,before);
  assert.equal(v.el("guestReady").disabled,false);
  v.el("guestRefresh").trigger("click");
  assert.deepEqual(v.messages.at(-1),{action:"guestRefresh",workflow:subscription});
});

test("changing a retained source clears readiness and blocks old workflow controls", () => {
  const v = view();
  const record = {id: subscription, phase: "provisioned", input: {source: {type: "csv"}}, guestReady: {checkedAt: "old-check"}, guestCommand: {phase: "finished"}};
  v.receive({kind: "record", record});
  assert.match(v.el("guestStatus").textContent, /old-check/);
  assert.equal(v.el("continueExecution").disabled, false);
  v.el("guestReady").trigger("click");
  assert.deepEqual(v.messages.at(-1), {action: "guestReady", workflow: subscription});
  v.receive({kind: "busy", value: false});
  v.choose("type", "neo4j");
  assert.doesNotMatch(v.el("guestStatus").textContent, /old-check|verified/);
  for (const id of ["guestReady", "guestRefresh", "reviewTarget", "continueExecution", "refresh", "deploy"]) assert.equal(v.el(id).disabled, true, id);
  // Even a synthetic stale click cannot send the previous workflow ID.
  v.el("continueExecution").trigger("click");
  assert.deepEqual(v.messages.at(-1), {action: "continueExecution"});
});

test("renewing a preview retains the workflow ID and requires fresh deployment consent", () => {
  const v = view();
  const input = { subscriptionId: subscription, resourceGroup: "migration-rg", region: "japaneast", zone: "1", size: "Standard_B2s_v2", subnetId: "retained-subnet", source: { type: "csv", location: "local" } };
  v.receive({ kind: "restoreInput", input, files: ["nodes.csv"] });
  v.receive({ kind: "record", record: { id: subscription, phase: "previewed", input, expiresAt: new Date(0).toISOString() } });
  v.el("networkApproved").checked = true; v.el("costApproved").checked = true;
  v.el("preview").trigger("click");
  assert.deepEqual(v.messages.at(-1), { action: "preview", draftId: subscription, input });
  assert.equal(v.el("networkApproved").checked, false);
  assert.equal(v.el("costApproved").checked, false);
  assert.equal(v.el("deploy").disabled, true);
  v.receive({ kind: "busy", value: false });
  v.choose("type", "neo4j");
  v.el("preview").trigger("click");
  assert.equal(v.messages.at(-1)?.draftId, undefined);
});

test("CSV and external source paths independently load runner RG/region dropdowns", () => {
  const v = view();
  v.choose("type", "csv");
  v.choose("runnerSubscription", subscription);
  assert.deepEqual(v.messages.at(-1), { action: "placementOptions", scope: "runner", subscription });
  v.fillCatalog();
  assert.equal(v.el('runnerGroup').tag, 'select'); assert.equal(v.el('region').tag, 'select');
  assert.ok(v.el('region').options.some(o => o.textContent === 'Japan East (japaneast)'));
  assert.equal(v.el('region').value, '');
  v.choose('runnerGroup','migration-rg'); v.choose('region','japaneast');
  assert.equal(v.el('preview').disabled, true);
  v.choose('zone','1');
  assert.equal(v.el('preview').disabled, false);
  v.choose('type','postgresql'); v.choose('location','on-premises');
  assert.equal(v.el('region').value, 'japaneast');
});
test("reconnect restores the saved source and placement without starting cloud work", () => {
  const v = view(), before = v.messages.length;
  v.receive({ kind: "restoreInput", input: { subscriptionId: subscription, resourceGroup: "migration-rg", region: "japaneast", zone: "2", size: "Standard_D2s_v5", subnetId: "retained-subnet", source: { type: "csv", location: "local" } }, files: ["nodes.csv"] });
  assert.equal(v.el("type").value, "csv"); assert.equal(v.el("location").value, "local");
  assert.equal(v.el("runnerGroup").value, "migration-rg"); assert.equal(v.el("region").value, "japaneast");
  assert.equal(v.el("runnerSubscription").value, subscription); assert.equal(v.el("subnet").value, "retained-subnet");
  assert.equal(v.el("zone").value, "2"); assert.equal(v.el("csvFiles").textContent, "nodes.csv");
  assert.equal(v.messages.length, before);
});

test("source RG suggests a default without overwriting a separate migration RG", () => {
  const v = view(); v.choose('subscription',subscription); v.fillCatalog('both');
  v.choose('sourceGroup','source-rg'); assert.equal(v.el('runnerGroup').value, 'source-rg');
  assert.equal(v.el('region').value, ''); // RG metadata is not compute placement.
  v.choose('runnerGroup','migration-rg'); v.choose('sourceGroup','migration-rg'); v.choose('sourceGroup','source-rg');
  assert.equal(v.el('runnerGroup').value, 'migration-rg');
});

test("source candidate selects an available region and refresh preserves the reviewed choice", () => {
  const v = view(); v.choose('subscription',subscription); v.fillCatalog('both'); v.choose('sourceGroup','source-rg');
  const id = `/subscriptions/${subscription}/resourceGroups/source-rg/providers/Microsoft.Compute/virtualMachines/source`;
  v.receive({ kind:'sources', subscription, group:'source-rg', type:'neo4j', values:[{ id, name:'Source', region:'japaneast', zone:'2', type:'Microsoft.Compute/virtualMachines' }] });
  v.choose('candidate', id);
  assert.equal(v.el('region').value, 'japaneast'); assert.equal(v.el('zone').value, '2');
  v.choose('region','japanwest'); v.fillCatalog(); assert.equal(v.el('region').value,'japanwest');
});

test("runner subscription changes clear old selections and ignore stale catalog responses", () => {
  const v = view(); v.choose('runnerSubscription',subscription); v.fillCatalog(); v.choose('runnerGroup','migration-rg'); v.choose('region','japaneast'); v.el('subnet').value='old-subnet';
  v.choose('runnerSubscription', otherSubscription);
  assert.equal(v.el('runnerGroup').value,''); assert.equal(v.el('region').value,''); assert.equal(v.el('subnet').value,'');
  v.fillCatalog('runner',subscription); assert.equal(v.el('region').options.length,1);
  v.fillCatalog('runner',otherSubscription); assert.ok(v.el('region').options.length>1);
});

test("empty or deleted options block preview instead of retaining stale selection", () => {
  const v = view(); v.choose('runnerSubscription', subscription); v.fillCatalog(); v.choose('runnerGroup','migration-rg'); v.choose('region','japaneast');
  v.receive({ kind:'placementOptions', scope:'runner', subscription, catalog:{groups:[],regions:[]} });
  assert.equal(v.el('region').value,''); assert.equal(v.el('runnerGroup').value,''); assert.equal(v.el('preview').disabled,true);
});

test("unknown source zone clears a previous known zone and requires explicit review", () => {
  const v = view(); v.choose('subscription',subscription); v.fillCatalog('both'); v.choose('sourceGroup','source-rg');
  const prefix = `/subscriptions/${subscription}/resourceGroups/source-rg/providers/Microsoft.Compute/virtualMachines/`;
  v.receive({kind:'sources',subscription,group:'source-rg',type:'neo4j',values:[
    {id:prefix+'known',region:'japaneast',zone:'2',type:'Microsoft.Compute/virtualMachines'},
    {id:prefix+'unknown',region:'japaneast',zone:'',type:'Microsoft.Compute/virtualMachines'}
  ]});
  v.choose('candidate',prefix+'known'); assert.equal(v.el('zone').value,'2');
  v.choose('candidate',prefix+'unknown'); assert.equal(v.el('zone').value,'');
  assert.equal(v.el('preview').disabled,true); assert.equal(v.el('configureSource').disabled,true);
  v.choose('zone','3'); assert.equal(v.el('preview').disabled,false);
  v.el('preview').trigger('click');
  assert.equal((v.messages.at(-1)!.input as RunnerInput).zone,'3');
});

test("source identity, location, region and runner subscription changes clear stale zones", () => {
  for (const [field,value] of [['type','csv'],['location','on-premises'],['sourceGroup','migration-rg'],['region','japanwest'],['runnerSubscription',otherSubscription],['sourceId','manual-source-id']]) {
    const v = view(); v.choose('subscription',subscription); v.fillCatalog('both'); v.choose('sourceGroup','source-rg');
    v.choose('region','japaneast'); v.choose('zone','2');
    v.choose(field!,value!);
    assert.equal(v.el('zone').value,'',field);
    assert.equal(v.el('preview').disabled,true,field);
  }
});

test("logical source zone is not copied across subscriptions", () => {
  const v = view(); v.choose('subscription',subscription); v.fillCatalog('both'); v.choose('sourceGroup','source-rg');
  v.choose('runnerSubscription',otherSubscription); v.fillCatalog('runner',otherSubscription);
  const id = `/subscriptions/${subscription}/resourceGroups/source-rg/providers/Microsoft.Compute/virtualMachines/source`;
  v.receive({kind:'sources',subscription,group:'source-rg',type:'neo4j',values:[{id,region:'japaneast',zone:'2',type:'Microsoft.Compute/virtualMachines'}]});
  v.choose('candidate',id);
  assert.equal(v.el('zone').value,'');
});

test("initial placement never silently chooses zone 1", () => {
  const v = view(); assert.equal(v.el('zone').value,'');
  assert.equal(v.el('preview').disabled,true);
});

test("manual region override replaces inferred guidance and survives catalog refresh", () => {
  const v = view(); v.choose('subscription',subscription); v.fillCatalog('both'); v.choose('sourceGroup','source-rg');
  const id = 'known-source';
  v.receive({kind:'sources',subscription,group:'source-rg',type:'neo4j',values:[{id,region:'japaneast',zone:'1',type:'Microsoft.Compute/virtualMachines'}]});
  v.choose('candidate',id); assert.match(v.el('regionNote').textContent,/Source region selected/);
  v.choose('region','japanwest');
  assert.match(v.el('regionNote').textContent,/explicitly selected/);
  assert.doesNotMatch(v.el('regionNote').textContent,/Source region selected/);
  v.fillCatalog(); assert.match(v.el('regionNote').textContent,/explicitly selected/);
  assert.equal(v.el('region').value,'japanwest'); assert.equal(v.el('zone').value,'');
  v.choose('region',''); assert.doesNotMatch(v.el('regionNote').textContent,/explicitly selected/);
});

test("source and placement identity changes remove stale inferred region guidance", () => {
  for (const [field,value] of [['type','csv'],['location','other-cloud'],['sourceGroup','migration-rg'],['runnerSubscription',otherSubscription],['sourceId','manual-id'],['candidate','']]) {
    const v = view(); v.choose('subscription',subscription); v.fillCatalog('both'); v.choose('sourceGroup','source-rg');
    v.receive({kind:'sources',subscription,group:'source-rg',type:'neo4j',values:[{id:'known',region:'japaneast',zone:'1',type:'Microsoft.Compute/virtualMachines'}]});
    v.choose('candidate','known'); assert.match(v.el('regionNote').textContent,/Source region selected/);
    v.choose(field!,value!);
    assert.doesNotMatch(v.el('regionNote').textContent,/Source region selected/,field);
    assert.match(v.el('regionNote').textContent,/Review/,field);
    assert.equal(v.el('preview').disabled,true,field);
  }
});

for (const scenario of ['cosmos','unknown-zone','unsupported-zone','unavailable-region','deselected']) test(`candidate ${scenario} requires reviewed zone`, () => {
  const v = view();
  if (scenario === 'cosmos') v.choose('type','cosmos-nosql');
  v.choose('subscription',subscription); v.fillCatalog('both'); v.choose('sourceGroup','source-rg');
  v.choose('region','japaneast'); v.choose('zone','2');
  const id = 'test-candidate';
  v.receive({kind:'sources',subscription,group:'source-rg',type:scenario==='cosmos'?'cosmos-nosql':'neo4j',values:[{
    id,region:scenario==='unavailable-region'?'unlisted':'japaneast',zone:scenario==='unknown-zone'?'':scenario==='unsupported-zone'?'4':'1',
    type:scenario==='cosmos'?'Microsoft.DocumentDB/databaseAccounts':'Microsoft.Compute/virtualMachines'
  }]});
  v.choose('candidate',scenario==='deselected'?'':id);
  assert.equal(v.el('zone').value,''); assert.equal(v.el('preview').disabled,true);
});
