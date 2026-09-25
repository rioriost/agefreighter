import { randomBytes } from "node:crypto";

export function runnerSourceHTML(): string {
  const nonce = randomBytes(24).toString("base64");
  return `<!doctype html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'nonce-${nonce}'; script-src 'nonce-${nonce}'">
<style nonce="${nonce}">body{max-width:1000px;margin:28px auto;padding:0 24px;font:14px var(--vscode-font-family);color:var(--vscode-foreground)}p,pre{line-height:1.6}section,fieldset{border:1px solid var(--vscode-widget-border);padding:18px;margin:18px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}label{display:block}input,select{display:block;box-sizing:border-box;width:100%;padding:8px;margin-top:5px;background:var(--vscode-input-background);color:var(--vscode-input-foreground);border:1px solid var(--vscode-input-border)}input[type=checkbox]{display:inline-block;width:auto;margin:0 8px 0 0}button{padding:9px 14px;margin:10px 8px 0 0;background:var(--vscode-button-background);color:var(--vscode-button-foreground);border:0}button:disabled{opacity:.5}pre{white-space:pre-wrap;overflow-wrap:anywhere}.muted{color:var(--vscode-descriptionForeground)}#error{color:var(--vscode-errorForeground)}[hidden]{display:none!important}</style></head><body>
<h1>Configure source assessment</h1><p id="sourceKind"></p><p class="muted">No desktop database connection or local CLI is used. This form prepares a read-only assessment on the selected Linux VM. It does not save the final LoadJob or deploy a target.</p>
<div id="error" role="alert"></div><section><h2>Connection and identity</h2><div class="grid">
<label>Migration name<input id="name" value="graph-migration"></label><label>Namespace<input id="namespace" value="migration"></label>
<label id="hostLabel">Host (no scheme or port)<input id="host"></label><label id="portLabel">Port<input id="port" type="number" value="7687"></label>
<label id="databaseLabel">Database<input id="database" value="neo4j"></label><label id="usernameLabel">Username<input id="username" value="neo4j"></label>
</div><p id="passwordNote">Passwords never enter this form. Use AGEFreighter: Prepare, Replace or Forget Source Credential before starting compute, or enter it after read approval. Optional encrypted reuse is limited to this workflow and connection, for up to 8 hours. TLS certificate validation is mandatory.</p>
<div id="sourceTLS"><button id="sourceCAButton">Select custom source CA bundle</button><pre id="sourceCA"></pre><p class="muted">Optional for a private PKI. The local path is never sent to the webview. The selected PEM is hash-bound, re-read for each approval, sent through the protected guest channel and removed from the Linux operation directory afterward.</p></div>
<div id="neo4j" class="grid"><label>Stable vertex key property<input id="vertexKey" value="id"></label><label>Stable edge key property<input id="edgeKey" value="id"></label></div>
<div id="cosmos"><label>Document format<select id="cosmosFormat"><option value="explicit">Explicit label/field mappings</option><option value="gremlin">Cosmos Gremlin documents over NoSQL API</option></select></label>
<p>The runner uses its managed identity, not your desktop Azure token. This guided flow does not accept Cosmos account keys. A ready grant means the account-scoped Data Reader assignment was verified through Azure Resource Manager, not that data-plane permissions have propagated. Only a successful source assessment proves data access; a denied assessment is retained for review, not automatically retried.</p><button id="cosmosAccessButton">Grant / verify Cosmos Data Reader</button><pre id="cosmosAccess"></pre>
<div id="gremlin" class="grid"><label>Container<input id="container"></label><label>Partition key property<input id="partitionKey" value="partitionKey"></label><label>Property types across all labels (optional)<input id="gremlinPropertyTypes" placeholder="score=float64,distance_km=float64"></label><p>Types apply after Gremlin property-wrapper decoding. Use string, int64, float64, boolean, or their [] arrays. Declare float64 when JSON may return 1 for 1.0. IDs retain both partition key and ID; a raw-ID-only comparison is not sufficient.</p></div>
<label id="labelFieldLabel">Document label field<input id="labelField" value="label"></label></div>
<div id="csv"><button id="pickFiles">Select local CSV files</button><button id="pickFolder">Select CSV folder</button><button id="uploadCSV">Upload reviewed CSV files</button><button id="importCSV">Import next CSV / refresh import</button><label>CSV null marker<input id="nullValue" value="\\N"></label><p>Headered UTF-8, comma-separated CSV. Folder selection lists up to 64 CSV files directly in that folder; it does not follow symlinks or upload files. Empty null marker means empty fields are null. Prepare transfer storage first. Upload verifies the selected bytes; import separately seals each file on the Linux VM. Maximum 2 GiB per file and 10 GiB per workflow. Review mappings again after all imports are verified.</p><pre id="files"></pre><pre id="csvStatus" role="status"></pre></div></section>
<section id="catalogSection" hidden><h2>PostgreSQL schema discovery (optional)</h2><label>Explicit schemas (comma separated)<input id="catalogSchemas" value="public"></label><p>Read metadata on the Linux VM before creating mappings. No row values or counts are collected. Requires a reviewed catalog-capable Linux artifact; one retained catalog per fresh workflow, without automatic retry. Manual mappings remain available.</p><button id="catalogStart" disabled>Discover reviewed schemas</button><button id="catalogRefresh" disabled>Refresh catalog status</button><button id="catalogReport" disabled>Transfer / open catalog recommendations</button><pre id="catalogStatus" role="status"></pre><pre id="catalogWarnings"></pre><div id="catalogChoices"></div><button id="catalogAdopt" disabled>Adopt selected mappings</button><p>Nothing is selected automatically. Select matching vertex endpoints for relationships. Only identity properties are proposed; review other properties and business direction. Adoption preserves existing mappings and requires source review and a new complete inventory.</p></section>
<section id="mappingSection"><h2>Vertex and edge mappings</h2><p id="mappingHelp"></p><p>For CSV or explicit Cosmos mappings, append :string, :int64, :float64, :boolean, or their [] array types. Example: score=score:float64. Cosmos JSON number spelling does not preserve schema intent; declare float64 to keep 1 as a floating-point property. Null remains null; incompatible values fail rather than being truncated. Changing declarations requires a new job, not resuming an old checkpoint.</p><p id="propertyProjectionHelp">Only explicitly mapped properties are copied. Stable IDs and endpoint fields establish identity and relationships; they are not automatically graph properties. Map those fields again as properties when required. Exact counts do not prove that all source properties were preserved.</p><div id="rows"></div><button id="addVertex">Add vertex mapping</button><button id="addEdge">Add edge mapping</button></section>
<button id="review">Review source settings</button><section id="reviewSection" hidden><h2>Reviewed assessment</h2><pre id="warnings"></pre><details><summary>Generated configuration (credentials are references only)</summary><pre id="configuration"></pre></details>
<p>Source data must remain unchanged. Discovery and ordered reads may scan more data than the 10,000-row sample. Cosmos reads consume RU. After credential entry, stale Linux readiness is refreshed automatically without reading source data. An unhealthy or changed boot blocks execution. Credential reuse is optional and encrypted, never saved in workflow JSON.</p><button id="assess" disabled>Approve sampled assessment</button><button id="inventory" disabled>Approve complete source inventory</button></section>
<section><h2>Private transfer storage</h2><pre id="storageStatus" role="status"></pre><button id="storage" disabled>Prepare / refresh transfer storage</button><p class="muted">A separate approval creates a workflow-owned storage account and grants your user data access on that account only. The HTTPS endpoint is network-public; anonymous access and shared keys are disabled. Storage/request/egress charges apply. No source firewall is changed.</p></section>
<section><h2>Retained operation</h2><pre id="assessment" role="status"></pre><button id="refresh">Refresh assessment status</button><button id="report" disabled>Transfer / open verified report</button><pre id="transferStatus" role="status"></pre><p class="muted">Active operations are watched automatically for up to 30 minutes while this panel is open. Cancel stops watching, not the guest job. One approved transfer exports and imports the sealed report; reopening reconciles the retained transfer without replay. A finished worker or imported report is not a successful migration. Capacity acceptance, target deployment and migration remain separate gates.</p></section>
<button id="retainRejectedExport" disabled>Retain rejected export / prepare fresh transfer</button><p class="muted">For a reviewed HTTP 409 export rejection only: verify a deallocated runner and absent command/blob after twenty minutes, allowing the original capability to expire. Evidence is preserved; no automatic retry. Fresh idle readiness and separate transfer approval are required afterward.</p>
<button id="retainFailure" disabled>Retain failed assessment / prepare fresh attempt</button><p class="muted">Review the failure evidence and refresh idle Linux guest readiness first. No automatic retry, evidence deletion or source read is performed.</p>
<script nonce="${nonce}">
const api=acquireVsCodeApi(),el=id=>document.getElementById(id);let type='',files=[],busy=false,reviewed=false,canAssess=false,retained=false,active=false,readyGate=false,inventoryReady=false,rows=[],transferEnabled=false,hasReport=false,currentOperation='',failedAssessment=false;
const fields=['name','namespace','host','port','database','username','vertexKey','edgeKey','cosmosFormat','container','partitionKey','gremlinPropertyTypes','labelField','nullValue'];
const send=(action,more={})=>{busy=true;update();api.postMessage({action,...more});};
function update(){for(const n of document.querySelectorAll('input,select,button'))n.disabled=busy;el('assess').disabled=busy||!reviewed||!canAssess||active||!readyGate;el('inventory').disabled=busy||!reviewed||!canAssess||!inventoryReady||active||!readyGate;el('refresh').disabled=busy||!retained;el('storage').disabled=busy||!transferEnabled;el('report').disabled=busy||!transferEnabled||!hasReport;el('uploadCSV').disabled=busy||!transferEnabled||active;el('importCSV').disabled=busy||!transferEnabled||active;el('cosmosAccessButton').disabled=busy||type!=='cosmos-nosql'||active||!readyGate;el('retainFailure').disabled=busy||!failedAssessment||!readyGate;catalogUpdate();}
let rejectedExportReview=false;
const baseUpdate=update;update=()=>{baseUpdate();el('retainRejectedExport').disabled=busy||!transferEnabled||!rejectedExportReview;};
window.addEventListener('message',event=>{if(event.data.kind==='init'){rejectedExportReview=event.data.rejectedExportReview===true;update();}});
function changed(){reviewed=false;el('reviewSection').hidden=true;update();}
function visibility(){el('neo4j').hidden=type!=='neo4j';el('cosmos').hidden=type!=='cosmos-nosql';el('csv').hidden=type!=='csv';el('sourceTLS').hidden=!['neo4j','postgresql'].includes(type);for(const id of ['hostLabel','databaseLabel'])el(id).hidden=type==='csv';for(const id of ['portLabel','usernameLabel','passwordNote'])el(id).hidden=!['neo4j','postgresql'].includes(type);const gremlin=type==='cosmos-nosql'&&el('cosmosFormat').value==='gremlin';el('gremlin').hidden=!gremlin;el('labelFieldLabel').hidden=gremlin;el('mappingSection').hidden=type==='neo4j'||gremlin;}
function input(parent,key,label,value,options){const wrap=document.createElement('label');wrap.textContent=label;const node=document.createElement(options?'select':'input');if(options)for(const option of options){const item=document.createElement('option');item.value=option.value;item.textContent=option.label;node.append(item);}node.value=value||'';node.addEventListener('change',changed);wrap.append(node);parent.append(wrap);return node;}
function add(kind,initial={}){const fieldset=document.createElement('fieldset'),legend=document.createElement('legend');legend.textContent=kind==='vertex'?'Vertex mapping':'Edge mapping';fieldset.append(legend);const grid=document.createElement('div');grid.className='grid';fieldset.append(grid);const controls={};
controls.label=input(grid,'label','Graph label',initial.label);
controls.collection=input(grid,'collection',type==='csv'?'Selected file':type==='postgresql'?'Table':'Container',initial.collection,type==='csv'?[{value:'',label:'Select a file'},...files.map(f=>({value:f.id,label:f.name}))]:undefined);
if(type==='postgresql')controls.schema=input(grid,'schema','Schema',initial.schema||'public');
controls.identity=input(grid,'identity','Stable ID column / field',initial.identity||'id');
if(kind==='edge')for(const [key,label] of [['startLabel','Start vertex label'],['startField','Start ID column / field'],['endLabel','End vertex label'],['endField','End ID column / field']])controls[key]=input(grid,key,label,initial[key]);
controls.properties=input(grid,'properties',(type==='csv'||type==='cosmos-nosql')?'Properties: name=field:type, ...':'Properties: graph_name=source_field, ...',initial.properties);
const row={kind,controls,fieldset};rows.push(row);const remove=document.createElement('button');remove.textContent='Remove mapping';remove.addEventListener('click',()=>{rows=rows.filter(r=>r!==row);fieldset.remove();changed();});fieldset.append(remove);el('rows').append(fieldset);}
function values(){const form={};for(const id of fields)form[id]=el(id).value;form.mappings=rows.map(row=>({kind:row.kind,...Object.fromEntries(Object.entries(row.controls).map(([key,n])=>[key,n.value]))}));return form;}
for(const id of fields)el(id).addEventListener('change',()=>{changed();visibility();});
el('addVertex').addEventListener('click',()=>{add('vertex');changed();});el('addEdge').addEventListener('click',()=>{add('edge');changed();});
el('pickFiles').addEventListener('click',()=>send('files'));el('pickFolder').addEventListener('click',()=>send('folder'));el('review').addEventListener('click',()=>send('review',{form:values()}));
el('sourceCAButton').addEventListener('click',()=>send('sourceCA'));
el('retainFailure').addEventListener('click',()=>send('retainFailure'));
el('cosmosAccessButton').addEventListener('click',()=>send('cosmosAccess'));
el('storage').addEventListener('click',()=>send('storage'));el('report').addEventListener('click',()=>send('report'));el('retainRejectedExport').addEventListener('click',()=>send('retainRejectedExport'));
el('uploadCSV').addEventListener('click',()=>send('uploadCSV'));el('importCSV').addEventListener('click',()=>send('importCSV'));
el('assess').addEventListener('click',()=>send('assess',{method:'profile'}));el('inventory').addEventListener('click',()=>send('assess',{method:'inventory'}));el('refresh').addEventListener('click',()=>send('refresh'));
window.addEventListener('message',event=>{const m=event.data;if(m.kind==='busy'){busy=m.value;update();}if(m.kind==='error')el('error').textContent=m.text;if(m.assessment&&m.assessment.operation!==currentOperation){currentOperation=m.assessment.operation;el('transferStatus').textContent='No report transferred';}if(m.kind==='init'){reviewed=false;el('reviewSection').hidden=true;active=false;readyGate=m.canStart===true;inventoryReady=m.inventoryReady===true;transferEnabled=m.transferEnabled===true;hasReport=!!m.assessment?.reportSHA256;el('storageStatus').textContent=m.storage||'Not prepared';el('transferStatus').textContent=m.transfer||'No report transferred';el('cosmosAccess').textContent=m.cosmosAccess||'Not granted';}if(m.assessment){retained=true;active=m.assessment.phase!=='finished'||!m.assessment.reportSHA256;hasReport=!!m.assessment.reportSHA256;}
if(m.kind==='init'){type=m.type;files=m.files||[];el('sourceKind').textContent=type+' — '+m.location;el('port').value=type==='postgresql'?'5432':'7687';el('database').value=type==='postgresql'?'postgres':'neo4j';el('username').value=type==='postgresql'?'postgres':'neo4j';if(m.form)for(const id of fields)el(id).value=String(m.form[id]??'');el('rows').replaceChildren();rows=[];for(const row of m.form?.mappings||[])add(row.kind,row);el('mappingHelp').textContent=type==='postgresql'?'Choose read-only source tables and stable ID/endpoint columns. Queries are generated. Optionally discover the reviewed schemas above on a catalog-capable Linux runner, then explicitly select mappings.':type==='csv'?'Map selected files, IDs and properties. Types: string, int64, float64, boolean and [] arrays.':'Map containers and top-level document fields. Each mapping filters the chosen label field to its graph label.';visibility();el('sourceCA').textContent=m.sourceCA?m.sourceCA.name+' — '+m.sourceCA.bytes+' bytes — SHA-256 '+m.sourceCA.sha256:'System trust store';el('files').textContent=files.map(f=>f.name).join('\\n');retained=!!m.assessment;el('assessment').textContent=m.assessment?JSON.stringify(m.assessment,null,2):'No assessment started.';update();}
if(m.kind==='review'){reviewed=true;canAssess=m.draft.canAssess;el('reviewSection').hidden=false;el('warnings').textContent=m.draft.warnings.join('\\n');el('configuration').textContent=JSON.stringify(m.draft.configuration,null,2);el('error').textContent='';update();}
if(m.kind==='assessment'){retained=true;el('assessment').textContent=JSON.stringify(m.assessment,null,2);update();}});
window.addEventListener('message',event=>{const m=event.data;if(m.kind==='init')el('csvStatus').textContent=JSON.stringify(m.csvTransfers||[],null,2);if(m.kind==='sourceCA'){el('sourceCA').textContent=m.sourceCA.name+' — '+m.sourceCA.bytes+' bytes — SHA-256 '+m.sourceCA.sha256;changed();}});
window.addEventListener('message',event=>{const m=event.data;if(m.kind==='init'||m.assessment){failedAssessment=m.assessment?.phase==='failed';update();}});
// A successful host reconciliation replaces a transient form error. Failed
// operations remain visible in the separately retained assessment/CSV status.
window.addEventListener('message',event=>{if(event.data.kind==='init')el('error').textContent='';});
let catalog, catalogAvailable=false, catalogFrozen=false, catalogChoices=[], catalogRecommendationHash='';
function catalogUpdate(){
  const pending=!!catalog&&(catalog.phase!=='finished'||!catalog.reportSHA256);
  el('catalogSection').hidden=type!=='postgresql';
  el('catalogStart').disabled=busy||type!=='postgresql'||!catalogAvailable||!!catalog||catalogFrozen;
  el('catalogRefresh').disabled=busy||!catalog;
  el('catalogReport').disabled=busy||!transferEnabled||catalog?.phase!=='finished'||!catalog.reportSHA256;
  el('catalogAdopt').disabled=busy||catalogFrozen||pending||!catalogChoices.some(x=>x.node.checked);
  if(pending){for(const id of ['review','assess','inventory','sourceCAButton'])el(id).disabled=true;}
}
const schemaValues=()=>el('catalogSchemas').value.split(',').map(x=>x.trim()).filter(Boolean);
el('catalogSchemas').addEventListener('change',changed);
el('catalogStart').addEventListener('click',()=>send('catalogStart',{form:values(),schemas:schemaValues()}));
el('catalogRefresh').addEventListener('click',()=>send('catalogRefresh'));
el('catalogReport').addEventListener('click',()=>send('catalogReport'));
el('catalogAdopt').addEventListener('click',()=>send('catalogAdopt',{form:values(),schemas:schemaValues(),selected:catalogChoices.filter(x=>x.node.checked).map(x=>x.id)}));
window.addEventListener('message',event=>{
  const m=event.data;
  if(m.assessment){catalogFrozen=true;catalogUpdate();}
  if(m.kind==='catalog'){
    if(m.catalog?.operation!==catalog?.operation&&m.catalog)el('catalogSchemas').value=m.catalog.configuration.schemas.join(', ');
    catalog=m.catalog;catalogAvailable=m.available===true;catalogFrozen=m.frozen===true;
    el('catalogStatus').textContent=catalog?JSON.stringify({operation:catalog.operation,phase:catalog.phase,schemas:catalog.configuration.schemas,reportSHA256:catalog.reportSHA256,transfer:m.transfer},null,2):catalogAvailable?'No catalog started.':'Catalog discovery requires a reviewed matching Linux artifact. Manual mappings remain available.';
    if(catalog&&catalog.phase!=='finished'){reviewed=false;el('reviewSection').hidden=true;}
    const digest=m.recommendations?.reportSHA256||'';
    if(digest!==catalogRecommendationHash){
      catalogRecommendationHash=digest;catalogChoices=[];el('catalogChoices').replaceChildren();
      el('catalogWarnings').textContent=(m.recommendations?.warnings||[]).join('\\n');
      for(const proposal of m.recommendations?.proposals||[]){
        const label=document.createElement('label'),node=document.createElement('input'),detail=document.createElement('span');
        node.type='checkbox';node.checked=false;
        detail.textContent=proposal.mapping.kind+' '+proposal.mapping.label+' — '+proposal.mapping.schema+'.'+proposal.mapping.collection+' — '+proposal.reason;
        node.addEventListener('change',catalogUpdate);label.append(node);label.append(detail);el('catalogChoices').append(label);catalogChoices.push({id:proposal.id,node});
      }
    }
    catalogUpdate();
  }
  if(m.kind==='catalogAdopted'){
    if(JSON.stringify(values())!==JSON.stringify(m.original)){el('error').textContent='Mappings were saved, but this form changed while awaiting approval. Reopen the source form to review retained settings; unsaved edits were preserved.';changed();return;}
    el('rows').replaceChildren();rows=[];for(const row of m.form.mappings)add(row.kind,row);
    for(const choice of catalogChoices)choice.node.checked=false;
    changed();el('error').textContent='';
  }
});
send('ready');
</script></body></html>`;
}
