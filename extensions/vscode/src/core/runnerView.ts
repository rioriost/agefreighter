import { randomBytes } from "node:crypto";

export function runnerHTML(cspSource: string): string {
  const nonce = randomBytes(24).toString("base64");
  return `<!doctype html><html lang="en"><head><meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'nonce-${nonce}'; script-src 'nonce-${nonce}'; img-src ${cspSource};">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Guided migration</title>
<style nonce="${nonce}">
body{max-width:1000px;margin:32px auto;padding:0 24px;font:14px var(--vscode-font-family);color:var(--vscode-foreground)}
h1{font-size:28px}p{line-height:1.6}.muted{color:var(--vscode-descriptionForeground)}section{padding:22px;border:1px solid var(--vscode-widget-border);border-radius:6px;margin:20px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}label{display:block}input,select{display:block;box-sizing:border-box;width:100%;margin-top:7px;padding:8px;background:var(--vscode-input-background);color:var(--vscode-input-foreground);border:1px solid var(--vscode-input-border)}
button{padding:9px 14px;margin:10px 8px 0 0;background:var(--vscode-button-background);color:var(--vscode-button-foreground);border:0;cursor:pointer}button:disabled{opacity:.5;cursor:default}input[type=checkbox]{display:inline;width:auto}.steps{line-height:1.9;color:var(--vscode-descriptionForeground)}pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5}#error{color:var(--vscode-errorForeground)}[hidden]{display:none!important}
</style></head><body>
<h1>New guided migration</h1><p>No desktop AGEFreighter installation is required. One Linux VM will host discovery, migration and verification.</p>
<p class="steps">1 · Source → 2 · Discovery VM → 3 · Assess → 4 · Target & resize → 5 · Migrate → 6 · Verify</p>
<p class="muted">Preview implementation: source selection and approval-gated runner provisioning. Remote assessment, CSV upload, resize and migration are not yet enabled. Existing local LoadJob commands are separate.</p>
<div id="error" role="alert"></div><p id="activity" role="status"></p>
<button id="accounts">Refresh Azure account</button><button id="restore">Reconnect to a saved workflow</button>
<section><h2>1. Select your source</h2><div class="grid">
<label>Source type<select id="type"><option value="neo4j">Neo4j</option><option value="postgresql">PostgreSQL</option><option value="cosmos-nosql">Azure Cosmos DB for NoSQL</option><option value="csv">CSV files</option></select></label>
<label>Source location<select id="location"></select></label>
</div><div id="azureSource"><div class="grid"><label>Azure subscription<select id="subscription"><option value="">Refresh Azure account</option></select></label>
<label>Source resource group<select id="sourceGroup"><option value="">Select subscription first</option></select></label>
<label>Source candidate<select id="candidate"><option value="">Discover candidates first</option></select></label></div>
<button id="discover">Discover Azure candidates</button><label>Source ARM resource ID<input id="sourceId" placeholder="/subscriptions/.../resourceGroups/.../providers/..."></label>
<p class="muted">VMs are candidates only: neither ARM names nor ports prove Neo4j/PostgreSQL identity. Database reachability and credentials will be checked from the runner, not from this desktop.</p></div>
<div id="externalSource" hidden><p>Database endpoint and credentials will be collected after the runner is ready. Provide a subnet with existing private connectivity (VPN/ExpressRoute/peering as applicable); no source exposure is added.</p></div>
<div id="csvSource" hidden><button id="csv">Select local CSV files</button><pre id="csvFiles"></pre><p class="muted">This selects files only. Upload requires a later explicit approval and checksum-verified transfer; it is not enabled yet.</p></div></section>
<section><h2>2. Review the discovery VM</h2><p class="muted">Choose an existing resource group and compute subnet. Region/zone should match the source where known. On-premises proximity must be reviewed; we do not guess location from a hostname.</p>
<div class="grid"><label>Runner subscription<select id="runnerSubscription"><option value="">Refresh Azure account</option></select></label>
<label>Existing runner resource group<input id="runnerGroup"></label><label>Azure region<input id="region" placeholder="japaneast"></label><label>Availability zone<select id="zone"><option>1</option><option>2</option><option>3</option></select></label>
<label>Discovery size<select id="size"><option>Standard_B2s_v2</option><option>Standard_D2s_v5</option><option>Standard_D4s_v5</option></select></label></div>
<label>Existing non-delegated compute subnet ARM ID<input id="subnet" placeholder="/subscriptions/.../providers/Microsoft.Network/virtualNetworks/.../subnets/..."></label>
<p>Burstable is a low-cost starting point, not a migration sizing result. A later approved resize preserves this VM, NIC, identity and persistent disk. It does not resize the source VM.</p>
<button id="preview">Check prerequisites & preview runner</button></section>
<section id="review" hidden><h2>Reviewed deployment</h2><pre id="reviewSummary" role="status"></pre><details><summary>Resource identities, pinned version and review details</summary><pre id="record"></pre></details>
<label><input type="checkbox" id="networkApproved"> I have reviewed source reachability, private DNS and outbound access for the VM agent and release download. No public IP, SSH ingress, peering or source firewall changes will be created.</label>
<label><input type="checkbox" id="costApproved"> I accept compute plus additional storage/network charges. VM/disk evidence is retained; closing VS Code does not stop resources.</label>
<button id="deploy" disabled>Approve & deploy discovery VM</button><button id="refresh">Refresh deployment status</button>
<p class="muted">ARM “provisioned” is not guest readiness or a successful assessment. Unknown status is reconciled by deployment ID, never automatically resubmitted.</p></section>
<section><h2>3–6. Assess, size, migrate and verify</h2><p>Next: verify Linux guest readiness → assess the selected source → review target and VM sizing → select output folder and save LoadJob → approve target deployment and same-VM resize → start durable migration → verify retained evidence.</p><button disabled>Assess on runner — not yet available</button><p class="muted">No source password is collected or saved by this preview. No target resources or migration jobs are started.</p></section>
<script nonce="${nonce}">
const api=acquireVsCodeApi(), el=id=>document.getElementById(id);let busy=false, record=null, candidates=[];
const send=(action,extra={})=>api.postMessage({action,...extra});
function options(id,values,placeholder){el(id).replaceChildren();if(placeholder){const o=document.createElement('option');o.value='';o.textContent=placeholder;el(id).append(o);}for(const v of values){const o=document.createElement('option');o.value=v.value;o.textContent=v.label;el(id).append(o);}}
function update(){el('deploy').disabled=busy||!record||record.phase!=='previewed'||!el('networkApproved').checked||!el('costApproved').checked;}
function invalidate(){record=null;el('review').hidden=true;el('networkApproved').checked=false;el('costApproved').checked=false;update();}
function source(){const type=el('type').value;options('location',(type==='csv'?['local']:type==='cosmos-nosql'?['azure']:['azure','on-premises','other-cloud']).map(v=>({value:v,label:v})));locationChanged();}
function locationChanged(){el('azureSource').hidden=el('location').value!=='azure';el('externalSource').hidden=!['on-premises','other-cloud'].includes(el('location').value);el('csvSource').hidden=el('type').value!=='csv';el('sourceId').value='';candidates=[];options('candidate',[],'Discover candidates first');invalidate();}
for(const id of ['accounts','restore','csv','refresh'])el(id).addEventListener('click',()=>send(id));
el('type').addEventListener('change',source);el('location').addEventListener('change',locationChanged);
el('subscription').addEventListener('change',()=>{invalidate();el('runnerSubscription').value=el('subscription').value;options('sourceGroup',[],'Loading...');options('candidate',[],'Discover candidates first');el('sourceId').value='';send('groups',{subscription:el('subscription').value});});
el('sourceGroup').addEventListener('change',()=>{el('runnerGroup').value=el('sourceGroup').value;el('sourceId').value='';options('candidate',[],'Discover candidates first');});
el('discover').addEventListener('click',()=>send('sources',{subscription:el('subscription').value,group:el('sourceGroup').value,type:el('type').value}));
el('candidate').addEventListener('change',()=>{const c=candidates.find(v=>v.id===el('candidate').value);if(c){el('sourceId').value=c.id;if(c.type.toLowerCase()!=='microsoft.documentdb/databaseaccounts')el('region').value=c.region;if(c.zone)el('zone').value=c.zone;}});
for(const node of document.querySelectorAll('input,select'))if(!['networkApproved','costApproved'].includes(node.id))node.addEventListener('change',invalidate);
el('networkApproved').addEventListener('change',update);el('costApproved').addEventListener('change',update);
el('preview').addEventListener('click',()=>{invalidate();send('preview',{input:{subscriptionId:el('runnerSubscription').value,resourceGroup:el('runnerGroup').value,region:el('region').value,zone:el('zone').value,size:el('size').value,subnetId:el('subnet').value,source:{type:el('type').value,location:el('location').value,...(el('location').value==='azure'?{resourceId:el('sourceId').value}:{})}}});});
el('deploy').addEventListener('click',()=>send('deploy',{hash:record?.previewHash,networkApproved:el('networkApproved').checked,costApproved:el('costApproved').checked}));
window.addEventListener('message',event=>{const m=event.data;if(m.kind==='busy'){busy=m.value;for(const n of document.querySelectorAll('button,input,select'))n.disabled=busy;document.querySelector('section:last-of-type button').disabled=true;el('activity').textContent=busy?'Checking…':'';update();}
if(m.kind==='error')el('error').textContent=m.text;
if(m.kind==='subscriptions'){const opts=m.values.map(s=>({value:s.id,label:s.name+' — '+s.accountLabel}));options('subscription',opts,'Select source subscription');options('runnerSubscription',opts,'Select runner subscription');el('error').textContent='';}
if(m.kind==='groups'&&m.subscription===el('subscription').value)options('sourceGroup',m.values.map(g=>({value:g.name,label:g.name})),'Select source resource group');
if(m.kind==='sources'&&m.subscription===el('subscription').value&&m.group===el('sourceGroup').value&&m.type===el('type').value){candidates=m.values;options('candidate',m.values.map(r=>({value:r.id,label:r.name+' ('+r.type+')'})),'Select a candidate');}
if(m.kind==='csv')el('csvFiles').textContent=m.files.join('\\n');
if(m.kind==='record'){record=m.record;el('review').hidden=false;const labels={previewed:'Awaiting your approval', 'deployment-submitted':'Azure deployment submitted — refresh for progress',provisioned:'VM provisioned — guest readiness is not yet verified',failed:'Azure deployment failed — resources/evidence retained',unknown:'Deployment status unknown — refresh, do not resubmit'};el('reviewSummary').textContent=[labels[record.phase],record.input.source.type+' → Linux discovery/migration VM',record.input.region+' / zone '+record.input.zone+' / '+record.input.size,'Resource group: '+record.input.resourceGroup,'Compute estimate: USD '+record.hourlyComputeUSD+'/hour (storage and network additional)','CLI: '+record.version,'Updated: '+record.updatedAt].join('\\n');el('record').textContent=JSON.stringify(record,null,2);el('networkApproved').checked=false;el('costApproved').checked=false;el('error').textContent='';update();}});
source();send('ready');
</script></body></html>`;
}
