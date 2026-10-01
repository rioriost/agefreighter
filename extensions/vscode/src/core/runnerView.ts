import { randomBytes } from "node:crypto";
import { guidedFeedbackCSS, guidedFeedbackHTML, guidedFeedbackScript } from "./guidedFeedback";
import { executionActions, ExecutionGroup } from "./runnerExecutionActions";

function executionButtons(group: ExecutionGroup): string {
  return `<ol class="execution-actions">${executionActions.filter(action => action.group === group).map(action =>
    `<li><button id="execute-${action.id}" class="secondary" aria-describedby="execute-${action.id}-status" disabled>${action.step}. ${group === "migrate" || group === "verify" ? "" : "(optional) "}${action.label}</button><p id="execute-${action.id}-status" class="muted">Select or reconnect to a saved workflow first.</p></li>`).join("")}</ol>`;
}

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
.execution-actions{list-style:none;margin:0;padding:0}.execution-actions li{padding:6px 0 12px;border-bottom:1px solid var(--vscode-widget-border)}.execution-actions p{margin:7px 0 0}.execution-actions button{font:inherit;max-width:100%;text-align:left;white-space:normal;overflow-wrap:anywhere}.execution-actions button:not(:disabled):hover{filter:brightness(1.1)}.execution-actions button:not(:disabled):active{outline:2px solid var(--vscode-focusBorder);outline-offset:1px}
${guidedFeedbackCSS}</style></head><body>
<h1>New guided migration</h1><p>No desktop AGEFreighter installation is required. One Linux VM will host discovery, migration and verification.</p>
<p class="steps">1 · Source → 2 · Discovery VM → 3 · Assess → 4 · Target & resize → 5 · Migrate → 6 · Verify</p>
<details><summary>Scope and safety</summary><p class="muted">Runner/storage provisioning, source assessment, private target review and separately approved CSV, Neo4j, PostgreSQL or Cosmos execution. Azure migration qualification is path-specific. Failed migrations require operator reconciliation; no automatic resume or replay. Existing local LoadJob commands are separate.</p></details>
${guidedFeedbackHTML}
<button id="accounts">Refresh Azure account</button><button id="restore">Reconnect to a saved workflow</button>
<section id="sourceStep"><h2>1. Select your source</h2><div class="grid">
<label>Source type<select id="type"><option value="neo4j">Neo4j</option><option value="postgresql">PostgreSQL</option><option value="cosmos-nosql">Azure Cosmos DB for NoSQL</option><option value="csv">CSV files</option></select></label>
<label>Source location<select id="location"></select></label>
</div><div id="azureSource"><div class="grid"><label>Azure subscription<select id="subscription"><option value="">Refresh Azure account</option></select></label>
<label>Source resource group<select id="sourceGroup"><option value="">Select subscription first</option></select></label>
<label>Source candidate<select id="candidate"><option value="">Discover candidates first</option></select></label></div>
<button id="discover">Discover Azure candidates</button><label>Source ARM resource ID<input id="sourceId" placeholder="/subscriptions/.../resourceGroups/.../providers/..."></label>
<p class="muted">VMs are candidates only: neither ARM names nor ports prove Neo4j/PostgreSQL identity. Database reachability and credentials will be checked from the runner, not from this desktop.</p></div>
<div id="externalSource" hidden><p>Database endpoint and credentials will be collected after the runner is ready. Provide a subnet with existing private connectivity (VPN/ExpressRoute/peering as applicable); no source exposure is added.</p></div>
<div id="csvSource" hidden><button id="csv">Select local CSV files</button><pre id="csvFiles"></pre><p class="muted">This selects files only. Review mappings in the source editor, then separately approve storage, upload and checksum-verified Linux import before assessment.</p></div></section>
<section><h2>2. Review the discovery VM</h2><p class="muted">Choose the existing migration resource group for the discovery VM and the later Flexible Server deployment. The source resource group may be different. Region/zone should match the source where known. On-premises proximity must be reviewed; we do not guess location from a hostname.</p>
<div class="grid"><label>Runner subscription<select id="runnerSubscription"><option value="">Refresh Azure account</option></select></label>
<label>Migration resource group (existing)<select id="runnerGroup"><option value="">Select runner subscription first</option></select></label><label>Azure region<select id="region"><option value="">Select runner subscription first</option></select></label><label>Availability zone<select id="zone"><option value="">Select a zone after reviewing placement</option><option>1</option><option>2</option><option>3</option></select></label>
<label>Discovery size<select id="size"><option>Standard_B2s_v2</option><option>Standard_D2s_v5</option><option>Standard_D4s_v5</option></select></label></div>
<button id="refreshPlacement">Refresh resource groups & regions</button>
<details><summary>Resource groups and network prerequisites</summary><p class="muted">Need a new resource group? Create it in Azure, then refresh this list. A resource group is not a network boundary. Peering is not required merely because groups differ. The private target uses the reviewed existing VNet, which may be in a different resource group. A separate delegated subnet is created there only after assessment, review of both resource scopes and approval.</p></details>
<p class="muted" id="regionNote">Region listing is not a capacity guarantee. SKU, zone and quota are checked before deployment. A resource group's metadata location does not determine the VM region.</p>
<p class="muted" id="zoneNote">Unknown source placement is not zone 1. Select a runner zone explicitly when no same-subscription source zone is known.</p>
<button id="selectSubnet">Choose runner subnet from Azure</button><p id="subnetSummary" class="next-action">Select the subnet intended for the runner. Do not use the future PostgreSQL delegated subnet.</p>
<details><summary>Advanced: existing non-delegated compute subnet ARM ID</summary><label>Subnet ARM ID<input id="subnet" placeholder="/subscriptions/.../providers/Microsoft.Network/virtualNetworks/.../subnets/..."></label></details>
<p>Burstable is a low-cost starting point, not a migration sizing result. A later approved resize preserves this VM, NIC, identity and persistent disk. It does not resize the source VM.</p>
<button id="preview">Check prerequisites & preview runner</button></section>
<section id="review" hidden><h2>Reviewed deployment</h2><pre id="reviewSummary" role="status"></pre><details><summary>Resource identities, pinned version and review details</summary><pre id="record"></pre></details>
<div id="approvalControls"><label><input type="checkbox" id="networkApproved"> I have reviewed source reachability, private DNS and outbound access for the VM agent and release download. No public IP, SSH ingress, peering or source firewall changes will be created.</label>
<label><input type="checkbox" id="costApproved"> I accept compute plus additional storage/network charges. VM/disk evidence is retained; closing VS Code does not stop resources.</label>
<button id="deploy" disabled>Approve & deploy discovery VM</button></div><button id="refresh" class="secondary">Refresh deployment status</button>
<p class="muted">Deployment and submitted readiness checks refresh automatically for up to 10 minutes. ARM “provisioned” is not guest readiness. Unknown status is reconciled by ID, never resubmitted.</p>
<button id="guestReady" disabled>Check Linux guest readiness</button><button id="guestRefresh" class="secondary" disabled>Refresh guest command</button><button id="stopWatch" class="secondary">Stop automatic refresh</button><pre id="guestStatus" role="status"></pre></section>
<section id="assessmentStep"><h2>3. Assess the source</h2><p id="assessmentSummary" class="next-action">Next: configure the connection, select a private CA if required, then approve a complete source inventory.</p><button id="configureSource" disabled>Configure source & assessment</button><p class="muted">Required transfer storage is prepared with your approval. Status updates and the approved report transfer follow automatically. Source passwords are requested privately, not entered into this page.</p></section>
<section id="targetStep"><h2>4. Review target and migration sizing</h2><p id="targetSummary">Requires a passing, hash-verified complete inventory. A sampled or incomplete report is not sufficient.</p><button id="reviewTarget" disabled>Review / reconcile private target</button><p class="muted">Review Flexible Server, the target subnet, output folder, LoadJob and same-VM resize. Each deployment and additional charge still requires approval.</p></section>
<section id="executionStep"><h2>5. Migrate</h2><p id="migrationSummary" class="next-action" role="status">Complete the private target in step 4, then follow the required steps below.</p>
<h3>Required steps - run in order</h3>${executionButtons("migrate")}
<p class="muted">Step 5-3 checks readiness again after resizing; it is valid for five minutes. Step 5-4 requires separate approval and generates complete counts evidence after the load. No automatic migration, resume or replay.</p>
<details id="migrationTools"><summary>Optional: cost authorization and status reconciliation</summary><p class="muted">Use only when the authorization needs renewal or an operation needs a manual status check. These are not extra required migration steps.</p>${executionButtons("migrationTools")}</details>
<details id="migrationRecovery"><summary>Optional: recovery and diagnostics</summary><p class="muted">Not needed for a successful first migration. For failed / interrupted jobs, check recovery readiness, inspect the checkpoint, then separately approve a resume. Never start a replacement job to bypass retained evidence.</p>${executionButtons("recovery")}</details></section>
<section id="verificationStep"><h2>6. Verify</h2><p id="verificationSummary" class="next-action" role="status">Wait for the migration's terminal verification report.</p>
<h3>Required result review</h3>${executionButtons("verify")}
<p class="muted">The Linux worker already runs counts verification after migration. This step transfers, hash-checks and evaluates that retained report; it never reruns the load. Migration finished, report imported and counts passed are distinct states.</p>
<details id="developmentVerification"><summary>Optional: full P1 qualification (development only)</summary><p class="muted">Only for the exact full P1 development fixture and explicit development opt-in, not ordinary migration or a smaller demo dataset. Counts verification is not a full-property digest.</p>${executionButtons("development")}</details></section>
<script nonce="${nonce}">
${guidedFeedbackScript}
const api=acquireVsCodeApi(), el=id=>document.getElementById(id);let busy=false, record=null, candidates=[],draftID=null;
const executionActions=${JSON.stringify(executionActions)};
function executionText(id,text){const node=el(id);if(node.textContent!==text)node.textContent=text;}
function updateExecution(){
  let next=null;
  const summary=record?.execution, expired=!summary?.migration&&summary?.readinessExpiresAt!==undefined&&Date.now()>summary.readinessExpiresAt;
  for(const action of executionActions){
    const button=el('execute-'+action.id),state=summary?.actions[action.id];
    let enabled=state?.enabled===true,detail=state?.detail||'Select or reconnect to a saved workflow first.',complete=state?.complete===true;
    if(expired&&action.id==='readiness'&&complete){complete=false;detail='Readiness expired. Run step 5-3 again before migration (valid for five minutes).';}
    if(expired&&action.id==='start'&&enabled){enabled=false;detail='Readiness expired. Run step 5-3 again before migration.';}
    if(action.id==='start'&&enabled&&Date.now()>=Date.parse(summary.costDeadline)){enabled=false;detail='Cost authorization expired. Review optional step 5-5.';}
    button.disabled=busy||!enabled;button.className='secondary';
    executionText('execute-'+action.id+'-status',detail);
    if(action.id==='start')executionText('execute-start','5-4. Start new '+(record?.input.source.type||'source')+' migration');
    if(action.group==='migrate'&&!complete&&!next)next={action,enabled,detail};
  }
  const migration=summary?.migration;
  executionText('migrationSummary',migration?'Migration: '+migration+'. '+(['finished','failed','interrupted'].includes(migration)?'Review section 6 if a report is available; use optional recovery for a failed / interrupted job.':'Status refresh does not start another job. Optional step 5-6 is the manual fallback.'):next?'Next: '+next.action.step+'. '+next.detail:'Complete the private target in step 4.');
  executionText('verificationSummary',summary?.verification?'Counts verification: '+summary.verification+'. Full-property P1 qualification is separate.':summary?.actions.verify?.detail||'Wait for the migration terminal report.');
  if(!migration&&next?.enabled)el('execute-'+next.action.id).className='';
  if(summary?.actions.verify?.enabled)el('execute-verify').className='';
}
const send=(action,extra={})=>{beginFeedback(action);setBusy(true);api.postMessage({action,...(record?{workflow:record.id}:{}),...extra});};
function options(id,values,placeholder){el(id).replaceChildren();if(placeholder){const o=document.createElement('option');o.value='';o.textContent=placeholder;el(id).append(o);}for(const v of values){const o=document.createElement('option');o.value=v.value;o.textContent=v.label;el(id).append(o);}}
function update(){el('deploy').disabled=busy||!record||record.phase!=='previewed'||!el('networkApproved').checked||!el('costApproved').checked;el('refreshPlacement').disabled=busy||!el('runnerSubscription').value;el('preview').disabled=busy||!el('runnerGroup').value||!el('region').value||!el('zone').value;el('guestReady').disabled=busy||!record||record.phase!=='provisioned'||['submitted','unknown'].includes(record.guestCommand?.phase);el('guestRefresh').disabled=busy||!record?.guestCommand;el('configureSource').disabled=busy||!record&&(!el('runnerGroup').value||!el('region').value||!el('zone').value);el('reviewTarget').disabled=busy||!record||record.phase!=='provisioned'||!['csv','neo4j','postgresql','cosmos-nosql'].includes(record.input.source.type);el('refresh').disabled=busy||!record;updateExecution();}
function setBusy(value){busy=value;for(const n of document.querySelectorAll('button,input,select'))n.disabled=busy;update();}
const baseUpdate=update;update=()=>{baseUpdate();el('selectSubnet').disabled=busy||!el('runnerSubscription').value||!el('region').value;el('stopWatch').disabled=false;el('reviewTarget').disabled=busy||feedbackWatching||!record||!(record.sourceReport?.canReviewTarget||record.targetPhase);};
function choose(id,value){if([...el(id).options].some(option=>option.value===value)){el(id).value=value;return true;}el(id).value='';return false;}
function clearZone(){el('zone').value='';el('zoneNote').textContent='Source zone is unknown or placement changed. Review and select a runner zone; logical zone numbers are subscription-specific.';}
function reviewRegion(){el('regionNote').textContent='Source or placement changed. Review the runner region; a retained selection does not prove source co-location. Current placement checks still apply.';}
function resetPlacement(){options('runnerGroup',[],'Select runner subscription first');options('region',[],'Select runner subscription first');el('subnet').value='';reviewRegion();clearZone();invalidate();}
function resetSource(){draftID=null;el('sourceId').value='';candidates=[];options('candidate',[],'Discover candidates first');reviewRegion();clearZone();}
function suggestSourceRegion(){const c=candidates.find(v=>v.id===el('sourceId').value);if(!c)return;if(c.type.toLowerCase()==='microsoft.documentdb/databaseaccounts'){el('regionNote').textContent='Select an actual Cosmos data region. The account metadata location is not used as a default.';return;}const found=choose('region',c.region);el('regionNote').textContent=found?'Source region selected. VM SKU, zone, subnet and quota checks still apply.':'The source region is not in the selected subscription list. Review subscription and region availability.';}
function invalidate(){record=null;el('review').hidden=true;el('guestStatus').textContent='Guest readiness has not been checked for this selection.';el('networkApproved').checked=false;el('costApproved').checked=false;update();}
function source(){const type=el('type').value;options('location',(type==='csv'?['local']:type==='cosmos-nosql'?['azure']:['azure','on-premises','other-cloud']).map(v=>({value:v,label:v})));locationChanged();}
function locationChanged(){el('azureSource').hidden=el('location').value!=='azure';el('externalSource').hidden=!['on-premises','other-cloud'].includes(el('location').value);el('csvSource').hidden=el('type').value!=='csv';resetSource();invalidate();}
for(const id of ['restore','csv','refresh','guestReady','guestRefresh'])el(id).addEventListener('click',()=>send(id));
el('stopWatch').addEventListener('click',()=>api.postMessage({action:'stopWatch'}));
el('selectSubnet').addEventListener('click',()=>send('selectSubnet',{subscription:el('runnerSubscription').value,region:el('region').value,sourceId:el('sourceId').value}));
function runnerInput(){return {subscriptionId:el('runnerSubscription').value,resourceGroup:el('runnerGroup').value,region:el('region').value,zone:el('zone').value,size:el('size').value,subnetId:el('subnet').value,source:{type:el('type').value,location:el('location').value,...(el('location').value==='azure'?{resourceId:el('sourceId').value}:{})}};}
el('configureSource').addEventListener('click',()=>send('configureSource',{workflow:record?.id,input:runnerInput()}));
el('reviewTarget').addEventListener('click',()=>send('reviewTarget'));
for(const action of executionActions)el('execute-'+action.id).addEventListener('click',()=>send('executionAction',{step:action.id}));
setInterval(updateExecution,1000);
el('accounts').addEventListener('click',()=>{resetSource();resetPlacement();options('sourceGroup',[],'Select subscription first');options('subscription',[],'Refresh Azure account');options('runnerSubscription',[],'Refresh Azure account');send('accounts');});
el('type').addEventListener('change',source);el('location').addEventListener('change',locationChanged);
el('subscription').addEventListener('change',()=>{resetSource();resetPlacement();el('runnerSubscription').value=el('subscription').value;options('sourceGroup',[],'Select subscription first');if(el('subscription').value)send('placementOptions',{scope:'both',subscription:el('subscription').value});});
el('runnerSubscription').addEventListener('change',()=>{resetPlacement();if(el('runnerSubscription').value)send('placementOptions',{scope:'runner',subscription:el('runnerSubscription').value});});
el('refreshPlacement').addEventListener('click',()=>{invalidate();send('placementOptions',{scope:'runner',subscription:el('runnerSubscription').value});});
el('sourceGroup').addEventListener('change',()=>{if(el('runnerSubscription').value===el('subscription').value&&!el('runnerGroup').value)choose('runnerGroup',el('sourceGroup').value);resetSource();});
el('discover').addEventListener('click',()=>send('sources',{subscription:el('subscription').value,group:el('sourceGroup').value,type:el('type').value}));
el('candidate').addEventListener('change',()=>{clearZone();const c=candidates.find(v=>v.id===el('candidate').value);if(c){el('sourceId').value=c.id;suggestSourceRegion();if(c.type.toLowerCase()!=='microsoft.documentdb/databaseaccounts'&&el('subscription').value===el('runnerSubscription').value&&c.region===el('region').value&&['1','2','3'].includes(c.zone)){el('zone').value=c.zone;el('zoneNote').textContent='Known source logical zone selected in the same subscription and region. Current placement is rechecked before deployment.';}}else{el('sourceId').value='';reviewRegion();}});
el('region').addEventListener('change',()=>{reviewRegion();if(el('region').value)el('regionNote').textContent='Runner region explicitly selected. Source co-location, subnet, SKU availability and quota still require preflight checks.';clearZone();});
el('sourceId').addEventListener('change',()=>{el('candidate').value='';reviewRegion();clearZone();});
el('zone').addEventListener('change',()=>{el('zoneNote').textContent=el('zone').value?'Runner zone explicitly selected. Source co-location, SKU availability and quota still require preflight checks.':'Review and select a runner zone before continuing.';});
for(const node of document.querySelectorAll('input,select'))if(!['networkApproved','costApproved'].includes(node.id)){node.addEventListener('change',()=>api.postMessage({action:'selectionChanged'}),true);node.addEventListener('change',invalidate);}
el('networkApproved').addEventListener('change',update);el('costApproved').addEventListener('change',update);
el('preview').addEventListener('click',()=>{invalidate();send('preview',{draftId:draftID||undefined,input:runnerInput()});});
el('deploy').addEventListener('click',()=>send('deploy',{hash:record?.previewHash,networkApproved:el('networkApproved').checked,costApproved:el('costApproved').checked}));
window.addEventListener('message',event=>{const m=event.data;feedbackMessage(m);if(m.kind==='busy')setBusy(m.value);if(m.kind==='record')draftID=['draft','previewed'].includes(m.record.phase)?m.record.id:null;
if(m.kind==='progress'||m.kind==='error')update();
if(m.kind==='subnet'&&m.subscription===el('runnerSubscription').value&&m.region===el('region').value&&m.sourceId===el('sourceId').value){el('subnet').value=m.subnet.id;el('subnetSummary').textContent=m.subnet.vnet+' / '+m.subnet.name+' — '+m.subnet.prefixes.join(', ')+(m.subnet.containsSource?' — contains the source VM; review source-subnet firewall rules.':' — review runner-to-source connectivity.');invalidate();}
if(m.kind==='restoreInput')el('subnetSummary').textContent='Saved subnet: '+m.input.subnetId;
if(m.kind==='record'){el('assessmentSummary').textContent=m.record.sourceReport?m.record.sourceReport.title+'. '+m.record.sourceReport.detail:m.record.assessment?'Source operation: '+m.record.assessment.phase+'. Open source assessment for status and report.':'Next: configure the connection and approve a complete inventory after Linux readiness.';el('targetSummary').textContent=m.record.targetMessage||(m.record.targetPhase?'Target: '+m.record.targetPhase:m.record.sourceReport?.canReviewTarget?'Complete inventory verified. Next: review target and migration sizing.':'Requires a passing, hash-verified complete inventory.');}
if(m.kind==='error')el('error').textContent=m.text;
if(m.kind==='record')el('approvalControls').hidden=m.record.phase!=='previewed';
if(m.kind==='restoreInput'){const i=m.input;el('type').value=i.source.type;source();el('location').value=i.source.location;locationChanged();for(const [id,value] of [['runnerSubscription',i.subscriptionId],['runnerGroup',i.resourceGroup],['region',i.region]]){if(!choose(id,value))options(id,[{value,label:value+' (saved — rechecked before deployment)'}]);}el('zone').value=i.zone;el('zoneNote').textContent='Saved runner zone restored, not newly inferred from the source. Placement is rechecked before deployment.';el('size').value=i.size;el('subnet').value=i.subnetId;el('sourceId').value=i.source.resourceId||'';el('csvFiles').textContent=m.files.join('\\n');el('regionNote').textContent='Saved placement restored. Current Azure resources, region, zone and quota are rechecked before deployment.';}
if(m.kind==='subscriptions'){const opts=m.values.map(s=>({value:s.id,label:s.name+' — '+s.accountLabel}));options('subscription',opts,'Select source subscription');options('runnerSubscription',opts,'Select runner subscription');el('error').textContent='';}
if(m.kind==='groups'&&m.subscription===el('subscription').value)options('sourceGroup',m.values.map(g=>({value:g.name,label:g.name})),'Select source resource group');
if(m.kind==='placementOptions'){if(m.scope==='both'&&m.subscription===el('subscription').value){const previous=el('sourceGroup').value;options('sourceGroup',m.catalog.groups.map(g=>({value:g.name,label:g.name})),'Select source resource group');choose('sourceGroup',previous);}if(m.subscription===el('runnerSubscription').value){const group=el('runnerGroup').value,region=el('region').value;options('runnerGroup',m.catalog.groups.map(g=>({value:g.name,label:g.name})),m.catalog.groups.length?'Select migration resource group':'No existing groups — create one in Azure, then refresh');options('region',m.catalog.regions.map(r=>({value:r.name,label:r.displayName+' ('+r.name+')'})),m.catalog.regions.length?'Select Azure region':'No Azure regions returned');choose('runnerGroup',group);if(!choose('region',region))suggestSourceRegion();el('error').textContent='';update();}}
if(m.kind==='sources'&&m.subscription===el('subscription').value&&m.group===el('sourceGroup').value&&m.type===el('type').value){candidates=m.values;options('candidate',m.values.map(r=>({value:r.id,label:r.name+' ('+r.type+')'})),'Select a candidate');}
if(m.kind==='csv')el('csvFiles').textContent=m.files.join('\\n');
if(m.kind==='record'){const c=m.record.guestCommand;el('guestStatus').textContent=c?.phase==='bootstrap-pending'?'Linux bootstrap is still running. Wait, then explicitly check Linux guest readiness again; refreshing this receipt does not retry it.':c?.phase==='failed'?(c.action==='ready'?'Linux readiness check failed.':(c.action||'Guest')+' command failed.')+' Review retained evidence; nothing was resubmitted.':c&&['submitted','unknown'].includes(c.phase)?c.action+' command '+c.phase+' — awaiting its retained receipt.':m.record.guestReady?'Pinned Linux guest verified at '+m.record.guestReady.checkedAt+'. Assessment outcome is shown separately.':'Guest readiness has not been checked.';}
if(m.kind==='record'){record=m.record;el('review').hidden=false;const labels={draft:'Local source draft — no Azure resources created',previewed:'Awaiting your approval', 'deployment-submitted':'Azure deployment submitted — refresh for progress',provisioned:'VM provisioned — check guest readiness before source reads',failed:'Azure deployment failed — resources/evidence retained',unknown:'Deployment status unknown — refresh, do not resubmit'};el('reviewSummary').textContent=[labels[record.phase],record.input.source.type+' → Linux discovery/migration VM',record.input.region+' / zone '+record.input.zone+' / '+record.input.size,'Resource group: '+record.input.resourceGroup,record.phase==='draft'?'Compute price and installation have not been checked.':'Compute estimate: USD '+record.hourlyComputeUSD+'/hour (storage and network additional)','CLI: '+(record.version||'Not selected yet'),'Updated: '+record.updatedAt].join('\\n');el('record').textContent=JSON.stringify(record,null,2);el('networkApproved').checked=false;el('costApproved').checked=false;el('error').textContent='';update();}});
source();send('ready');
</script></body></html>`;
}
