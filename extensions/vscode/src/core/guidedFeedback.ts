/** Shared, theme-aware feedback for the two long-running migration workbenches. */
export const guidedFeedbackCSS = `
html{scroll-padding-bottom:min(42vh,264px)}
body{padding-bottom:min(42vh,264px)!important}
.workflow-feedback{position:fixed;bottom:0;left:0;right:0;z-index:20;max-height:40vh;overflow:auto;box-sizing:border-box;padding:12px 24px;background:var(--vscode-editor-background);border-top:2px solid var(--vscode-focusBorder);box-shadow:0 -2px 8px var(--vscode-widget-shadow)}
.workflow-feedback p,.workflow-feedback div{margin:4px auto;max-width:1000px;overflow-wrap:anywhere}
.workflow-feedback progress{display:block;width:min(100%,1000px);height:4px;margin:6px auto;accent-color:var(--vscode-progressBar-background)}
.workflow-feedback [hidden]{display:none!important}
.inline-error{color:var(--vscode-errorForeground);border-left:3px solid currentColor;padding:8px 12px;white-space:pre-wrap;overflow-wrap:anywhere}
.next-action{border-left:3px solid var(--vscode-focusBorder);padding:10px 14px}
button.secondary{background:var(--vscode-button-secondaryBackground);color:var(--vscode-button-secondaryForeground)}
button:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible,a:focus-visible{outline:2px solid var(--vscode-focusBorder);outline-offset:3px}
details{margin:12px 0}summary{cursor:pointer}section{scroll-margin-top:16px}
p,label,summary,h1,h2,h3{overflow-wrap:anywhere}.grid>*{min-width:0}
@media(max-width:600px){body{padding-left:12px!important;padding-right:12px!important}.workflow-feedback{padding:8px 12px}.grid{grid-template-columns:minmax(0,1fr)!important}button{max-width:100%;white-space:normal}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;animation:none!important;transition:none!important}}
`;

export const guidedFeedbackHTML = `<aside class="workflow-feedback" aria-label="Current operation">
<p id="activity" role="status" aria-live="polite" aria-atomic="true">Ready. Follow the steps above.</p>
<progress id="operationProgress" aria-label="Operation in progress" hidden></progress>
<div id="error" role="alert" aria-atomic="true"></div>
</aside>`;

export const guidedFeedbackScript = `
let feedbackAction='',feedbackOrigin=null,feedbackInline=null,feedbackWatching=false;
function beginFeedback(action){
  feedbackAction=action;
  const focused=document.activeElement;
  feedbackOrigin=focused&&focused!==document.body?focused:null;
  if(feedbackInline){feedbackInline.remove();feedbackInline=null;}
  document.getElementById('error').textContent='';
  document.getElementById('activity').textContent=(feedbackOrigin?.tagName==='BUTTON'?feedbackOrigin.textContent.trim():action)+' — working…';
  document.getElementById('operationProgress').hidden=false;
}
function feedbackMessage(m){
  const activity=document.getElementById('activity'),progress=document.getElementById('operationProgress');
  if(m.kind==='error'){
    document.getElementById('error').textContent=m.text;
    progress.hidden=true;feedbackWatching=false;
    activity.textContent='Action needs attention. No automatic replay was attempted.';
    if(feedbackInline){feedbackInline.remove();feedbackInline=null;}
    const section=feedbackOrigin?.closest('section');
    if(section){feedbackInline=document.createElement('p');feedbackInline.className='inline-error';feedbackInline.textContent=m.text;section.append(feedbackInline);}
  }
  if(m.kind==='progress'){
    feedbackWatching=m.active!==false;activity.textContent=m.text;progress.hidden=!feedbackWatching;
  }
  if(m.kind==='busy'){
    progress.hidden=!m.value&&!feedbackWatching;
    if(!m.value&&!feedbackWatching&&!document.getElementById('error').textContent&&activity.textContent.endsWith(' — working…'))activity.textContent='Action finished. Review the current step status.';
    if(!m.value&&feedbackOrigin&&document.activeElement===document.body)queueMicrotask(()=>{if(!feedbackOrigin.disabled&&document.activeElement===document.body)feedbackOrigin.focus();});
  }
}
`;
