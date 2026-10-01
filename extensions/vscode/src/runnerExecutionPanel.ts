import * as vscode from "vscode";
import {createHash} from "node:crypto";
import {readFile} from "node:fs/promises";
import {RunnerControl} from "./core/runnerLifecycle";
import {RunnerStore} from "./guided/runnerStore";
import {AzureSession} from "./guided/azure";
import {startResize,advanceResize} from "./core/runnerResize";
import {migrationPreflight,startMigration,refreshMigration,verifyMigrationReport,applyTargetPreload,checkMigrationReadiness} from "./core/runnerExecution";
import {escapeHTML} from "./core/report";
import {targetComputeRate} from "./core/runnerTargetPreflight";
import {renewTargetAuthorization} from "./core/runnerTarget";
import {diagnoseTarget,archiveEmptyTargetFailure,needsTargetDiagnosis} from "./core/runnerDiagnostic";
import {qualifyP1} from "./p1QualificationPanel";
import {diagnoseP1} from "./p1DiagnosticPanel";
import {inspectSourceCA} from "./core/runnerSource";
import {ensureAssessmentReadiness} from "./core/runnerAssessment";
import {inspectResume,recoveryReadiness,resumeAdmission,resumeMigration} from "./core/runnerResume";
import {showMigrationVerification} from "./migrationVerificationPanel";
import { sourceCredential, invalidateStaleSourceCredential } from "./sourceCredentialPanel";
import { watchRetainedOperation } from "./runnerWatch";
import { transferApprovedReport } from "./runnerReportFlow";
import { authorizeResize, resizeAuthorized } from "./core/resizeAuthorization";
import { boundedWatch } from "./core/boundedWatch";
import { pickTargetDeadline } from "./runnerTargetInputs";
import { executionActions, ExecutionAction, executionActionLabel, executionActionState, parseExecutionAction, resizeStatusMessage } from "./core/runnerExecutionActions";
import { reconcileGuest } from "./core/runnerGuest";
import { RunnerRecord } from "./core/runner";

export interface ExecutionProgress { text: string; active: boolean }
class ExecutionMonitoringStopped extends Error {}

/** A bounded resize may group its explicitly approved steps. Reconnecting alone
 * never launches/resumes a migration or replays an uncertain operation. */
export async function continueRunnerExecution(context:vscode.ExtensionContext,control:RunnerControl,store:RunnerStore,azure:AzureSession,workflow?:string,requestedAction?:ExecutionAction,onUpdate?:(record:RunnerRecord,progress?:ExecutionProgress)=>Promise<void>,cancelled:()=>boolean=()=>false):Promise<void>{
  if(!vscode.workspace.isTrusted)throw new Error("Trust this workspace before controlling migration resources.");
  if(requestedAction!==undefined)parseExecutionAction(requestedAction);
  const assertActive=()=>{if(cancelled()||!vscode.workspace.isTrusted)throw new Error("Execution review stopped or workspace trust changed. Reconcile any retained operation; nothing will be replayed.");};
  const originalControl=control;
  control={...control,request:(...args)=>{assertActive();return originalControl.request(...args);}};
  const selected=workflow?{id:workflow}:await vscode.window.showQuickPick((await store.list()).filter(r=>r.target?.phase==="provisioned").map(r=>({label:r.id,description:`${r.input.source.type} — ${r.input.resourceGroup} — ${r.migration?.phase??r.resize?.phase??"resize required"}`,id:r.id})),{placeHolder:"Select the retained private target"});
  if(!selected)return;
  let r=await store.read(selected.id);
  const startLabel=`Start new ${r.input.source.type} migration and counts verification`;
  const renewLabel="Review a new cost authorization (no Azure mutation)";
  const resumeInspectionLabel="Inspect same-job recovery (read only; does not resume)";
  const resumeLabel="Explicitly resume the retained job and counts verification";
  const recoveryReadyLabel="Refresh recovery readiness (read only)";
  const label=(id:ExecutionAction)=>id==="start"?startLabel:id==="renew"?renewLabel:id==="verify"?"Transfer / open migration verification":executionActionLabel(id);
  const action=requestedAction?label(requestedAction):await vscode.window.showQuickPick(executionActions.map(item=>label(item.id)),{placeHolder:`Runner: ${r.resize?.phase??"not resized"}; migration: ${r.migration?.phase??"not started"}`});
  if(!action)return;
  assertActive();
  if(requestedAction){
    r=await store.read(r.id);
    const state=executionActionState(r,requestedAction);
    if(!state.enabled)throw new Error(state.detail);
  }
  const confirm=async(title:string,detail:string)=>{const result=await vscode.window.showWarningMessage(title,{modal:true,detail},"Approve this step");assertActive();return result;};
  const price=async()=>{if(!r.target)throw new Error("No retained target plan.");const input=r.target.input;if(targetComputeRate(await azure.retailRates(r.input.region,[input.loaderSize,input.postgresSKU]),input)!==input.hourlyUSD)throw new Error("Compute price changed; review the cost plan before further mutation.");};
  if(action===renewLabel){
    if(!r.target)throw new Error("No retained target plan.");
    const ask=(prompt:string,value:string)=>vscode.window.showInputBox({prompt,value,ignoreFocusOut:true});
    const deadline=await pickTargetDeadline();if(deadline===undefined)return;
    const budget=await ask("Approved cumulative workflow cost ceiling, USD (not an additional allowance)",String(r.target.input.budgetUSD));if(budget===undefined)return;
    const reserve=await ask("Reserve within this ceiling for accrued charges, retained storage/network and delayed billing, USD",String(r.target.input.additionalReserveUSD));if(reserve===undefined)return;
    const candidate={deadline,budgetUSD:Number(budget),additionalReserveUSD:Number(reserve),hourlyUSD:targetComputeRate(await azure.retailRates(r.input.region,[r.target.input.loaderSize,r.target.input.postgresSKU]),r.target.input)};
    renewTargetAuthorization(r,candidate);
    if(await confirm("Record this new cost authorization?",`No Azure operation is performed by this step. Previous deadline and ceiling remain in local audit history. Existing migration and verification evidence are preserved; this does not replay or resume a job.\nNew deadline ${candidate.deadline}; cumulative ceiling USD ${candidate.budgetUSD}; accrued/non-compute reserve USD ${candidate.additionalReserveUSD}; current compute USD ${candidate.hourlyUSD}/hour.`)!=="Approve this step")return;
    r=await store.exclusive(r.id,async()=>{const latest=await store.read(r.id);const next=renewTargetAuthorization(latest,candidate);await control.persist(next);return next;});
  }else if(action==="Apply / reconcile AGE preload restart"){
    const approved=!!r.targetRestart || await confirm("Apply the AGE preload configuration?",`Restart only ${r.target?.serverId} if its approved preload parameter requires it. This is before migration. Existing data is retained. An already submitted restart is reconciled by read only.`)==="Approve this step";
    const initial=r;
    let receipt=r.targetRestart?.submittedAt;
    await vscode.window.withProgress({location:vscode.ProgressLocation.Notification,title:"5-1. AGE preload / PostgreSQL restart — cancel stops monitoring only",cancellable:true},async(progress,token)=>{
      const stopped=()=>token.isCancellationRequested||cancelled()||!vscode.workspace.isTrusted;
      const assertMonitoring=()=>{if(stopped())throw new ExecutionMonitoringStopped("AGE preload monitoring stopped.");};
      const publish=async(current:RunnerRecord,text:string,active:boolean)=>{
        progress.report({message:text});
        await onUpdate?.(current,{text,active});
      };
      const status=(current:RunnerRecord)=>executionActionState(current,"preload").detail;
      const preloadControl:RunnerControl={...control,
        request:(...args)=>{assertMonitoring();return control.request(...args);},
        persist:async current=>{
          if(!receipt && current.targetRestart?.phase==="submitted")assertMonitoring();
          await control.persist(current);
          receipt??=current.targetRestart?.submittedAt;
          if(!stopped())await publish(current,status(current),current.targetRestart?.phase!=="finished");
        }
      };
      const reconcile=(allowRestart:boolean)=>store.exclusive(initial.id,async()=>{
        assertMonitoring();
        const current=await store.read(initial.id);
        assertMonitoring();
        if(current.vmId!==initial.vmId || current.artifact.sha256!==initial.artifact.sha256 ||
          current.target?.serverId!==initial.target?.serverId || current.target?.hash!==initial.target?.hash ||
          JSON.stringify(current.target?.input)!==JSON.stringify(initial.target?.input) ||
          receipt!==undefined && current.targetRestart?.submittedAt!==receipt)
          throw new Error("Preload restart scope or retained receipt changed; monitoring stopped without replay.");
        receipt??=current.targetRestart?.submittedAt;
        return applyTargetPreload(preloadControl,current,allowRestart);
      });
      try{
        assertMonitoring();
        await publish(r,r.targetRestart?status(r):"Checking the private PostgreSQL target and approved AGE preload setting...",true);
        r=await reconcile(approved);
        if(r.targetRestart && r.targetRestart.phase!=="finished"){
          await publish(r,status(r),!stopped());
          await boundedWatch({deadline:Date.now()+30*60000,intervalMs:15000,maxSteps:120,sleep:control.sleep,cancelled:stopped,
            step:()=>reconcile(false),done:current=>current.targetRestart?.phase==="finished",
            progress:async current=>{r=current;if(!stopped())await publish(current,status(current),current.targetRestart?.phase!=="finished");}});
        }
      }catch(error){if(!(error instanceof ExecutionMonitoringStopped))throw error;}
      r=await store.read(initial.id);
      const text=r.targetRestart?.phase==="finished"?status(r)+" Continue with the next required migration step."
        :r.targetRestart?status(r)+` Monitoring ${stopped()?"stopped":"reached its time limit"}; Azure work was not cancelled. Use 5-1 to resume read-only monitoring.`
        :"AGE preload preparation stopped. No restart was submitted.";
      await publish(r,text,false);
    });
  }else if(action==="Check Linux guest readiness again"){
    r=await vscode.window.withProgress({location:vscode.ProgressLocation.Notification,title:"5-3. Checking Linux readiness after resize",cancellable:true},
      async(_p,token)=>store.exclusive(r.id,async()=>checkMigrationReadiness(control,await store.read(r.id),()=>cancelled()||token.isCancellationRequested||!vscode.workspace.isTrusted)));
  }else if(action==="Refresh pending guest command"){
    r=await store.exclusive(r.id,async()=>(await reconcileGuest(control,await store.read(r.id))).record);
  }else if(action==="Reconcile resize (read only)"){
    r=await store.exclusive(r.id,async()=>advanceResize(control,await store.read(r.id)));
  }else if(action==="Approve / continue same-VM resize sequence"){
    if(!r.target)throw new Error("Review the private target first.");
    if(!resizeAuthorized(r) && await confirm("Complete the bounded same-VM resize sequence?",`${r.vmId}\n${r.resize?.phase??"Deallocate before resizing"} → ${r.target.input.loaderSize}. This one approval covers deallocation, size change and restart of this exact idle VM, for up to 20 minutes or the earlier target deadline. The same NIC, disk and identity are preserved. No source VM, credentials or migration are changed. Deadline ${r.target.input.deadline}; cumulative ceiling USD ${r.target.input.budgetUSD}. Uncertain submissions are not replayed. Cancel stops the sequence monitor; reopen to reconcile retained state.`)!=="Approve this step")return;
    const reviewed=r;
    await vscode.window.withProgress({location:vscode.ProgressLocation.Notification,title:"5-2. Working on approved same-VM resize",cancellable:true},async(progress,token)=>{
      const stopped=()=>token.isCancellationRequested||cancelled()||!vscode.workspace.isTrusted;
      const assertMonitoring=()=>{if(stopped())throw new ExecutionMonitoringStopped("Resize sequence monitoring stopped.");};
      const publish=async(current:RunnerRecord,text:string,active:boolean)=>{
        progress.report({message:text});await onUpdate?.(current,{text,active});
      };
      const report=async(current:RunnerRecord)=>{
        const active=current.resize?.phase!=="finished" && !current.resize?.unknown;
        await publish(current,(active?"Working... ":"")+resizeStatusMessage(current),active);
      };
      try{
        assertMonitoring();
        await publish(r,"Working... Checking current prices for the approved resize...",true);
        await price();assertMonitoring();
        r=await store.exclusive(reviewed.id,async()=>{
          assertMonitoring();
          const latest=await store.read(reviewed.id);
          assertMonitoring();
          if(latest.vmId!==reviewed.vmId || JSON.stringify(latest.input)!==JSON.stringify(reviewed.input) ||
            JSON.stringify(latest.artifact)!==JSON.stringify(reviewed.artifact) || latest.target?.serverId!==reviewed.target?.serverId ||
            latest.target?.hash!==reviewed.target?.hash || JSON.stringify(latest.target?.input)!==JSON.stringify(reviewed.target?.input))
            throw new Error("Resize scope changed during review.");
          const next=resizeAuthorized(latest)?latest:authorizeResize(latest);await control.persist(next);return next;
        });
        const binding=r.resizeAuthorization!.binding;
        const assertAuthorization=()=>{
          assertMonitoring();
          if(!resizeAuthorized(r) || r.resizeAuthorization!.binding!==binding)throw new Error("Resize authorization expired or changed; no further operation was submitted.");
        };
        const resizeControl:RunnerControl={...control,
          request:(...args)=>{assertAuthorization();return control.request(...args);},
          list:(...args)=>{assertAuthorization();return control.list(...args);},
          persist:async current=>{
            if(current.resize && ["deallocating","resizing","starting"].includes(current.resize.phase) &&
              (current.resize.phase!==r.resize?.phase || current.resize.startedAt!==r.resize?.startedAt))assertAuthorization();
            await control.persist(current);r=current;if(!stopped())await report(current);
          }
        };
        await report(r);
        await boundedWatch({deadline:Date.parse(r.resizeAuthorization!.deadline),intervalMs:15000,maxSteps:80,sleep:control.sleep,cancelled:stopped,
          step:()=>store.exclusive(reviewed.id,async()=>{
            assertMonitoring();r=await store.read(reviewed.id);assertAuthorization();
            if(r.resize)return advanceResize(resizeControl,r,true);
            await publish(r,"Working... Checking Linux guest readiness before resizing...",true);
            r=await ensureAssessmentReadiness(resizeControl,r,stopped);
            assertAuthorization();
            await publish(r,"Working... Checking runner placement, size availability and quota...",true);
            return startResize(resizeControl,r);
          }),
          progress:async current=>{r=current;if(!stopped())await report(current);},
          done:current=>current.resize?.phase==="finished"||current.resize?.unknown===true});
      }catch(error){if(!(error instanceof ExecutionMonitoringStopped))throw error;}
      r=await store.read(reviewed.id);
      const text=r.resize?.phase==="finished" || r.resize?.unknown ? resizeStatusMessage(r)
        :`${r.resize?resizeStatusMessage(r):"No VM resize was submitted."} Sequence monitoring ${stopped()?"stopped":"reached its authorization or time limit"}. Submitted Azure work was not cancelled. Use 5-2 to review and continue; completed operations will not be replayed.`;
      await publish(r,text,false);
    });
  }else if(action===startLabel||action===resumeLabel){
    const a=r.assessment;if(!a?.reportSHA256 || !a.reportBytes)throw new Error("Import complete source inventory first.");
    const report=await store.readReport(r.id,{operation:a.operation,sha256:a.reportSHA256,bytes:a.reportBytes});
    const resuming=action===resumeLabel;
    const evidence=resuming?(resumeAdmission(r),r.migration!.evidence):await migrationPreflight(control,r,report);
    const detail=resuming?`Resume job ${r.migration!.jobId}, generation ${r.resumeInspection!.generationId}, fingerprint ${r.resumeInspection!.fingerprint}, from ${r.resumeInspection!.committedRows} rows. Preserve the existing graph, configuration, previous operation and evidence. Create only a new continuation operation. The guest must prove the old service inactive before replacing its retained lease. No AGE preparation, replacement graph, automatic retry or new job. After resume, check the same generation and complete counts. Full P1 canonical verification remains separate.`:`${evidence.rows} mapped rows. Inventory ${evidence.reportSHA256}. Prepare AGE, create the new graph, migrate, then run complete counts verification. The new job UUID is retained before writes. No replace, delete, automatic resume or retry. A counts pass is distinct from the independent P1 property digest.`;
    if(await confirm(resuming?"Explicitly resume this retained Linux migration?":`Start this new ${r.input.source.type} migration on the Linux runner?`,`${detail}\nLinux ${r.artifact.version}, archive ${r.artifact.sha256}.\n${r.target!.serverId}\nSource and target credentials use protected transport; only the target secret is retained in SecretStorage.`)!=="Approve this step")return;
    await price();
    let sourcePassword:string|undefined;
    if(r.input.source.type==="neo4j"||r.input.source.type==="postgresql"){
      sourcePassword=await sourceCredential(context,r,r.sourceDraft!.form);
      if(sourcePassword===undefined)return;
    }
    try { r=await store.exclusive(r.id,async()=>{
      assertActive();
      let latest=await store.read(r.id);
      const key=`runner-target/${r.id}/${createHash("sha256").update(latest.target!.serverId).digest("hex")}`,password=await context.secrets.get(key);
      if(!password)throw new Error("The retained target credential is unavailable.");
      let sourceCAPEM:string|undefined;
      if(latest.sourceCA){const data=await readFile(latest.sourceCA.path),checked=inspectSourceCA(latest.sourceCA.path,latest.sourceCA.name,data);if(checked.bytes!==latest.sourceCA.bytes||checked.sha256!==latest.sourceCA.sha256||latest.sourceDraft?.sourceCASHA256!==checked.sha256)throw new Error("The selected source CA changed; select and review it again.");sourceCAPEM=data.toString("utf8");}
      if(resuming)return resumeMigration(control,latest,password,sourcePassword,sourceCAPEM);
      latest=await vscode.window.withProgress({location:vscode.ProgressLocation.Notification,title:"Checking Linux readiness before migration",cancellable:false},
        ()=>ensureAssessmentReadiness(control,latest));
      return startMigration(control,latest,report,password,sourcePassword,sourceCAPEM);
    }); } finally { sourcePassword=undefined; }
  }else if(action===recoveryReadyLabel){
    r=await store.exclusive(r.id,async()=>recoveryReadiness(control,await store.read(r.id)));
  }else if(action===resumeInspectionLabel){
    if(await confirm("Inspect the retained checkpoint without resuming?","Reads the existing guest configuration and target metadata only. No source connection, worker start, lease removal, graph creation or data changes. A successful inspection is not permission to resume. Requires a reviewed runner advertising resume-inspection-v1 and fresh readiness.")!=="Approve this step")return;
    const checked=await store.exclusive(r.id,async()=>{
      const latest=await store.read(r.id);
      if(!latest.target)throw new Error("No retained target.");
      const key=`runner-target/${r.id}/${createHash("sha256").update(latest.target.serverId).digest("hex")}`;
      return inspectResume(control,latest,await context.secrets.get(key));
    });
    r=checked.record;
    if(checked.inspection){
      const view=vscode.window.createWebviewPanel("agefreighter.resumeInspection","AGEFreighter recovery inspection",vscode.ViewColumn.Beside,{enableScripts:false,localResourceRoots:[]});
      view.webview.html=`<!doctype html><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'"><h1>Recovery review required — no job resumed</h1><pre>${escapeHTML(JSON.stringify(checked.inspection,null,2))}</pre>`;
    }else void vscode.window.showInformationMessage("Inspection submitted or still pending. Select the same inspection action to reconcile; it never starts a migration.");
  }else if(action==="Refresh retained migration (never replay)"){
    r=await store.exclusive(r.id,async()=>refreshMigration(control,await store.read(r.id)));
  }else if(action==="Diagnose retained P1 failure (read only)"){
    r=await diagnoseP1(context,control,store,azure,r.id);
  }else if(action==="Qualify / reconcile full P1 digest (development only)"){
    r=await qualifyP1(context,control,store,azure,r.id);
  }else if(action==="Requalify with reviewed P1 ordering fix (read only)"){
    r=await qualifyP1(context,control,store,azure,r.id,true);
  }else if(action==="Archive empty-target preparation failure"){
    if(await confirm("Archive this preparation failure without deleting or resuming anything?",`Requires fresh read-only proof that the target graph and metadata schema are absent. Retains the failed job and diagnostic in history and all guest evidence. This permits a separately approved runner repair and a new create-only job, not replay or replacement of existing data.`)!=="Approve this step")return;
    r=await store.exclusive(r.id,async()=>archiveEmptyTargetFailure(control,await store.read(r.id)));
  }else if(action==="Diagnose retained target (read only)"){
    const newRead=needsTargetDiagnosis(r);
    if(newRead && await confirm("Read-only diagnosis of the retained target?",`Run the pinned Linux CLI doctor without --persist against the existing target. No load, resume, AGE preparation or resource changes. Prior expired diagnostics remain in history. Private credentials remain in SecretStorage/protected transport. Raw redacted evidence stays on the guest.`)!=="Approve this step")return;
    r=await store.exclusive(r.id,async()=>{
      const latest=await store.read(r.id);
      const key=`runner-target/${r.id}/${createHash("sha256").update(latest.target!.serverId).digest("hex")}`;
      if(needsTargetDiagnosis(latest)&&!newRead)throw new Error("Diagnostic expired; select diagnosis again to approve a fresh read.");
      return diagnoseTarget(control,latest,needsTargetDiagnosis(latest)?await context.secrets.get(key):undefined);
    });
    if(r.targetDiagnostic?.result){
      const view=vscode.window.createWebviewPanel("agefreighter.targetDiagnostic","AGEFreighter target diagnostic",vscode.ViewColumn.Beside,{enableScripts:false,localResourceRoots:[]});
      view.webview.html=`<!doctype html><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'"><h1>Read-only target diagnostic</h1><pre>${escapeHTML(JSON.stringify(r.targetDiagnostic.result,null,2))}</pre>`;
    }
  }else if(action==="Transfer / open migration verification"){
    const m=r.migration;if(!m?.reportSHA256 || !m.reportBytes)throw new Error("Refresh the terminal migration verification manifest first.");
    const manifest={operation:m.operation,sha256:m.reportSHA256,bytes:m.reportBytes};
    if(!r.reportTransfers?.some(t=>t.operation===m.operation) && await confirm("Transfer and verify the retained migration report?",`${m.jobId}\n${manifest.bytes} bytes; SHA-256 ${manifest.sha256}. Retained privately on this Mac, not sent to an AI model. This never reruns migration.`)!=="Approve this step")return;
    r=await transferApprovedReport(control,store,r.id,manifest,(...args)=>azure.reportCapability(...args),()=>!vscode.workspace.isTrusted);
    if(r.reportTransfers?.find(t=>t.operation===m.operation)?.phase!=="imported"){
      void vscode.window.showInformationMessage("Report transfer remains retained. Reopen to reconcile without another export or approval.");return;
    }
    r=await store.exclusive(r.id,async()=>{
      let latest=await store.read(r.id);
      if(latest.migration?.operation!==m.operation || latest.migration.reportSHA256!==manifest.sha256)throw new Error("Migration report changed during review.");
      const text=await store.readReport(latest.id,manifest),next=verifyMigrationReport(latest,text);await control.persist(next);return next;
    });
    if(r.migration?.verification){
      const text=await store.readReport(r.id,manifest);
      showMigrationVerification(r.migration.verification,text);
    }
  }else throw new Error("Unsupported migration or verification step.");
  await onUpdate?.(r);
  if(r.migration && !["finished","failed","interrupted"].includes(r.migration.phase)){
    await watchRetainedOperation(control,store,r.id,"migration",cancelled,async current=>{
      if(["failed","interrupted"].includes(current.migration?.phase??""))await invalidateStaleSourceCredential(context,await store.read(current.id));
      await onUpdate?.(current);
    });
    r=await store.read(r.id);
  }
  if(!requestedAction)void vscode.window.showInformationMessage(`Target preload: ${r.targetRestart?.phase??"not checked"}; runner resize: ${r.resize?.phase??"not started"}; migration: ${r.migration?.phase??"not started"}; counts: ${r.migration?.verification?.outcome??"not verified"}. Independent P1 property digest is a separate qualification gate.`);
}
