import { RunnerRecord, SourceKind } from "./runner";
import { assertIdleHealth } from "./runnerGuest";
import { assessmentActive } from "./runnerAssessment";

export const executionActions = [
  { id: "preload", step: "5-1", group: "migrate", label: "Apply / reconcile AGE preload restart" },
  { id: "resize", step: "5-2", group: "migrate", label: "Approve / continue same-VM resize sequence" },
  { id: "readiness", step: "5-3", group: "migrate", label: "Check Linux guest readiness again" },
  { id: "start", step: "5-4", group: "migrate", label: "Start new migration" },
  { id: "renew", step: "5-5", group: "migrationTools", label: "Review a new cost authorization" },
  { id: "migrationRefresh", step: "5-6", group: "migrationTools", label: "Refresh retained migration (never replay)" },
  { id: "resizeRefresh", step: "5-7", group: "migrationTools", label: "Reconcile resize (read only)" },
  { id: "guestRefresh", step: "5-8", group: "migrationTools", label: "Refresh pending guest command" },
  { id: "recoveryReady", step: "5-9", group: "recovery", label: "Refresh recovery readiness (read only)" },
  { id: "inspectResume", step: "5-10", group: "recovery", label: "Inspect same-job recovery (read only; does not resume)" },
  { id: "resume", step: "5-11", group: "recovery", label: "Explicitly resume the retained job and counts verification" },
  { id: "diagnoseTarget", step: "5-12", group: "recovery", label: "Diagnose retained target (read only)" },
  { id: "archiveFailure", step: "5-13", group: "recovery", label: "Archive empty-target preparation failure" },
  { id: "verify", step: "6-1", group: "verify", label: "Verify the migration" },
  { id: "qualifyP1", step: "6-2", group: "development", label: "Qualify / reconcile full P1 digest (development only)" },
  { id: "diagnoseP1", step: "6-3", group: "development", label: "Diagnose retained P1 failure (read only)" },
  { id: "requalifyP1", step: "6-4", group: "development", label: "Requalify with reviewed P1 ordering fix (read only)" }
] as const;

export type ExecutionAction = typeof executionActions[number]["id"];
export type ExecutionGroup = typeof executionActions[number]["group"];
export interface ExecutionActionState { enabled: boolean; detail: string; complete?: boolean }

export function parseExecutionAction(value: unknown): ExecutionAction {
  const action = executionActions.find(action => action.id === value);
  if (!action) throw new Error("Unsupported migration or verification step.");
  return action.id;
}

export function executionActionLabel(action: ExecutionAction, source?: SourceKind): string {
  return action === "start" ? `Start new ${source ?? ""} migration`.replace("  ", " ")
    : executionActions.find(item => item.id === action)!.label;
}

export function resizeStatusMessage(r: RunnerRecord): string {
  const resize = r.resize;
  if (!resize) return "Checking Linux guest readiness before resizing the runner...";
  if (resize.unknown) return `Resize ${resize.phase} response is uncertain. The automatic sequence is paused. Use 5-7 for read-only reconciliation; no request will be replayed.`;
  switch (resize.phase) {
    case "deallocating": return "Stopping and deallocating the runner VM...";
    case "ready-to-resize": return "Runner VM is deallocated and ready for the approved size change.";
    case "resizing": return `Changing the runner VM size to ${resize.size}...`;
    case "ready-to-start": return "Size change confirmed. The runner VM is ready to start.";
    case "starting": return "Starting the resized runner VM...";
    case "finished": return "Same-VM resize complete. Continue with 5-3. Check Linux guest readiness again.";
  }
  throw new Error("Unsupported retained resize phase.");
}

/** Presentation guidance only. Live ownership, budget, health and approval gates
 * remain authoritative in the execution controllers. No state here is persisted. */
export function executionActionState(r: RunnerRecord | undefined, action: ExecutionAction, now = Date.now()): ExecutionActionState {
  const blocked = (detail: string): ExecutionActionState => ({ enabled: false, detail });
  const available = (detail: string, complete = false): ExecutionActionState => ({ enabled: true, detail, complete });
  if (!r) return blocked("Select or reconnect to a saved workflow first.");
  const m = r.migration, pending = r.guestCommand && ["submitted", "unknown"].includes(r.guestCommand.phase);
  const stopped = !!m && ["failed", "interrupted"].includes(m.phase);
  if (action === "renew") return r.target ? available("Optional: review a new deadline / cumulative ceiling. No Azure mutation or automatic extension.") : blocked("Review a target cost plan in step 4 first.");
  if (action === "guestRefresh") return r.guestCommand ? available("Reconcile the retained receipt only; never resubmit the command.") : blocked("No retained guest command.");
  if (action === "migrationRefresh") return m ? available(`Migration ${m.phase}. Refresh status without starting another job.`) : blocked("Available after step 5-4 submits a migration.");
  if (action === "verify") {
    if (!m) return blocked("Start the migration in step 5-4 first.");
    if (!["finished", "failed", "interrupted"].includes(m.phase)) return blocked(`Migration ${m.phase}. Wait for the terminal report; use step 5-6 to refresh.`);
    if (!m.reportSHA256 || !m.reportBytes) return blocked("No sealed verification report yet. Refresh in step 5-6; a failed job may have no report.");
    return available(m.verification ? `Counts: ${m.verification.outcome}. Open the retained hash-verified report; never rerun migration.`
      : "Transfer the retained report, check its SHA-256 and complete counts, then display the result. No load or verification worker is rerun.", m.verification?.outcome === "pass");
  }
  if (r.phase !== "provisioned" || r.target?.phase !== "provisioned") return blocked("Complete private target provisioning in step 4 first.");
  if (action === "resizeRefresh") return r.resize ? available(`Resize ${r.resize.phase}. Read-only reconciliation; no new resize approval.`) : blocked("No retained resize sequence. Use step 5-2 first.");
  if (action === "recoveryReady") return stopped ? available("Read current runner health for this failed / interrupted job. Does not resume.") : blocked("Only needed for a failed or interrupted migration.");
  if (action === "inspectResume") return stopped ? available("After recovery readiness (5-9), inspect the retained checkpoint without resuming.") : blocked("Only needed for a failed or interrupted migration.");
  if (action === "resume") return stopped && r.resumeInspection ? available("Requires fresh recovery readiness, a matching inspection and separate approval. Preserves the same job.") : blocked("Requires a failed / interrupted job and a matching recovery inspection (5-10).");
  if (action === "diagnoseTarget") return m && ["failed", "finished"].includes(m.phase) ? available("Read-only diagnosis for a terminal migration; approval required for a new read.") : blocked("Available for a failed or finished migration.");
  if (action === "archiveFailure") return m?.phase === "failed" && r.targetDiagnostic?.phase === "finished" ? available("Only with fresh proof of an absent target graph and metadata. Retains all failed-job evidence.") : blocked("Requires a failed preparation and fresh empty-target diagnostic proof from step 5-12.");
  if (action === "qualifyP1") return m?.verification?.outcome === "pass" ? available("Development opt-in and the exact full P1 fixture are required. Not needed for general migration counts.") : blocked("Requires passing counts verification in step 6-1 and the full P1 development fixture.");
  if (action === "diagnoseP1") return r.p1Qualification?.phase === "failed" ? available("Development only: diagnose the retained P1 failure without replay.") : blocked("Only needed after a retained P1 qualification failure.");
  if (action === "requalifyP1") return r.p1Diagnostic?.phase === "finished" ? available("Development only: requires the reviewed source-key ordering diagnosis and separate approval.") : blocked("Requires the reviewed P1 ordering-failure diagnosis from step 6-3.");
  if (m) {
    if (action === "preload" && r.targetRestart?.phase === "finished") return { enabled: false, complete: true, detail: "Complete for the retained migration: AGE preload reconciled." };
    if (action === "resize" && r.resize?.phase === "finished") return { enabled: false, complete: true, detail: "Complete for the retained migration: same-VM resize finished." };
    if (action === "readiness") return { enabled: false, complete: true, detail: "Readiness was checked for the retained job. A stopped job uses optional recovery readiness, not this new-migration step." };
    return { enabled: false, complete: m.phase === "finished", detail: `Migration already ${m.phase}. Use step 5-6 / section 6, or optional recovery; never start another job.` };
  }
  if (r.upgrade && r.upgrade.phase !== "finished") return blocked("Reconcile the retained runner upgrade first.");
  if (action === "preload") return available(r.targetRestart?.phase === "finished" ? "Complete: AGE preload is applied and PostgreSQL is Ready."
    : r.targetRestart?.phase === "submitted" ? "Restarting PostgreSQL... Waiting for Azure Ready and the AGE preload setting to take effect. No restart will be resubmitted."
    : r.targetRestart ? "Restart request status is uncertain. Waiting for read-only confirmation; the restart will not be resubmitted."
    : "Required: approve the AGE preload restart, or reconcile if it is already applied.", r.targetRestart?.phase === "finished");
  if (r.targetRestart?.phase !== "finished") return blocked("Complete AGE preload preparation in step 5-1 first.");
  if (action === "resize") {
    if (r.resize?.phase === "finished") return { enabled: false, complete: true, detail: "Complete: the same runner is at the reviewed migration size." };
    if (assessmentActive(r)) return blocked("Reconcile the retained source operation before resizing.");
    return available(r.resize ? resizeStatusMessage(r)
      : "Required: approve deallocation, resize and restart of this runner only.");
  }
  if (r.resize?.phase !== "finished") return blocked("Complete the same-VM resize in step 5-2 first.");
  if (action === "readiness") {
    if (pending) return r.guestCommand?.action === "ready" ? available("Readiness check pending. Click to reconcile the same receipt; no new check is submitted.")
      : blocked("Reconcile the other pending guest command in step 5-8 first.");
    if (assessmentActive(r)) return blocked("Reconcile the retained source operation before checking migration readiness.");
    if (r.guestCommand?.action === "ready" && r.guestCommand.phase === "failed") return available("Last readiness check failed. Review retained evidence before explicitly checking again.");
    if (!r.guestReady) return available("Required after resize: check installation, idle worker, disk below 80%, and no swap / OOM.");
    try { assertIdleHealth(r, now); }
    catch (error) { return available(error instanceof Error ? error.message : "Readiness could not be verified. Review retained evidence."); }
    return available(`Ready: checked ${r.guestReady.checkedAt}; valid for five minutes. Continue to step 5-4.`, true);
  }
  if (pending) return blocked("Reconcile the pending guest command in step 5-3 (readiness) or 5-8 before migration.");
  if (assessmentActive(r) || !r.assessment?.reportSHA256 || !r.assessment.reportBytes) return blocked("Import a complete source inventory in step 3 first.");
  if (!Number.isFinite(Date.parse(r.target.input.deadline)) || Date.parse(r.target.input.deadline) <= now) return blocked("Cost authorization expired. Review a new authorization in optional step 5-5.");
  try { assertIdleHealth(r, now); }
  catch { return blocked("Run step 5-3: fresh, matching, idle Linux readiness is required (valid for five minutes)."); }
  if (!r.guestReady?.capabilities?.includes(`${r.input.source.type}-migration-v1`)) return blocked("The reviewed runner lacks migration support for this source. Review its pinned artifact and capabilities.");
  return available("Ready for separate approval. Creates one new job, migrates, then generates complete counts evidence for step 6-1.");
}

export function executionSummary(r: RunnerRecord, now = Date.now()) {
  return {
    actions: Object.fromEntries(executionActions.map(action => [action.id, executionActionState(r, action.id, now)])),
    readinessExpiresAt: r.guestReady ? Date.parse(r.guestReady.checkedAt) + 300000 : undefined,
    costDeadline: r.target?.input.deadline,
    migration: r.migration?.phase,
    verification: r.migration?.verification?.outcome
  };
}
