import * as vscode from "vscode";
import { RunnerControl } from "./core/runnerLifecycle";
import { RunnerStore } from "./guided/runnerStore";
import { RunnerRecord } from "./core/runner";
import { refreshAssessment } from "./core/runnerAssessment";
import { refreshCatalog } from "./core/runnerCatalog";
import { refreshMigration } from "./core/runnerExecution";
import { boundedWatch } from "./core/boundedWatch";
import { refreshRunner } from "./core/runnerLifecycle";
import { reconcileGuest } from "./core/runnerGuest";
import { refreshTarget, targetPending, targetStatusMessage } from "./core/runnerTarget";

const watchers = new Set<string>();
export type WatchKind = "assessment" | "postgresCatalog" | "migration";
/** Only read-only status commands may be submitted. No work dispatch, replay,
 * source re-read, resize or credential access is reachable from this watcher. */
export async function watchRetainedOperation(control: RunnerControl, store: RunnerStore, workflow: string, kind: WatchKind, cancelled: () => boolean = () => false,
  progress?: (record: RunnerRecord) => Promise<void>): Promise<RunnerRecord | undefined> {
  const key = `${workflow}/${kind}`;
  if (watchers.has(key)) return;
  watchers.add(key);
  try {
    const initial = await store.read(workflow), op = initial[kind];
    if (!op) return;
    if (["finished", "failed", "interrupted"].includes(op.phase)) return cancelled() ? undefined : initial;
    const deadline = Math.min(Date.now() + 30 * 60000, initial.target ? Date.parse(initial.target.input.deadline) : Infinity);
    return await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: "Watching retained operation — cancel stops monitoring only", cancellable: true }, async (_p, token) => {
      const result = await boundedWatch({ deadline, intervalMs: 15000, maxSteps: 120, sleep: control.sleep,
        cancelled: () => token.isCancellationRequested || cancelled() || !vscode.workspace.isTrusted,
        step: () => store.exclusive(workflow, async () => {
          const current = await store.read(workflow);
          if (current[kind]?.operation !== op.operation || current.vmId !== initial.vmId || current.artifact.sha256 !== initial.artifact.sha256) throw new Error("Retained operation changed; monitoring stopped without replay.");
          if (["finished", "failed", "interrupted"].includes(current[kind]!.phase)) return current;
          const command = current.guestCommand;
          if (command?.phase === "failed") throw new Error(`The retained ${command.action} command failed. Automatic monitoring stopped; review its evidence before another action.`);
          // Poll an existing ARM receipt promptly, but do not consume the
          // 25-command budget by creating a new status command every tick.
          if (command?.action === "status" && command.operation === op.operation && command.phase === "finished" &&
              Date.now() - Date.parse(command.submittedAt) < 120000) return current;
          return kind === "migration" ? refreshMigration(control, current) : kind === "postgresCatalog" ? refreshCatalog(control, current) : refreshAssessment(control, current);
        }), done: r => ["finished", "failed", "interrupted"].includes(r[kind]!.phase), progress });
      return token.isCancellationRequested || cancelled() || !vscode.workspace.isTrusted ? undefined : result;
    });
  } finally { watchers.delete(key); }
}

/** Reconcile only already-submitted ARM operations; never start guest work. */
export async function watchRunnerState(control: RunnerControl, store: RunnerStore, workflow: string,
  cancelled: () => boolean, progress: (record: RunnerRecord) => Promise<void>): Promise<void> {
  const initial = await store.read(workflow);
  await boundedWatch({ deadline: Date.now() + 10 * 60000, intervalMs: 5000, maxSteps: 120, sleep: control.sleep, cancelled,
    step: () => store.exclusive(workflow, async () => {
      let r = await store.read(workflow);
      if (r.vmId !== initial.vmId || r.deploymentId !== initial.deploymentId || r.artifact.sha256 !== initial.artifact.sha256) throw new Error("Runner changed; automatic monitoring stopped.");
      if (["deployment-submitted", "unknown"].includes(r.phase)) r = await refreshRunner(control, r);
      if (r.guestCommand && ["submitted", "unknown"].includes(r.guestCommand.phase)) r = (await reconcileGuest(control, r)).record;
      return r;
    }),
    done: r => !["deployment-submitted", "unknown"].includes(r.phase) && !["submitted", "unknown"].includes(r.guestCommand?.phase ?? ""),
    progress });
}

/** GET-only reconciliation of the retained target, including an approved repair.
 * Cancellation stops monitoring, never the Azure deployment. */
export async function watchTargetState(control: RunnerControl, store: RunnerStore, workflow: string,
  cancelled: () => boolean, progress: (record: RunnerRecord) => Promise<void>): Promise<RunnerRecord | undefined> {
  const initial = await store.read(workflow), target = initial.target;
  if (!target || target.phase === "previewed") throw new Error("No submitted private target to monitor.");
  return vscode.window.withProgress({
    location: vscode.ProgressLocation.Notification, title: "Watching private target deployment — cancel stops monitoring only", cancellable: true
  }, async (notification, token) => {
    const stopped = () => token.isCancellationRequested || cancelled() || !vscode.workspace.isTrusted;
    const result = await boundedWatch({
      deadline: Date.now() + 30 * 60000, intervalMs: 15000, maxSteps: 120, sleep: control.sleep, cancelled: stopped,
      step: () => store.exclusive(workflow, async () => {
        const current = await store.read(workflow);
        if (stopped()) return current;
        if (current.vmId !== initial.vmId || current.artifact.sha256 !== initial.artifact.sha256 ||
            current.target?.deploymentId !== target.deploymentId || current.target.serverId !== target.serverId ||
            current.target.hash !== target.hash || current.target.phase === "previewed")
          throw new Error("Retained target identity changed; monitoring stopped without replay.");
        return refreshTarget(control, current);
      }),
      done: record => !targetPending(record),
      progress: async record => {
        if (stopped()) return;
        notification.report({ message: targetStatusMessage(record) });
        await progress(record);
      }
    });
    return stopped() ? undefined : result;
  });
}
