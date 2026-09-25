import * as vscode from "vscode";
import { RunnerControl } from "./core/runnerLifecycle";
import { RunnerStore } from "./guided/runnerStore";
import { RunnerRecord } from "./core/runner";
import { refreshAssessment } from "./core/runnerAssessment";
import { refreshCatalog } from "./core/runnerCatalog";
import { refreshMigration } from "./core/runnerExecution";
import { boundedWatch } from "./core/boundedWatch";

const watchers = new Set<string>();
export type WatchKind = "assessment" | "postgresCatalog" | "migration";
/** Only read-only status commands may be submitted. No work dispatch, replay,
 * source re-read, resize or credential access is reachable from this watcher. */
export async function watchRetainedOperation(control: RunnerControl, store: RunnerStore, workflow: string, kind: WatchKind, cancelled: () => boolean = () => false,
  progress?: (record: RunnerRecord) => Promise<void>): Promise<void> {
  const key = `${workflow}/${kind}`;
  if (watchers.has(key)) return;
  watchers.add(key);
  try {
    const initial = await store.read(workflow), op = initial[kind];
    if (!op || ["finished", "failed", "interrupted"].includes(op.phase)) return;
    const deadline = Math.min(Date.now() + 30 * 60000, initial.target ? Date.parse(initial.target.input.deadline) : Infinity);
    await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: "Watching retained operation — no automatic retry", cancellable: true }, async (_p, token) => {
      await boundedWatch({ deadline, intervalMs: 120000, maxSteps: 15, sleep: control.sleep,
        cancelled: () => token.isCancellationRequested || cancelled() || !vscode.workspace.isTrusted,
        step: () => store.exclusive(workflow, async () => {
          const current = await store.read(workflow);
          if (current[kind]?.operation !== op.operation || current.vmId !== initial.vmId || current.artifact.sha256 !== initial.artifact.sha256) throw new Error("Retained operation changed; monitoring stopped without replay.");
          if (["finished", "failed", "interrupted"].includes(current[kind]!.phase)) return current;
          return kind === "migration" ? refreshMigration(control, current) : kind === "postgresCatalog" ? refreshCatalog(control, current) : refreshAssessment(control, current);
        }), done: r => ["finished", "failed", "interrupted"].includes(r[kind]!.phase), progress });
    });
  } finally { watchers.delete(key); }
}
