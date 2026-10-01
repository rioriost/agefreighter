import { createHash } from "node:crypto";
import { RunnerRecord } from "./core/runner";
import { RunnerControl } from "./core/runnerLifecycle";
import { boundedWatch } from "./core/boundedWatch";
import { refreshStorage, storageDraft, submitStorage } from "./core/runnerStorageLifecycle";
import { verifyTransferStorage } from "./core/runnerReportStorage";
import { RunnerStore } from "./guided/runnerStore";

/** Provision once, with native consent; subsequent calls only reconcile. */
export async function prepareRequiredStorage(control: RunnerControl, store: RunnerStore, workflow: string,
  principal: (subscription: string) => Promise<string>,
  approve: (record: RunnerRecord, principal: string) => Promise<boolean>,
  cancelled: () => boolean,
  progress: (record: RunnerRecord) => Promise<void>): Promise<RunnerRecord | undefined> {
  const initial = await store.read(workflow);
  const binding = (r: RunnerRecord) => createHash("sha256").update(JSON.stringify([r.id, r.input, r.vmId])).digest("hex");
  if (!initial.storageDeployment || initial.storageDeployment.phase === "previewed") {
    const user = await principal(initial.input.subscriptionId);
    if (cancelled() || !await approve(initial, user) || cancelled()) return;
    await store.exclusive(workflow, async () => {
      const current = await store.read(workflow);
      if (binding(current) !== binding(initial)) throw new Error("Runner placement changed during storage approval. Review it again.");
      if (current.storageDeployment && current.storageDeployment.phase !== "previewed") throw new Error("Storage was already submitted. Refresh its retained deployment; do not submit it again.");
      const check = () => { if (cancelled()) throw new Error("Storage preparation stopped. No source read was submitted."); };
      check();
      const approvedControl: RunnerControl = { ...control,
        persist: async r => { check(); await control.persist(r); },
        request: (...args) => { check(); return control.request(...args); }
      };
      const draft = { ...current, storageDeployment: storageDraft(current, user) };
      await approvedControl.persist(draft);
      await submitStorage(approvedControl, draft);
    });
  }
  const result = await boundedWatch({ deadline: Date.now() + 5 * 60000, intervalMs: 5000, maxSteps: 60, sleep: control.sleep, cancelled,
    step: () => store.exclusive(workflow, async () => {
      const current = await store.read(workflow);
      if (binding(current) !== binding(initial)) throw new Error("Runner placement changed. Storage monitoring stopped.");
      return current.storageDeployment?.phase === "ready" || current.storageDeployment?.phase === "failed" ? current : refreshStorage(control, current);
    }),
    done: r => r.storageDeployment?.phase === "ready" || r.storageDeployment?.phase === "failed", progress });
  if (cancelled()) return;
  if (result?.storageDeployment?.phase !== "ready") throw new Error(`Transfer storage is ${result?.storageDeployment?.phase ?? "unavailable"}. No source read was submitted. Use Prepare / refresh transfer storage to reconcile; no deployment was replayed.`);
  await verifyTransferStorage(control, result);
  return cancelled() ? undefined : result;
}
