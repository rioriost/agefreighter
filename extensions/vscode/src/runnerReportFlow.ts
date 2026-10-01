import { RunnerRecord } from "./core/runner";
import { RunnerControl } from "./core/runnerLifecycle";
import { RunnerStore } from "./guided/runnerStore";
import { ReportManifest } from "./core/runnerBlob";
import { startReportExport, refreshReportExport, importReport } from "./core/runnerReport";
import { boundedWatch } from "./core/boundedWatch";
import { reconcileGuest } from "./core/runnerGuest";

/** Caller has approved this sealed report and destination. Retained transfer
 * intent is the receipt of that approval, including after a window reload. */
export async function transferApprovedReport(control: RunnerControl, store: RunnerStore, workflow: string, manifest: ReportManifest,
  capability: (r: RunnerRecord, operation: string, permission: "r" | "c") => Promise<string>,
  cancelled: () => boolean = () => false, validate?: (r: RunnerRecord, text: string) => void): Promise<RunnerRecord> {
  const initial = await store.read(workflow);
  const deadline = Math.min(Date.now() + 120000, initial.target ? Date.parse(initial.target.input.deadline) : Infinity);
  return await boundedWatch({ deadline, intervalMs: 10000, maxSteps: 12, sleep: control.sleep, cancelled,
    step: () => store.exclusive(workflow, async () => {
      let r = await store.read(workflow);
      const operation = [r.assessment, r.postgresCatalog, r.migration].find(x => x?.operation === manifest.operation);
      if (!operation || operation.reportSHA256 !== manifest.sha256 || operation.reportBytes !== manifest.bytes || r.vmId !== initial.vmId || JSON.stringify(r.storageDeployment) !== JSON.stringify(initial.storageDeployment)) throw new Error("Approved report or destination changed; transfer stopped.");
      const transfer = r.reportTransfers?.find(t => t.operation === manifest.operation);
      if (transfer?.phase === "imported") return r;
      if (!transfer) {
        if (r.guestCommand && ["submitted", "unknown"].includes(r.guestCommand.phase)) {
          r = (await reconcileGuest(control, r)).record;
          if (["submitted", "unknown"].includes(r.guestCommand!.phase)) return r;
          if (r.guestCommand?.phase === "failed") throw new Error(`The retained ${r.guestCommand.action} command failed. Review its evidence before transferring the report; nothing was resubmitted.`);
        }
        const grant = await capability(r, manifest.operation, "c");
        if (cancelled()) return r;
        const check = () => { if (cancelled()) throw new Error("Report transfer monitoring stopped. Retained work was not replayed."); };
        return startReportExport({ ...control,
          persist: async next => { check(); await control.persist(next); },
          request: (...args) => { check(); return control.request(...args); }
        }, r, grant, manifest.operation);
      }
      if (r.guestCommand?.action === "export-report" && r.guestCommand.operation === manifest.operation) {
        r = await refreshReportExport(control, r);
        if (r.guestCommand?.phase === "failed") throw new Error(`Retained report export failed; no retry was submitted. This is a report-transfer failure, not a Linux readiness result. Review the command evidence and storage access from runner subnet ${r.input.subnetId}; a storage firewall or RBAC restriction may require administrator approval.`);
      }
      if (r.reportTransfers?.find(t => t.operation === manifest.operation)?.phase !== "exported") return r;
      return importReport(control, r, manifest.operation, await capability(r, manifest.operation, "r"), async (id, m, text) => { validate?.(r, text); await store.retainReport(id, m, text); });
    }), done: r => r.reportTransfers?.find(t => t.operation === manifest.operation)?.phase === "imported"
  }) ?? initial;
}
