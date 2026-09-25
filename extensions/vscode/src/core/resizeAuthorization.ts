import { createHash } from "node:crypto";
import { RunnerRecord } from "./runner";
import { targetBudget } from "./runnerTarget";

function binding(r: RunnerRecord): string {
  return createHash("sha256").update(JSON.stringify([r.id,r.vmId,r.artifact,r.input,r.target?.serverId,r.target?.hash,r.target?.input])).digest("hex");
}
export function resizeAuthorized(r: RunnerRecord, now=Date.now()): boolean {
  return !!r.resizeAuthorization && r.resizeAuthorization.binding===binding(r) && Date.parse(r.resizeAuthorization.deadline)>now;
}
/** One native approval covers only deallocate -> resize -> start of this exact
 * idle runner. No AGE restart, credential access or migration is included. */
export function authorizeResize(r: RunnerRecord, now=Date.now()): RunnerRecord {
  if(!r.target || r.target.phase!=="provisioned" || r.migration)throw new Error("Review the idle runner and provisioned target first.");
  targetBudget(r.target.input,now);
  return {...r,resizeAuthorization:{binding:binding(r),approvedAt:new Date(now).toISOString(),deadline:new Date(Math.min(now+20*60000,Date.parse(r.target.input.deadline))).toISOString()}};
}
