/** Accept only an unchanged, effect-free draft/optional storage preview. */
import assert from "node:assert/strict";
import { RunnerInput, RunnerRecord, sourceWorkflowDraft } from "../../core/runner";
import { storageDraft } from "../../core/runnerStorageLifecycle";
import { reportStorageNames } from "../../core/runnerReportStorage";
import { lostResponseHash } from "./deploymentLostResponse";
export function validateLostResponseSeed(record:RunnerRecord,scope:{workflow:string;input:RunnerInput},preservedStorageIntent?:unknown):void {
  const {storageDeployment,...base}=record;
  const canonical=sourceWorkflowDraft(scope.workflow,scope.input); canonical.updatedAt=record.updatedAt;
  assert.ok(Number.isFinite(Date.parse(record.updatedAt)));
  assert.deepEqual(base,canonical,"Continuation must contain only the original effect-free draft");
  if(!storageDeployment){assert.equal(preservedStorageIntent,undefined);return;}
  const d=storageDeployment,names=reportStorageNames(record),prefix=`${names.id}/providers/Microsoft.Authorization/roleAssignments/`;
  assert.equal(d.phase,preservedStorageIntent===undefined?"previewed":"ready");assert.ok(d.roleId.startsWith(prefix));const role=d.roleId.slice(prefix.length);assert.match(role,/^[a-f0-9-]{36}$/);
  const expected=storageDraft(record,d.principalId);(expected.template.resources as Record<string,unknown>[])[2]!.name=role;
  expected.roleId=d.roleId;expected.expiresAt=d.expiresAt;
  expected.hash=lostResponseHash({id:expected.id,principalId:expected.principalId,roleId:expected.roleId,template:expected.template});
  if(preservedStorageIntent!==undefined){
    expected.phase="ready";
    assert.ok(["Enabled","Disabled","SecuredByPerimeter","Unknown"].includes(d.networkAccess??""));expected.networkAccess=d.networkAccess;
    assert.deepEqual(preservedStorageIntent,{path:`${d.id}?api-version=2022-09-01`,bodySHA256:lostResponseHash({properties:{mode:"Incremental",template:expected.template}}),roleId:expected.roleId},
      "Ready storage requires its exact prior PUT claim; no replay or new role is authorized");
  }
  assert.ok(Number.isFinite(Date.parse(d.expiresAt)));assert.deepEqual(d,expected,"Continuation storage preview changed");
}
