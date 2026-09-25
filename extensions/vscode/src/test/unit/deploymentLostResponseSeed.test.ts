import assert from "node:assert/strict";
import test from "node:test";
import { validateLostResponseSeed } from "../helpers/deploymentLostResponseSeed";
import { otherCancellationFixture,otherNativeCancelCases } from "../helpers/nativeCancelOtherScenarios";
import { sourceWorkflowDraft } from "../../core/runner";
import { storageDraft } from "../../core/runnerStorageLifecycle";
function fixture(){const f=otherCancellationFixture(otherNativeCancelCases[0]);const scope={workflow:f.record.id,input:f.record.input},r=sourceWorkflowDraft(scope.workflow,scope.input);r.storageDeployment=storageDraft(r,r.id);return {scope,r};}
test("continuation preserves the exact effect-free draft and reviewed storage preview",()=>{const {scope,r}=fixture();const original=JSON.stringify(r);assert.doesNotThrow(()=>validateLostResponseSeed(r,scope));assert.equal(JSON.stringify(r),original);});
for(const field of ["sourceDraft","guestCommand","guestReady","developmentUpload","migration"] as const)test(`continuation rejects ${field}`,()=>{const {scope,r}=fixture();Object.assign(r,{[field]:{}});assert.throws(()=>validateLostResponseSeed(r,scope));});
test("continuation rejects submitted/ready states, extra permissions and altered storage hash",()=>{for(const mutation of [(r:any)=>r.phase="unknown",(r:any)=>r.storageDeployment.phase="submitted",(r:any)=>r.storageDeployment.hash="a".repeat(64),(r:any)=>r.storageDeployment.template.resources.push({type:"Microsoft.Authorization/roleAssignments"})]){const {scope,r}=fixture();mutation(r);assert.throws(()=>validateLostResponseSeed(r,scope));}});
