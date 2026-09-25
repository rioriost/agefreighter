import assert from "node:assert/strict";
import test from "node:test";
import { bootstrapScript, runnerTemplate, sourceWorkflowDraft } from "../../core/runner";
import { developmentArtifact, developmentDownload } from "../../core/runnerDevelopment";
import { deploymentResources } from "../../core/runnerLifecycle";
const id="11111111-1111-4111-8111-111111111111",commit="a".repeat(40),sha="b".repeat(64);
const record=()=>sourceWorkflowDraft(id,{subscriptionId:id,resourceGroup:"test",region:"japaneast",zone:"1",subnetId:`/subscriptions/${id}/resourceGroups/test/providers/Microsoft.Network/virtualNetworks/test/subnets/test`,size:"Standard_B2s_v2",source:{type:"csv",location:"local"}});
test("development artifacts bind exact workflow, commit, version, hash and size without weakening official release URLs",()=>{
 const r=record(),raw={schemaVersion:1,platform:"linux-amd64",version:`2.4.0-dev.${commit.slice(0,12)}`,commit,sha256:sha,bytes:42};
 r.artifact=developmentArtifact(r,raw);const script=bootstrapScript(r.artifact);
 assert.match(script,/169\.254\.169\.254/);assert.match(script,/sha256sum --check/);assert.doesNotMatch(script,/sig=|AccountKey|git clone/);assert.match(script,/count > 42/);assert.match(script,/deadline = time.monotonic/);
 const resources=runnerTemplate(id,r.input,r.artifact,"ssh-ed25519 AAAA").resources as any[];
 assert.equal(resources.length,4);assert.equal(resources[3].type,"Microsoft.Authorization/roleAssignments");assert.match(resources[3].scope,new RegExp(`/containers/af-${id}$`));assert.match(resources[3].properties.roleDefinitionId,/2a2b9908-6ea1-4ae2-8e65-a410df84e7d1$/);
 assert.equal(deploymentResources(r).length,4);
 for(const bad of [{...raw,version:"2.4.0"},{...raw,commit:"main"},{...raw,bytes:128*1024*1024+1},{...raw,sha256:"bad"}])assert.throws(()=>developmentArtifact(r,bad));
 assert.throws(()=>developmentDownload({...r.artifact,url:r.artifact.url+"?sig=secret"}));
 assert.throws(()=>runnerTemplate("22222222-2222-4222-8222-222222222222",r.input,r.artifact,"ssh-ed25519 AAAA"));
 assert.throws(()=>bootstrapScript({...r.artifact,development:undefined}));
});
