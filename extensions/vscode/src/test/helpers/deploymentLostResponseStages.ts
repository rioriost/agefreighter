/** Disposable B09 companion stage gates; never imported by the released extension. */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { RunnerInput, RunnerRecord, object, runnerTemplate } from "../../core/runner";
import { reportStorageNames } from "../../core/runnerReportStorage";
import { guestReadinessScript } from "../../core/runnerGuest";
import { storageDraft } from "../../core/runnerStorageLifecycle";
import { lostResponseHash, retainLostResponseFile } from "./deploymentLostResponse";
import { describeLostResponsePoll, validateLostResponsePoll } from "./deploymentLostResponsePoll";
import type { AzureSession } from "../../guided/azure";
export interface LostResponseScope {
  workflow: string; input: RunnerInput; artifactSHA256: string; artifactBytes: number;
  manifestPath: string; archivePath: string; annotation: string; expiresAt: string;
}
export class LostResponseStages {
  private sequence = 0;
  private readonly polls = new Map<string, { remaining: number; deadline: number }>();
  private approvedPreview?: string;
  constructor(readonly root: string, readonly scope: LostResponseScope, private readonly current: () => Promise<RunnerRecord>, private readonly actual: AzureSession) {}
  private fresh() { assert.ok(Date.now() < Date.parse(this.scope.expiresAt), "Companion authorization window expired"); }
  private async event(kind: string, details: object = {}) {
    this.fresh(); assert.ok(++this.sequence <= 300, "Companion observation budget exhausted");
    await retainLostResponseFile(this.root, `stage-${String(this.sequence).padStart(4, "0")}.json`, { kind, ...details, at: new Date().toISOString() });
  }
  private async record() {
    const r = await this.current(); assert.equal(r.id, this.scope.workflow); assert.deepEqual(r.input, this.scope.input);
    assert.ok(!r.target && !r.migration && !r.assessment && !r.sourceDraft && !r.guestReady, "Unexpected source/ready state in negative fixture");
    return r;
  }
  async approvePreview() {
    const r = await this.record(); assert.equal(r.phase, "previewed"); this.checkTemplate(r.template, r);
    this.approvedPreview = lostResponseHash(r);
    await retainLostResponseFile(this.root, "native-approved-preview.json", { record: r, hash: this.approvedPreview, at: new Date().toISOString(), annotation: this.scope.annotation });
  }
  private checkTemplate(template: unknown, record: RunnerRecord) {
    assert.equal(record.artifact.sha256, this.scope.artifactSHA256);
    const vm = (object(template).resources as unknown[]).map(object).find(r => r.type === "Microsoft.Compute/virtualMachines"); assert.ok(vm);
    const keys = object(object(object(vm.properties).osProfile).linuxConfiguration).ssh;
    const key = object((object(keys).publicKeys as unknown[])?.[0]).keyData;
    assert.equal(typeof key, "string");
    assert.equal(lostResponseHash(template), lostResponseHash(runnerTemplate(record.id, record.input, record.artifact, key as string)));
  }
  private checkStorage(record: RunnerRecord) {
    const d = record.storageDeployment; assert.ok(d);
    const names = reportStorageNames(record), role = d.roleId.slice(`${names.id}/providers/Microsoft.Authorization/roleAssignments/`.length);
    assert.match(role, /^[a-f0-9-]{36}$/);
    assert.equal(d.roleId, `${names.id}/providers/Microsoft.Authorization/roleAssignments/${role}`);
    const canonical = storageDraft(record, d.principalId);
    (canonical.template.resources as Record<string, unknown>[])[2]!.name = role;
    assert.equal(d.id, canonical.id); assert.equal(lostResponseHash(d.template), lostResponseHash(canonical.template));
  }
  private async learnPoll(response: { poll?: string }, sourcePath: string) {
    if (!response.poll) return;
    const description = describeLostResponsePoll(response.poll, this.scope.input.subscriptionId, this.scope.input.region);
    await this.event("whatIf-poll-header", { sourceSHA256: lostResponseHash(sourcePath), ...description });
    validateLostResponsePoll(response.poll, this.scope.input.subscriptionId, this.scope.input.region);
    assert.ok(this.polls.has(response.poll) || this.polls.size < 8, "What-if poll URL budget exhausted");
    this.polls.set(response.poll, this.polls.get(sourcePath) ?? this.polls.get(response.poll) ?? { remaining: 30, deadline: Math.min(Date.now() + 120000, Date.parse(this.scope.expiresAt)) });
  }
  private paths(r: RunnerRecord) {
    const i = r.input, base = `/subscriptions/${i.subscriptionId}`, group = `${base}/resourceGroups/${i.resourceGroup}`, names = reportStorageNames(r);
    return { group, names, storage: `${group}/providers/Microsoft.Resources/deployments/${names.account}-transfer`,
      get: new Set([`${i.subnetId}?api-version=2024-05-01`, `${i.subnetId.replace(/\/subnets\/[^/]+$/, "")}?api-version=2024-05-01`, `${group}?api-version=2021-04-01`,
        `${names.id}?api-version=2023-05-01`, `${names.containerId}?api-version=2023-05-01`, `${r.deploymentId}?api-version=2022-09-01`,
        `${group}/providers/Microsoft.Resources/deployments/${names.account}-transfer?api-version=2022-09-01`, `${r.vmId}?api-version=2024-07-01`,
        `${group}/providers/Microsoft.Network/networkSecurityGroups/${r.vmId.split('/').at(-1)}?api-version=2024-05-01`,
        `${group}/providers/Microsoft.Network/networkInterfaces/${r.vmId.split('/').at(-1)}?api-version=2024-05-01`,
        `${group}/providers/Microsoft.Compute/disks/${r.vmId.split('/').at(-1)}-os?api-version=2024-03-02`,
        `${names.containerId}/providers/Microsoft.Authorization/roleAssignments/${r.id}?api-version=2022-04-01`,
        ...(r.storageDeployment ? [`${r.storageDeployment.roleId}?api-version=2022-04-01`] : [])]),
      list: new Set([`${group}/resources?api-version=2021-04-01`, `${base}/resourcegroups?api-version=2021-04-01`,
        `${base}/providers/Microsoft.Compute/skus?api-version=2021-07-01&$filter=${encodeURIComponent(`location eq '${i.region}'`)}`,
        `${base}/providers/Microsoft.Compute/locations/${i.region}/usages?api-version=2025-04-01`, `${r.vmId}/runCommands?api-version=2024-07-01`]) };
  }
  async request(subscription: string, path: string, method: "GET"|"POST"|"PUT"|"PATCH"|"DELETE" = "GET", body?: unknown) {
    body = body === undefined ? undefined : structuredClone(body);
    this.fresh(); assert.equal(subscription, this.scope.input.subscriptionId);
    const r = await this.record(), p = this.paths(r);
    if (method === "GET") {
      assert.equal(body, undefined);
      const guest = new RegExp(`^${r.vmId.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}/runCommands/af-[a-f0-9-]{36}\\?api-version=2024-07-01(?:&\\$expand=instanceView)?$`);
      assert.ok(p.get.has(path) || this.polls.has(path) || guest.test(path), "GET outside exact trial scope");
      const polling = this.polls.get(path);
      if (polling) { assert.ok(polling.remaining > 0 && Date.now() < polling.deadline, "What-if polling budget expired"); polling.remaining--; }
      await this.event("GET", this.polls.has(path) ? describeLostResponsePoll(path, this.scope.input.subscriptionId, this.scope.input.region) : { path });
      const response = await this.actual.runnerRequest(subscription, path);
      if (this.polls.has(path)) await this.learnPoll(response, path);
      return response;
    }
    if (method === "POST") {
      const storage = path === `${p.storage}/whatIf?api-version=2022-09-01`;
      assert.ok(storage || path === `${r.deploymentId}/whatIf?api-version=2022-09-01`);
      const template = object(object(body).properties).template;
      if (storage) { this.checkStorage(r); assert.equal(lostResponseHash(template), lostResponseHash(r.storageDeployment!.template)); }
      else this.checkTemplate(template, r);
      assert.equal(lostResponseHash(body), lostResponseHash({ properties: { mode: "Incremental", template, whatIfSettings: { resultFormat: "ResourceIdOnly" } } }));
      await this.event("POST-whatIf", { path, bodySHA256: lostResponseHash(body) });
      const response = await this.actual.runnerRequest(subscription, path, method, body); await this.learnPoll(response, path); return response;
    }
    assert.equal(method, "PUT", "No PATCH/DELETE or other effect permitted");
    if (path === `${p.storage}?api-version=2022-09-01`) {
      this.checkStorage(r);
      assert.equal(r.storageDeployment?.phase, "submitted");
      assert.equal(lostResponseHash(body), lostResponseHash({ properties: { mode: "Incremental", template: r.storageDeployment!.template } }));
      await retainLostResponseFile(this.root, "storage-put-intent.json", { path, bodySHA256: lostResponseHash(body), roleId: r.storageDeployment!.roleId });
      return this.actual.runnerRequest(subscription, path, method, body);
    }
    if (path === `${r.deploymentId}?api-version=2022-09-01`) {
      assert.equal(r.phase, "deployment-submitted"); this.checkTemplate(r.template, r);
      const approval = JSON.parse(await readFile(join(this.root, "native-approved-preview.json"), "utf8"));
      assert.equal(approval.hash, this.approvedPreview);
      assert.equal(lostResponseHash({ ...r, phase: "previewed", updatedAt: approval.record.updatedAt }), approval.hash);
      assert.equal(lostResponseHash(body), lostResponseHash({ properties: { mode: "Incremental", template: r.template } }));
      await retainLostResponseFile(this.root, "runner-put-intent.json", { path, templateSHA256: lostResponseHash(r.template), at: new Date().toISOString() });
      const response = await this.actual.runnerRequest(subscription, path, method, body);
      assert.ok([200,201,202].includes(response.status), "No accepted deployment response");
      await retainLostResponseFile(this.root, "azure-acceptance.json", { deploymentId: r.deploymentId, status: response.status, at: new Date().toISOString(),
        mechanism: "adapter-received-response-withheld-from-controller", realServiceQualifiedByThisFile: false, independentARMObservationRequired: true });
      throw Error("B09 companion intentionally withheld accepted deployment response; GET reconcile only");
    }
    assert.equal(r.guestCommand?.action, "ready"); assert.equal(r.guestCommand?.phase, "submitted");
    assert.equal(path, `${r.guestCommand!.id}?api-version=2024-07-01`);
    const properties = object(object(body).properties), parameters = properties.protectedParameters as {name:string;value:string}[];
    assert.equal(parameters.length, 1); assert.equal(parameters[0]!.name, "AF_RUNNER_REQUEST");
    const payload = JSON.parse(Buffer.from(parameters[0]!.value, "base64").toString("utf8"));
    assert.deepEqual(payload, { version: 1, workflow: r.id, operation: r.guestCommand!.operation, action: "ready" });
    assert.deepEqual(body, { location: r.input.region, properties: { source: { script: guestReadinessScript }, protectedParameters: parameters, timeoutInSeconds: 60, asyncExecution: false } });
    await retainLostResponseFile(this.root, "readiness-put-intent.json", { path, action: "ready", operation: r.guestCommand!.operation, at: new Date().toISOString() });
    return this.actual.runnerRequest(subscription, path, method, body);
  }
  async list(subscription: string, path: string) {
    assert.equal(subscription, this.scope.input.subscriptionId); assert.ok(this.paths(await this.record()).list.has(path));
    await this.event("LIST", { path }); return this.actual.runnerList(subscription, path);
  }
  async invoke(method: string, args: unknown[]): Promise<unknown> {
    if (method === "dispose") { this.actual.dispose(); return; }
    if (method === "runnerRequest") return this.request(...args as Parameters<AzureSession["runnerRequest"]>);
    if (method === "runnerList") return this.list(...args as Parameters<AzureSession["runnerList"]>);
    this.fresh();
    if (method === "subscriptions") { assert.equal(args.length,0); await this.event(method); return (await this.actual.subscriptions()).filter(s => s.id === this.scope.input.subscriptionId); }
    if (method === "locations" || method === "storagePrincipal") {
      assert.deepEqual(args,[this.scope.input.subscriptionId]); await this.event(method);
      return method === "locations" ? this.actual.locations(this.scope.input.subscriptionId) : this.actual.storagePrincipal(this.scope.input.subscriptionId);
    }
    if (method === "retailRates") {
      assert.deepEqual(args,[this.scope.input.region,[this.scope.input.size]]); await this.event(method);
      return this.actual.retailRates(this.scope.input.region,[this.scope.input.size]);
    }
    assert.equal(method,"uploadRunnerArchive", "Unapproved Azure method");
    const r = await this.record(); assert.equal(lostResponseHash(args[0]),lostResponseHash(r)); assert.equal(args[1],this.scope.archivePath);
    const manifest = args[2] as {file:string;sha256:string;bytes:number};
    assert.deepEqual(manifest,{file:r.id,sha256:this.scope.artifactSHA256,bytes:this.scope.artifactBytes});
    assert.equal(r.developmentUpload?.phase,"prepared"); assert.equal(r.developmentUpload?.artifact.sha256,this.scope.artifactSHA256);
    await retainLostResponseFile(this.root,"artifact-upload-intent.json",{workflow:r.id,sha256:manifest.sha256,bytes:manifest.bytes});
    return this.actual.uploadRunnerArchive(r,this.scope.archivePath,manifest);
  }
}
