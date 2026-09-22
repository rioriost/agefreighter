/** Explicitly opted-in B02 read-only checks against the dedicated trial scope.
 * Not part of npm test. No deployment, what-if, guest execution or source data.
 * Production preflight runs unchanged; CLI supplies fresh real ARM responses.
 */
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { preflightRunner, RunnerControl } from "../core/runnerLifecycle";
import { RunnerInput } from "../core/runner";

const sub = "67c417f3-5a13-446c-afb9-40cd87f2fdb7";
const sourceGroup = "rg-af-vscode-p1-20260905-a", migrationGroup = "rg-af-vscode-p1-b01-20260921";
const base = `/subscriptions/${sub}`, group = `${base}/resourceGroups/${sourceGroup}`;
const vnet = `${group}/providers/Microsoft.Network/virtualNetworks/vnet-af-vscode-p1`;
const baseline: RunnerInput = {subscriptionId: sub, resourceGroup: migrationGroup,
  region: "japaneast", zone: "1", size: "Standard_B2s_v2", subnetId: `${vnet}/subnets/runner`,
  source: {type: "neo4j", location: "azure", resourceId: `${group}/providers/Microsoft.Compute/virtualMachines/af-n44-source`}};
const execute = promisify(execFile);

async function main(): Promise<void> {
  assert.equal(process.env.AF_B02_LIVE_READ_ONLY, "1", "Explicit B02 read-only opt-in required");
  const output = await mkdtemp(join(tmpdir(), "af-b02-arm-readonly-"));
  const traces: {method: string; path: string; status: number}[] = [];
  async function get(path: string): Promise<{status: number; value: unknown}> {
    const url = new URL(path.startsWith("https:") ? path : `https://management.azure.com${path}`);
    assert.equal(url.origin, "https://management.azure.com");
    const allowed = [
      `${vnet}`, `${vnet}/subnets/runner`, `${vnet}/subnets/afpg-5cb990c12a254de5a10d`,
      `${vnet}/subnets/af-b02-nonexistent-20260922`, `${base}/resourceGroups/${migrationGroup}`,
      `${base}/resourceGroups/rg-af-b02-nonexistent-20260922`,
      `${group}/providers/Microsoft.Compute/virtualMachines/af-n44-source`,
      `${base}/providers/Microsoft.Compute/skus`,
      `${base}/providers/Microsoft.Compute/locations/japaneast/usages`
    ];
    assert.ok(allowed.some(p => p.toLowerCase() === url.pathname.toLowerCase()), "Unreviewed ARM path refused");
    try {
      const {stdout} = await execute("az", ["rest", "--method", "get", "--url", url.href,
        "--subscription", sub, "--output", "json", "--only-show-errors"], {timeout: 60000, maxBuffer: 32 * 1024 * 1024});
      traces.push({method: "GET", path: url.pathname, status: 200});
      return {status: 200, value: JSON.parse(stdout)};
    } catch (e) {
      const error = e as {stderr?: string};
      if (/\b(ResourceNotFound|ResourceGroupNotFound|NotFound)\b/.test(error.stderr ?? "")) {
        traces.push({method: "GET", path: url.pathname, status: 404});
        return {status: 404, value: {}};
      }
      throw Error("ARM GET failed; no retry, fallback, credential or raw response disclosure");
    }
  }
  const control: RunnerControl = {
    request: async (subscription, path, method = "GET") => {
      assert.equal(subscription, sub); assert.equal(method, "GET"); return get(path);
    },
    list: async (subscription, path) => {
      assert.equal(subscription, sub);
      const rows: unknown[] = [];
      for (let page = 0; page < 30; page++) {
        const response = await get(path); assert.equal(response.status, 200);
        const value = response.value as {value: unknown[]; nextLink?: string};
        assert.ok(Array.isArray(value.value)); rows.push(...value.value);
        if (!value.nextLink) return rows;
        path = value.nextLink;
      }
      throw Error("ARM pagination bound reached");
    },
    persist: async () => {throw Error("No local workflow persistence allowed");},
    sleep: async () => {throw Error("No polling allowed");}
  };
  const cases: {name: string; patch?: Partial<RunnerInput>; refusal?: RegExp}[] = [
    {name: "valid existing private placement"},
    {name: "nonexistent compute subnet", patch: {subnetId: `${vnet}/subnets/af-b02-nonexistent-20260922`}, refusal: /subnet does not exist/},
    {name: "existing PostgreSQL-delegated subnet", patch: {subnetId: `${vnet}/subnets/afpg-5cb990c12a254de5a10d`}, refusal: /non-delegated compute subnet/},
    {name: "runner region differs from actual VNet", patch: {region: "japanwest"}, refusal: /region must match the existing VNet/},
    {name: "nonexistent migration group", patch: {resourceGroup: "rg-af-b02-nonexistent-20260922"}, refusal: /existing resource group/},
    {name: "runner zone differs from actual source VM", patch: {zone: "2"}, refusal: /source availability zone/}
  ];
  const results = [];
  for (const c of cases) {
    const start = traces.length;
    let message: string | undefined;
    try {await preflightRunner(control, {...structuredClone(baseline), ...c.patch});}
    catch (e) {message = e instanceof Error ? e.message : "unknown";}
    const pass = c.refusal ? !!message && c.refusal.test(message) : message === undefined;
    results.push({name: c.name, pass, outcome: message ?? "accepted read-only preflight", reads: traces.length - start});
    console.log(JSON.stringify(results.at(-1)));
    if (!pass) break; // Unexpected live state must not be relabeled as coverage.
  }
  const text = JSON.stringify({generatedAt: new Date().toISOString(), scope: "B02 unchanged production preflight with real ARM GETs, not installed GUI qualification",
    sourceReads: "ARM metadata only; no database data", mutations: 0, results, traces}, null, 2);
  await writeFile(join(output, "result.json"), text, {flag: "wx", mode: 0o600});
  console.log(JSON.stringify({evidence: output, sha256: createHash("sha256").update(text).digest("hex")}));
  assert.equal(results.length, cases.length); assert.ok(results.every(r => r.pass));
}
main().catch(error => {console.error(error.message); process.exitCode = 1;});
