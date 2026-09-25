import test from "node:test";
import assert from "node:assert/strict";
import { armToken } from "../../guided/armToken";

test("ARM obtains a refreshed same-account token for every request or page", async () => {
  let calls = 0;
  const session = async (scopes: string[]) => {
    assert.deepEqual(scopes, ["https://management.azure.com/.default"]);
    return { accessToken: `refreshed-${++calls}`, account: { id: "selected" } };
  };
  assert.equal(await armToken("selected", "https://management.azure.com/", session), "refreshed-1");
  assert.equal(await armToken("selected", "https://management.azure.com", session), "refreshed-2");
  assert.equal(calls, 2);
});

test("Missing or different-account ARM sessions fail once without fallback or token disclosure", async () => {
  for (const value of [undefined, null, { accessToken: "", account: { id: "selected" } },
    { accessToken: "sensitive", account: { id: "other" } }]) {
    let calls = 0;
    await assert.rejects(armToken("selected", "https://management.azure.com", async () => { calls++; return value; }),
      error => error instanceof Error && /selected VS Code account/.test(error.message) && !error.message.includes("sensitive"));
    assert.equal(calls, 1);
  }
});
