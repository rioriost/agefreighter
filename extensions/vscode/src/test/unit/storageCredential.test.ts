import test from "node:test";
import assert from "node:assert/strict";
import { storageCredential, storageScope } from "../../guided/storageCredential";

test("Blob requests use a fresh Storage session for the same account, not the fixed ARM credential", async () => {
  let calls = 0;
  const credential = storageCredential("selected", async scopes => {
    assert.deepEqual(scopes, [storageScope]);
    return { accessToken: `storage-${++calls}`, account: { id: "selected" } };
  });
  assert.equal((await credential.getToken(storageScope))?.token, "storage-1");
  assert.equal((await credential.getToken([storageScope]))?.token, "storage-2");
  assert.equal(calls, 2);
});

test("Missing or foreign Storage sessions fail without ARM or account fallback", async () => {
  for (const session of [undefined, { accessToken: "", account: { id: "selected" } },
    { accessToken: "private-token", account: { id: "other" } }]) {
    let calls = 0;
    const credential = storageCredential("selected", async () => { calls++; return session; });
    await assert.rejects(credential.getToken(storageScope), error => error instanceof Error &&
      /Storage access/.test(error.message) && !error.message.includes("private-token"));
    assert.equal(calls, 1);
  }
});

test("Storage credentials reject non-Storage or mixed scopes before account access", async () => {
  const credential = storageCredential("selected", async () => { assert.fail("No account access expected"); });
  for (const scopes of [[], "https://management.azure.com/.default", [storageScope, "other"]])
    await assert.rejects(credential.getToken(scopes), /Only the Azure Storage audience/);
});
