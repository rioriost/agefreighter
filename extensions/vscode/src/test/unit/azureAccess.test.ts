import assert from "node:assert/strict";
import test from "node:test";
import { AzureAccessError, existingAzureAccess } from "../../core/azureAccess";

test("no VS Code accounts is signed out and never requests a new login", async () => {
  assert.equal(await existingAzureAccess({
    accounts: async () => [],
    session: async () => { throw new Error("must not request a session"); }
  }), "signedOut");
});

test("Azure Resources account without AGEFreighter permission requests access, not login", async () => {
  const calls: unknown[] = [];
  assert.equal(await existingAzureAccess({
    accounts: async () => ["existing-account"],
    session: async (account, options) => { calls.push({ account, ...options }); return false; }
  }), "accessRequired");
  assert.deepEqual(calls, [
    { account: "existing-account", createIfNone: false, silent: true },
    { account: "existing-account", createIfNone: false, silent: false }
  ]);
  assert.match(new AzureAccessError("accessRequired").message, /already signed in/);
});

test("refresh after granting account access is ready without another request", async () => {
  const calls: boolean[] = [];
  assert.equal(await existingAzureAccess({
    accounts: async () => ["existing-account"],
    session: async (_account, options) => { calls.push(options.silent); return true; }
  }), "ready");
  assert.deepEqual(calls, [true]);
});

test("uses an accessible second account without prompting for the first", async () => {
  assert.equal(await existingAzureAccess({
    accounts: async () => ["other", "azure"],
    session: async (account, options) => {
      assert.equal(options.silent, true);
      return account === "azure";
    }
  }), "ready");
});

test("a session granted during the access request is immediately usable", async () => {
  assert.equal(await existingAzureAccess({
    accounts: async () => ["azure"],
    session: async (_account, options) => !options.silent
  }), "ready");
});

test("provider and permission failures are not mislabeled as signed out", async () => {
  const failure = new Error("Authentication provider unavailable");
  await assert.rejects(existingAzureAccess({
    accounts: async () => { throw failure; }, session: async () => false
  }), (error) => error === failure && !(error instanceof AzureAccessError));
  await assert.rejects(existingAzureAccess({
    accounts: async () => ["azure"], session: async () => { throw failure; }
  }), (error) => error === failure);
});
