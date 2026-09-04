import assert from "node:assert/strict";
import test from "node:test";
import { CLIError, runJSON, runText } from "../../core/process";

const options = {
  cwd: process.cwd(),
  timeoutMs: 5000,
  maxOutputBytes: 64 * 1024,
  env: { ...process.env }
};

test("captures JSON without invoking a shell", async () => {
  const value = await runJSON(process.execPath, ["-e", "process.stdout.write(JSON.stringify({ok:true}))"], options);
  assert.deepEqual(value, { ok: true });
});

test("reports non-zero exits without mixing stderr into JSON", async () => {
  await assert.rejects(
    runText(process.execPath, ["-e", "process.stderr.write('failed');process.exit(7)"], options),
    (error: unknown) => error instanceof CLIError && error.code === 7 && error.stderr === "failed"
  );
});

test("rejects malformed and oversized output", async () => {
  await assert.rejects(
    runJSON(process.execPath, ["-e", "process.stdout.write('not-json')"], options),
    /invalid JSON/
  );
  await assert.rejects(
    runText(process.execPath, ["-e", "process.stdout.write('x'.repeat(100))"], {
      ...options,
      maxOutputBytes: 10
    }),
    /output limit/
  );
});

test("times out bounded commands", async () => {
  await assert.rejects(
    runText(process.execPath, ["-e", "setInterval(()=>{}, 1000)"], {
      ...options,
      timeoutMs: 20
    }),
    /timed out/
  );
});
