import { spawn } from "node:child_process";

export interface ProcessOptions {
  cwd: string;
  timeoutMs: number;
  maxOutputBytes: number;
  env?: NodeJS.ProcessEnv;
  signal?: AbortSignal;
}

export interface ProcessResult {
  stdout: string;
  stderr: string;
}

export class CLIError extends Error {
  public constructor(
    message: string,
    public readonly code?: number,
    public readonly stderr = ""
  ) {
    super(message);
    this.name = "CLIError";
  }
}

export async function runText(
  executable: string,
  args: readonly string[],
  options: ProcessOptions
): Promise<ProcessResult> {
  if (!executable.trim()) {
    throw new Error("AGEFreighter binary path is empty.");
  }
  if (!Number.isSafeInteger(options.maxOutputBytes) || options.maxOutputBytes < 1) {
    throw new Error("maxOutputBytes must be a positive safe integer.");
  }
  if (!Number.isSafeInteger(options.timeoutMs) || options.timeoutMs < 1) {
    throw new Error("timeoutMs must be a positive safe integer.");
  }

  return await new Promise<ProcessResult>((resolve, reject) => {
    let settled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let stdoutBytes = 0;
    let stderrBytes = 0;
    const stdout: Buffer[] = [];
    const stderr: Buffer[] = [];
    const child = spawn(executable, [...args], {
      cwd: options.cwd,
      env: options.env ?? { ...process.env },
      shell: false,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"]
    });

    const finish = (error?: Error, result?: ProcessResult): void => {
      if (settled) {
        return;
      }
      settled = true;
      if (timer) {
        clearTimeout(timer);
      }
      options.signal?.removeEventListener("abort", abort);
      if (error) {
        reject(error);
      } else if (result) {
        resolve(result);
      }
    };

    const terminate = (error: Error): void => {
      child.kill("SIGTERM");
      finish(error);
    };

    const append = (chunks: Buffer[], chunk: Buffer, isStdout: boolean): void => {
      const next = (isStdout ? stdoutBytes : stderrBytes) + chunk.byteLength;
      if (next > options.maxOutputBytes) {
        terminate(new CLIError(
          `AGEFreighter ${isStdout ? "stdout" : "stderr"} exceeded the configured output limit.`
        ));
        return;
      }
      if (isStdout) {
        stdoutBytes = next;
      } else {
        stderrBytes = next;
      }
      chunks.push(chunk);
    };

    child.stdout.on("data", (chunk: Buffer) => append(stdout, chunk, true));
    child.stderr.on("data", (chunk: Buffer) => append(stderr, chunk, false));
    child.on("error", (error) => finish(new CLIError(
      `Unable to start AGEFreighter: ${error.message}`
    )));
    child.on("close", (code, signal) => {
      const result = {
        stdout: Buffer.concat(stdout).toString("utf8"),
        stderr: Buffer.concat(stderr).toString("utf8")
      };
      if (code === 0) {
        finish(undefined, result);
        return;
      }
      const suffix = signal ? `signal ${signal}` : `exit code ${code ?? "unknown"}`;
      finish(new CLIError(`AGEFreighter failed with ${suffix}.`, code ?? undefined, result.stderr));
    });

    const abort = (): void => terminate(new CLIError("AGEFreighter command was cancelled."));
    if (options.signal?.aborted) {
      abort();
      return;
    }
    options.signal?.addEventListener("abort", abort, { once: true });
    timer = setTimeout(() => terminate(new CLIError(
      `AGEFreighter command timed out after ${options.timeoutMs} ms.`
    )), options.timeoutMs);
  });
}

export async function runJSON(
  executable: string,
  args: readonly string[],
  options: ProcessOptions
): Promise<unknown> {
  const result = await runText(executable, args, options);
  try {
    return JSON.parse(result.stdout) as unknown;
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new CLIError(`AGEFreighter returned invalid JSON: ${detail}`);
  }
}
