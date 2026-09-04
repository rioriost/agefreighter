import * as path from "node:path";
import * as vscode from "vscode";
import {
  isConnectedReadOperation,
  isReadOperation,
  ReadOperation,
  requiresJobId,
  validateJobId
} from "./core/operations";
import { redactForModel } from "./core/security";
import { ExtensionController } from "./controller";
import { JobsProvider } from "./jobsView";

interface ToolInput {
  operation: string;
  job: string;
  jobId?: string;
}

class ReadEvidenceTool implements vscode.LanguageModelTool<ToolInput> {
  public constructor(private readonly controller: ExtensionController) {}

  public async prepareInvocation(
    options: vscode.LanguageModelToolInvocationPrepareOptions<ToolInput>,
    _token: vscode.CancellationToken
  ): Promise<vscode.PreparedToolInvocation> {
    const operation = isReadOperation(options.input.operation)
      ? options.input.operation
      : "read";
    const connection = isReadOperation(operation) && isConnectedReadOperation(operation)
      ? " This opens the source or target connection configured by the selected job."
      : "";
    return {
      invocationMessage: `Reading AGEFreighter ${operation} evidence`,
      confirmationMessages: {
        title: `Run read-only AGEFreighter ${operation}?`,
        message: new vscode.MarkdownString(
          `Run the read-only \`${operation}\` operation for \`${path.basename(options.input.job)}\`?${connection}`
        )
      }
    };
  }

  public async invoke(
    options: vscode.LanguageModelToolInvocationOptions<ToolInput>,
    _token: vscode.CancellationToken
  ): Promise<vscode.LanguageModelToolResult> {
    if (!isReadOperation(options.input.operation)) {
      throw new Error("Unsupported operation. Use validate, plan, profile, doctor, status, report, or optimize.");
    }
    const operation = options.input.operation;
    const jobId = requiresJobId(operation) ? validateJobId(options.input.jobId) : undefined;
    const uri = await this.controller.resolveToolJob(options.input.job);
    const result = await this.controller.runEvidence(operation, uri, jobId, {
      confirmConnection: false,
      showPanel: false
    });
    if (result === undefined) {
      throw new Error("AGEFreighter evidence was not produced.");
    }
    return new vscode.LanguageModelToolResult([
      new vscode.LanguageModelTextPart(JSON.stringify(redactForModel(result)))
    ]);
  }
}

export function registerAI(
  context: vscode.ExtensionContext,
  controller: ExtensionController,
  jobs: JobsProvider
): void {
  context.subscriptions.push(
    vscode.lm.registerTool("agefreighter_read", new ReadEvidenceTool(controller))
  );

  const participant = vscode.chat.createChatParticipant(
    "agefreighter.chat",
    async (request, _chatContext, stream, token) => {
      if (request.command === "help") {
        stream.markdown(helpText());
        return;
      }

      const operation = commandOperation(request.command);
      if (operation) {
        const uri = await controller.pickJob();
        if (!uri) {
          stream.markdown("No migration job was selected.");
          return;
        }
        const jobId = requiresJobId(operation) ? await controller.promptJobId() : undefined;
        if (requiresJobId(operation) && !jobId) {
          stream.markdown("A durable job ID is required for that operation.");
          return;
        }
        stream.progress(`Running read-only AGEFreighter ${operation}…`);
        const result = await controller.runEvidence(operation, uri, jobId, {
          confirmConnection: true,
          showPanel: false
        });
        if (result === undefined) {
          stream.markdown("The operation was cancelled.");
          return;
        }
        const evidence = JSON.stringify(redactForModel(result), null, 2);
        stream.reference(uri);
        await explainWithModel(
          request,
          stream,
          token,
          `Explain this bounded AGEFreighter ${operation} evidence. Distinguish verified facts, warnings, and recommended next actions. Do not claim that a migration was executed unless the evidence says so.\n\n${evidence}`
        );
        return;
      }

      const discovered = await jobs.getJobs();
      const inventory = discovered.map((job) => job.label?.toString() ?? path.basename(job.uri.fsPath));
      await explainWithModel(
        request,
        stream,
        token,
        `You are the AGEFreighter migration assistant inside VS Code. Deterministic work is performed only by the AGEFreighter CLI. You may explain, compare, and recommend, but never claim that a command ran or a target changed. Do not request credential values. Available workspace migration jobs: ${JSON.stringify(inventory)}. User request: ${request.prompt}`
      );
    }
  );
  participant.iconPath = vscode.Uri.joinPath(context.extensionUri, "images", "icon.png");
  context.subscriptions.push(participant);
}

function commandOperation(command: string | undefined): ReadOperation | undefined {
  return isReadOperation(command) && command !== "optimize" ? command : undefined;
}

async function explainWithModel(
  request: vscode.ChatRequest,
  stream: vscode.ChatResponseStream,
  token: vscode.CancellationToken,
  prompt: string
): Promise<void> {
  try {
    const response = await request.model.sendRequest(
      [vscode.LanguageModelChatMessage.User(prompt)],
      {},
      token
    );
    for await (const fragment of response.text) {
      stream.markdown(fragment);
    }
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    stream.markdown(`The selected chat model could not answer: ${detail}\n\n${helpText()}`);
  }
}

function helpText(): string {
  return [
    "AGEFreighter keeps migration execution deterministic in the CLI.",
    "",
    "Use `/validate` and `/plan` before connecting to a database. Use `/profile` and `/doctor` for bounded source and target evidence. `/status` and `/report` require the durable job UUID.",
    "",
    "Start, resume, verify, and cleanup are intentionally available only as explicit commands from the AGEFreighter view or Command Palette. They always require direct confirmation and run in a visible terminal."
  ].join("\n");
}
