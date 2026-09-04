# AGEFreighter 2.3.0 VS Code extension delivery plan

Status: reviewed and approved for implementation on the `2.3.0` branch  
Target: Visual Studio Marketplace public listing  
Extension source: `extensions/vscode`  
Extension identifier: `rioriost.agefreighter`

## Outcome

AGEFreighter 2.3.0 adds an open-source VS Code extension that guides graph
migrations without moving deterministic migration logic out of the Go CLI.
The extension makes project setup, validation, planning, execution, recovery,
verification, and report review accessible from VS Code. When a VS Code chat
model is available, it also supplies an `@agefreighter` participant and
read-only language-model tools for interpreting bounded, redacted evidence.

The Marketplace release is not complete until a clean VSIX installs, activates,
passes its smoke tests, is published under the expected publisher, is publicly
visible, and its public page and installation command are verified.

## Architecture and trust boundary

```text
User / VS Code UI / optional chat model
                  |
                  v
       VS Code extension (TypeScript)
       - gathers paths and intent
       - renders bounded JSON reports
       - asks for confirmation before mutations
       - never reads or expands secret references
                  |
         typed child-process adapter
                  |
                  v
       agefreighter 2.3.0 CLI
       - validates configuration
       - connects to source and target
       - owns checkpoints and job identity
       - performs every migration mutation
       - produces deterministic evidence
```

The model may explain evidence or propose a next action, but it cannot invent a
job ID, alter a migration configuration, execute a mutating command, or bypass
the extension's explicit confirmation. Credentials stay in the existing CLI
environment/file-reference model and are never added to a model prompt,
telemetry event, log, or Marketplace asset.

## Scope

### Included in 2.3.0

- A repository-local TypeScript VS Code extension.
- Workspace discovery for AGEFreighter YAML and JSON job files.
- Commands for binary selection, job selection, validate, plan, profile,
  doctor, load, resume, status, verify, report, and optimize.
- A tree view showing configured jobs and their available operations.
- A report viewer using VS Code-native webview content with strict escaping and
  no scripts or remote content.
- A terminal-backed path for long-running mutating operations so users can see
  output and retain control when VS Code reloads.
- JSON output for CLI validation while preserving the existing text default.
- Optional `@agefreighter` chat participant and read-only language-model tools
  for validation, static plans, profiles, status, reports, and diagnostics.
- Explicit confirmation for load, resume, verification that may query the
  target, optimizer `ANALYZE`, and cleanup. No model tool performs these writes.
- Unit tests, extension-host smoke tests, packaging checks, security/privacy
  documentation, release notes, and CI.
- A packaged VSIX and a public Visual Studio Marketplace listing.

### Deferred

- An MCP server or independent hosted control plane.
- Bundling platform-specific AGEFreighter binaries in the VSIX.
- Editing migration mappings with a model without a visible diff and approval.
- Automatic background resume, unattended fault recovery, or secret storage.
- Marketplace release automation requiring a long-lived PAT. The first release
  is published interactively; later automation should use Microsoft Entra ID.
- Windows Authenticode signing. The extension documents the same unsigned
  Windows CLI status as the 2.2.0 release.

## Milestones

### M0 — plan, review, and release contract

Deliver this document, verify current VS Code and Marketplace requirements,
reserve the extension identifier, and record the publisher/authentication
preconditions. Review the plan against the existing CLI contracts, the
production-simulation recovery findings, Marketplace security constraints, and
the optional nature of AI.

Acceptance:

- The plan has explicit scope, threat boundaries, release gates, and rollback.
- No implementation begins before the review findings below are incorporated.

### M1 — stable CLI integration contract

Add schema-versioned JSON output to `agefreighter validate`, document the
extension-facing command contract, and add Go tests proving stdout is valid JSON
and errors remain on stderr/non-zero exit. The extension must accept additional
JSON fields for forward compatibility and reject malformed or oversized output.

Acceptance:

- Existing `agefreighter validate JOB` output remains unchanged.
- `agefreighter validate --format json JOB` is deterministic and tested.
- No credential value or credential-reference name appears in the result.

### M2 — deterministic VS Code core

Create the extension manifest, TypeScript build, configuration, job discovery,
tree view, command adapter, output channel, terminal execution, and report
viewer. Read-only commands use cancellable child processes with bounded stdout,
timeouts, and an explicitly constructed environment. Mutating commands are
shown in a terminal and require a VS Code confirmation dialog.

Acceptance:

- The binary path is configurable and its version can be checked.
- YAML/JSON jobs are found without traversing ignored dependency/build trees.
- Paths with spaces and shell metacharacters are passed as argument arrays, not
  interpolated shell commands.
- Load/resume/cleanup cannot run without direct user confirmation.
- Reports are rendered from escaped text with no script execution.

### M3 — optional AI assistance

Register `@agefreighter` and read-only language-model tools. Feed only bounded
CLI results and a minimal migration-domain instruction to the user-selected VS
Code model. Keep deterministic intent routing for slash commands and gracefully
fall back to the command palette when no model is available.

Acceptance:

- The extension has useful non-AI functionality with no Copilot dependency.
- Model tools are read-only and declare strict JSON input schemas.
- Prompts exclude environment variables, raw configuration, queries, source
  values, credentials, and unbounded logs.
- Prompt construction and response handling are unit tested without a live
  model; CI never spends model quota.

### M4 — quality, security, and user documentation

Add unit tests for argument construction, discovery, redaction, output limits,
webview escaping, and failure classification. Add an Extension Host smoke test
for activation and command registration. Document installation, prerequisites,
privacy, trust, recovery, supported platforms, and unsigned Windows binaries.

Acceptance:

- Go checks and extension lint/type/unit tests pass.
- The VSIX contents contain no test fixtures, credentials, local paths, or
  unnecessary source/build artifacts.
- Marketplace README, CHANGELOG, LICENSE, icon, categories, keywords,
  repository, issue tracker, and privacy/security disclosures are complete.

### M5 — CI, packaging, and release readiness

Add a GitHub Actions workflow that tests the extension on supported desktop
platforms, packages the VSIX, and uploads it as an artifact. Build locally,
inspect the manifest and archive contents, install in an isolated Extension
Host, and run the Marketplace packaging validation.

Acceptance:

- The package is reproducible from the committed lockfile.
- `vsce package` succeeds without warnings that affect trust or listing quality.
- A clean installation activates against an AGEFreighter 2.3.0 binary.
- The extension workflow and existing repository checks pass on the pushed
  branch.

### M6 — Marketplace registration and public release

Create or select the `rioriost` publisher, accept the Publisher Agreement,
publish the reviewed VSIX, and verify the public listing. If the exact publisher
ID is unavailable, update the manifest and links in one reviewed commit rather
than impersonating or approximating the identity.

Acceptance:

- `publisher.extension` matches the public listing exactly.
- The Marketplace page is public, the version is 2.3.0, and installation from
  the Marketplace succeeds.
- Listing links resolve to the repository, issues, license, privacy/security
  documentation, and release notes.
- The published VSIX checksum and public URL are recorded in the release notes.

### M7 — repository release

Update the root documentation and version references, merge the reviewed branch
through the repository's normal protected-branch path, tag `v2.3.0`, create the
GitHub release, and verify its existing signed/notarized platform artifacts.
The Marketplace listing must reference the final documentation and must not
claim Windows Authenticode signing.

Acceptance:

- Required GitHub checks pass and the branch is merged without bypassing branch
  protection.
- `v2.3.0` release artifacts, checksums, provenance, macOS notarization, and
  Homebrew update complete under the existing release workflow.
- Windows binaries remain available and are explicitly documented as unsigned.

## Test matrix

| Layer | Required evidence |
|---|---|
| Go CLI | JSON validation contract and existing unit/race/coverage gates |
| TypeScript unit | adapter arguments, bounds, errors, discovery, escaping, prompts |
| VS Code host | activation, commands, view registration, configuration |
| Packaging | manifest validation, VSIX allowlist inspection, license and icon |
| Platforms | macOS, Linux, and Windows extension build/smoke; CLI invocation mocked where necessary |
| Manual integration | validate/plan/profile against a safe sample; confirmed terminal launch for mutations |
| Marketplace | public page, install, activate, uninstall/reinstall, links and version |

Database integration remains owned by the existing AGEFreighter test matrix.
The extension does not duplicate production-scale qualification or connect to a
real database during Marketplace packaging tests.

## Threat model and safeguards

- **Command injection:** use executable plus argument arrays and never a shell
  for read-only commands; terminal commands use VS Code terminal shell
  integration with platform-specific quoting tested for adversarial paths.
- **Secret disclosure:** never parse referenced secret values, do not send job
  files or process environments to a model, and redact CLI errors before model
  use or rendering.
- **Prompt injection from source data:** only documented bounded reports may be
  provided to a model; source rows and queries are excluded by CLI contract.
- **Unintended mutation:** keep model tools read-only and require a direct modal
  user confirmation before every mutating command.
- **Resource exhaustion:** cap stdout/stderr, enforce read-only command timeouts,
  support cancellation, and use terminals for long-running work.
- **Webview injection:** escape all dynamic text, forbid scripts and remote
  resources, and apply a restrictive content security policy.
- **Supply chain:** lock npm dependencies, minimize runtime dependencies, scan
  the packed file list, and publish the checksum of the exact VSIX.

## Rollback and incident response

- Stop rollout before publication if any release gate fails.
- If a published version is defective, publish a corrected patch; unpublish only
  for a security or legal incident because existing installations depend on the
  identifier.
- Disable AI contributions in a patch without disabling deterministic commands
  if a model API regression occurs.
- A failed terminal migration follows the CLI's durable status/report/resume
  procedure; the extension never deletes checkpoints automatically.

## Plan review

The initial concept was technically feasible but too broad in four places. The
review made the following binding corrections:

1. **No duplicated migration engine.** A webview-native workflow engine would
   create a second source of truth. All state and mutations remain in the Go
   CLI; the extension is an adapter and presentation layer.
2. **No AI write tools.** Model confirmation UI is not a substitute for a user
   decision. Model-visible tools are read-only, while load/resume/cleanup and
   optimizer mutation are explicit commands with modal confirmation.
3. **No bundled binaries in the first release.** Bundling multiplies notarized,
   signed, and unsigned platform artifact policy inside the VSIX. Version 2.3.0
   locates a separately installed CLI and gives actionable installation help.
4. **No PAT-first automation.** Marketplace global PAT retirement is imminent.
   The first publication uses an interactive account; future CI publication
   must use Microsoft Entra ID after the publisher identity is proven.

Two usability gaps were also added to scope: long-running mutations must be
terminal-visible and reload-tolerant, and AI absence must never disable the
deterministic workflow. With those corrections, the plan is approved for M1.

## Current platform references

- [VS Code AI extensibility overview](https://code.visualstudio.com/api/extension-guides/ai/ai-extensibility-overview)
- [Language model tools](https://code.visualstudio.com/api/extension-guides/ai/tools)
- [Chat participant API](https://code.visualstudio.com/api/extension-guides/ai/chat)
- [Language Model API](https://code.visualstudio.com/api/extension-guides/ai/language-model)
- [Publishing VS Code extensions](https://code.visualstudio.com/api/working-with-extensions/publishing-extension)
- [Continuous integration for extensions](https://code.visualstudio.com/api/working-with-extensions/continuous-integration)
- [Visual Studio Marketplace publisher creation](https://learn.microsoft.com/azure/devops/extend/publish/overview)
