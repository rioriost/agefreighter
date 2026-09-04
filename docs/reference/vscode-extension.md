# VS Code extension

AGEFreighter 2.3.0 includes an open-source VS Code extension in
[`extensions/vscode`](../../extensions/vscode). It is a presentation and
orchestration layer for a separately installed AGEFreighter CLI, not a second
migration engine.

## Install

Install the CLI first and verify `agefreighter version` reports 2.3.0 or newer.
Then install **AGEFreighter** from the Visual Studio Marketplace, or install a
reviewed local package:

```sh
code --install-extension agefreighter-2.3.0.vsix
```

If the CLI is not on `PATH`, run **AGEFreighter: Select CLI Binary**. In Remote
SSH, Dev Containers, or Codespaces, install and select the binary in the remote
extension-host environment.

## Deterministic workflow

The Migration Jobs view discovers workspace YAML and JSON documents with
`apiVersion: agefreighter.io/v2` and `kind: LoadJob`.

1. Validate the job and inspect the static plan. These do not connect to a
   source or target.
2. Run the bounded profile and target doctor after accepting the connection
   notice.
3. Start the migration after reviewing the modal confirmation. The command runs
   in a visible terminal.
4. Retain the durable job UUID. Use it for status, resume, verify, report, and
   cleanup.
5. Review reports locally. Dynamic report content is HTML-escaped and the
   webview has scripts and remote content disabled.

Long-running and mutating commands use a terminal because their lifetime and
output should remain visible to the operator. The extension does not
automatically resume a migration after reload or failure.

## AI assistance

AI is optional. `@agefreighter` uses the model selected by the user in VS Code
to explain bounded CLI evidence. The globally contributed `agefreighter_read`
tool supports only validate, plan, profile, doctor, status, report, and
optimization advice.

The extension does not give a model access to load, resume, cleanup,
verification, or optimizer mutation. VS Code displays a confirmation for every
tool invocation, and connected read operations say that they open a configured
source or target connection.

See the extension's [privacy disclosure](../../extensions/vscode/PRIVACY.md)
and the [CLI integration contract](vscode-extension-cli-contract.md) for the
redaction and process boundaries.

## Build and test

```sh
cd extensions/vscode
npm ci
npm run check
npm run test:host
npm run package
```

The Extension Host test downloads a matching VS Code build. Ordinary CI unit
tests do not call a language model or connect to a database.

The VSIX contains the bundled extension JavaScript, documentation, license,
icons, and manifest only. It does not contain `node_modules`, source fixtures,
the CLI, credentials, or local paths.

## Windows binary status

The extension itself is TypeScript/JavaScript. The separately downloaded
AGEFreighter 2.3.0 Windows CLI binaries remain unsigned because the SignPath
Foundation application was not approved. They are still distributed; verify
the release checksum and GitHub provenance before selecting the executable.
