# VS Code extension

AGEFreighter includes an open-source VS Code extension in
[`extensions/vscode`](../../extensions/vscode). It is a presentation and
orchestration layer for a separately installed AGEFreighter CLI, not a second
migration engine.

## Guided source setup (2.4.0)

Run **AGEFreighter: New Guided Migration** to enter a Neo4j endpoint and source
placement in a form. The extension uses the Azure account already signed into
VS Code, stores the Neo4j password outside the workspace through SecretStorage,
generates a protected draft, and runs CLI validation and bounded profiling.
The operator signs in before opening the workflow; the extension has no
secondary Azure login action.

For an Azure-hosted source, select the target subscription and provide the
source ARM resource ID. Its region and logical availability zone are accepted
only when Azure Resource Manager verifies them. For an on-premises or other
cloud source, enter its physical location; the eventual nearest-region value is
a recommendation that the operator must confirm. The extension then checks
PostgreSQL 18 capabilities, PostgreSQL and Compute quotas, zonal VM SKUs, and
bounded USD retail rates and saves a 24-hour proposal without deploying it.

The source profile preserves lower-bound and incomplete status. Azure resource
creation stays disabled until sizing totals, SKU/quota/AGE availability, cost,
network reachability, and an ARM what-if result have passed their review gates.
See the [guided migration contract](vscode-guided-migration-contract.md).

This development build ends at the proposal. Guided Azure deployment, automatic
migration, and final verification are still being implemented. The generated
draft is kept out of the existing executable-job list. Chat `/help` and
**AGEFreighter: Open Documentation** describe the installed build's workflow;
the documentation command opens its bundled guide.

## Install

Install the CLI first. The guided path requires the 2.4.0 `inventory` command;
the existing LoadJob-first features remain compatible with 2.3.0 or newer.
Then install **AGEFreighter** from the Visual Studio Marketplace, or install a
reviewed local package:

```sh
code --install-extension agefreighter-2.4.0.vsix
```

If the CLI is not on `PATH`, run **AGEFreighter: Select CLI Binary**. In Remote
SSH, Dev Containers, or Codespaces, install and select the binary in the remote
extension-host environment.

## Deterministic workflow

The existing Migration Jobs view discovers workspace YAML and JSON documents with
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
AGEFreighter 2.4.0 Windows CLI binaries remain unsigned because the SignPath
Foundation application was not approved. They are still distributed; verify
the release checksum and GitHub provenance before selecting the executable.
