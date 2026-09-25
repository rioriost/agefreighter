# VS Code extension

AGEFreighter includes an open-source VS Code extension in
[`extensions/vscode`](../../extensions/vscode). It presents and orchestrates the
deterministic Go engine; it is not a second migration engine. Guided operations
run on a dedicated Linux Azure VM. Existing LoadJob commands use a separately
installed CLI in the extension-host environment.

## Guided migration (2.4.0)

1. Run **AGEFreighter: New Guided Migration** without first selecting a project
   folder or CLI. Choose CSV, Neo4j, PostgreSQL or Cosmos DB for NoSQL. For
   Neo4j/PostgreSQL, choose Azure, on-premises or another cloud; Cosmos uses
   Azure and CSV uses local files.
2. Reuse the Azure account signed into VS Code. Select the source and reviewed
   runner subscription, existing resource group/subnet, region and zone. ARM
   discovery identifies candidates and placement, not database readiness.
   On-premises/other-cloud endpoints require explicit operator placement and
   existing private connectivity; no automatic peering/firewall change occurs.
3. Save the source form and explicit mappings. Review and approve transfer
   storage/file uploads where needed, then the pinned Linux runner deployment
   after fresh what-if, quota, SKU, network and cost checks. Credentials use a
   native private prompt; workflow-scoped encrypted reuse is optional. No
   source password belongs in the form, YAML, logs or model context.
4. Verify Linux guest readiness and installation. PostgreSQL catalog discovery
   can propose mappings for explicit adoption. Review source settings and
   approve complete source inventory; import its size/hash-verified report.
   Cosmos requires a separately reviewed managed-identity Data Reader grant and
   a source-immutability window. CSV requires full-hash guest file seals.
5. Review a new private PostgreSQL 18/AGE target, non-overlapping delegated
   subnet, same-VM migration size, cumulative budget/reserve and deadline.
   Select an output folder only when saving the reviewed LoadJob and target
   plan. Save-only makes no deployment; target creation is separately approved.
   The target is single-server/HA-off and create-only. An existing VNet may be
   in a separate network resource group, with permissions checked in both groups.
6. Separately approve AGE readiness/restart, idle same-VM resize and migration.
   Durable operation/job IDs precede writes. Lost responses are reconciled by
   ID without automatic replay; active inventory/migration blocks resize.
7. Reconnect to the retained operation, import counts/full verification reports,
   and review complete evidence. Failed jobs require explicit same-job recovery
   inspection and approval; reload does not automatically resume a load.

The operator signs in through VS Code; the extension does not start an
independent Azure CLI login. Neither a provisioned VM, a completed inventory,
a successful report transfer nor a counts pass alone is full migration
qualification. The pinned Linux release must match the extension requirements;
unavailable release artifacts block runner creation. The development-artifact
opt-in is for qualification only and does not publish or attest a release.

See the [full installed guide](../../extensions/vscode/README.md) for exact
buttons, CA/credential handling, bounded polling, report recovery, command
capacity and supported limitations. The [version-2 runner contract](vscode-runner-contract.md)
is active; the [version-1 contract](vscode-guided-migration-contract.md) is
retained as history. **Open Documentation** opens the bundled guide.

The defined P1 qualification matrix has nine base routes and twelve finite
extended branches passing. The [evidence ledger](../../production-simulation/vscode-e2e/remaining-validation.md)
keeps their evidence layers and limits, including endpoint-only network
simulations and mixed-layer negatives. This is not every source/network/auth
combination or production-scale throughput qualification. Packaging,
compatibility and release publication are separate M6 checks.

## Install

Install the matching reviewed release VSIX:

```sh
code --install-extension agefreighter-2.4.0.vsix
```

GitHub release assets and Marketplace publication have separate availability.
Check the actual publication result rather than assuming one implies the other.
Use VS Code 1.105 or newer. The guided workflow needs no local CLI; the advanced
LoadJob workflow needs a compatible installed CLI (2.4.0 is recommended).
PostgreSQL checkpoints predating the 2.3.1 floating-point fix must not be resumed
with the changed fingerprint.

If the advanced CLI is not on `PATH`, run **AGEFreighter: Select CLI Binary**.
In Remote SSH, Dev Containers or Codespaces, advanced commands use a CLI in that
remote extension-host environment. The guided local-file flow and each remote
host topology have their own support/qualification boundaries.

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

## Windows lifecycle limits

The guided cloud worker is Linux. Windows local-lock recovery is unsupported:
the extension cannot obtain the boot identity needed to review an interrupted
lock safely. It preserves that lock and requires manual investigation.
Readiness-control removal also fails closed because this implementation cannot
establish durable POSIX evidence-directory synchronization on Windows. An
archive may remain, but no control DELETE is sent. Ordinary workflow lock
release and read-only reconciliation are separate from these limitations and
from the CLI signing status below.

## Windows binary status

The extension itself is TypeScript/JavaScript. The separately downloaded
AGEFreighter 2.4.0 Windows CLI binaries remain unsigned because the SignPath
Foundation application was not approved. They are still distributed; verify
the release checksum and GitHub provenance before selecting the executable.
