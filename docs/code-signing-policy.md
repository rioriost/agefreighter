# Code signing policy

## Scope and provider

Official Windows release archives contain `agefreighter.exe` and
`agefreighter-tools.exe`. Both executables are built from this repository by
GitHub Actions and submitted to SignPath before checksums, SBOMs, provenance,
and the GitHub release are created.

Free code signing provided by SignPath.io, certificate by SignPath Foundation.
The Authenticode publisher shown by Windows is SignPath Foundation.

## Trusted build and release rules

- Only version-tag builds from `rioriost/agefreighter` may request a production
  signature.
- The source checkout, Windows cross-build, and SignPath submission run only on
  GitHub-hosted runners.
- The raw unsigned ZIP is uploaded by GitHub Actions and referenced by its
  immutable artifact ID. Locally supplied binaries are not accepted.
- Both executables carry the product name `agefreighter`, the release version,
  their expected original filename, and a command-specific description.
- Every production signing request requires manual approval in SignPath.
- The signed ZIP is verified on a separate Windows runner with Authenticode
  policy before it can reach checksum, SBOM, provenance, or publication jobs.

## Team roles

- Committer and reviewer: [Rio Fujita (`rioriost`)](https://github.com/rioriost)
- Signing approver: [Rio Fujita (`rioriost`)](https://github.com/rioriost)

Multi-factor authentication is required for GitHub and SignPath access. Changes
to the release workflow or this policy are reviewed as security-sensitive
release changes.

## Privacy

agefreighter does not transfer information to networked systems unless the
user or operator specifically configures and requests that transfer. Source,
target, observability, and optional AI-service endpoints are selected by the
operator. Credentials are supplied externally and are not embedded in release
artifacts.

## Installation and removal

The project distributes portable command-line executables and does not install
services or modify Windows system configuration. Installation consists of
extracting the ZIP and optionally adding its directory to `PATH`. To uninstall,
remove that `PATH` entry and delete the extracted directory. Migration output
or operator-created configuration is not removed automatically.
