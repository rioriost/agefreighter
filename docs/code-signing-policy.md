# Code signing policy

## Scope and provider

Official Windows release archives contain `agefreighter.exe` and
`agefreighter-tools.exe`. Both executables are built from this repository by
GitHub Actions before checksums, SBOMs, provenance, and the GitHub release are
created.

The Windows executables in v2.0.0 are intentionally unsigned while SignPath
Foundation enrollment is pending. Code signing remains planned for a later
release. Once enabled, signing will be provided by SignPath.io using a
certificate from SignPath Foundation, and the Authenticode publisher shown by
Windows will be SignPath Foundation.

## v2.0.0 unsigned-release controls

- The Windows cross-build runs on a GitHub-hosted runner from the tagged
  repository commit.
- A separate Windows runner confirms that both executables are unsigned and
  executes their version commands before publication.
- The ZIP is covered by the release checksum, SPDX SBOM, and GitHub
  build-provenance attestation.

## Planned signing rules

- Only version-tag builds from `rioriost/agefreighter` may request a production
  signature.
- The source checkout, Windows cross-build, and SignPath submission will run
  only on GitHub-hosted runners.
- The raw unsigned ZIP will be uploaded by GitHub Actions and referenced by its
  immutable artifact ID. Locally supplied binaries will not be accepted.
- Both executables will carry the product name `agefreighter`, the release
  version, their expected original filename, and a command-specific
  description.
- Every production signing request will require manual approval in SignPath.
- The signed ZIP will be verified on a separate Windows runner with
  Authenticode policy before it can reach checksum, SBOM, provenance, or
  publication jobs.

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
