# Installation and release verification

## Supported binaries

Each stable 2.x release contains `agefreighter`, `agefreighter-tools`, the
project `LICENSE`, and `THIRD_PARTY_NOTICES.txt` for:

| Operating system | Architecture | Archive |
|---|---|---|
| macOS | amd64 | `agefreighter_vVERSION_darwin_amd64.tar.gz` |
| macOS | arm64 | `agefreighter_vVERSION_darwin_arm64.tar.gz` |
| Linux | amd64 | `agefreighter_vVERSION_linux_amd64.tar.gz` |
| Linux | arm64 | `agefreighter_vVERSION_linux_arm64.tar.gz` |
| Windows | amd64 | `agefreighter_vVERSION_windows_amd64.zip` |

`vVERSION` includes the leading `v`, for example `v2.2.0`.

The Windows archive is included in v2.2.0, but its executables are intentionally
unsigned. The SignPath Foundation application was not approved, so
Authenticode signing remains planned for a later release through a future
eligible application or another signing arrangement. Windows may display an
unknown-publisher or SmartScreen warning.

## Verify a release

Download the archive, its `.spdx.json` SBOM, `checksums.txt`, and the
`agefreighter-vVERSION.intoto.jsonl` attestation bundle from the same GitHub
release.

Verify the archive checksum:

```sh
grep 'agefreighter_v2.2.0_darwin_arm64.tar.gz$' checksums.txt |
  shasum -a 256 -c -
```

On Linux, use `sha256sum -c` instead. Verify GitHub's keyless build-provenance
attestation with GitHub CLI:

```sh
gh attestation verify agefreighter_v2.2.0_darwin_arm64.tar.gz \
  --repo rioriost/agefreighter
```

The release workflow uses short-lived GitHub OIDC identity for provenance.
macOS binaries are additionally signed with a Developer ID Application
certificate, hardened-runtime enabled, securely timestamped, and accepted by
Apple's notarization service before checksums and attestations are generated.
The v2.2.0 Windows executables are unsigned. Verify their ZIP checksum and
GitHub build-provenance attestation before installation; these authenticate the
release workflow and archive but do not provide an Authenticode publisher
identity. See the [code signing policy](../code-signing-policy.md). Apple
credentials are restricted to the protected `release` environment.

## Install an archive

macOS and Linux:

```sh
tar -xzf agefreighter_v2.2.0_darwin_arm64.tar.gz
install -m 0755 agefreighter agefreighter-tools /usr/local/bin/
agefreighter version
agefreighter-tools version
```

For Windows, extract the zip and place `agefreighter.exe` and
`agefreighter-tools.exe` in a directory on `PATH`. For v2.2.0,
`Get-AuthenticodeSignature` reports `NotSigned` by design.

## Homebrew

Every release includes a generated, checksum-bound `agefreighter.rb`. After the
GitHub release is published, automation commits that Formula to
[`rioriost/homebrew-cask`](https://github.com/rioriost/homebrew-cask), which
Homebrew resolves as the `rioriost/cask` tap:

```sh
brew install rioriost/cask/agefreighter
```

The Formula installs and tests both executables. The agefreighter repository
stores only the generator, not a placeholder Formula that could be installed
without valid release checksums. A fully qualified install trusts only this
Formula rather than every package in the third-party tap.

## Build from source

```sh
git clone https://github.com/rioriost/agefreighter.git
cd agefreighter
git checkout v2.2.0
make build VERSION=2.2.0
```

Source builds require the Go version declared in `go.mod`. Database services
are not required to compile the binaries.

Existing 1.x installations are not upgraded in place. Follow the
[1.x to 2.0 migration guide](../migration-1.x-to-2.0.md), validate a new v2 job,
and complete a trial migration before replacing the 1.x command.

## Native PostgreSQL property-graph target

The `postgresql-property-graph` target does not require a separate extension,
but it does require the exact PostgreSQL 19 SQL/PGQ build listed in the
[compatibility matrix](compatibility.md). The binary probes the server version
and property-graph DDL before creating metadata or graph objects. PostgreSQL 19
Beta 3 is for evaluation only; installing an agefreighter binary does not make
that pre-release database suitable for production. Requalify against the
published PostgreSQL 19 GA image digest before production use.
