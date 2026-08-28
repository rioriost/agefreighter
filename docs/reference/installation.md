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

`vVERSION` includes the leading `v`, for example `v2.0.0`.

Prereleases published while SignPath Foundation enrollment is pending may omit
the Windows archive. Official Windows binaries are never published unsigned;
check the assets and notes for the release you are installing.

## Verify a release

Download the archive, its `.spdx.json` SBOM, `checksums.txt`, and the
`agefreighter-vVERSION.intoto.jsonl` attestation bundle from the same GitHub
release.

Verify the archive checksum:

```sh
grep 'agefreighter_v2.0.0_darwin_arm64.tar.gz$' checksums.txt |
  shasum -a 256 -c -
```

On Linux, use `sha256sum -c` instead. Verify GitHub's keyless build-provenance
attestation with GitHub CLI:

```sh
gh attestation verify agefreighter_v2.0.0_darwin_arm64.tar.gz \
  --repo rioriost/agefreighter
```

The release workflow uses short-lived GitHub OIDC identity for provenance.
macOS binaries are additionally signed with a Developer ID Application
certificate, hardened-runtime enabled, securely timestamped, and accepted by
Apple's notarization service before checksums and attestations are generated.
Windows executables are signed with a publicly trusted Authenticode certificate
provided by SignPath Foundation and an RFC 3161 timestamp, then verified with
Windows Authenticode policy before the ZIP, SBOM, checksums, and attestations
are generated. See the [code signing policy](../code-signing-policy.md). Apple
credentials and the SignPath submission token are restricted to the protected
`release` environment. Verify both the checksum and provenance before
installation.

## Install an archive

macOS and Linux:

```sh
tar -xzf agefreighter_v2.0.0_darwin_arm64.tar.gz
install -m 0755 agefreighter agefreighter-tools /usr/local/bin/
agefreighter version
agefreighter-tools version
```

For Windows, extract the zip and place `agefreighter.exe` and
`agefreighter-tools.exe` in a directory on `PATH`.

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
git checkout v2.0.0
make build VERSION=2.0.0
```

Source builds require the Go version declared in `go.mod`. Database services
are not required to compile the binaries.

Existing 1.x installations are not upgraded in place. Follow the
[1.x to 2.0 migration guide](../migration-1.x-to-2.0.md), validate a new v2 job,
and complete a trial migration before replacing the 1.x command.
