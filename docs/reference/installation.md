# Installation and release verification

## Supported binaries

Each 2.x release contains both `agefreighter` and `agefreighter-tools` for:

| Operating system | Architecture | Archive |
|---|---|---|
| macOS | amd64 | `agefreighter_vVERSION_darwin_amd64.tar.gz` |
| macOS | arm64 | `agefreighter_vVERSION_darwin_arm64.tar.gz` |
| Linux | amd64 | `agefreighter_vVERSION_linux_amd64.tar.gz` |
| Linux | arm64 | `agefreighter_vVERSION_linux_arm64.tar.gz` |
| Windows | amd64 | `agefreighter_vVERSION_windows_amd64.zip` |

`vVERSION` includes the leading `v`, for example `v2.0.0`.

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

The release workflow uses short-lived GitHub OIDC identity. It does not use or
store a signing key. Verify both the checksum and provenance before
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

Every release includes a generated, checksum-bound `agefreighter.rb`:

```sh
brew install --formula ./agefreighter.rb
```

The formula installs and tests both executables. The repository stores only the
generator, not a placeholder formula that could be installed without valid
release checksums.

## Build from source

```sh
git clone https://github.com/rioriost/agefreighter.git
cd agefreighter
git checkout v2.0.0
make build VERSION=2.0.0
```

Source builds require the Go version declared in `go.mod`. Database services
are not required to compile the binaries.
