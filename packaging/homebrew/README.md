# Homebrew packaging

Release automation generates `agefreighter.rb` from the signed release
archives and `checksums.txt`:

```sh
scripts/release/generate-formula.sh v2.0.0 OWNER/REPOSITORY dist agefreighter.rb
```

The generator fails unless all four macOS/Linux archives have exactly one
valid SHA-256 entry. The resulting Formula installs both `agefreighter` and
`agefreighter-tools` and tests each binary's `version` command. There is no
checked-in production Formula with placeholder checksums; only generated,
checksum-bound Formulae are publishable.

After a tagged GitHub release succeeds, the release workflow publishes the
generated file as `Formula/agefreighter.rb` in
[`rioriost/homebrew-cask`](https://github.com/rioriost/homebrew-cask). The
repository name maps to the `rioriost/cask` tap, so users install it with:

```sh
brew install rioriost/cask/agefreighter
```

The protected GitHub `release` environment must define:

| Secret | Purpose |
|---|---|
| `APPLE_DEVELOPER_ID_CERTIFICATE_P12` | Base64-encoded Developer ID Application certificate and private key |
| `APPLE_DEVELOPER_ID_CERTIFICATE_PASSWORD` | Password protecting the PKCS#12 file |
| `APPLE_SIGNING_IDENTITY` | Exact `Developer ID Application: ...` identity imported from the certificate |
| `APPLE_API_KEY_P8` | Base64-encoded App Store Connect API private key |
| `APPLE_API_KEY_ID` | App Store Connect API key ID |
| `APPLE_API_ISSUER_ID` | App Store Connect API issuer ID |
| `HOMEBREW_TAP_DEPLOY_KEY` | Private half of a write-enabled deploy key registered only on `rioriost/homebrew-cask` |

macOS archives are built on native GitHub-hosted runners. Both executables are
signed with hardened runtime and a secure timestamp, submitted together to
Apple with `notarytool`, and repackaged without modifying their signed bytes.
The final signed archive is used for its SBOM, checksum, provenance attestation,
and Homebrew Formula.
