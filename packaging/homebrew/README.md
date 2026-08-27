# Homebrew packaging

Release automation generates `agefreighter.rb` from the signed release
archives and `checksums.txt`:

```sh
scripts/release/generate-formula.sh v2.0.0 OWNER/REPOSITORY dist agefreighter.rb
```

The generator fails unless all four macOS/Linux archives have exactly one
valid SHA-256 entry. The resulting formula installs both `agefreighter` and
`agefreighter-tools` and tests each binary's `version` command. There is no
checked-in production formula with placeholder checksums; only generated,
checksum-bound formulae are publishable.
