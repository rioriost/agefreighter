# Security policy

## Supported release lines

| Release line | Status |
|---|---|
| 2.x | Current feature and security maintenance on `main` |
| 1.x | Security and critical maintenance on `release/1.x` |
| 2.x prereleases | Evaluation only; replace with the next RC or stable release |

Security fixes are released from the affected maintenance line. This policy
does not promise a fixed support end date; any retirement will be announced in
release notes before support ends.

## Reporting a vulnerability

Use GitHub's **Report a vulnerability** form in the repository Security tab.
That creates a private vulnerability report visible only to the maintainer and
the reporter. Do not open a public issue for a suspected vulnerability.

Include the affected version and platform, impact, reproduction steps or a
proof of concept, and any known mitigations. Remove credentials, personal data,
database contents, and other secrets from the report.

The maintainer will acknowledge the report, investigate affected release
lines, coordinate a fix and advisory, and credit the reporter when requested.
Public disclosure should wait until a fixed release or an agreed disclosure
date.

## Release security

Release archives include checksums, SPDX SBOMs, third-party license notices,
and GitHub build-provenance attestations. macOS binaries are Developer ID
signed and notarized. Windows binaries are published only after Authenticode
signing succeeds. Follow the [installation verification procedure](docs/reference/installation.md)
before installing a release.
