# B12 fresh-download integration — September 23

Status at11:09UTC: **authorized setup complete; authentication blocked before
either report download**. Neither negative case has passed or failed its report
admission check. No VM/database was started and no source operation was submitted.

## Approved boundary and immutable inputs

The user approved a separately signed-in disposable VS Code profile reading only
the accepted B03 PostgreSQL qualification report from existing workflow storage.
Each case changes the copied expected seal, not the remote Blob. There is no
credential-cache/SecretStorage copy, source read, upload, RBAC/network/tag change
or accepted-record edit.

- Workflow `cb2ef280-a891-4edf-b45b-de75e33eb7b8`.
- Qualification `028ec524-990f-4ed3-8265-6eb024d5cf57`.
- Migration job `d707adf8-7cab-4d73-8e38-ec141e72ac18`.
- Original result:23,224bytes; SHA-256
  `00938de546f94d1fe50e372044bef9b423e104919855fc8d934a06b4b42f77fc`.
- Snapshot SHA-256:
  `58f295e4e6c72a66d31dd90cafdaf3ec761f7a2a89d2e7bcbab988428755590f`.
- Independent six-field guest-export receipt SHA-256:
  `feac8b05ecd8178ab8a91c69a6e98c1abb09dc5cc836952686e7644776a9bda0`.

The independent receipt came from this task's retained successful ARM export
response observed08:56:47UTC (export succeeded08:56:24–08:56:26UTC, exit0), not
from copying the operator record's seal. A fresh ARM read of that stopped guest's
historical command no longer returned output, so the VM was not restarted to
recover it. Private provenance retains the historical response locator; raw
logs, tokens and credentials are not published.

The reviewed helper created two separate absent-before-creation, owner-only
profile directories. Originals are read-only; no report files were copied.

| Case | Expected seal change, besides phase pass → exported | Copied record SHA-256 |
| --- | --- | --- |
| wrong-sha256 | First SHA character `0` → `1`; length unchanged | `5e1f53791b8edd49a0befda6a663011439c354242e56cfba08a53e258b5e548b` |
| wrong-length | Expected length differs by one byte; SHA unchanged | `14ddab0047f428a5973f0e3bf2fe7819f79b6a5bf86273343256dd976f5d5ed5` |

## Native profile setup and authentication boundary

The first case installed the previously approved3bc0069 VSIX, not the in-progress
B09 follow-up. Its package SHA-256 is
`1a3121acbd22f6938f960b058939e62a226355022213d7d84110c742b1d5e507`.
Azure Resources0.13.0 was installed from the official Marketplace. An unmodified
copy of VS Code1.139.0 gives native automation an unambiguous app path; its
executable SHA matches the normal official application:
`1b58953da3281bddea360f21198ab80f8a5caa72c77f7cf996bd3c9fa74dfb3c`.
No workspace trust override or authentication-policy change was made.

The actual Azure view's **Sign in to Azure** action and native Microsoft sign-in
consent were used. The view remained at **Waiting for Azure sign-in**. Sanitized
Microsoft authentication diagnostics recorded broker key error`-34018` and
silent-acquisition interaction/conditional-access categories. This does not prove
which policy or platform condition caused the failure. Interactive user sign-in
was requested; no broker/keychain repair, policy bypass, repeated login attempt,
credential-cache copy or substitute CLI authentication was used.

The AGEFreighter report command has not been invoked. Ownership admission,
read-only report capability issuance and fresh Blob GET are not yet qualified.
Authentication failure is not a report rejection PASS.

At11:08UTC, both copied records retained their initial hashes/`exported` state,
with no additional store files. All91normal operator files retained aggregate
path/mode/content SHA-256
`794d6a7e3c88b12a2b70b0373064bd40f2c746318104d0c82697ea8741551709`.
Fresh read-only Azure observations confirmed the B03 runner deallocated and its
target Stopped. Other stopped guests were not probed or restarted.

## Finite completion criterion and evidence limits

The original plan's invalid-verification classes are counts/digest mismatch,
rejects, missing, truncated, stale, wrong-job and incomplete evidence. Existing
production-controller/real isolated-host tests cover those semantic classes,
and a signed-in retained-failure refusal is already recorded. The remaining
integration check is the normal signed-in native download path refusing both
deliberately wrong expectations with unchanged records, no retained report and
no new PASS. See [retained-host coverage](p1-retained-host-20260922.md).

After those two observed refusals, B12 may be reported complete for this defined
scope using explicitly mixed-layer evidence. It must not be described as all
eight cases executed against Azure. Wrong expected length can be rejected at
Content-Length before body reading; it is not observed network truncation.
Wrong expected SHA is not observed remote corruption. No corrupted remote Blob,
Azure outage or source/SKU Cartesian product is added to the original requirement.
