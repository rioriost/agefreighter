#!/bin/bash
set -euo pipefail
umask 077

commit="${CANDIDATE_COMMIT:?exact candidate required}"
lease="${QUALIFICATION_LEASE_ID:?exact lease required}"
[[ "$commit" =~ ^[a-f0-9]{40}$ ]]
[[ "$lease" =~ ^[a-f0-9]{8}(-[a-f0-9]{4}){3}-[a-f0-9]{12}$ ]]
[[ "${AGEFREIGHTER_HORIZON_P1_REFRESH_HOST:?}" =~ ^afhs-20261118-pg17\.rw\.[a-z0-9]+\.japaneast\.horizondb\.azure\.com$ ]]
[[ "${AGEFREIGHTER_HORIZON_P1_REFRESH_CLIENT_ID:?}" =~ ^[a-f0-9]{8}(-[a-f0-9]{4}){3}-[a-f0-9]{12}$ ]]
[[ "${QUALIFICATION_EVIDENCE_ACCOUNT:?}" =~ ^[a-z0-9]{3,24}$ ]]
test "${AGEFREIGHTER_HORIZON_P1_REFRESH_USER:?}" = afhs-20261118-p1-csv
test "$(hostname -s)" = horizon-tiny
test -r /proc/swaps
test "$(wc -l </proc/swaps)" -eq 1
test "$(df --output=pcent /var/lib/agefreighter-horizon-tiny | tail -1 | tr -dc 0-9)" -lt 80

root="/var/lib/agefreighter-horizon-tiny/p1-renewal-$lease"
original=/opt/agefreighter-horizon-tiny
image='golang@sha256:205c7636d74e0ffbe3d007162df7d74f859f5a8c8322dfb9f907977686c126cf'
mkdir -m 0700 "$root"
mkdir -m 0700 "$root/build"
if ! timeout --kill-after=15s 12m bash -s -- "$original" "$root" "$commit" "$image" >"$root/build.log" 2>&1 <<'SH'
set -euo pipefail
git -C "$1" fetch --quiet https://github.com/rioriost/agefreighter.git refs/heads/feat/horizondb-scale-qualification
test "$(git -C "$1" rev-parse FETCH_HEAD)" = "$3"
git -C "$1" worktree add --quiet --detach "$2/candidate" "$3"
test "$(git -C "$2/candidate" rev-parse HEAD)" = "$3"
docker run --rm --network host \
  --mount type=bind,source="$2/candidate",target=/workspace,readonly \
  --mount type=bind,source="$2/build",target=/output \
  --mount type=volume,source=ht-go-mod,target=/go/pkg/mod \
  --mount type=volume,source=ht-go-build,target=/root/.cache/go-build \
  --workdir /workspace --env GOTOOLCHAIN=local --env CGO_ENABLED=0 "$4" \
  go test -trimpath -buildvcs=false -c -o /output/renewal.test ./internal/app
git -C "$1" worktree remove "$2/candidate"
SH
then
  echo 'P1_RENEWAL_BUILD_FAILED'
  exit 1
fi

export AGEFREIGHTER_REQUIRE_HORIZON_P1_REFRESH=1
status=0
"$root/build/renewal.test" -test.v -test.timeout=80m \
    -test.run '^TestHorizonDBP1ManagedIdentityRenewalIntegration$' >"$root/renewal.log" 2>&1 || status=$?
python3 - "$root" "$commit" "$lease" "$status" <<'PY'
import datetime as dt
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import sys
import urllib.request

root = Path(sys.argv[1])
log = (root / "renewal.log").read_bytes()
if not 0 < len(log) <= 65536 or re.search(rb"postgres(?:ql)?://|(?i:Bearer)\s+\S+|eyJ[a-zA-Z0-9_-]+\.", log):
    raise SystemExit("Malformed or credential-shaped probe output refused")
verify = re.search(rb"fresh pooled connection verified after initial credential expiration; replacement expires "
                   rb"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)", log)
passed = (sys.argv[4] == "0" and verify is not None
          and b"--- PASS: TestHorizonDBP1ManagedIdentityRenewalIntegration" in log and b"--- SKIP" not in log)
with (root / "build" / "renewal.test").open("rb") as binary:
    binary_hash = hashlib.file_digest(binary, "sha256").hexdigest()
proof = {
    "candidateCommit": sys.argv[2], "leaseId": sys.argv[3], "executableSha256": binary_hash,
    "writerHost": os.environ["AGEFREIGHTER_HORIZON_P1_REFRESH_HOST"],
    "identityClientId": os.environ["AGEFREIGHTER_HORIZON_P1_REFRESH_CLIENT_ID"],
    "initialExpiryUtc": os.environ["AGEFREIGHTER_HORIZON_P1_REFRESH_INITIAL_EXPIRY_UTC"],
    "logSha256": hashlib.sha256(log).hexdigest(),
    "outcome": "passed" if passed else "failed",
    "coverage": "live same-identity renewal only, no migration",
}
if passed:
    proof.update(replacementExpiryUtc=verify[1].decode("ascii"), verifiedTLS=True, readOnly=True, newBackend=True)
with (root / "renewal-proof.json").open("x") as f:
    json.dump(proof, f, sort_keys=True)
    f.write("\n")

host = os.environ["QUALIFICATION_EVIDENCE_ACCOUNT"] + ".blob.core.windows.net"
addresses = {r[4][0] for r in socket.getaddrinfo(host, 443, socket.AF_INET)}
if not addresses or any(ipaddress.ip_address(a) not in ipaddress.ip_network("10.86.0.0/16") for a in addresses):
    raise SystemExit("Renewal evidence storage must resolve inside the approved VNet")

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError("Authenticated request redirect refused")

http = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
imds = ("http://169.254.169.254/metadata/identity/oauth2/token?"
        "api-version=2018-02-01&resource=https%3A%2F%2Fstorage.azure.com%2F")
with http.open(urllib.request.Request(imds, headers={"Metadata": "true"}), timeout=15) as result:
    token = json.load(result)["access_token"]
blob = "afhs-20261118/p1-csv/" + sys.argv[3] + "/credential-renewal.json"
url = "https://" + host + "/qualification/" + blob
headers = {"Authorization": "Bearer " + token, "x-ms-version": "2023-11-03",
           "x-ms-date": dt.datetime.now(dt.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")}
body = json.dumps({"proof": proof, "log": log.decode("utf-8")}, sort_keys=True).encode()
with http.open(urllib.request.Request(url, data=body, method="PUT", headers={
    **headers, "x-ms-blob-type": "BlockBlob", "If-None-Match": "*", "Content-Type": "application/json",
}), timeout=20) as result:
    if result.status != 201:
        raise SystemExit("Private evidence was not newly created")
digest = hashlib.sha256(body).hexdigest()
with http.open(urllib.request.Request(url, headers=headers), timeout=20) as result:
    if hashlib.sha256(result.read(len(body) + 1)).hexdigest() != digest:
        raise SystemExit("Private renewal evidence SHA-256 readback differs")
proof.update(blob=blob, evidenceSha256=digest, privateReadback=True)
print("P1_RENEWAL_PROOF " + json.dumps(proof, sort_keys=True))
if not passed:
    for line in log.decode().splitlines():
        if re.fullmatch(r"\s*target_connection_test.go:\d+: [A-Za-z0-9 ,_-]+", line):
            print(line)
    raise SystemExit("Live same-identity renewal did not pass")
PY
