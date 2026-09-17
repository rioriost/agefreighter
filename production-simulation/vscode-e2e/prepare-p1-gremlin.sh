#!/usr/bin/env bash
# Dedicated, create-only guest staging. Run under a bounded systemd unit.
# Never invoke the historical prepare-p1-cosmos.sh against the accepted graph.
set -euo pipefail
umask 077
readonly work=/var/lib/agefreighter-gremlin-p1-20260917
readonly target_container=graph-gremlin-p1-20260917
readonly endpoint=https://afcosmosp120260907.documents.azure.com:443/
test "$(date -u +%s)" -lt 1789888475
test "$(df --output=pcent /var/lib | tail -1 | tr -dc '0-9')" -lt 80
test "$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)" -gt 1048576
test "$(awk 'NR>1{s+=$4}END{print s+0}' /proc/swaps)" -eq 0
if pgrep -x agefreighter >/dev/null || pgrep -f '^/[^ ]*/cosmosfixtureload( |$)' >/dev/null; then
  echo 'Existing loader process; refusing preparation.' >&2
  exit 1
fi
# mkdir without -p is intentional: never overwrite or automatically resume.
mkdir -m 0700 "$work"
mkdir -m 0700 "$work/documents" "$work/evidence"
date -u +%FT%TZ > "$work/evidence/started-at.txt"
cat /proc/sys/kernel/random/boot_id > "$work/evidence/boot-id.txt"

python3 - "$work" <<'PY'
import hashlib, json, os, pathlib, sys, tarfile, urllib.request

work = pathlib.Path(sys.argv[1])
base = ('https://afbd3b66801e184d788f36f4.blob.core.windows.net/'
        'af-bd3b6680-1e18-4d78-8f36-f43467a09a0a/gremlin-p1-20260917/')
archive_hash = 'f01f2044429b3a8cb2f1d123d0b2e41ec20bc1610c022e949ae5ec087219dcc6'
loader_hash = '93570ccebac744503e94775614cc50befb9ef2fea032d79a851a6fa6afd0cfb9'
manifest_hash = '7cb9642cd3c3a85fb6503b727697a3178ed43269156086590eea1ca211936860'

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError('Redirect refused')

imds = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
req = urllib.request.Request(
    'http://169.254.169.254/metadata/identity/oauth2/token'
    '?api-version=2018-02-01&resource=https%3A%2F%2Fstorage.azure.com%2F',
    headers={'Metadata': 'true'})
with imds.open(req, timeout=30) as response:
    token = json.load(response)['access_token']
https = urllib.request.build_opener(NoRedirect())
for name, expected, size in [
    ('p1-gremlin-documents.tar.gz', archive_hash, 246433349),
    ('cosmosfixtureload', loader_hash, 11502980),
]:
    request = urllib.request.Request(base + expected + '/' + name,
        headers={'Authorization': 'Bearer ' + token, 'x-ms-version': '2023-11-03'})
    digest, received = hashlib.sha256(), 0
    partial = work / (name + '.partial')
    with https.open(request, timeout=120) as response, partial.open('xb') as out:
        if int(response.headers.get('Content-Length', '-1')) != size:
            raise RuntimeError('Artifact length mismatch')
        while True:
            chunk = response.read(1048576)
            if not chunk:
                break
            received += len(chunk)
            if received > size:
                raise RuntimeError('Artifact size exceeded')
            digest.update(chunk)
            out.write(chunk)
    if received != size or digest.hexdigest() != expected:
        raise RuntimeError('Artifact integrity mismatch')
    os.link(partial, work / name)  # create-only publication; retain partial evidence
del token

with tarfile.open(work / 'p1-gremlin-documents.tar.gz', 'r:gz') as archive:
    members = archive.getmembers()
    names = [m.name for m in members]
    if len(members) != 19 or len(set(names)) != 19 or any(
        not m.isfile() or '/' in m.name or m.size > 1024**3 for m in members
    ):
        raise RuntimeError('Unexpected archive members')
    member = archive.getmember('portable-manifest.json')
    if member.size > 1024**2:
        raise RuntimeError('Manifest too large')
    raw = archive.extractfile(member).read()
    if hashlib.sha256(raw).hexdigest() != manifest_hash:
        raise RuntimeError('Manifest integrity mismatch')
    manifest = json.loads(raw)
    expected_names = {'portable-manifest.json'} | {t['documents'] for t in manifest['tables']}
    if set(names) != expected_names or len(manifest['tables']) != 18:
        raise RuntimeError('Unexpected fixture files')
    if manifest['documentFormat'] != 'cosmos-gremlin-p1-partition64-v1':
        raise RuntimeError('Wrong document representation')
    for table in manifest['tables']:
        digest, rows = hashlib.sha256(), 0
        with archive.extractfile(table['documents']) as src, (work/'documents'/table['documents']).open('xb') as out:
            while True:
                chunk = src.read(1048576)
                if not chunk:
                    break
                digest.update(chunk)
                rows += chunk.count(b'\n')
                out.write(chunk)
        if digest.hexdigest() != table['documentsSha256'] or rows != table['rows']:
            raise RuntimeError('Document hash/count mismatch')
    with (work/'portable-manifest.json').open('xb') as out:
        out.write(raw)
print('Pinned Gremlin fixture and loader verified; no source writes yet.')
PY

chmod 0500 "$work/cosmosfixtureload"
timeout --signal=TERM --kill-after=30s 7h "$work/cosmosfixtureload" \
  -endpoint "$endpoint" -database p1 -container "$target_container" \
  -input "$work/documents" -workers 96 -require-empty \
  > "$work/evidence/load.json" 2> "$work/evidence/load-progress.log"
jq -e --slurpfile manifest "$work/portable-manifest.json" '
  . as $result | $result.rows == 5600000 and $result.remoteRows == 5600000 and
  all($manifest[0].tables[]; . as $table | $result.files[$table.documents] == $table.rows)
' "$work/evidence/load.json" >/dev/null
sha256sum "$work/evidence/load.json" > "$work/evidence/load.sha256"
date -u +%FT%TZ > "$work/evidence/completed-at.txt"
echo 'Gremlin source preparation and exact remote count passed; not migration qualification.'
