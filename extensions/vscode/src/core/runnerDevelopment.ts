import { object, RunnerArtifact, RunnerRecord } from "./runner";
import { reportStorageNames } from "./runnerReportStorage";

export function developmentArtifact(record: RunnerRecord, raw: unknown): RunnerArtifact {
  const value = object(raw), names = reportStorageNames(record);
  if (value.schemaVersion !== 1 || value.platform !== "linux-amd64" || typeof value.commit !== "string" || !/^[a-f0-9]{40}$/.test(value.commit) ||
    value.version !== `2.4.0-dev.${value.commit.slice(0,12)}` || typeof value.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(value.sha256) ||
    !Number.isSafeInteger(value.bytes) || Number(value.bytes) < 1 || Number(value.bytes) > 128 * 1024 * 1024) throw new Error("A commit-pinned Linux development archive manifest is required.");
  return { version: String(value.version), sha256: value.sha256, url: `${names.origin}/${names.container}/artifacts/${value.sha256}.tar.gz`, development: { commit: value.commit, bytes: Number(value.bytes) } };
}

/** No shared key, SAS or bearer token in cloud-init, command lines or logs.
 * The VM receives a container-scoped Blob Reader grant; RBAC propagation is bounded.
 * Only reviewed content-addressed bytes are executable. No remote build is used. */
export function developmentDownload(artifact: RunnerArtifact): string {
  if (!artifact.development || !/^https:\/\/af[a-f0-9]{22}\.blob\.core\.windows\.net\/af-[a-f0-9-]{36}\/artifacts\/[a-f0-9]{64}\.tar\.gz$/.test(artifact.url) ||
    !artifact.url.endsWith(`/${artifact.sha256}.tar.gz`) || !/^[a-f0-9]{40}$/.test(artifact.development.commit) || artifact.version !== `2.4.0-dev.${artifact.development.commit.slice(0,12)}` ||
    !Number.isSafeInteger(artifact.development.bytes) || artifact.development.bytes < 1 || artifact.development.bytes > 128 * 1024 * 1024) throw new Error("Invalid pinned development artifact.");
  return `python3 - "$work/archive.tar.gz" <<'AF_PINNED_DOWNLOAD'
import json, sys, time, urllib.request
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args): return None
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
deadline = time.monotonic() + 900
for attempt in range(60):
    try:
        if time.monotonic() >= deadline: sys.exit('Pinned download deadline exceeded')
        req = urllib.request.Request('http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https%3A%2F%2Fstorage.azure.com%2F', headers={'Metadata':'true'})
        with opener.open(req, timeout=10) as response:
            token = json.loads(response.read(65536))['access_token']
        req = urllib.request.Request('${artifact.url}', headers={'Authorization':'Bearer '+token, 'x-ms-version':'2023-11-03', 'Accept-Encoding':'identity'})
        with opener.open(req, timeout=30) as response:
            if response.status != 200 or response.headers.get('Content-Length') != '${artifact.development.bytes}': raise ValueError()
            with open(sys.argv[1], 'wb') as target:
                count = 0
                while True:
                    if time.monotonic() >= deadline: sys.exit('Pinned download deadline exceeded')
                    chunk = response.read(1048576)
                    if not chunk: break
                    count += len(chunk)
                    if count > ${artifact.development.bytes}: raise ValueError()
                    target.write(chunk)
                if count != ${artifact.development.bytes}: raise ValueError()
        break
    except Exception:
        if attempt == 59: sys.exit('Pinned artifact download failed; no alternate source was used')
        time.sleep(10)
AF_PINNED_DOWNLOAD`;
}
