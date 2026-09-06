import {createHash} from "node:crypto";
import {object,RunnerArtifact,RunnerRecord} from "./runner";
import {developmentDownload} from "./runnerDevelopment";

export const p1Root="bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70";
export const p1FixtureRoot="f74220f6c58f0c1a62f80a567520ffcde43a2499ba48100667ee7b78ff4e2e2f";
export interface P1Qualification {operation:string;commandId:string;jobId:string;artifact:RunnerArtifact;startedAt:string;phase:"submitted"|"unknown"|"verified"|"exporting"|"exported"|"pass"|"failed";exportCommandId?:string;sha256?:string;bytes?:number}

/** Recompute the canonical root from every leaf; do not trust a summary pass. */
export function verifyP1(text:string,jobId:string):void{
  const d=object(JSON.parse(text)),e=object(d.expected),a=object(d.actual),c=object(d.comparison);
  if(d.version!==1 || d.jobId!==jobId || d.readOnly!==true || a.jobId!==jobId || e.source!=="fixture" || a.source!=="apache-age" || c.status!=="pass")throw new Error("P1 report identity or outcome differs.");
  for(const m of [e,a]){
    if(m.version!==1 || m.canonicalVersion!=="agefreighter-production-simulation-v1" || m.fixtureRootSha256!==p1FixtureRoot || m.rootSha256!==p1Root || m.rangeRows!==100000 || m.recordCount!==5600000 || !Array.isArray(m.leaves) || m.leaves.length!==64)throw new Error("P1 coverage is incomplete.");
    const root=createHash("sha256");let rows=0;
    for(const item of m.leaves){const l=object(item);
      if(!["v","e"].includes(String(l.kind)) || typeof l.name!=="string" || !/^[A-Za-z_]+$/.test(l.name) || typeof l.sha256!=="string" || !/^[a-f0-9]{64}$/.test(l.sha256) || [l.rangeIndex,l.startKey,l.endKey,l.rows].some(v=>!Number.isSafeInteger(v)||Number(v)<0) || Number(l.rows)<1 || Number(l.rows)>100000)throw new Error("Invalid P1 range.");
      root.update([l.kind,l.name,l.rangeIndex,l.startKey,l.endKey,l.rows,l.sha256].join("\0")+"\n");rows+=Number(l.rows);
    }
    if(rows!==5600000 || root.digest("hex")!==p1Root)throw new Error("P1 canonical root mismatch.");
  }
  for(let i=0;i<64;i++)for(const k of ["kind","name","rangeIndex","startKey","endKey","rows","sha256"]){if(object((e.leaves as unknown[])[i])[k]!==object((a.leaves as unknown[])[i])[k])throw new Error("P1 leaf mismatch.");}
}

/** Isolated executable; does not replace the qualified loader installation. */
export function p1Script(r:RunnerRecord,q:P1Qualification):string{
  if(!/^[a-f0-9-]{36}$/.test(r.id)||!/^[a-f0-9-]{36}$/.test(q.operation)||q.jobId!==r.migration?.jobId||!/^[a-f0-9-]{36}$/.test(q.jobId)||!r.target||!/^[a-z][a-z0-9-]+$/.test(r.target.input.serverName))throw new Error("Invalid qualification identity.");
  return `#!/bin/bash
set -euo pipefail
set +x
umask 077
root=/var/lib/agefreighter/workflows/${r.id}
test ! -e "$root/active"
test "$(cat /var/lib/agefreighter/evidence/archive.sha256)" = '${r.artifact.sha256}'
test "$(df -P "$root" | awk 'NR==2 {gsub(/%/,"",$5); print $5}')" -lt 75
(set -o noclobber; printf '%s' '${q.operation}' > "$root/active")
work="$root/p1-${q.operation}"
mkdir -m 0700 "$work"
${developmentDownload(q.artifact)}
printf '%s  %s\\n' '${q.artifact.sha256}' "$work/archive.tar.gz" | sha256sum --check --status
python3 - "$work" <<'AF_EXTRACT_P1'
import pathlib,sys,tarfile
p=pathlib.Path(sys.argv[1])
with tarfile.open(p/'archive.tar.gz') as t:
    members=t.getmembers()
    assert len(members)==1 and members[0].name=='p1runnerverify' and members[0].isfile() and 0<members[0].size<64*1024*1024
    with open(p/'p1runnerverify','xb') as f:f.write(t.extractfile(members[0]).read())
(p/'p1runnerverify').chmod(0o700)
AF_EXTRACT_P1
set +e
printf '%s' "$AF_P1_DSN" | systemd-run --quiet --wait --pipe --unit=af-p1-${q.operation} --working-directory="$work" -p MemoryMax=4G -p MemorySwapMax=0 -p RuntimeMaxSec=1500 -p NoNewPrivileges=yes "$work/p1runnerverify" '${q.jobId}' '${r.target.input.serverName}.postgres.database.azure.com' >"$work/stdout.log" 2>"$work/stderr.log"
result=$?
unset AF_P1_DSN
set -e
test "$result" = 0 || { echo 'Independent qualification failed; inspect retained evidence'; exit 1; }
python3 - "$work/result.json" <<'AF_RECEIPT'
import sys,pathlib,hashlib,json
data=pathlib.Path(sys.argv[1]).read_bytes()
assert 0<len(data)<4*1024*1024
print(json.dumps({'workflow':'${r.id}','operation':'${q.operation}','jobId':'${q.jobId}','sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'verified':True},separators=(',',':')))
AF_RECEIPT
test "$(cat "$root/active")" = '${q.operation}'
rm "$root/active"
`;
}

/** Export only after verification; issue the 15-minute capability just in time. */
export function p1ExportScript(r:RunnerRecord,q:P1Qualification):string{
  p1Script(r,q);
  if(!/^[a-f0-9]{64}$/.test(q.sha256??'')||!Number.isSafeInteger(q.bytes)||q.bytes!<1)throw new Error('Missing verified result receipt.');
  return `#!/bin/bash
set -euo pipefail
set +x
umask 077
work=/var/lib/agefreighter/workflows/${r.id}/p1-${q.operation}
python3 - "$work/result.json" <<'AF_EXPORT_P1'
import os,sys,urllib.request,pathlib,hashlib,json
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*a):return None
data=pathlib.Path(sys.argv[1]).read_bytes()
assert 0<len(data)<4*1024*1024
assert len(data)==${q.bytes} and hashlib.sha256(data).hexdigest()=='${q.sha256}'
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
try:
    req=urllib.request.Request(os.environ.pop('AF_P1_REPORT'),data=data,method='PUT',headers={'x-ms-version':'2023-11-03','x-ms-blob-type':'BlockBlob','Content-Type':'application/json','If-None-Match':'*'})
    with opener.open(req,timeout=60) as response:assert response.status==201
except Exception:sys.exit('P1 export not confirmed; do not replay verification')
print(json.dumps({'workflow':'${r.id}','operation':'${q.operation}','jobId':'${q.jobId}','sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'exported':True},separators=(',',':')))
AF_EXPORT_P1
`;
}
