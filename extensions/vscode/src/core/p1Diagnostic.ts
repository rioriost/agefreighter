import {RunnerRecord,RunnerArtifact,object} from "./runner";
import {developmentDownload} from "./runnerDevelopment";
import {targetBudget} from "./runnerTarget";

export interface P1Diagnostic {
  operation:string;commandId:string;jobId:string;failedOperation:string;
  phase:"submitted"|"unknown"|"finished"|"failed";startedAt:string;
  artifact:RunnerArtifact;result?:Record<string,unknown>;
}
const uuid=/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/;
export function diagnosticGate(r:RunnerRecord):void {
  if(r.p1Qualification?.profile==="gremlin-partition64")throw new Error("Gremlin qualification failure requires a profile-specific evidence review; the legacy raw-ID diagnostic must not run.");
  const g=r.guestReady,h=g?.health,age=Date.now()-Date.parse(g?.checkedAt??"");
  if(r.p1Qualification?.phase!=="failed" || r.migration?.phase!=="finished" || r.migration.verification?.outcome!=="pass" || r.p1Qualification.jobId!==r.migration.jobId || r.target?.phase!=="provisioned")throw new Error("Retained failed P1 verification and passing counts are required.");
  targetBudget(r.target.input);
  if(!g || !h || !Number.isFinite(age)||age<0||age>300000 || g.archiveSha256!==r.artifact.sha256 || g.cliVersion!==r.artifact.version || r.artifact.development&&g.commit!==r.artifact.development.commit || !Number.isFinite(h.storageUsedPercent)||h.storageUsedPercent>=75 || h.swapUsedBytes!==0 || h.oomEvents!==0 || r.guestCommand&&["submitted","unknown"].includes(r.guestCommand.phase))throw new Error("Refresh and reconcile pinned Linux health before diagnosis.");
  // The old failure marker deliberately remains. Only the fixed guest script
  // may distinguish that marker from an active worker; no idle status is forged.
}
export function diagnosticReceipt(value:unknown,r:RunnerRecord,d:P1Diagnostic):Record<string,unknown>{
  const v=object(value),f=object(v.failure??{});
  const allowed:Record<string,string[]>={setup:["invalid-input-or-output"],"fixture-generation":["fixture-generation-or-root"],"fixture-digest":["fixture-digest-or-coverage"],"target-digest":["source-key-order","target-read-or-canonicalization"],comparison:["canonical-mismatch"]};
  if(v.workflow!==r.id||v.operation!==d.operation||v.jobId!==d.jobId||v.failedOperation!==d.failedOperation||v.readOnly!==true||v.exitCode!==1||!allowed[String(f.stage)]?.includes(String(f.code))||f.version!==1||f.outcome!=="fail"||!Number.isSafeInteger(v.bytes)||Number(v.bytes)<1||Number(v.bytes)>1024||typeof v.sha256!=="string"||!/^[a-f0-9]{64}$/.test(v.sha256))throw new Error("Diagnostic receipt is incomplete; retain evidence, do not replay.");
  return {workflow:r.id,operation:d.operation,jobId:d.jobId,failedOperation:d.failedOperation,readOnly:true,exitCode:1,bytes:v.bytes,sha256:v.sha256,failure:{version:1,outcome:"fail",stage:f.stage,code:f.code}};
}
export function p1DiagnosticScript(r:RunnerRecord,d:P1Diagnostic):string {
  if(r.p1Qualification?.profile==="gremlin-partition64")throw new Error("Legacy raw-ID diagnostic cannot inspect the Gremlin profile.");
  if(![r.id,d.operation,d.jobId,d.failedOperation,r.guestReady?.bootId].every(x=>typeof x==="string"&&uuid.test(x)) || d.operation===d.failedOperation || d.jobId!==r.migration?.jobId || d.failedOperation!==r.p1Qualification?.operation || !r.target || !/^[a-z][a-z0-9-]+$/.test(r.target.input.serverName))throw new Error("Invalid diagnostic identity.");
  return `#!/bin/bash
set -euo pipefail
set +x
umask 077
root=/var/lib/agefreighter/workflows/${r.id}
test "$(cat /proc/sys/kernel/random/boot_id)" = '${r.guestReady!.bootId}'
test "$(cat /var/lib/agefreighter/evidence/archive.sha256)" = '${r.artifact.sha256}'
test "$(df -P "$root" | awk 'NR==2 {gsub(/%/,"",$5); print $5}')" -lt 75
exec 9>"$root/p1-diagnostic.lock"
flock -n 9
if test -e "$root/active"; then test "$(cat "$root/active")" = '${d.failedOperation}'; fi
! systemctl is-active --quiet af-p1-${d.failedOperation}
! pgrep -x agefreighter >/dev/null
! pgrep -x p1runnerverify >/dev/null
work="$root/p1-diagnostic-${d.operation}"
mkdir -m 0700 "$work"
${developmentDownload(d.artifact)}
printf '%s  %s\\n' '${d.artifact.sha256}' "$work/archive.tar.gz" | sha256sum --check --status
python3 - "$work" <<'AF_DIAG_EXTRACT'
import pathlib,sys,tarfile
p=pathlib.Path(sys.argv[1])
with tarfile.open(p/'archive.tar.gz') as t:
    m=t.getmembers()
    assert len(m)==1 and m[0].name=='p1runnerverify' and m[0].isfile() and 0<m[0].size<64*1024*1024
    with open(p/'p1runnerverify','xb') as f:f.write(t.extractfile(m[0]).read())
(p/'p1runnerverify').chmod(0o700)
AF_DIAG_EXTRACT
set +e
printf '%s' "$AF_P1_DSN" | systemd-run --quiet --wait --pipe --unit=af-p1-diagnostic-${d.operation} --working-directory="$work" -p MemoryMax=4G -p MemorySwapMax=0 -p RuntimeMaxSec=1500 -p NoNewPrivileges=yes "$work/p1runnerverify" '${d.jobId}' '${r.target.input.serverName}.postgres.database.azure.com' >"$work/stdout.log" 2>"$work/stderr.log"
result=$?
unset AF_P1_DSN
set -e
test "$result" = 1 || { echo 'Unexpected diagnostic exit; evidence retained'; exit 1; }
python3 - "$work/failure.json" <<'AF_DIAG_RECEIPT'
import pathlib,sys,json,hashlib
b=pathlib.Path(sys.argv[1]).read_bytes()
assert 0<len(b)<=1024
f=json.loads(b)
allowed={'setup':['invalid-input-or-output'],'fixture-generation':['fixture-generation-or-root'],'fixture-digest':['fixture-digest-or-coverage'],'target-digest':['source-key-order','target-read-or-canonicalization'],'comparison':['canonical-mismatch']}
assert f.get('version')==1 and f.get('outcome')=='fail' and f.get('code') in allowed.get(f.get('stage'),[])
safe={k:f[k] for k in ['version','outcome','stage','code']}
print(json.dumps({'workflow':'${r.id}','operation':'${d.operation}','jobId':'${d.jobId}','failedOperation':'${d.failedOperation}','readOnly':True,'exitCode':1,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'failure':safe},separators=(',',':')))
AF_DIAG_RECEIPT
`;
}
