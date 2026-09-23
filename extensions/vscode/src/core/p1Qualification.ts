import {createHash} from "node:crypto";
import {object,RunnerArtifact,RunnerRecord} from "./runner";
import {developmentDownload} from "./runnerDevelopment";
import {diagnosticGate} from "./p1Diagnostic";
import {ReportManifest,reportManifest} from "./runnerBlob";

export const p1Root="bf6bb2aa48ffb240333f0a9e3e12aa62086e4f99c9f083b5432f42be9e08bf70";
export const p1FixtureRoot="f74220f6c58f0c1a62f80a567520ffcde43a2499ba48100667ee7b78ff4e2e2f";
export type P1Profile="raw-id"|"gremlin-partition64";
export function p1ProfileForConfiguration(configuration:unknown):P1Profile{
  const source=object(object(configuration??{}).source??{});
  return source.type==="cosmos-nosql"&&object(object(source.cosmos??{}).gremlin??{}).enabled===true?"gremlin-partition64":"raw-id";
}
export function p1ProfileSpec(profile:P1Profile):{root:string;version:string}{
  if(profile==="raw-id")return {root:p1Root,version:"agefreighter-production-simulation-v1"};
  if(profile==="gremlin-partition64")return {root:"8a048faa36fad90404c263d3ce75073d117e5d96a15f8a614a42347cbd7a0ef4",version:"agefreighter-production-simulation-gremlin-partition64-v1"};
  throw new Error("Unknown P1 qualification profile.");
}
export function assertP1VerifierManifest(raw:Record<string,unknown>,profile:P1Profile):void{
  const spec=p1ProfileSpec(profile);
  if(raw.purpose!=="p1-read-only-verifier"||raw.fixtureRoot!==p1FixtureRoot||raw.canonicalRoot!==spec.root||(raw.qualificationProfile??"raw-id")!==profile||(raw.canonicalVersion??(profile==="raw-id"?spec.version:""))!==spec.version||typeof raw.archive!=="string"||!/^[A-Za-z0-9_.-]+\.tar\.gz$/.test(raw.archive))throw new Error("Not the reviewed P1 verifier profile.");
}
export interface P1Qualification {profile?:P1Profile;operation:string;commandId:string;jobId:string;artifact:RunnerArtifact;startedAt:string;phase:"submitted"|"unknown"|"verified"|"exporting"|"exported"|"pass"|"failed";exportCommandId?:string;sha256?:string;bytes?:number;replacesFailedOperation?:string}

export type P1RejectionCategory="json-shape"|"profile"|"identity-outcome"|"coverage"|"range"|"canonical-root"|"leaf";
const p1RejectionMessages:Record<P1RejectionCategory,string>={
  "json-shape":"P1 report is not a JSON object with the required object fields.",
  profile:"P1 report profile differs from the reviewed source.",
  "identity-outcome":"P1 report identity or outcome differs.",
  coverage:"P1 coverage is incomplete.",
  range:"Invalid P1 range.",
  "canonical-root":"P1 canonical root mismatch.",
  leaf:"P1 leaf mismatch."
};
export class P1RejectionError extends Error {
  constructor(readonly category:P1RejectionCategory){super(p1RejectionMessages[category]);this.name="P1RejectionError";}
}
/** The immutable report is evidence of a rejection, never an accepted result.
 * Contains the guest seal and reviewed identity, not report contents or SAS. */
export class P1RejectedImportError extends P1RejectionError {
  readonly evidence:Readonly<{status:"rejected";retained:true;manifest:Readonly<ReportManifest>;jobId:string;profile:P1Profile}>;
  constructor(category:P1RejectionCategory,manifest:ReportManifest,jobId:string,profile:P1Profile){
    super(category);this.name="P1RejectedImportError";
    this.evidence=Object.freeze({status:"rejected",retained:true,manifest:Object.freeze(reportManifest(manifest)),jobId,profile});
    this.message+=` Rejected evidence is retained; it is not an accepted P1 result. Category: ${category}.`;
  }
}

export function requalificationGate(r:RunnerRecord):void {
  diagnosticGate(r);
  const d=r.p1Diagnostic, f=object(d?.result?.failure??{});
  if(d?.phase!=="finished"||d.failedOperation!==r.p1Qualification?.operation||d.jobId!==r.migration?.jobId||f.stage!=="target-digest"||f.code!=="source-key-order")throw new Error("Reviewed retained source-key-order diagnosis is required; no automatic retry.");
}

/** A failed ARM command can return plain text or no JSON. Let the caller seal
 * terminal failure instead of leaving it submitted after a parser exception. */
export function parseP1Receipt(output:unknown):Record<string,unknown>{
  try{const value:unknown=JSON.parse(String(output));return value!==null&&typeof value==="object"&&!Array.isArray(value)?value as Record<string,unknown>:{};}catch{return {};}
}

/** Frozen P1-only admission, not a general migration restriction. Stable IDs
 * correlate records but are not automatically copied into graph properties. */
export function assertP1Projection(configuration:unknown):void{
  const source=object(object(configuration).source);
  if(source.type==="neo4j")return; // Discovery determines properties; full digest remains mandatory.
  const mappings=object(source.type==="postgresql"?source.postgresql:source.type==="csv"?source.csv:source.cosmos);
  if(source.type==="cosmos-nosql" && object(mappings.gremlin??{}).enabled===true){
    const g=object(mappings.gremlin),types=object(g.propertyTypes??{});
    if(g.partitionKeyProperty!=="partitionKey"||types.score!=="float64"||types.distance_km!=="float64")throw new Error("Frozen Gremlin P1 requires partitionKey and explicit float64 score/distance_km.");
    return;
  }
  if(!["postgresql","csv","cosmos-nosql"].includes(String(source.type)))throw new Error("Unsupported P1 source configuration.");
  const groups=[
    {key:"vertices",names:"Supplier Facility Product PurchaseOrder Shipment Lot Location Carrier Customer",properties:"source_key external_id name region created_at status score active tags quantities description"},
    {key:"edges",names:"SUPPLIES PRODUCED_AT PLACED_WITH CONTAINS FULFILLS ORIGINATES_AT DESTINED_FOR CARRIED_BY INCLUDED_IN",properties:"source_key relationship_id occurred_at quantity status distance_km notes"},
  ];
  for(const group of groups){
    const rows=mappings[group.key],names=group.names.split(" "),properties=group.properties.split(" ");
    if(!Array.isArray(rows)||rows.length!==names.length)throw new Error("Frozen P1 requires all nine vertex and nine edge mappings.");
    for(const name of names){
      const matches=rows.filter(row=>object(row).label===name);
      if(matches.length!==1)throw new Error(`Frozen P1 mapping missing or duplicated: ${name}.`);
      const actual=Object.keys(object(object(matches[0]).properties));
      const missing=properties.filter(p=>!actual.includes(p));
      if(missing.length || actual.length!==properties.length)throw new Error(`P1 property projection differs for ${name}: missing ${missing.join(", ")||"none"}; unexpected properties ${actual.filter(p=>!properties.includes(p)).join(", ")||"none"}. Identity fields must also be explicitly mapped as properties. Retain this job; review a new migration, never patch the committed graph.`);
    }
  }
}

/** Recompute the canonical root from every leaf; do not trust a summary pass. */
export function verifyP1(text:string,jobId:string,profile:P1Profile="raw-id"):void{
  const spec=p1ProfileSpec(profile);
  let d:Record<string,unknown>,e:Record<string,unknown>,a:Record<string,unknown>,c:Record<string,unknown>;
  try{d=object(JSON.parse(text));e=object(d.expected);a=object(d.actual);c=object(d.comparison);}catch{throw new P1RejectionError("json-shape");}
  if((d.qualificationProfile??"raw-id")!==profile)throw new P1RejectionError("profile");
  if(d.version!==1 || d.jobId!==jobId || d.readOnly!==true || a.jobId!==jobId || e.source!=="fixture" || a.source!=="apache-age" || c.status!=="pass")throw new P1RejectionError("identity-outcome");
  for(const m of [e,a]){
    if(m.version!==1 || m.canonicalVersion!==spec.version || m.fixtureRootSha256!==p1FixtureRoot || m.rootSha256!==spec.root || m.rangeRows!==100000 || m.recordCount!==5600000 || !Array.isArray(m.leaves) || m.leaves.length!==64)throw new P1RejectionError("coverage");
    const root=createHash("sha256");let rows=0;
    for(const item of m.leaves){let l:Record<string,unknown>;try{l=object(item);}catch{throw new P1RejectionError("range");}
      if(!["v","e"].includes(String(l.kind)) || typeof l.name!=="string" || !/^[A-Za-z_]+$/.test(l.name) || typeof l.sha256!=="string" || !/^[a-f0-9]{64}$/.test(l.sha256) || [l.rangeIndex,l.startKey,l.endKey,l.rows].some(v=>!Number.isSafeInteger(v)||Number(v)<0) || Number(l.rows)<1 || Number(l.rows)>100000)throw new P1RejectionError("range");
      root.update([l.kind,l.name,l.rangeIndex,l.startKey,l.endKey,l.rows,l.sha256].join("\0")+"\n");rows+=Number(l.rows);
    }
    if(rows!==5600000 || root.digest("hex")!==spec.root)throw new P1RejectionError("canonical-root");
  }
  for(let i=0;i<64;i++)for(const k of ["kind","name","rangeIndex","startKey","endKey","rows","sha256"]){if(object((e.leaves as unknown[])[i])[k]!==object((a.leaves as unknown[])[i])[k])throw new P1RejectionError("leaf");}
}

/** Isolated executable; does not replace the qualified loader installation. */
export function p1Script(r:RunnerRecord,q:P1Qualification):string{
  const profile=q.profile??"raw-id";p1ProfileSpec(profile);
  if(profile!==p1ProfileForConfiguration(r.sourceDraft?.configuration))throw new Error("Retained P1 profile differs from source configuration.");
  if(!/^[a-f0-9-]{36}$/.test(r.id)||!/^[a-f0-9-]{36}$/.test(q.operation)||q.jobId!==r.migration?.jobId||!/^[a-f0-9-]{36}$/.test(q.jobId)||!r.target||!/^[a-z][a-z0-9-]+$/.test(r.target.input.serverName))throw new Error("Invalid qualification identity.");
  const old=q.replacesFailedOperation;
  if(old && (!/^[a-f0-9-]{36}$/.test(old)||old===q.operation||!r.p1QualificationHistory?.some(h=>h.operation===old&&h.phase==="failed"&&h.jobId===q.jobId)||!r.guestReady?.bootId||!/^[a-f0-9-]{36}$/.test(r.guestReady.bootId)))throw new Error("Missing retained failed qualification identity.");
  return `#!/bin/bash
set -euo pipefail
set +x
umask 077
root=/var/lib/agefreighter/workflows/${r.id}
exec 9>"$root/p1-diagnostic.lock"
flock -n 9
test "$(cat /var/lib/agefreighter/evidence/archive.sha256)" = '${r.artifact.sha256}'
test "$(df -P "$root" | awk 'NR==2 {gsub(/%/,"",$5); print $5}')" -lt 75
${old?`test "$(cat /proc/sys/kernel/random/boot_id)" = '${r.guestReady!.bootId}'
test "$(cat "$root/active")" = '${old}'
! systemctl is-active --quiet af-p1-${old}
! pgrep -x agefreighter >/dev/null
! pgrep -x p1runnerverify >/dev/null
test -d "$root/p1-${old}"
test ! -e "$root/p1-${old}/active.retained"
mv "$root/active" "$root/p1-${old}/active.retained"`:""}
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
printf '%s' "$AF_P1_DSN" | systemd-run --quiet --wait --pipe --unit=af-p1-${q.operation} --working-directory="$work" -p MemoryMax=4G -p MemorySwapMax=0 -p RuntimeMaxSec=1500 -p NoNewPrivileges=yes "$work/p1runnerverify" '${q.jobId}' '${r.target.input.serverName}.postgres.database.azure.com'${profile==="gremlin-partition64"?" 'gremlin-partition64'":""} >"$work/stdout.log" 2>"$work/stderr.log"
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
