/** Test-only ARM polling receipt and guard. Signed query values stay in memory. */
import { createHash } from "node:crypto";
const digest=(raw:string)=>createHash("sha256").update(raw).digest("hex");
export function describeLostResponsePoll(raw:string,subscription:string,region:string) {
  let url:URL;try{url=new URL(raw,"https://management.azure.com");}catch{return {urlSHA256:digest(raw),pathKind:"invalid",queryKeys:[]};}
  const base=`/subscriptions/${subscription}/`,path=url.pathname.toLowerCase();
  const pathKind=path.startsWith((base+"operationresults/").toLowerCase())?"subscription-operation-results":path.startsWith((base+`providers/Microsoft.Resources/locations/${region}/`).toLowerCase())?"regional-operation":"unrecognized";
  return {urlSHA256:digest(raw),pathKind,armOrigin:url.origin==="https://management.azure.com",subscriptionMatches:path.startsWith(base.toLowerCase()),
    queryKeys:[...url.searchParams.keys()].map(k=>/^[a-z-]{1,32}$/.test(k)?k:"unrecognized"),
    apiVersion:url.searchParams.get("api-version")==="2022-09-01"?"2022-09-01":"unrecognized",opaquePathAndQueryValuesRetained:false};
}
export function validateLostResponsePoll(raw:string,subscription:string,region:string):void {
  const fail=()=>{throw Error("Unapproved what-if polling URL; sanitized shape receipt retained");};
  let url:URL;try{url=new URL(raw,"https://management.azure.com");}catch{return fail();}
  if(raw.length>16384||url.origin!=="https://management.azure.com"||url.username||url.password||url.hash||/[\\\r\n]/.test(raw))return fail();
  const base=`/subscriptions/${subscription}/`.toLowerCase(),path=url.pathname.toLowerCase();
  if(!path.startsWith(base)||url.searchParams.get("api-version")!=="2022-09-01")return fail();
  const remainder=url.pathname.slice(base.length),keys=[...url.searchParams.keys()];
  if(new Set(keys).size!==keys.length)return fail();
  if(/^operationresults\/[A-Za-z0-9_-]{1,1024}$/i.test(remainder)) {
    if(keys.sort().join(",")!==["api-version","t","c","s","h"].sort().join(","))return fail();
    for(const [name,max] of [["t",80],["c",8192],["s",1024],["h",256]] as const){
      const value=url.searchParams.get(name)!;
      if(!value||value.length>max||!(name==="t"?/^[A-Za-z0-9:._+-]+$/:/^[A-Za-z0-9+/=_-]+$/).test(value))return fail();
    }
    return;
  }
  const prefix=`providers/Microsoft.Resources/locations/${region}/`;
  if(remainder.toLowerCase().startsWith(prefix.toLowerCase())&&/^operation(?:Statuses|Results)\/[a-zA-Z0-9-]+$/i.test(remainder.slice(prefix.length))&&keys.join()==="api-version")return;
  return fail();
}
