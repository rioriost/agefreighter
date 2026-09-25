// Separate, unpublished VSIX staging only. The regular build never loads this
// plugin or instrumentation. Never use this artifact for Marketplace publication.
import * as esbuild from 'esbuild';
import {readFile,writeFile,mkdir,copyFile,access} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';

const trial='/Users/rifujita/Git_Managed/agefreighter/production-simulation/work/csv-lost-ack-20260917.Svcntw';
const destination=join(trial,'qualification-extension');
await assert.rejects(access(destination),{code:'ENOENT'});
const source=await readFile(resolve('src/guided/azure.ts'),'utf8');
const old='return uploadCSV(record, path, manifest, this.storageCredential(subscription), fetch, progress, signal);';
assert.equal(source.split(old).length,2,'exactly one reviewed upload call expected');
const helper=`
// AF_CSV_ACK_QUALIFICATION_ONLY_20260917 - never publish
import {appendFile as afAckAppend,writeFile as afAckWrite} from 'node:fs/promises';
let afAckConsumed=false;
function afAckTransport(record: RunnerRecord,path: string,manifest: CSVManifest): typeof fetch {
 if(record.id!=='bd3b6680-1e18-4d78-8f36-f43467a09a0a'||path!==${JSON.stringify(join(trial,'Ack-Supplier.csv'))})return fetch;
 const endpoint='https://afbd3b66801e184d788f36f4.blob.core.windows.net/af-bd3b6680-1e18-4d78-8f36-f43467a09a0a/uploads/'+manifest.file+'/'+manifest.sha256+'.csv';
 if(manifest.bytes!==8797607||manifest.sha256!=='0ecaaca3879b11a4bc76835c23f37cda9bfac7d5ed457ad50937170876d861fe')throw Error('Qualification manifest mismatch');
 return async(input,init)=>{
  const url=new URL(String(input));
  if(url.origin+url.pathname!==endpoint||Date.now()>Date.parse('2026-09-20T07:14:35.311Z'))throw Error('Qualification scope or deadline mismatch');
  const response=await fetch(input,init);
  const event={at:new Date().toISOString(),file:manifest.file,method:init?.method,kind:url.searchParams.get('comp')??'blob',status:response.status,etag:response.headers.get('etag')};
  await afAckAppend(${JSON.stringify(join(trial,'gui-transport.jsonl'))},JSON.stringify(event)+'\\n',{mode:0o600});
  if(init?.method==='PUT'&&url.searchParams.get('comp')==='blocklist'&&response.status===201&&!afAckConsumed){
   afAckConsumed=true;
   await afAckWrite(${JSON.stringify(join(trial,'gui-commit-witness.json'))},JSON.stringify({...event,workflow:record.id,manifest,qualification:'AF_CSV_ACK_QUALIFICATION_ONLY_20260917'},null,2)+'\\n',{flag:'wx',mode:0o600});
   await response.body?.cancel();throw Error('Injected lost commit acknowledgement');
  }
  return response;
 };
}
`;
await mkdir(join(destination,'dist'),{recursive:true});
await mkdir(join(destination,'images'));
await mkdir(join(destination,'resources'));
await esbuild.build({entryPoints:[resolve('src/extension.ts')],bundle:true,external:['vscode'],format:'cjs',platform:'node',target:'node20',outfile:join(destination,'dist/extension.js'),plugins:[{name:'isolated-ack-loss',setup(build){build.onLoad({filter:/[/\\]guided[/\\]azure\.ts$/},()=>({contents:source.replace(old,'return uploadCSV(record, path, manifest, this.storageCredential(subscription), afAckTransport(record,path,manifest), progress, signal);')+helper,loader:'ts',resolveDir:resolve('src/guided')}));}}]});
const bundle=await readFile(join(destination,'dist/extension.js'));
assert.ok(bundle.includes(Buffer.from('AF_CSV_ACK_QUALIFICATION_ONLY_20260917')));
const manifest=JSON.parse(await readFile('package.json','utf8'));
manifest.displayName='AGEFreighter (CSV ACK qualification only)';
manifest.description='Unpublished isolated CSV response-loss qualification. Never publish. Restore normal candidate after the trial.';
delete manifest.scripts;delete manifest.devDependencies;delete manifest.dependencies;
manifest.files=['dist/extension.js','images/icon.png','resources/activity.svg','README.md','LICENSE'];
await writeFile(join(destination,'package.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});
await writeFile(join(destination,'README.md'),'# UNPUBLISHED CSV ACK QUALIFICATION ONLY\n\nNever publish. Drops one successful block-list response only for the approved isolated workflow and Ack-Supplier.csv. No source or target operations are added. Restore the normal candidate after qualification.\n',{flag:'wx'});
await copyFile('LICENSE',join(destination,'LICENSE'));
await copyFile('images/icon.png',join(destination,'images/icon.png'));
await copyFile('resources/activity.svg',join(destination,'resources/activity.svg'));
console.log(JSON.stringify({destination,bundleSHA256:createHash('sha256').update(bundle).digest('hex'),qualificationOnly:true}));
