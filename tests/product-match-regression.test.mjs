import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {productResults,searchLibrary,candidateVerdict,READY,DINO_READY} from '../site-source/search.mjs';
import {parseSpecText,compareSpecs,compareGroupSpecs,summarize} from '../site-source/specs.mjs';
const model='Xenova/clip-vit-base-patch32';
const vec=(n,s=1)=>[s,Math.sqrt(1-s*s),...Array(n-2).fill(0)];
const row=(id,clip=.9,dino=.8,group=id)=>({id,name:id,group,category:'Herramientas',ai:{model,embedding:vec(512,clip),...(dino===null?{}:{dinoEmbedding:vec(384,dino)})}});
const q={embedding:vec(512),dinoEmbedding:vec(384)};
const hash='a'.repeat(64);

test('one missing DINO does not disable other groups',()=>{
 const r=productResults([row('a'),row('b',.8,null)],q.embedding,q.dinoEmbedding);
 assert.equal(r.find(x=>x.image.id==='a').fused,true);
 assert.equal(r.find(x=>x.image.id==='b').fused,false);
});
test('mixed views retain available DINO evidence',()=>{
 const r=productResults([row('a',.9,.8,'g'),row('b',.8,null,'g')],q.embedding,q.dinoEmbedding);
 assert.equal(r.length,1);assert.equal(r[0].dinoCoverage,.5);assert.equal(r[0].fused,true);
});
test('strong CLIP-only candidates are not buried by weak dual-model entries',()=>{
 const r=productResults([row('a',.99,null),row('b',.1,.1)],q.embedding,q.dinoEmbedding);
 assert.equal(r[0].image.id,'a');assert.equal(r[0].fused,false);
});
test('CLIP-only and empty queries remain finite and conservative',()=>{
 const r=searchLibrary({embedding:q.embedding},[row('a',.95,null)]);
 assert.equal(r.fused,false);assert.equal(r.confidence,'low');assert.equal(candidateVerdict(r.results[0]).ok,false);
 assert.equal(searchLibrary({},[row('a')]).results.length,0);
});
test('invalid vectors and incompatible DINO metadata are ignored',()=>{
 const a=row('a');a.ai.embedding[0]=NaN;assert.equal(READY(a),false);
 const b=row('b');b.ai.dinoPooling='another-pool';assert.equal(DINO_READY(b),false);
 assert.equal(productResults([b],q.embedding,q.dinoEmbedding)[0].fused,false);
});
test('rank fusion ties do not depend on input ordering',()=>{
 const a=row('a'),b=row('b');
 const r1=productResults([a,b],q.embedding,q.dinoEmbedding),r2=productResults([b,a],q.embedding,q.dinoEmbedding);
 assert.deepEqual(r1.map(x=>[x.image.id,x.score]),r2.map(x=>[x.image.id,x.score]));
});
test('huge or negative ranking scores cannot certify identity',()=>{
 for(const score of [-20,0,10000])assert.equal(candidateVerdict({productScore:score,visual:.2,dinoVisual:.1}).ok,false);
 assert.equal(candidateVerdict({productScore:10000,visual:.95,dinoVisual:.9}).tentative,true);
});
test('exact query feedback persists across filename changes and affects retrieval',()=>{
 const a=row('a');a.confirmations=[{decision:'different',query:'old.png',querySha256:hash,groupId:'a',at:'2026-11-26T01:00:00Z'}];
 assert.equal(searchLibrary(q,[a],{querySha256:hash}).results.length,0);
 assert.equal(searchLibrary(q,[a],{querySha256:'b'.repeat(64)}).results.length,1);
 a.confirmations.push({decision:'same',querySha256:hash,groupId:'a',at:'2026-11-26T02:00:00Z'});
 const hit=searchLibrary(q,[a],{querySha256:hash}).results[0];
 assert.equal(candidateVerdict(hit).tentative,false);assert.equal(candidateVerdict(hit).ok,true);
});
test('legacy filename-only feedback is not mistaken for an exact image decision',()=>{
 const a=row('a');a.confirmations=[{decision:'different',query:'a.png',at:'2026-11-26T01:00:00Z'}];
 assert.equal(searchLibrary(q,[a],{querySha256:hash}).results.length,1);
});
test('feedback applies to a complete multi-view product group',()=>{
 const a=row('a',.9,.8,'g'),b=row('b',.8,.7,'g');
 b.confirmations=[{decision:'different',querySha256:hash,groupId:'g',at:'2026-11-26T01:00:00Z'}];
 assert.equal(searchLibrary(q,[a,b],{querySha256:hash}).results.length,0);
});
test('repeated OCR words are not package quantity',()=>{
 const a=parseSpecText('C16 C16 C16'),b=parseSpecText('C16');
 assert.equal(compareSpecs(a,b).conflict,null);assert.equal(summarize(a),'C16');
});
test('unverified OCR disagreement requests review, not exclusion',()=>{
 const r=compareSpecs(parseSpecText('C16'),parseSpecText('C20'));
 assert.equal(r.conflict,null);assert.equal(r.review,true);
});
test('verified unambiguous specifications can exclude a mismatch',()=>{
 const r=compareSpecs(parseSpecText('C16',{verified:true}),parseSpecText('C20',{verified:true}));
 assert.equal(r.level,'spec');assert.ok(r.conflict);
});
test('ambiguous multi-spec images do not exclude',()=>{
 const r=compareSpecs(parseSpecText('C16 C20',{verified:true}),parseSpecText('C32',{verified:true}));
 assert.equal(r.conflict,null);assert.equal(r.review,true);
});
test('only explicit verified quantities can differ',()=>{
 const a=parseSpecText('C16',{verified:true,quantity:2}),b=parseSpecText('C16',{verified:true,quantity:1});
 assert.equal(compareSpecs(a,b).level,'count');
 assert.equal(compareSpecs({...a,verified:false},b).conflict,null);
});
test('missing or agreeing angle keeps a group eligible',()=>{
 const a=parseSpecText('C16',{verified:true}),bad=parseSpecText('C20',{verified:true});
 assert.equal(compareGroupSpecs(a,[bad,null]).conflict,null);
 assert.equal(compareGroupSpecs(a,[bad,a]).conflict,null);
 assert.ok(compareGroupSpecs(a,[bad,bad]).conflict);
});

let workerPromise;
async function worker(){
 if(!workerPromise)workerPromise=(async()=>{
  const source=(await readFile(new URL('../site-source/worker/index.js',import.meta.url),'utf8')).replace(/\/\*[A-Z_]+\*\//g,JSON.stringify('{}'));
  return (await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'))).default;
 })();return workerPromise;
}
function bucket(initial){const entries=new Map(Object.entries(initial));return {
 async get(k){const s=entries.get(k);return s===undefined?null:{text:async()=>s}},
 async put(k,s){entries.set(k,s)},async list(){return {objects:[...entries.keys()].map(key=>({key})),truncated:false}}
};}
const id='11111111-1111-1111-1111-111111111111';
async function patch(body,initial=row(id)){
 const BUCKET=bucket({['meta/'+id]:JSON.stringify({...initial,created:'2026-11-26'})});
 const handler=await worker();const response=await handler.fetch(new Request('https://test.local/api/images/'+id,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),{BUCKET});
 return {response,BUCKET,handler};
}
test('API stores DINO and returns it again after a fresh GET',async()=>{
 const ai={...row(id).ai,status:'已识别',dinoModel:'onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX',dinoPooling:'cls-v1'};
 const {response,BUCKET,handler}=await patch({ai},row(id,.9,null));assert.equal(response.status,200);
 const rows=await (await handler.fetch(new Request('https://test.local/api/images'),{BUCKET})).json();
 assert.deepEqual(rows[0].ai.dinoEmbedding,ai.dinoEmbedding);assert.equal(rows[0].ai.dinoPooling,ai.dinoPooling);
});
test('CLIP refresh and label edits preserve saved DINO',async()=>{
 const {response}=await patch({ai:{model,embedding:vec(512),status:'已识别'}});
 assert.equal((await response.json()).ai.dinoEmbedding.length,384);
 const r=await patch({name:'renamed'});assert.equal((await r.response.json()).ai.dinoEmbedding.length,384);
});
test('API rejects invalid DINO instead of persisting a corrupt index',async()=>{
 for(const dinoEmbedding of [[1],Array(384).fill(0),[2,...Array(383).fill(0)]]){
  const {response}=await patch({ai:{model,embedding:vec(512),status:'已识别',dinoEmbedding}});assert.equal(response.status,400);
 }
});
test('API preserves stable feedback identity',async()=>{
 const {response}=await patch({confirmations:[{decision:'same',querySha256:hash,groupId:id,at:'2026-11-26T01:00:00Z'}]});
 assert.equal((await response.json()).confirmations[0].querySha256,hash);
});
