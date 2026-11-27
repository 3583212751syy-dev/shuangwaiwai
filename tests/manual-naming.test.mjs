import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

let workerPromise;
async function worker(){
 if(!workerPromise)workerPromise=(async()=>{
  const source=(await readFile(new URL('../site-source/worker/index.js',import.meta.url),'utf8')).replace(/\/\*[A-Z_]+\*\//g,JSON.stringify('{}'));
  return (await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'))).default;
 })();return workerPromise;
}
function bucket(){const entries=new Map();return {
 async get(k){const s=entries.get(k);return s===undefined?null:{text:async()=>s}},
 async put(k,s){entries.set(k,s)},async list(){return {objects:[...entries.keys()].map(key=>({key})),truncated:false}}
};}
const envOf=async()=>({BUCKET:bucket()});
const post=(handler,body,env)=>handler.fetch(new Request('https://test.local/api/translate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),env);
const namesGet=async(env)=>{const h=await worker();const r=await h.fetch(new Request('https://test.local/api/names'),env);return {status:r.status,body:await r.json()}};
const namesPost=async(body,env)=>{const h=await worker();const r=await h.fetch(new Request('https://test.local/api/names',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),env);return {status:r.status,body:await r.json()}};

test('names roundtrip: POST persists and GET returns merged map',async()=>{
 const env=await envOf();
 let r=await namesPost({group:'row:473',name:'摩托车座套',original:'Funda para sillin de moto'},env);
 assert.equal(r.status,201);
 assert.equal(r.body['row:473'].name,'摩托车座套');
 r=await namesPost({group:'group:自定义分组',name:'防臭地漏'},env);
 assert.equal(r.body['group:自定义分组'].name,'防臭地漏');
 const g=await namesGet(env);
 assert.equal(g.status,200);
 assert.equal(g.body['row:473'].name,'摩托车座套');
 assert.equal(g.body['group:自定义分组'].original,undefined);
 assert.ok(g.body['row:473'].at);
});

test('names POST rejects empty group or name',async()=>{
 const env=await envOf();
 assert.equal((await namesPost({group:'',name:'x'},env)).status,400);
 assert.equal((await namesPost({group:'row:1',name:'  '},env)).status,400);
});

test('translate: Chinese input returns as-is without network',async()=>{
 const h=await worker(),env=await envOf();
 const r=await post(h,{text:'摩托车座套'},env);
 assert.equal(r.status,200);
 const d=await r.json();
 assert.equal(d.translated,'摩托车座套');
 assert.equal(d.original,'摩托车座套');
 assert.equal(d.provider,'原文已是中文');
});

test('translate: rejects empty text',async()=>{
 const h=await worker(),env=await envOf();
 const r=await post(h,{text:''},env);
 assert.equal(r.status,400);
});

test('translate: Spanish listing name becomes Chinese (real network, MyMemory fallback chain)',async()=>{
 const h=await worker(),env=await envOf();
 const r=await post(h,{text:'Funda para sillin de moto'},env);
 const d=await r.json();
 assert.equal(r.status,200,'实际响应: '+JSON.stringify(d));
 assert.ok(/[\u4e00-\u9fff]/.test(d.translated),'翻译结果应含中文: '+d.translated);
 assert.equal(d.original,'Funda para sillin de moto');
 assert.ok(d.provider);
},30000);

test('translate: English input becomes Chinese (real network)',async()=>{
 const h=await worker(),env=await envOf();
 const r=await post(h,{text:'motorcycle saddle cover'},env);
 const d=await r.json();
 assert.equal(r.status,200,'实际响应: '+JSON.stringify(d));
 assert.ok(/[\u4e00-\u9fff]/.test(d.translated),'翻译结果应含中文: '+d.translated);
},30000);
