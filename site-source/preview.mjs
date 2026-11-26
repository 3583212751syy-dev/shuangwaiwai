// 本地可运行版：复用线上同一份 Worker 代码，内存图库 + 本机 149MB CLIP 模型，无需部署凭证。
import http from 'node:http';
import {readFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const root=path.dirname(fileURLToPath(import.meta.url));
const work=path.resolve(root,'..');
const modelDir=path.join(work,'model-runtime','model-cache');
const imageDir=path.join(work,'export','包装-产品图');
const port=Number(process.env.PORT||8788);
const {default:app}=await import('./dist/server/index.js');

const objects=new Map();
const env={BUCKET:{
 async put(k,b,o){objects.set(k,{buf:Buffer.isBuffer(b)?b:Buffer.from(b),httpMetadata:o?.httpMetadata})},
 async get(k){const v=objects.get(k);return v?{text:async()=>v.buf.toString('utf8'),body:new Uint8Array(v.buf),httpMetadata:v.httpMetadata}:null},
 async list({prefix}={}){return {objects:[...objects.keys()].filter(k=>k.startsWith(prefix||'')).sort().map(key=>({key})),truncated:false}}
}};

const meta=JSON.parse(await readFile(path.join(work,'model-runtime','metadata-before-ai.json'),'utf8'));
const analyses=JSON.parse(await readFile(path.join(work,'model-runtime','analyses.json'),'utf8'));
let loaded=0,missing=0;
for(const m of meta){
 const a=analyses[m.filename];
 const full={...m,ai:a?{model:'Xenova/clip-vit-base-patch32',version:'3.8.1',status:a.status,suggestedName:a.suggestedName,suggestedCategory:a.suggestedCategory,score:a.score,embedding:a.embedding,method:'CLIP 文本分类',at:new Date().toISOString()}:undefined};
 let img;try{img=await readFile(path.join(imageDir,m.filename))}catch{missing++;continue}
 objects.set('images/'+m.id,{buf:img,httpMetadata:{contentType:'image/jpeg'}});
 objects.set('meta/'+m.id,{buf:Buffer.from(JSON.stringify(full))});
 loaded++;
}
const groups=new Set(meta.map(m=>m.group)).size;

const server=http.createServer(async(req,res)=>{
 try{
  const u=new URL(req.url,'http://127.0.0.1:'+port);
  if(u.pathname.startsWith('/models/')){
   const rel=u.pathname.slice(8),file=path.resolve(modelDir,rel);
   if(!file.startsWith(path.resolve(modelDir))){res.writeHead(403).end('Forbidden');return}
   const data=await readFile(file);
   res.writeHead(200,{'Content-Type':rel.endsWith('.json')?'application/json; charset=utf-8':'application/octet-stream','Content-Length':data.length,'Cache-Control':'public, max-age=86400'});
   res.end(data);return;
  }
  const chunks=[];for await(const c of req)chunks.push(c);
  const request=new Request(u.href,{method:req.method,headers:req.headers,body:['GET','HEAD'].includes(req.method)?undefined:Buffer.concat(chunks)});
  const resp=await app.fetch(request,env);
  res.writeHead(resp.status,Object.fromEntries(resp.headers));
  res.end(Buffer.from(await resp.arrayBuffer()));
 }catch(e){res.writeHead(500,{'Content-Type':'text/plain; charset=utf-8'}).end(String(e?.stack||e))}
});

server.listen(port,'127.0.0.1',()=>{
 console.log('IMAGE-MATCH LOCAL READY http://127.0.0.1:'+port);
 console.log('images='+loaded+' missing='+missing+' rawGroups='+groups);
 console.log('models='+modelDir);
});
