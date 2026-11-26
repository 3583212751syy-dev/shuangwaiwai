// 自包含在线版：Worker 代码 + 内存索引 + 本机模型缓存，单端口 HTTP 服务。
// 图片按需从磁盘读取（不常驻内存），避免 2430 张原图撑爆沙箱内存。
import http from 'node:http';
import {readFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import Module,{createRequire} from 'node:module';
// 发布工具会剔除名为 node_modules 的目录，所以 OCR 依赖平铺放在 vendor/tess_pkgs，
// 这里显式把它加入模块搜索路径。
process.env.NODE_PATH=path.join(path.dirname(fileURLToPath(import.meta.url)),'vendor','tess_pkgs');
Module._initPaths();
const require=createRequire(import.meta.url);

const root=path.dirname(fileURLToPath(import.meta.url));
const modelDir=path.join(root,'model-cache');
const imageDir=path.join(root,'images');
const port=Number(process.env.PORT||3000);
const {default:app}=await import('./app-worker.mjs');

const TYPE={'.webp':'image/webp','.png':'image/png','.jpg':'image/jpeg','.jpeg':'image/jpeg','.gif':'image/gif','.avif':'image/avif'};

/* OCR 规格识别：tesseract.js（Apache-2.0）在服务端运行，浏览器只上传图片二进制。
 * 语言包与 wasm 全部在 vendor/ 本地，不依赖外网。 */
let ocrWorker=null;
// 线上沙箱内存有限，wasm OCR 会导致服务不稳定；默认关闭，本地用 OCR=1 开启做验证。
const OCR_ENABLED=!!process.env.OCR;
async function readSpecs(buf){
 if(!OCR_ENABLED)return [];
 if(!ocrWorker){
  const {createWorker}=require('tesseract.js');
  // Node 端用 tesseract.js 自带 worker（传浏览器版 worker.min.js 会崩）；
  // 只把语言包路径指向本地 vendor，避免运行时访问外网 CDN。
  ocrWorker=await createWorker('eng',1,{langPath:path.join(root,'vendor'),logger:()=>{}});
 }
 const {data}=await ocrWorker.recognize(buf);
 const s=String(data.text||'').toUpperCase(),out=new Set();
 for(const m of s.matchAll(/\b([BCD])(\d{1,2})\b/g))out.add(m[0]);
 for(const m of s.matchAll(/\b[A-Z]{1,5}[- ]?\d{1,5}[A-Z]?\b/g))if(m[0].length<=11)out.add(m[0].replace(/\s/g,''));
 return [...out];
}

const objects=new Map();
const idToFile=new Map();
const env={BUCKET:{
 async put(k,b,o){objects.set(k,{buf:Buffer.isBuffer(b)?b:Buffer.from(b),httpMetadata:o?.httpMetadata})},
 async get(k){const v=objects.get(k);return v?{text:async()=>v.buf.toString('utf8'),body:new Uint8Array(v.buf),httpMetadata:v.httpMetadata}:null},
 async list({prefix}={}){return {objects:[...objects.keys()].filter(k=>k.startsWith(prefix||'')).sort().map(key=>({key})),truncated:false}}
}};

const meta=JSON.parse(await readFile(path.join(root,'metadata.json'),'utf8'));
const analyses=JSON.parse(await readFile(path.join(root,'analyses.json'),'utf8'));
let loaded=0;
for(const m of meta){
 const a=analyses[m.filename];
 const full={...m,ai:a?{model:'Xenova/clip-vit-base-patch32',version:'3.8.1',status:a.status,suggestedName:a.suggestedName,suggestedCategory:a.suggestedCategory,score:a.score,embedding:a.embedding,dinoEmbedding:a.dinoEmbedding,method:'CLIP 文本分类',at:new Date().toISOString()}:undefined};
 objects.set('meta/'+m.id,{buf:Buffer.from(JSON.stringify(full))});
 idToFile.set(m.id,m.filename);
 loaded++;
}
console.log('IMAGE-MATCH SERVER meta='+loaded+' port='+port);

const server=http.createServer(async(req,res)=>{
 try{
  const u=new URL(req.url,'http://127.0.0.1:'+port);
  if(u.pathname==='/overrides.json'){
   try{const data=await readFile(path.join(root,'overrides.json'));res.writeHead(200,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-cache'});res.end(data)}catch{res.writeHead(404).end('not found')}
   return;
  }
  // 浏览器端规格 OCR 的模型与 wasm（与 CLIP/DINO 的 model-cache 分开存放）
  if(u.pathname.startsWith('/models/ocr/')){
   const rel=u.pathname.slice('/models/ocr/'.length),base=path.resolve(root,'models','ocr'),file=path.resolve(base,rel);
   if(!file.startsWith(base)){res.writeHead(403).end('Forbidden');return}
   try{
    const data=await readFile(file);
    const ct=rel.endsWith('.wasm')?'application/wasm':(rel.endsWith('.js')||rel.endsWith('.mjs'))?'text/javascript; charset=utf-8':'application/octet-stream';
    res.writeHead(200,{'Content-Type':ct,'Content-Length':data.length,'Cache-Control':'public, max-age=86400'});
    res.end(data);
   }catch{res.writeHead(404).end('Not found')}
   return;
  }
  if(u.pathname.startsWith('/models/')){
   const rel=u.pathname.slice(8),file=path.resolve(modelDir,rel);
   if(!file.startsWith(path.resolve(modelDir))){res.writeHead(403).end('Forbidden');return}
   const data=await readFile(file);
   res.writeHead(200,{'Content-Type':rel.endsWith('.json')?'application/json; charset=utf-8':'application/octet-stream','Content-Length':data.length,'Cache-Control':'public, max-age=86400'});
   res.end(data);return;
  }
  // OCR 规格识别接口：接收图片二进制，返回可读到的型号/规格 token
  if(u.pathname==='/api/specs'&&req.method==='POST'){
   const chunks=[];for await(const c of req)chunks.push(c);
   try{const specs=await readSpecs(Buffer.concat(chunks));res.writeHead(200,{'Content-Type':'application/json; charset=utf-8'});res.end(JSON.stringify({specs}))}
   catch(e){res.writeHead(200,{'Content-Type':'application/json; charset=utf-8'});res.end(JSON.stringify({specs:[],error:String(e.message||e).slice(0,120)}))}
   return;
  }
  // OCR 运行核心（tesseract.js / wasm），必须在 worker 的页面兜底之前返回
  if(u.pathname.startsWith('/vendor/')){
   const rel=u.pathname.slice(8),base=path.resolve(root,'vendor'),file=path.resolve(base,rel);
   if(!file.startsWith(base)){res.writeHead(403).end('Forbidden');return}
   try{
    const data=await readFile(file);
    const ct=rel.endsWith('.wasm')?'application/wasm':rel.endsWith('.js')?'text/javascript; charset=utf-8':'application/octet-stream';
    res.writeHead(200,{'Content-Type':ct,'Content-Length':data.length,'Cache-Control':'public, max-age=86400'});
    res.end(data);
   }catch{res.writeHead(404).end('Not found')}
   return;
  }
  // 底图按需读盘：请求 /images/<id> 时再读文件
  const im=/^\/images\/([a-f0-9-]+)$/.exec(u.pathname);
  if(im&&req.method==='GET'){
   const fn=idToFile.get(im[1]);
   if(!fn){res.writeHead(404).end('Not found');return}
   try{
    const data=await readFile(path.join(imageDir,fn));
    res.writeHead(200,{'Content-Type':TYPE[path.extname(fn).toLowerCase()]||'image/jpeg','Content-Length':data.length,'Cache-Control':'public, max-age=86400','X-Content-Type-Options':'nosniff'});
    res.end(data);
   }catch{res.writeHead(404).end('Not found')}
   return;
  }
  const chunks=[];for await(const c of req)chunks.push(c);
  const request=new Request(u.href,{method:req.method,headers:req.headers,body:['GET','HEAD'].includes(req.method)?undefined:Buffer.concat(chunks)});
  const resp=await app.fetch(request,env);
  res.writeHead(resp.status,Object.fromEntries(resp.headers));
  res.end(Buffer.from(await resp.arrayBuffer()));
 }catch(e){res.writeHead(500,{'Content-Type':'text/plain; charset=utf-8'}).end(String(e?.stack||e))}
});

server.listen(port,'0.0.0.0',()=>console.log('IMAGE-MATCH ONLINE READY'));
