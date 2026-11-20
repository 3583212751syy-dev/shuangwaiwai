import http from 'node:http';
import {readFile,writeFile,mkdir,open} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {randomBytes,randomInt} from 'node:crypto';
import {defaults,settings,readJSON,writeJSON,safeId,safeFile,checkPassword,encryptSecret,decryptSecret,normalizeAnalysis,taskPrompt,layersFor,cleanString} from './lib/core.mjs';
import {analyze,generateLocal,generateOpenAI,retrieveLocal,health,validateLocalEndpoint} from './lib/providers.mjs';
import {normalizeImage,renderImage,validateLayers} from './lib/render.mjs';
import {sharp,JSZip} from './lib/deps.mjs';
export const initialConfig={comfyURL:'http://127.0.0.1:8188',ollamaURL:'http://127.0.0.1:11434',visionModel:'qwen2.5vl:3b',diffusionModel:'qwen-image-edit-2511-Q4_K_M.gguf',textEncoder:'qwen_2.5_vl_7b_fp8_scaled.safetensors',vae:'qwen_image_vae.safetensors',lightning:'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors',imageAPIModel:'gpt-image-1.5',visionAPIModel:'gpt-4.1-mini'};
export async function createApp({root=process.env.MX_ROOT||'D:\\图片AI',providerOverrides={}}={}){
 const dataDir=path.join(root,'data'),projectsDir=path.join(root,'projects');await mkdir(dataDir,{recursive:true});await mkdir(projectsDir,{recursive:true});
 let config={...initialConfig,...await readJSON(path.join(dataDir,'config.json'),{})};
 let preferences=settings(await readJSON(path.join(dataDir,'preferences.json'),defaults));
 let key;try{key=await readFile(path.join(dataDir,'secret.key'))}catch(e){if(e.code!=='ENOENT')throw e;key=randomBytes(32);await writeFile(path.join(dataDir,'secret.key'),key,{flag:'wx',mode:0o600})}
 const manifest=await readJSON(path.join(dataDir,'projects.json'),[]);const projects=new Map();for(const id of manifest){const p=await readJSON(path.join(projectsDir,safeId(id),'project.json'),null);if(p){for(const task of p.tasks||[])if(['queued','running'].includes(task.state)){task.state='interrupted';task.error='服务重启，任务状态待核查；本地已提交任务可点击取回结果，避免重复生成。';}if(p.analysisState==='running')p.analysisState='interrupted';projects.set(id,p)}}
 let persistChain=Promise.resolve();function persist(){persistChain=persistChain.catch(()=>{}).then(async()=>{await writeJSON(path.join(dataDir,'preferences.json'),preferences);await writeJSON(path.join(dataDir,'config.json'),config);await writeJSON(path.join(dataDir,'projects.json'),[...projects.keys()]);for(const p of projects.values())await writeJSON(path.join(projectsDir,p.id,'project.json'),p)});return persistChain;}
 const sessions=new Map(),attempts=new Map(),jobs=[];let draining=false,closed=false;const providers={analyze,generateLocal,generateOpenAI,retrieveLocal,health,...providerOverrides};
 const touch=p=>{p.updatedAt=Date.now()};const pdir=p=>path.join(projectsDir,p.id);const refs=p=>p.references.map(f=>path.join(pdir(p),f));
 function secret(){return decryptSecret(config.apiKeyEncrypted,key)}
 function safeError(e){let message=String(e.message||e).slice(0,1200);const apiKey=secret();if(apiKey)message=message.split(apiKey).join('[redacted]');return message;}
 async function saveVersion(p,t,base,layers,note){
  const version=t.versions.length+1,stem=`${t.id}-v${version}`,baseFile=stem+'-base.png',file=stem+'.png';
  const normalized=await normalizeImage(base);await writeFile(path.join(pdir(p),baseFile),normalized);await writeFile(path.join(pdir(p),file),await renderImage(normalized,layers));
  t.versions.push({version,file,baseFile,layers,review:'pending',createdAt:Date.now(),note});t.currentVersion=version;t.state='complete';t.error='';touch(p);await persist();
 }
 async function drain(){if(draining||closed)return;draining=true;try{while(jobs.length&&!closed){const job=jobs.shift();try{await job()}catch(e){console.error('Background task failed:',safeError(e))}}}finally{draining=false}}
 function enqueue(job){jobs.push(job);setImmediate(drain)}
 function analyzeProject(p){p.analysisState='queued';enqueue(async()=>{p.analysisState='running';p.analysisError='';touch(p);await persist();try{const a=await providers.analyze({...config,provider:p.preferences.provider,language:p.preferences.language},refs(p),secret());p.analysis=a;p.analysisState='complete';if(p.name==='自动识别产品')p.name=a.name;for(const task of p.tasks)if(!task.versions.length&&!task.customPrompt)task.prompt=taskPrompt(p,task.type,task.index,task.change||'');}catch(e){p.analysisState='error';p.analysisError=safeError(e)}touch(p);await persist()})}
 function enqueueTask(p,t,recover=false){
  t.state='queued';t.error='';touch(p);
  enqueue(async()=>{t.state='running';touch(p);await persist();try{
   let bytes;
   if(recover){if(!t.promptId)throw Error('没有可取回的本地任务编号');bytes=await providers.retrieveLocal(config,t.promptId)}
   else if(p.preferences.provider==='local')bytes=await providers.generateLocal(config,p,t,refs(p),async({promptId,graph})=>{t.promptId=promptId;await writeJSON(path.join(pdir(p),`${t.id}-workflow.json`),graph);await persist()});
   else bytes=await providers.generateOpenAI(config,p,t,refs(p),secret());
   await saveVersion(p,t,bytes,layersFor(p,t.type),recover?'取回本地任务':'AI 生成');
  }catch(e){t.state='error';t.error=safeError(e);touch(p);await persist()}});
 }
 function newTasks(p){const tasks=[];for(const [type,count]of Object.entries(p.preferences.counts))for(let i=0;i<count;i++){if(type==='size'&&!p.dimensions.trim())throw Error('尺寸图必须填写卖家确认的实际数字与单位');tasks.push({id:`${type}-${i+1}`,type,index:i,state:'ready',prompt:taskPrompt(p,type,i),seed:randomInt(1,2147483647),versions:[],currentVersion:0})}if(!tasks.length)throw Error('请至少选择一张图片');return tasks}
 function project(id){const p=projects.get(safeId(id));if(!p)throw Error('项目不存在');return p}
 function task(p,id){const t=p.tasks.find(t=>t.id===id);if(!t)throw Error('图片任务不存在');return t}
 function mutable(p){if(p.tasks.some(t=>['queued','running'].includes(t.state))||['queued','running'].includes(p.analysisState))throw Error('项目正在处理，请等待完成后修改');}
 function json(res,status,data,headers={}){res.writeHead(status,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store',...headers});res.end(JSON.stringify(data));}
 async function input(req,max=36*1024*1024){let length=0;const chunks=[];for await(const c of req){length+=c.length;if(length>max)throw Error('请求过大，请将上传图片合计控制在 24MB 内');chunks.push(c)}try{return JSON.parse(Buffer.concat(chunks))}catch{throw Error('请求格式错误')}}
 function auth(req){const token=/\bmx_session=([a-f0-9]{64})\b/.exec(req.headers.cookie||'')?.[1];const session=sessions.get(token);if(!session||session.expires<Date.now()){sessions.delete(token);return null}return {token,...session};}
 async function origins(){const remote=await readJSON(path.join(dataDir,'remote.json'),{});const list=new Set(['http://127.0.0.1:3188','http://localhost:3188']);if(/^https:\/\/[a-z0-9-]+\.trycloudflare\.com$/.test(remote.url||''))list.add(remote.url);return list;}
 const webDir=path.join(path.dirname(fileURLToPath(import.meta.url)),'web');
 const server=http.createServer(async(req,res)=>{
  res.setHeader('X-Content-Type-Options','nosniff');res.setHeader('Referrer-Policy','no-referrer');res.setHeader('X-Frame-Options','DENY');res.setHeader('Content-Security-Policy',"default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'");res.setHeader('Cache-Control','no-store');
  try{
   const url=new URL(req.url,'http://127.0.0.1'),route=url.pathname,method=req.method;
   const allowed=await origins();const host=req.headers.host||'';const hostAllowed=[...allowed].some(o=>new URL(o).host===host);if(!hostAllowed)return json(res,403,{error:'访问地址未被允许'});
   if(!['GET','HEAD'].includes(method)&&!allowed.has(req.headers.origin))return json(res,403,{error:'请求来源不被允许'});
   if(method==='GET'&&['/','/app.js','/style.css'].includes(route)){const name=route==='/'?'index.html':route.slice(1);const b=await readFile(path.join(webDir,name));res.writeHead(200,{'Content-Type':route.endsWith('.js')?'text/javascript; charset=utf-8':route.endsWith('.css')?'text/css; charset=utf-8':'text/html; charset=utf-8'});return res.end(b)}
   if(method==='GET'&&route==='/api/session')return json(res,200,{authenticated:!!auth(req),configured:!!await readJSON(path.join(dataDir,'owner.json'),null)});
   if(method==='POST'&&route==='/api/login'){
    const bucket=attempts.get('owner')||{count:0,start:Date.now()};if(Date.now()-bucket.start>600000){bucket.count=0;bucket.start=Date.now()}if(bucket.count>=10)return json(res,429,{error:'登录尝试过多，请 10 分钟后再试'});bucket.count++;attempts.set('owner',bucket);
    const {username,password}=await input(req,2048);const owner=await readJSON(path.join(dataDir,'owner.json'),null);if(!owner)return json(res,403,{error:'请在本机运行安装程序，先设置账号密码'});
    if(!checkPassword(password,owner)||username!==owner.username)return json(res,401,{error:'账号或密码错误'});
    attempts.delete('owner');const token=randomBytes(32).toString('hex');sessions.set(token,{expires:Date.now()+12*3600000});if(sessions.size>100)sessions.delete(sessions.keys().next().value);
    const remote=req.headers.origin?.startsWith('https://');return json(res,200,{ok:true},{'Set-Cookie':`mx_session=${token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=43200${remote?'; Secure':''}`});
   }
   const session=auth(req);if(!session)return json(res,401,{error:'请先登录'});
   if(method==='POST'&&route==='/api/logout'){sessions.delete(session.token);return json(res,200,{ok:true},{'Set-Cookie':'mx_session=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0'})}
   if(method==='GET'&&route==='/api/state'){const remote=await readJSON(path.join(dataDir,'remote.json'),{});return json(res,200,{preferences,config:{...config,apiKeyEncrypted:undefined,apiConfigured:!!config.apiKeyEncrypted},remote,root,projects:[...projects.values()].map(p=>({id:p.id,name:p.name,updatedAt:p.updatedAt,analysisState:p.analysisState,complete:p.tasks.filter(t=>t.versions.length).length,total:p.tasks.length})).sort((a,b)=>b.updatedAt-a.updatedAt),queue:jobs.length,working:draining})}
   if(method==='GET'&&route==='/api/health')return json(res,200,await providers.health(config));
   if(method==='PUT'&&route==='/api/preferences'){preferences=settings(await input(req,20000));await persist();return json(res,200,preferences)}
   if(method==='PUT'&&route==='/api/config'){
    if(draining||jobs.length)throw Error('有任务正在运行，请完成后再修改连接');const body=await input(req,20000);const next={...config};
    for(const name of ['comfyURL','ollamaURL'])if(body[name]!==undefined)next[name]=validateLocalEndpoint(body[name]);
    for(const name of ['visionModel','diffusionModel','textEncoder','vae','lightning','imageAPIModel','visionAPIModel'])if(body[name]!==undefined){if(typeof body[name]!=='string'||!body[name].trim()||body[name].length>180||/[\r\n]/.test(body[name]))throw Error('模型名称无效');next[name]=body[name]}
    if(body.apiKey!==undefined){if(typeof body.apiKey!=='string'||body.apiKey.length>500)throw Error('密钥无效');if(body.apiKey)next.apiKeyEncrypted=encryptSecret(body.apiKey,key)}if(body.clearKey)delete next.apiKeyEncrypted;config=next;await persist();return json(res,200,{ok:true})
   }
   if(method==='POST'&&route==='/api/projects'){
    const body=await input(req);const pref=settings(body.preferences||preferences);if(pref.provider==='openai'&&pref.autoAnalyze&&!body.cloudConsent)throw Error('云端识别需明确确认发送产品图及 API 费用');if(!Array.isArray(body.images)||!body.images.length||body.images.length>3)throw Error('请上传 1–3 张同款产品参考图');
    const buffers=[];let total=0;for(const img of body.images){if(typeof img!=='string')throw Error('图片格式无效');const m=/^data:image\/(png|jpeg|webp);base64,([A-Za-z0-9+/=]+)$/.exec(img);if(!m)throw Error('仅支持 PNG、JPEG、WebP');const bytes=Buffer.from(m[2],'base64');total+=bytes.length;if(bytes.length>8*1024*1024||total>24*1024*1024)throw Error('每张最多 8MB，合计最多 24MB');buffers.push(await sharp(bytes,{limitInputPixels:40000000}).rotate().resize(2400,2400,{fit:'inside',withoutEnlargement:true}).png().toBuffer())}
    const id=`${Date.now()}-${randomBytes(6).toString('hex')}`;const p={id,name:cleanString(body.name,150)||'自动识别产品',facts:cleanString(body.facts,4000),dimensions:cleanString(body.dimensions,1000),preferences:pref,analysis:null,analysisState:'ready',references:buffers.map((_,i)=>`reference-${i+1}.png`),tasks:[],createdAt:Date.now(),updatedAt:Date.now()};p.tasks=newTasks(p);await mkdir(pdir(p),{recursive:true});for(let i=0;i<buffers.length;i++)await writeFile(path.join(pdir(p),p.references[i]),buffers[i]);projects.set(id,p);preferences=pref;if(pref.autoAnalyze)analyzeProject(p);await persist();return json(res,201,p);
   }
   const match=/^\/api\/projects\/(\d{13}-[a-f0-9]{12})(?:\/(.*))?$/.exec(route);
   if(match){const p=project(match[1]),rest=match[2]||'';
    if(method==='GET'&&!rest)return json(res,200,p);
    if(method==='PATCH'&&!rest){mutable(p);const b=await input(req,20000);for(const k of ['name','facts','dimensions'])if(b[k]!==undefined)p[k]=cleanString(b[k],k==='name'?150:4000);if(b.analysis)p.analysis=normalizeAnalysis(b.analysis);if(b.preferences)p.preferences=settings(b.preferences);for(const t of p.tasks)if(!t.customPrompt)t.prompt=taskPrompt(p,t.type,t.index,t.change||'');touch(p);await persist();return json(res,200,p)}
    if(method==='POST'&&rest==='analyze'){mutable(p);const b=await input(req,2000);if(p.preferences.provider==='openai'&&!b.cloudConsent)throw Error('云端识别需明确确认发送产品图及 API 费用');analyzeProject(p);await persist();return json(res,202,{ok:true})}
    if(method==='POST'&&rest==='replan'){mutable(p);if(p.tasks.some(t=>t.versions.length))throw Error('已有历史版本，请保留本项目并新建项目调整图片数量');const b=await input(req,10000);p.preferences=settings(b.preferences||p.preferences);p.tasks=newTasks(p);preferences=p.preferences;touch(p);await persist();return json(res,200,p)}
    if(method==='POST'&&rest==='generate'){
     const b=await input(req,20000);if(p.preferences.provider==='openai'&&!b.cloudConsent)throw Error('云端生成需明确确认发送图片及 API 费用');
     const selected=b.taskIds||p.tasks.filter(t=>!t.versions.length&&!['running','queued'].includes(t.state)).map(t=>t.id);if(!Array.isArray(selected)||selected.length>20)throw Error('任务数量无效');
     const ts=[...new Set(selected)].map(id=>task(p,id));for(const t of ts){if(['queued','running'].includes(t.state))continue;if(t.promptId&&t.state!=='complete'&&!b.regenerateConfirmed)throw Error('该任务已有 ComfyUI 编号，请先取回结果，或明确确认重新生成');if(b.prompt&&ts.length===1){t.prompt=cleanString(b.prompt,14000);t.customPrompt=true}if(b.change&&ts.length===1){t.change=cleanString(b.change,2000);t.prompt=taskPrompt(p,t.type,t.index,t.change);t.customPrompt=false}t.promptId=null;t.seed=randomInt(1,2147483647);enqueueTask(p,t)}await persist();return json(res,202,{queued:ts.map(t=>t.id)})
    }
    if(method==='GET'&&rest==='export.zip'){const zip=new JSZip();for(const t of p.tasks)for(const v of t.versions)zip.file(`${t.id}/v${v.version}-${v.review}.png`,await readFile(path.join(pdir(p),v.file)));zip.file('project.json',JSON.stringify(p,null,2));const bytes=await zip.generateAsync({type:'nodebuffer'});res.writeHead(200,{'Content-Type':'application/zip','Content-Disposition':`attachment; filename="mx-${p.id}.zip"`});return res.end(bytes)}
    const fileMatch=/^files\/([a-zA-Z0-9_-]+\.png)$/.exec(rest);
    if(method==='GET'&&fileMatch){const file=safeFile(fileMatch[1]);const known=p.references.includes(file)||p.tasks.some(t=>t.versions.some(v=>v.file===file||v.baseFile===file));if(!known)throw Error('图片不存在');const bytes=await readFile(path.join(pdir(p),file));res.writeHead(200,{'Content-Type':'image/png',...(url.searchParams.has('download')?{'Content-Disposition':`attachment; filename="${file}"`}:{})});return res.end(bytes)}
    const tm=/^tasks\/([a-z]+-\d+)\/(review|edit|recover|select)$/.exec(rest);
    if(method==='POST'&&tm){const t=task(p,tm[1]),b=await input(req,20000);if(['running','queued'].includes(t.state))throw Error('图片正在生成，请等待');
     if(tm[2]==='recover'){if(p.preferences.provider!=='local'||!t.promptId)throw Error('没有可以取回的本地任务');enqueueTask(p,t,true);await persist();return json(res,202,{ok:true})}
     const v=t.versions.find(v=>v.version===Number(b.version||t.currentVersion));if(!v)throw Error('版本不存在');
     if(tm[2]==='review'){if(!['pending','approved','rejected'].includes(b.review))throw Error('审核状态无效');v.review=b.review;v.reviewNote=cleanString(b.note,2000)}
     if(tm[2]==='select')t.currentVersion=v.version;
     if(tm[2]==='edit'){const layers=validateLayers(b.layers);if(t.type==='main'&&layers.some(l=>l.text))throw Error('默认主图不添加文字，请在详情图中编辑文案');await saveVersion(p,t,await readFile(path.join(pdir(p),v.baseFile)),layers,'文字层编辑')}
     touch(p);await persist();return json(res,200,p)
    }
   }
   return json(res,404,{error:'接口不存在'});
  }catch(e){return json(res,e.code==='ENOENT'?404:400,{error:safeError(e)})}
 });
 return {server,root,projects,persist,async close(){closed=true;while(draining)await new Promise(r=>setTimeout(r,10));await persistChain;await new Promise(r=>server.close(r))}};
}
if(process.argv[1]===fileURLToPath(import.meta.url)){
 const app=await createApp();const lockPath=path.join(app.root,'data','server.lock');let lock;
 try{lock=await open(lockPath,'wx')}catch{console.error('已有运行锁。请先运行停止程序，或确认没有服务运行后删除 data/server.lock。');process.exit(1)}
 await lock.writeFile(String(process.pid));app.server.listen(3188,'127.0.0.1',()=>console.log('MX Studio ready: http://127.0.0.1:3188'));
 const stop=async()=>{await app.close();await lock.close();const {unlink}=await import('node:fs/promises');await unlink(lockPath).catch(()=>{});process.exit(0)};process.on('SIGINT',stop);process.on('SIGTERM',stop);
}
