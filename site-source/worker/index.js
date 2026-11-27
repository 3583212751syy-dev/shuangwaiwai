const html=/*HTML*/;
const aiCore=/*AI_CORE*/,aiClient=/*AI_CLIENT*/,catalogModule=/*CATALOG*/;
const searchModule=/*SEARCH*/;
const specsModule=/*SPECS*/,ocrWeb=/*OCR_WEB*/,specsJson=/*SPECS_JSON*/,referencesJson=/*REFERENCES_JSON*/;
const overridesJson=/*OVERRIDES*/;
const json=(v,status=200)=>Response.json(v,{status,headers:{'Cache-Control':'no-store'}});
const safeUrl=raw=>{try{const u=new URL(raw);if(u.protocol!=='https:')return null;const h=u.hostname.toLowerCase();if(h==='localhost'||h==='0.0.0.0'||h==='127.0.0.1'||h==='::1'||h.endsWith('.local')||h.endsWith('.internal')||/^10\./.test(h)||/^192\.168\./.test(h)||/^169\.254\./.test(h)||/^172\.(1[6-9]|2\d|3[01])\./.test(h))return null;return u}catch{return null}};
export default {async fetch(req,env){
 const url=new URL(req.url),p=url.pathname;
 try{
 if(p==='/')return new Response(html,{headers:{'Content-Type':'text/html; charset=utf-8'}});
 const scripts={'/ai.mjs':aiCore,'/ai-client.mjs':aiClient,'/search.mjs':searchModule,'/catalog.mjs':catalogModule,'/specs.mjs':specsModule,'/ocr-web.js':ocrWeb};
 if(scripts[p])return new Response(scripts[p],{headers:{'Content-Type':'text/javascript; charset=utf-8','Cache-Control':'no-cache'}});
 // 查证参考库：平台 OCR + 人工目录录入，用于同款判定前的关键规格硬对比
 if(p==='/references.json')return new Response(referencesJson,{headers:{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-cache'}});
 if(p==='/specs.json')return new Response(specsJson,{headers:{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-cache'}});
 // 底库覆盖表：线上桶里的类目/向量是旧版，前端拿这张表覆盖品名、类目和 DINOv2 向量
 if(p==='/overrides.json')return new Response(overridesJson,{headers:{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-cache'}});
 if(p==='/api/proxy-image'&&req.method==='GET'){
  const u=safeUrl(url.searchParams.get('url')||'');if(!u)return json({error:'仅支持公网 https 图片地址'},400);
  const r=await fetch(u.href,{headers:{Accept:'image/*','User-Agent':'Mozilla/5.0'},signal:AbortSignal.timeout(20000),redirect:'follow'});
  if(!r.ok)return json({error:'图片地址返回 '+r.status},502);
  const type=(r.headers.get('content-type')||'').split(';')[0];
  if(!type.startsWith('image/'))return json({error:'地址未返回图片'},415);
  const size=Number(r.headers.get('content-length')||0);if(size>20*1024*1024)return json({error:'图片超过 20 MB'},413);
  const buf=await r.arrayBuffer();if(buf.byteLength>20*1024*1024)return json({error:'图片超过 20 MB'},413);
  return new Response(buf,{headers:{'Content-Type':type,'Cache-Control':'private, max-age=600','X-Content-Type-Options':'nosniff'}});
 }
 // 产品名称翻译：任意语言 → 中文。主源 Google 免费接口，备源 MyMemory（含 Autodetect 语种），
 // 都不可用时报错让用户直接填中文，绝不把失败包装成成功。
 if(p==='/api/translate'&&req.method==='POST'){
  const b=await req.json().catch(()=>null);
  const text=String(b?.text||'').trim().replace(/\s+/g,' ').slice(0,300);
  if(!text)return json({error:'请输入产品名称'},400);
  const cjk=(text.match(/[\u4e00-\u9fff]/g)||[]).length;
  if(cjk*2>=text.length)return json({original:text,translated:text,provider:'原文已是中文'});
  const providers=[
   async()=>{
    // MyMemory 先：支持 CORS 且国内可达；线上沙箱无外网时前端会直连兜底
    const r=await fetch('https://api.mymemory.translated.net/get?q='+encodeURIComponent(text)+'&langpair=Autodetect|zh-CN',{signal:AbortSignal.timeout(8000)});
    if(!r.ok)throw Error('mymemory '+r.status);
    const d=await r.json();
    const t=String(d?.responseData?.translatedText||'').trim();
    if(d?.responseStatus!==200||!t||t===text)throw Error('mymemory bad');
    const det=d.responseData.detectedLanguage&&d.responseData.detectedLanguage!=='zh-CN'?'（检测语言 '+d.responseData.detectedLanguage+'）':'';
    return [t,'MyMemory'+det];
   },
   async()=>{
    const r=await fetch('https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=zh-CN&dt=t&q='+encodeURIComponent(text),{signal:AbortSignal.timeout(8000)});
    if(!r.ok)throw Error('google '+r.status);
    const d=await r.json();
    const t=(d?.[0]||[]).map(x=>x?.[0]||'').join('').trim();
    if(!t)throw Error('google empty');
    return [t,'google'];
   },
  ];
  for(const fn of providers){
   try{const [translated,provider]=await fn();return json({original:text,translated:String(translated).trim().slice(0,300),provider})}
   catch(e){/* 换下一个源 */}
  }
  return json({error:'翻译服务暂不可用，可直接填写中文名称保存'},502);
 }
 // 人工命名覆盖表：识别不准时用户填的名称（任意语言自动翻译成中文后保存），按同款分组生效。
 // 存 BUCKET 'names/all' 一个 JSON；server 端写穿到磁盘 names-overrides.json，重启不丢。
 if(!env.BUCKET)return json({error:'图库存储尚未连接'},503);
 if(p==='/api/names'&&req.method==='GET'){
  const o=await env.BUCKET.get('names/all');
  return json(o?JSON.parse(await o.text()):{});
 }
 if(p==='/api/names'&&req.method==='POST'){
  const b=await req.json().catch(()=>null);
  const group=String(b?.group||'').trim().slice(0,180),name=String(b?.name||'').trim().slice(0,180),original=String(b?.original||'').trim().slice(0,300);
  if(!group||!name)return json({error:'需要分组名和产品名称'},400);
  const old=await env.BUCKET.get('names/all');
  const names=old?JSON.parse(await old.text()):{};
  names[group]={name,original:original||undefined,at:new Date().toISOString()};
  await env.BUCKET.put('names/all',JSON.stringify(names));
  return json(names,201);
 }
 if(p==='/api/images'&&req.method==='GET'){
  let cursor,items=[];
  do{const page=await env.BUCKET.list({prefix:'meta/',cursor});const data=await Promise.all(page.objects.map(async o=>{const r=await env.BUCKET.get(o.key);return r?JSON.parse(await r.text()):null}));items.push(...data.filter(Boolean));cursor=page.truncated?page.cursor:undefined}while(cursor);
  const slim=url.searchParams.get('slim')==='1';
  const rows=items.filter(x=>url.searchParams.get('trash')==='1'?!!x.deletedAt:!x.deletedAt).sort((a,b)=>b.created.localeCompare(a.created));
  return json(slim?rows.map(x=>({...x,ai:x.ai?{model:x.ai.model,status:x.ai.status,suggestedName:x.ai.suggestedName,suggestedCategory:x.ai.suggestedCategory,method:x.ai.method}:undefined})):rows);
 }
 if(p==='/api/images'&&req.method==='POST'){
  const f=await req.formData(),image=f.get('image');
  if(!image||!['image/jpeg','image/png','image/webp'].includes(image.type))return json({error:'仅支持 JPG、PNG、WebP 产品图片'},400);
  if(image.size>20*1024*1024)return json({error:'单张图片不能超过 20MB'},400);
  const id=crypto.randomUUID(),name=String(f.get('name')||image.name).slice(0,180),category=String(f.get('category')||'待确认').slice(0,60),group=String(f.get('group')||name).slice(0,180);
  const m={id,name,category,group,filename:image.name,size:image.size,created:new Date().toISOString(),source:'产品图',classification:'人工添加 / 人工确认'};
  await env.BUCKET.put('images/'+id,await image.arrayBuffer(),{httpMetadata:{contentType:image.type}});
  await env.BUCKET.put('meta/'+id,JSON.stringify(m));return json(m,201);
 }
 const match=p.match(/^\/api\/images\/([a-f0-9-]+)$/);
 if(match&&req.method==='PATCH'){
  const old=await env.BUCKET.get('meta/'+match[1]);if(!old)return json({error:'图片不存在'},404);
  const m=JSON.parse(await old.text()),body=await req.json();for(const k of ['name','category','group'])if(typeof body[k]==='string'&&body[k].trim())m[k]=body[k].trim().slice(0,180);
  if(typeof body.deleted==='boolean'){if(body.deleted)m.deletedAt=new Date().toISOString();else delete m.deletedAt;}
  if(Array.isArray(body.confirmations))m.confirmations=body.confirmations.slice(-200).filter(x=>x&&['same','different'].includes(x.decision)).map(x=>({decision:x.decision,query:String(x.query||'').slice(0,180),querySha256:/^[a-f0-9]{64}$/.test(x.querySha256||'')?x.querySha256:undefined,groupId:typeof x.groupId==='string'?x.groupId.slice(0,180):undefined,at:typeof x.at==='string'?x.at:new Date().toISOString()}));
  if(body.ai){
   const a=body.ai;
   if(a.model!=='Xenova/clip-vit-base-patch32'||!['待确认','已识别','人工确认'].includes(a.status)||!Array.isArray(a.embedding)||a.embedding.length!==512||a.embedding.some(x=>!Number.isFinite(x)||Math.abs(x)>1.01))return json({error:'识图数据格式不正确'},400);
   const preservedDino=Object.fromEntries(['dinoEmbedding','dinoModel','dinoPooling'].filter(k=>m.ai?.[k]!==undefined).map(k=>[k,m.ai[k]]));
   // MobileCLIP 检索主向量：512 维 fp32 归一化；只更新 CLIP/标签时保留已有 MC 数据
   const preservedMc=Object.fromEntries(['mcEmbedding','mcModel'].filter(k=>m.ai?.[k]!==undefined).map(k=>[k,m.ai[k]]));
   if(a.mcEmbedding!==undefined){
    if(!Array.isArray(a.mcEmbedding)||a.mcEmbedding.length!==512||a.mcEmbedding.some(x=>!Number.isFinite(x)||Math.abs(x)>1.01)||!a.mcEmbedding.some(x=>x!==0)||
       (a.mcModel&&a.mcModel!=='Xenova/mobileclip_s0'))return json({error:'MobileCLIP 数据格式不正确'},400);
    Object.assign(preservedMc,{mcEmbedding:a.mcEmbedding,mcModel:'Xenova/mobileclip_s0'});
   }
   if(a.dinoEmbedding!==undefined){
    if(!Array.isArray(a.dinoEmbedding)||a.dinoEmbedding.length!==384||a.dinoEmbedding.some(x=>!Number.isFinite(x)||Math.abs(x)>1.01)||!a.dinoEmbedding.some(x=>x!==0)||
       (a.dinoModel&&a.dinoModel!=='onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX')||(a.dinoPooling&&a.dinoPooling!=='cls-v1'))return json({error:'DINO 数据格式不正确'},400);
    Object.assign(preservedDino,{dinoEmbedding:a.dinoEmbedding,dinoModel:'onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX',dinoPooling:'cls-v1'});
   }
   m.ai={...preservedDino,...preservedMc,model:a.model,version:'3.8.1',status:a.status,suggestedName:String(a.suggestedName||'').slice(0,100),suggestedCategory:String(a.suggestedCategory||'待确认').slice(0,50),score:Math.min(1,Math.max(0,Number(a.score)||0)),embedding:a.embedding,at:new Date().toISOString()};
   if(typeof a.method==='string')m.ai.method=a.method.slice(0,40);
   m.classification='CLIP 图片内容识别';
  }
  await env.BUCKET.put('meta/'+m.id,JSON.stringify(m));return json(m);
 }
 const im=p.match(/^\/images\/([a-f0-9-]+)$/);if(im){const o=await env.BUCKET.get('images/'+im[1]);if(!o)return new Response('Not found',{status:404});return new Response(o.body,{headers:{'Content-Type':o.httpMetadata?.contentType||'image/jpeg','Cache-Control':'private, max-age=3600','X-Content-Type-Options':'nosniff'}})}
 // 未匹配的页面路径一律回首页，避免从应用面板等非根路径入口打开时出现 "Not found"。
 // API 与图片路径保持真实 404，不吞掉错误。
 if((req.method==='GET'||req.method==='HEAD')&&!p.startsWith('/api/'))return new Response(html,{headers:{'Content-Type':'text/html; charset=utf-8'}});
 return json({error:'Not found'},404);
 }catch(e){return json({error:'操作失败：'+String(e?.message||e).slice(0,120)},500)}
}};

