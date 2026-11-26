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
 // ��ͼ���⣺���� OCR + ����Ŀ��¼�룬����ͬ���ж�ǰ�Ĺؼ����Ӳ�Ա�
 if(p==='/references.json')return new Response(referencesJson,{headers:{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-cache'}});
 if(p==='/specs.json')return new Response(specsJson,{headers:{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-cache'}});
 // �׿⸲�Ǳ�������Ͱ�����Ŀ/�����Ǿɰ棬ǰ�������ű�����Ʒ������Ŀ�� DINOv2 ����
 if(p==='/overrides.json')return new Response(overridesJson,{headers:{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-cache'}});
 if(p==='/api/proxy-image'&&req.method==='GET'){
  const u=safeUrl(url.searchParams.get('url')||'');if(!u)return json({error:'��֧�ֹ��� https ͼƬ��ַ'},400);
  const r=await fetch(u.href,{headers:{Accept:'image/*','User-Agent':'Mozilla/5.0'},signal:AbortSignal.timeout(20000),redirect:'follow'});
  if(!r.ok)return json({error:'ͼƬ��ַ���� '+r.status},502);
  const type=(r.headers.get('content-type')||'').split(';')[0];
  if(!type.startsWith('image/'))return json({error:'��ַδ����ͼƬ'},415);
  const size=Number(r.headers.get('content-length')||0);if(size>20*1024*1024)return json({error:'ͼƬ���� 20 MB'},413);
  const buf=await r.arrayBuffer();if(buf.byteLength>20*1024*1024)return json({error:'ͼƬ���� 20 MB'},413);
  return new Response(buf,{headers:{'Content-Type':type,'Cache-Control':'private, max-age=600','X-Content-Type-Options':'nosniff'}});
 }
 if(!env.BUCKET)return json({error:'ͼ��洢��δ����'},503);
 if(p==='/api/images'&&req.method==='GET'){
  let cursor,items=[];
  do{const page=await env.BUCKET.list({prefix:'meta/',cursor});const data=await Promise.all(page.objects.map(async o=>{const r=await env.BUCKET.get(o.key);return r?JSON.parse(await r.text()):null}));items.push(...data.filter(Boolean));cursor=page.truncated?page.cursor:undefined}while(cursor);
  const slim=url.searchParams.get('slim')==='1';
  const rows=items.filter(x=>url.searchParams.get('trash')==='1'?!!x.deletedAt:!x.deletedAt).sort((a,b)=>b.created.localeCompare(a.created));
  return json(slim?rows.map(x=>({...x,ai:x.ai?{model:x.ai.model,status:x.ai.status,suggestedName:x.ai.suggestedName,suggestedCategory:x.ai.suggestedCategory,method:x.ai.method}:undefined})):rows);
 }
 if(p==='/api/images'&&req.method==='POST'){
  const f=await req.formData(),image=f.get('image');
  if(!image||!['image/jpeg','image/png','image/webp'].includes(image.type))return json({error:'��֧�� JPG��PNG��WebP ��ƷͼƬ'},400);
  if(image.size>20*1024*1024)return json({error:'����ͼƬ���ܳ��� 20MB'},400);
  const id=crypto.randomUUID(),name=String(f.get('name')||image.name).slice(0,180),category=String(f.get('category')||'������').slice(0,60),group=String(f.get('group')||name).slice(0,180);
  const m={id,name,category,group,filename:image.name,size:image.size,created:new Date().toISOString(),source:'��Ʒͼ',classification:'���ƹ��� / �˹�ȷ��'};
  await env.BUCKET.put('images/'+id,await image.arrayBuffer(),{httpMetadata:{contentType:image.type}});
  await env.BUCKET.put('meta/'+id,JSON.stringify(m));return json(m,201);
 }
 const match=p.match(/^\/api\/images\/([a-f0-9-]+)$/);
 if(match&&req.method==='PATCH'){
  const old=await env.BUCKET.get('meta/'+match[1]);if(!old)return json({error:'ͼƬ������'},404);
  const m=JSON.parse(await old.text()),body=await req.json();for(const k of ['name','category','group'])if(typeof body[k]==='string'&&body[k].trim())m[k]=body[k].trim().slice(0,180);
  if(typeof body.deleted==='boolean'){if(body.deleted)m.deletedAt=new Date().toISOString();else delete m.deletedAt;}
  if(Array.isArray(body.confirmations))m.confirmations=body.confirmations.slice(-200).filter(x=>x&&['same','different'].includes(x.decision)).map(x=>({decision:x.decision,query:String(x.query||'').slice(0,180),querySha256:/^[a-f0-9]{64}$/.test(x.querySha256||'')?x.querySha256:undefined,groupId:typeof x.groupId==='string'?x.groupId.slice(0,180):undefined,at:typeof x.at==='string'?x.at:new Date().toISOString()}));
  if(body.ai){
   const a=body.ai;
   if(a.model!=='Xenova/clip-vit-base-patch32'||!['��ȷ��','��ʶ��','�˹�ȷ��'].includes(a.status)||!Array.isArray(a.embedding)||a.embedding.length!==512||a.embedding.some(x=>!Number.isFinite(x)||Math.abs(x)>1.01))return json({error:'ʶͼ�����ʽ����ȷ'},400);
   const preservedDino=Object.fromEntries(['dinoEmbedding','dinoModel','dinoPooling'].filter(k=>m.ai?.[k]!==undefined).map(k=>[k,m.ai[k]]));
   if(a.dinoEmbedding!==undefined){
    if(!Array.isArray(a.dinoEmbedding)||a.dinoEmbedding.length!==384||a.dinoEmbedding.some(x=>!Number.isFinite(x)||Math.abs(x)>1.01)||!a.dinoEmbedding.some(x=>x!==0)||
       (a.dinoModel&&a.dinoModel!=='onnx-community/dinov2-small')||(a.dinoPooling&&a.dinoPooling!=='all-tokens-mean-v1'))return json({error:'DINO ������ʽ����ȷ'},400);
    Object.assign(preservedDino,{dinoEmbedding:a.dinoEmbedding,dinoModel:'onnx-community/dinov2-small',dinoPooling:'all-tokens-mean-v1'});
   }
   m.ai={...preservedDino,model:a.model,version:'3.8.1',status:a.status,suggestedName:String(a.suggestedName||'').slice(0,100),suggestedCategory:String(a.suggestedCategory||'������').slice(0,50),score:Math.min(1,Math.max(0,Number(a.score)||0)),embedding:a.embedding,at:new Date().toISOString()};
   if(typeof a.method==='string')m.ai.method=a.method.slice(0,40);
   m.classification='CLIP ͼƬ����ʶ��';
  }
  await env.BUCKET.put('meta/'+m.id,JSON.stringify(m));return json(m);
 }
 const im=p.match(/^\/images\/([a-f0-9-]+)$/);if(im){const o=await env.BUCKET.get('images/'+im[1]);if(!o)return new Response('Not found',{status:404});return new Response(o.body,{headers:{'Content-Type':o.httpMetadata?.contentType||'image/jpeg','Cache-Control':'private, max-age=3600','X-Content-Type-Options':'nosniff'}})}
 // δƥ���ҳ��·��һ�ɻ���ҳ�������Ӧ�����ȷǸ�·����ڴ�ʱ���� "Not found"��
 // API ��ͼƬ·��������ʵ 404�����̵�����
 if((req.method==='GET'||req.method==='HEAD')&&!p.startsWith('/api/'))return new Response(html,{headers:{'Content-Type':'text/html; charset=utf-8'}});
 return json({error:'Not found'},404);
 }catch(e){return json({error:'����ʧ�ܣ�'+String(e?.message||e).slice(0,120)},500)}
}};

