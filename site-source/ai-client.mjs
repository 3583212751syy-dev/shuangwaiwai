import {createRecognizer,createDinoEmbedder,sourceGroup,identify,catalogAnalysis} from './ai.mjs';
import {searchLibrary,READY,DINO_READY,candidateVerdict} from './search.mjs';
import {effectiveLabel,catalogFor,CATEGORIES,groupProducts,isDefaultName} from './catalog.mjs';
import {parseSpecText,summarize as summarizeSpecs,compareSpecs,compareGroupSpecs} from './specs.mjs';

const $=id=>document.getElementById(id);
let library=[],queries=[],recognizer,loading,working=false,indexing=false,baseQueue=[],page=36,pasteTarget='query';
let grouped=false,pendingOnly=false,selected=new Set(),topK=8,overlay=false,compareState=null;

const el=(tag,text,cls)=>{const x=document.createElement(tag);if(text!==undefined&&text!==null)x.textContent=text;if(cls)x.className=cls;return x};
const btn=(text,fn,cls)=>{const b=el('button',text,cls);b.onclick=fn;return b};
function status(id,text,err=false){const n=$(id);n.textContent=text;n.classList.toggle('error',err)}
async function api(p,opt={}){const r=await fetch(p,opt);if(!r.ok){let msg;try{msg=(await r.json()).error}catch{}throw Error(msg||'请求失败（'+r.status+'），请重试')}return r.json()}
const patch=(id,body)=>api('/api/images/'+id,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
function image(src,alt){const im=el('img');im.src=src;im.alt=alt;im.loading='lazy';return im}
function readyLabel(x){return READY(x)?'已保存 · 可检索':'已保存 · 正在准备检索'}

/* ---------- 入库与索引 ---------- */
function initFilters(){
 const cats=[...new Set([...CATEGORIES,...library.map(x=>effectiveLabel(x).category)])].filter(c=>!LEGACY_CATEGORY[c]);
 for(const id of ['category-filter','edit-category','batch-category']){
  const s=$(id),old=s.value;s.replaceChildren();
  if(id!=='category-filter'&&id==='batch-category')s.append(el('option','不修改',''));
  if(id==='category-filter')s.append(el('option','全部类别',''));
  for(const c of cats){const o=el('option',c,c);s.append(o)}
  s.value=old||'';
 }
}
function filtered(){
 const q=$('library-search').value.trim().toLowerCase(),cat=$('category-filter').value;
 return library.filter(x=>{
  const l=effectiveLabel(x);
  if(cat&&l.category!==cat)return false;
  if(pendingOnly&&!(x.ai?.status==='待确认'||l.category==='Otras Categorías'||l.name==='产品待确认'))return false;
  return [l.name,l.category,l.group,x.filename].join(' ').toLowerCase().includes(q);
 });
}
function card(x){
 const l=effectiveLabel(x),c=el('article',null,'card');
 const pick=el('input');pick.type='checkbox';pick.className='pick';pick.checked=selected.has(x.id);
 Object.assign(pick.style,{position:'absolute',top:'8px',left:'8px',zIndex:'2',width:'18px',height:'18px'});
 pick.onchange=()=>{pick.checked?selected.add(x.id):selected.delete(x.id)};
 const im=image('/images/'+x.id,l.name);im.onclick=()=>view(im.src,l.name);
 const body=el('div',null,'body');body.append(el('h3',l.name));
 const badges=el('div');const cat=el('span',l.category,'badge');body.append(badges);
 badges.append(cat,el('span',READY(x)?'可检索':'待准备',READY(x)?'badge':'badge warn'));
 if(x.ai?.status==='待确认')badges.append(el('span','待确认','badge warn'));
 body.append(el('div',l.group+(l.view?' · '+l.view+'视图':''),'sub'));
 body.append(el('div',readyLabel(x),'sub'));
 body.append(btn('编辑信息',()=>openEdit(x)),btn('移出底图库',async()=>{remove.disabled=true;try{await patch(x.id,{deleted:true});selected.delete(x.id);await refresh()}catch(e){status('base-status',e.message,true)}finally{remove.disabled=false}}));
 const remove=body.lastChild;
 c.append(pick,im,body);return c;
}
function renderLibrary(){
 const rows=filtered(),groups=groupProducts(library);
 $('library-count').textContent=library.length+' 张底图 · '+groups.length+' 个同款分组 · '+library.filter(READY).length+' 张可检索';
 const pending=library.filter(x=>x.ai?.status==='待确认'||effectiveLabel(x).category==='Otras Categorías').length;
 $('library-stat').textContent='待确认 '+pending+' 张 · 来源：'+(library.filter(x=>effectiveLabel(x).source==='产品目录').length)+' 张来自产品目录 · '+library.filter(x=>effectiveLabel(x).source==='人工编辑').length+' 张人工编辑';
 const box=$('library');box.replaceChildren();
 if(!rows.length){box.append(el('div',library.length?'没有匹配的底图':'底图库为空，请先添加产品图片。','empty'));}
 else if(grouped){
  for(const g of groupProducts(rows)){
   const head=el('div',null,'group-head');head.append(el('span',g.name),el('span',g.category+' · '+g.items.length+' 张多角度'));
   const grid=el('div',null,'grid');for(const item of g.items)grid.append(card(item));
   box.append(head,grid);
  }
  box.style.display='block';
 }else{
  box.style.display='grid';for(const x of rows.slice(0,page))box.append(card(x));
 }
 $('library-more').classList.toggle('hidden',grouped||rows.length<=page);
 $('catalog-sync').classList.toggle('hidden',!library.some(needsSync));
}
function needsSync(x){const c=catalogFor(x.filename);return isDefaultName(x.name)&&c&&c.name!=='产品待确认'}

/* ---------- 底库覆盖表（线上桶里的类目/向量是旧版时用它兜底） ---------- */
// 早期底库用的自拟中文类目，显示/筛选时统一折算成美客多官方一级类目
const LEGACY_CATEGORY={'汽配':'Accesorios para Vehículos','五金工具':'Herramientas','电子电气':'Electrónica, Audio y Video','家居用品':'Hogar, Muebles y Jardín','宠物用品':'Mascotas','待分类':'Otras Categorías'};
let overridesLoaded=null;
async function loadOverrides(){
 if(overridesLoaded!==null)return overridesLoaded;
 try{const r=await fetch('/overrides.json',{cache:'no-store'});overridesLoaded=r.ok?await r.json():null}catch{overridesLoaded=null}
 return overridesLoaded;
}
function decodeDino(d){
 if(!d?.b64||!d.files?.length)return {};
 const bin=atob(d.b64),dim=d.dim||384,out={};
 for(let i=0;i<d.files.length;i++){
  const v=new Array(dim);
  for(let k=0;k<dim;k++){const b=bin.charCodeAt(i*dim+k);v[k]=(b>127?b-256:b)/127}
  out[d.files[i]]=v;
 }
 return out;
}
async function applyOverrides(list){
 const o=await loadOverrides();if(!o)return list;
 const dino=o.dino?decodeDino(o.dino):{};
 let relabelled=0,withDino=0;
 for(const x of list){
  const lab=o.labels&&x.row!=null?o.labels[String(x.row)]:null;
  if(lab&&lab[0]){x.name=lab[0];x.category=lab[1]||x.category;x.group='第'+x.row+'行 · '+lab[0];x.labelSource='底图库标签';relabelled++;
 // 覆盖表同步确认状态：人工已定名的不再显示「待确认」，待复核组如实保留
   if(lab[2]&&x.ai)x.ai.status=lab[2]}
 // 覆盖表没覆盖到的（新上传、老数据）也把旧中文类目折算成美客多官方口径
  if(LEGACY_CATEGORY[x.category])x.category=LEGACY_CATEGORY[x.category];
  const vec=dino[x.filename];
  if(vec&&x.ai&&!DINO_READY(x)){x.ai.dinoEmbedding=vec;x.ai.dinoModel='onnx-community/dinov2-small';x.ai.dinoPooling='all-tokens-mean-v1';withDino++}
 }
 if(relabelled||withDino)console.log('底库覆盖表已应用：标签 '+relabelled+' 条、DINOv2 向量 '+withDino+' 条');
 return list;
}
async function refresh(){library=await applyOverrides(await api('/api/images'));initFilters();renderLibrary()}

/* ---------- 编辑 / 批量 / 导出 ---------- */
function openEdit(x){
 const l=effectiveLabel(x);$('edit-name').value=l.name;$('edit-category').value=l.category;$('edit-group').value=l.group;
 $('edit-dialog').dataset.id=x.id;$('edit-dialog').showModal();
}
async function savePatch(id,body){const updated=await patch(id,body),i=library.findIndex(x=>x.id===id);if(i>=0)library[i]=updated;return updated}
function exportCsv(){
 const head=['文件名','产品名称','类别','同款分组','识别状态','识别方式','标签来源','可检索','创建时间'];
 const rows=[head];
 for(const x of library){const l=effectiveLabel(x);rows.push([x.filename,l.name,l.category,l.group,x.ai?.status||'未识别',x.ai?.method||x.ai?.model||'',l.source,READY(x)?'是':'否',x.created])}
 const csv='\ufeff'+rows.map(r=>r.map(v=>'"'+String(v??'').replace(/"/g,'""')+'"').join(',')).join('\r\n');
 const a=el('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));a.download='底图分类清单-'+new Date().toISOString().slice(0,10)+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),2000);
}
async function mapLimit(items,limit,fn){
 const out=[];let i=0;
 const workers=Array.from({length:Math.min(limit,items.length)},async()=>{while(i<items.length){const n=i++;try{out[n]=await fn(items[n],n)}catch(e){out[n]={error:e.message}}}});
 await Promise.all(workers);return out;
}
async function syncCatalog(){
 const todo=library.filter(needsSync);if(!todo.length)return status('base-status','没有需要回填的底图');
 $('catalog-sync').disabled=true;let done=0;
 await mapLimit(todo,3,async x=>{
  const c=catalogFor(x.filename),body={name:c.name,category:c.category,group:c.group};
  if(READY(x))body.ai={...x.ai,suggestedName:c.name,suggestedCategory:c.category,status:'已识别'};
  await savePatch(x.id,body);done++;status('base-status','正在回填产品目录标签 '+done+' / '+todo.length);
 });
 $('catalog-sync').disabled=false;renderLibrary();status('base-status','已回填 '+done+' 张底图的真实产品名称与类目。');
}

/* ---------- 入库与索引 ---------- */
async function localModelBase(){
 try{const r=await fetch('/models/Xenova/clip-vit-base-patch32/config.json',{signal:AbortSignal.timeout(4000)});if(r.ok){const t=await r.text();if(t.includes('model_type')||t.includes('clip'))return '/models/'}}catch{}
 return null;
}
async function ready(report){
 if(recognizer)return recognizer;if(loading)return loading;
 loading=(async()=>{
  const base=await localModelBase();
  report(base?'正在从本机加载识图模型…':'首次加载识图模型，约 155 MB，请保持页面打开…');
  const lib=await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1');
  recognizer=await createRecognizer(lib,p=>{if(p.status==='progress')report('正在加载识图模型 '+Math.round(p.progress||0)+'%')},'wasm',base);
  return recognizer;
 })().catch(e=>{loading=null;throw Error('识图模型加载失败，请检查网络后重试。'+e.message)});
 return loading;
}
let dinoEmbedder,dinoLoading;
async function readyDino(report){
 if(dinoEmbedder)return dinoEmbedder;if(dinoLoading)return dinoLoading;
 dinoLoading=(async()=>{
  const base=await localModelBase();if(!base)return null;
  const lib=await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1');
  dinoEmbedder=await createDinoEmbedder(lib,p=>{if(p.status==='progress')report('正在加载第二视觉模型 '+Math.round(p.progress||0)+'%')},'wasm',base);
  return dinoEmbedder;
 })().catch(e=>{dinoLoading=null;console.warn('DINOv2 加载失败，退回单模型：'+e.message);return null});
 return dinoLoading;
}
async function rawAnalyze(source,report){
 const analysis=await (await ready(report)).analyze(source);
 try{
  const dino=await readyDino(report);
  if(dino){analysis.dinoEmbedding=await dino.embed(source);analysis.dinoModel='onnx-community/dinov2-small';analysis.dinoPooling='all-tokens-mean-v1'}
 }catch(e){console.warn('DINOv2 特征失败，本次只用 CLIP：'+e.message)}
 return analysis;
}
// 平台查证参考库：人工确认过的真实产品图（1688/美客多查证）做视觉锚点。
// CLIP 零样本在细粒度工业件上只有 0.2 分（等同瞎猜），所以先用锚点余弦比对，命中就用查证过的品名/类目。
let referencesLib=null;
async function loadReferences(){
 if(referencesLib!==null)return referencesLib;
 try{const r=await fetch('/references.json',{cache:'no-store'});referencesLib=r.ok?await r.json():null}catch{referencesLib=null}
 return referencesLib;
}
function cosine(a,b){if(a?.length!==b?.length||!a?.length)return -1;let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s}
const REF_MIN=0.82; // 同一产品不同角度典型 >0.85；形近件 <0.8
async function matchReference(q){
 try{
  const lib=await loadReferences();if(!lib?.items?.length||!q.embedding)return null;
  let best=null;
  for(const it of lib.items){
   const s=cosine(q.embedding,it.embedding);
   if(!best||s>best.score)best={item:it,score:s};
  }
  if(!best||best.score<REF_MIN)return null;
  return {name:best.item.name,category:best.item.category,sub:best.item.sub,subZh:best.item.subZh,source:best.item.source,score:best.score};
 }catch(e){console.warn('参考库匹配失败：',e.message);return null}
}
async function analyzeQuery(source,report){
 const a=identify(await rawAnalyze(source,report),library);
 const ref=await matchReference(a);
 if(ref){
  a.referenceMatch=ref;
  a.suggestedName=ref.name;a.suggestedCategory=ref.category;a.referenceSub=ref.sub;a.referenceSubZh=ref.subZh;
  a.method='平台查证参考图匹配';a.status='已识别';a.confident=true;
 }
 return a;
}

/* ---------- 入库与索引 ---------- */
function syncBody(x){const c=catalogFor(x.filename);if(!c||c.name==='产品待确认'||!isDefaultName(x.name))return {};return {name:c.name,category:c.category,group:c.group}}
async function prepareIndex(){
 if(indexing||working)return;
 const todo=library.filter(x=>!READY(x)||!DINO_READY(x));
 if(!todo.length){$('index-retry').classList.add('hidden');return}
 indexing=true;renderQueries();$('index-retry').classList.add('hidden');
 let done=0;
 try{
  const outcome=await mapLimit(todo,2,async(x,n)=>{
   for(let attempt=0;attempt<2;attempt++){
    try{
     const a=await rawAnalyze('/images/'+x.id,t=>status('base-status',t));
     const body={...syncBody(x),ai:catalogAnalysis(x,a)};
     Object.assign(x,await patch(x.id,body));done++;
     status('base-status','底图已保存，正在准备检索 '+done+' / '+todo.length);
     if(done%5===0)renderLibrary();return;
    }catch(e){if(attempt===1)throw e}
   }
  });
  const failed=outcome.filter(r=>r&&r.error);
  if(failed.length)throw Error(failed.length+' 张底图准备失败：'+failed[0].error);
  status('base-status','底图已保存并可用于检索，无需逐张确认。');
 }catch(e){status('base-status','原图已保存；检索准备未完成：'+e.message,true);$('index-retry').classList.remove('hidden')}
 finally{indexing=false;renderQueries();if(baseQueue.length)drainBase()}
}
function validate(files){
 const good=[],bad=[];
 for(const f of files){if(!/^image\/(jpeg|png|webp|gif|bmp)$/i.test(f.type)){bad.push(f.name+'：不支持此格式');continue}if(f.size>20*1024*1024){bad.push(f.name+'：超过 20 MB');continue}good.push(f)}
 return {good,bad};
}
async function jpeg(file){if(['image/jpeg','image/png','image/webp'].includes(file.type))return file;const bitmap=await createImageBitmap(file),canvas=document.createElement('canvas');canvas.width=bitmap.width;canvas.height=bitmap.height;canvas.getContext('2d').drawImage(bitmap,0,0);bitmap.close();const blob=await new Promise(r=>canvas.toBlob(r,'image/png'));if(!blob)throw Error('图片转换失败');return new File([blob],file.name.replace(/\.[^.]+$/,'')+'.png',{type:'image/png'})}
async function drainBase(){
 if(indexing||working)return;working=true;renderQueries();
 const batch=baseQueue.shift();if(!batch){working=false;return}
 const files=batch.files;let ok=0,failed=[];
 $('base-progress').classList.remove('hidden');$('base-progress').max=files.length;
 try{
  for(let i=0;i<files.length;i++){
   try{
    const f=await jpeg(files[i]),data=new FormData();
    data.append('image',f);data.append('name',f.name.replace(/\.[^.]+$/,''));data.append('category','参考底图');
    data.append('group',sourceGroup(f.name)||batch.group);
    status('base-status','正在保存底图 '+(i+1)+' / '+files.length);
    library.unshift(await api('/api/images',{method:'POST',body:data}));ok++;
   }catch(e){failed.push(files[i]);status('base-status',e.message,true)}
   $('base-progress').value=i+1;
  }
  renderLibrary();status('base-status','已保存 '+ok+' 张底图'+(failed.length?'；'+failed.length+' 张失败，可重试':'，正在准备检索。'),!!failed.length);
  if(failed.length)baseQueue.unshift({files:failed,group:batch.group});
 }finally{
  working=false;$('base-progress').classList.add('hidden');renderQueries();
  if(baseQueue.length)drainBase();else prepareIndex();
 }
}
function addBase(files){
 const {good,bad}=validate(files);if(bad.length)status('base-status',bad.join('；'),true);if(!good.length)return;
 baseQueue.push({files:good,group:'产品 '+new Date().toLocaleString('zh-CN',{hour12:false}).replace(/[^\d]/g,'').slice(0,14)+'-'+Math.random().toString(36).slice(2,6)});
 if(working||indexing)status('base-status','已加入底图保存队列 '+baseQueue.length+' 批');else drainBase();
}
function addQueries(files){
 const {good,bad}=validate(files);if(bad.length)status('query-status',bad.join('；'),true);
 for(const file of good)queries.push({file,url:URL.createObjectURL(file),analysis:null});
 renderQueries();
 if(good.length){status('query-status','已加入 '+queries.length+' 张对比图片，点击“开始查找”。');$('query-drop').focus()}
}
// 单张对比图片右上角的叉：立即移除，不必清空全部。
function removeQuery(i){
 if(working)return status('query-status','正在查找中，暂不能移除图片。',true);
 const q=queries[i];if(!q)return;
 URL.revokeObjectURL(q.url);queries.splice(i,1);renderQueries();
 status('query-status','已移除 1 张对比图片，剩余 '+queries.length+' 张。');
}
function renderQueries(){
 const box=$('query-preview');
 box.replaceChildren(...queries.map((x,i)=>{
  const wrap=el('div',null,'qitem'),im=image(x.url,x.file.name),del=el('button','×','qdel');
  del.title='移除这张对比图片';del.setAttribute('aria-label','移除这张对比图片');
  del.onclick=()=>removeQuery(i);
  wrap.append(im,del);return wrap;
 }));
 $('query-run').disabled=!queries.length||working||indexing;$('query-clear').disabled=working;
}

/* ---------- 放大 / 对比 ---------- */
function view(src,title){compareState=null;$('viewer-title').textContent=title;$('viewer-body').replaceChildren(Object.assign(image(src,title),{className:'full'}));$('opacity-row').classList.add('hidden');$('viewer-mode').classList.add('hidden');$('viewer').showModal()}
function renderCompare(){
 if(!compareState)return;
 $('viewer-title').textContent='对比产品外形与细节（重点核对型号 / 规格 / 数量）';
 $('viewer-body').replaceChildren();$('viewer-mode').classList.remove('hidden');
 if(!overlay){
  const box=el('div',null,'compare');box.append(image(compareState.q,'上传的对比图片'),image(compareState.c,'图库产品底图'));$('viewer-body').append(box);$('opacity-row').classList.add('hidden');
 }else{
  const wrap=el('div',null,'overlay-wrap'),a=image(compareState.c,'图库产品底图'),b=image(compareState.q,'叠加的对比图片');b.className='over';b.style.opacity=$('opacity').value/100;
  wrap.append(a,b);$('viewer-body').append(wrap);$('opacity-row').classList.remove('hidden');
 }
 $('viewer-mode').textContent=overlay?'切换并排对比':'切换叠加对比';
}
function compare(q,candidate){compareState={q:q.url,c:'/images/'+candidate.id};overlay=false;renderCompare();$('viewer').showModal()}

/* ---------- 读图规格硬否决 ----------
 * 参考 ebiarian/product-image-description-alignment（MIT）：关键特征不一致即否决。
 * 不同于旧版 tesseract wasm（线上不稳定）：这里用 ppu-paddle-ocr/web + PP-OCRv6 tiny
 * 直接在浏览器读上传图，型号出现次数同时作为「画面里有几只」的信号。
 * 底图规格来自离线 OCR + 逐张目检生成的 /specs.json，不占线上服务资源。
 */
let specsLib=null,ocrModule=null;
async function loadSpecsLib(){
 if(specsLib!==null)return specsLib;
 try{const r=await fetch('/specs.json',{cache:'no-store'});specsLib=r.ok?await r.json():null}catch{specsLib=null}
 return specsLib;
}
function entryToParsed(e){
 if(!e)return null;
 const counts={...(e.counts||{})};
 for(const t of [...(e.curves||[]),...(e.models||[])])counts[t]=counts[t]||1;
 return {tokens:Object.keys(counts),counts,curves:e.curves||[],models:e.models||[],poles:e.poles||[],verified:e.verified===true,quantity:e.quantity??null};
}
async function readSpecs(file,report=()=>{}){
 try{
  if(!ocrModule)ocrModule=await import('/ocr-web.js');
  const buf=file instanceof ArrayBuffer?file:await file.arrayBuffer();
  const text=await ocrModule.recognizeText(buf,report);
  const p=parseSpecText(text);
  return {...p,text:String(text||'')};
 }catch(e){console.warn('规格 OCR 不可用：',e.message);return null}
}
async function librarySpecs(filename){
 const lib=await loadSpecsLib();
 return entryToParsed(lib?.specs?.[filename]);
}

/* ---------- 检索结果 ---------- */
 // 同款判定：看「首位领先第二位的幅度」。真同款必然明显领先；
// 库里没有同款时，前几名会挤在一起（实测三张非同款图领先幅度都只有 1.02~1.11 倍，真同款是 2.4 倍）。
function verdict(hit){return candidateVerdict(hit)}
// 读到的电气型号能纠正 CLIP 的类目幻觉（实测把 DZ47 断路器判成「健康医疗」）
function specCategory(p){
 const m=(p?.models||[]).join(' '),c=p?.curves||[],text=(p?.text||'')+' '+m;
 if(/DZ47|DZ808|ZBM1|iC65|5SX|NC65|BREAKER|断路器/i.test(text)&&c.some(x=>/^[BCD]\d{1,2}$/.test(x)))
  return {category:'Electrónica, Audio y Video',categoryZh:'电子音频视频',sub:'Componentes Electrónicos',subZh:'电子元件',
   name:'小型断路器 '+c.filter(x=>/^[BCD]\d{1,2}$/.test(x)).join('/')};
 return null;
}
// ① 识别结果三行：类别 / 分类 / 名称。命中同款用底图库人工标签，否则读图规格优先于 CLIP 零样本。
function idRows(a,top,vd,q){
 const rows=[];
 const libCat=vd.ok&&top?top.label.category:null;
 const specCat=libCat?null:specCategory(q?._specCheck?.querySpecs);
 const cands=a?.mlCandidates||[],ml=cands[0];
 const ref=a?.referenceMatch;
 const cat=libCat||ref?.category||specCat?.category||ml?.category||a?.mlCategory||null;
 const catZh=libCat?null:(ref?(ref.category==='Hogar, Muebles y Jardín'?'家居家具园艺':(ref.category==='Accesorios para Vehículos'?'汽车配件':null)):(specCat?.categoryZh||ml?.categoryZh||a?.mlCategoryZh));
 const conf=ml?.score;
 // 低置信不采信：top1 <0.30 时 CLIP 等同瞎猜（实测防火毯被判"美术文具"0.23），不报肯定类目，只列最接近候选
 const noTrust=!libCat&&!ref&&!specCat&&conf!=null&&conf<0.30;
 const low=!noTrust&&!libCat&&!ref&&!specCat&&conf!=null&&conf<0.35;
 const alt=(low&&cands.length>1?'；其他可能 '+(cands.slice(1,3).map(c=>c.category.split(',')[0]+' '+c.score.toFixed(2)).join(' / ')):'')
  +(noTrust&&cands.length?'，最接近 '+cands.slice(0,3).map(c=>c.category.split(',')[0]+' '+c.score.toFixed(2)).join(' / ')+'（置信度过低不采信）':'');
 const basis=libCat?'底图库同款分组的人工标签':ref?('平台查证参考图匹配（余弦 '+ref.score.toFixed(3)+'，≥'+REF_MIN+'）：'+ref.source):specCat?('读图规格 '+summarizeSpecs(q._specCheck.querySpecs)):
  ('CLIP 零样本判断'+(conf!=null?'，置信度 '+conf.toFixed(2)+(noTrust?'（过低，不采信）':low?'（偏低，仅供参考）':''):''));
 rows.push('类别：'+(noTrust?'未判定':(cat?cat+(catZh?'（'+catZh+'）':''):'未判定'))+' · 依据：'+basis+alt);
 // 二级类目按「最终采用的大类」取，避免跨类目错配
 const subs=(a?.mlSubsByParent&&cat&&a.mlSubsByParent[cat])||a?.mlSubs||[];
 const s1=ref?{sub:ref.sub,subZh:ref.subZh,score:null}:specCat?.sub?{sub:specCat.sub,subZh:specCat.subZh,score:null}:(subs[0]);
 const subNoTrust=!ref&&!specCat&&s1&&s1.score!=null&&s1.score<0.30;
 rows.push('分类：'+(subNoTrust?'未判定（低置信不采信'+(s1?'，最接近 '+s1.sub+' '+s1.score.toFixed(2)+(subs[1]?' / '+subs[1].sub+' '+subs[1].score.toFixed(2):'')+'）':''):(s1?(s1.sub+'（'+s1.subZh+'）'+(s1.score!=null?' · 置信度 '+s1.score.toFixed(2)+(s1.score<0.35?'（偏低，仅供参考）':''):' · 依据读图规格')):'未判定'))+(!specCat&&!ref&&subs[1]&&!subNoTrust?' · 其他可能 '+subs[1].sub+'（'+subs[1].subZh+'） '+subs[1].score.toFixed(2):''));
 // 命中同款：名称直接用底图库同款分组的人工标签（与同款卡片/汇总表一致），不再附「形近参考不是结论」
 const libName=vd.ok&&top?top.label.name:null;
 const name=libName||specCat?.name||(a?.suggestedName&&a.suggestedName!=='未识别'?a.suggestedName:(a?.referenceName&&!vd.ok?'未识别':(a?.suggestedName||'未识别')));
 const nameBasis=libName?'底图库同款分组标签':(specCat?.name?'读图规格':(a?.method||'CLIP'));
 rows.push('名称：'+name+(!libName&&!ref&&a?.referenceName&&name!==a.referenceName?' · 形近参考：'+a.referenceName+'（底图分组 '+a.referenceGroup+'，不是识别结论）':'')+' · 判定方式 '+nameBasis);
 return rows;
}
function result(q,forceAll=false){
 const found=searchLibrary(q.analysis,library,{forceAll,topK:Math.min(64,topK*4),querySha256:q.sha256}),a=q.analysis;
 // Preserve visual order unless a confirmed conflict excludes a product group.
 found.results.sort((x,y)=>Number(!!q._specChecks?.[x.label.group||x.image.id]?.conflict)-Number(!!q._specChecks?.[y.label.group||y.image.id]?.conflict));
 found.results=found.results.slice(0,topK);
 const selectedCheck=found.results[0]&&q._specChecks?.[found.results[0].label.group||found.results[0].image.id];
 q._specCheck=selectedCheck?{...selectedCheck,topId:found.results[0].image.id}:null;
 const section=el('section',null,'result-block panel'),head=el('div',null,'result-title');
 head.append(image(q.url,q.file.name),el('h2',q.file.name));section.append(head);
 const top=found.results[0],second=found.results[1],vd=verdict(top,second,a);
 // OCR 关键规格硬否决：外观一致但型号/规格不同，不能算同款
  if(q._specCheck?.conflict&&top&&q._specCheck.topId===top.image.id){
   vd.ok=false;vd.text='规格不符（'+q._specCheck.conflict+'）';
   vd.reason=q._specCheck.reason||'读图规格与底图规格不一致，按非同款处理';
  }
  // ① 先给识别结果：类别（美客多大类）+ 分类（二级）+ 名称 + 读图规格
  const idBox=el('div',null,'recognition');
  idBox.append(el('div','① 识别结果','recog-title'));
  for(const line of idRows(a,top,vd,q))idBox.append(el('div',line));
  if(q._specCheck?.querySpecs){
   const ls=q._specCheck.libSpecs;
   idBox.append(el('div','规格（读图）：'+summarizeSpecs(q._specCheck.querySpecs)
    +(ls?' · 底图：'+summarizeSpecs(ls)+(q._specCheck.libSource?'（'+q._specCheck.libSource+'）':''):' · 底图未读到可读规格，无法自动核对')));
  }
 // ② 再给同款检索结论
 const rec=el('div',null,'recognition'+(vd.ok?'':' warn'));
 rec.append(el('div','② 同款检索：'+(found.fused?'CLIP + DINOv2 双模型融合':'CLIP 单模型')+'，全库 '+found.total+' 张底图按组比对（同一行=同一产品多角度）。','recog-title'));
 rec.append(el('div',vd.ok?(vd.tentative?'⚠ 疑似同款：'+vd.text+'。'+vd.reason+'。外观一样不代表同款，请重点核对型号/规格/数量后再确认。':'✓ 找到同款：'+vd.text+'。'+vd.reason):('候选（'+vd.text+'）。'+vd.reason+'，当前证据不足以确认，请人工核对是否同款。')));
 if(q._specCheck?.review)rec.append(el('div',q._specCheck.reason||'规格证据待人工核对','sub'));
 idBox.append(rec);
 section.append(idBox);
 if(a.candidates?.length){
  const line=el('div',null,'candidates');line.append(el('span','候选名称（点击人工确认）：','sub'));
  for(const c of a.candidates){
   line.append(btn(c.name+' '+c.score.toFixed(3),()=>{q.analysis={...q.analysis,suggestedName:c.name,suggestedCategory:c.category,status:'人工确认',confident:true,method:'人工选择',matchedGroup:c.group};section.replaceWith(result(q,forceAll))},'pill'));
  }
  idBox.append(line);
 }
 const grid=el('div',null,'grid');
 for(let hi=0;hi<found.results.length;hi++){
  const hit=found.results[hi],l=effectiveLabel(hit.image),c=el('article',null,'card'),im=image('/images/'+hit.image.id,l.name),body=el('div',null,'body');
  im.onclick=()=>compare(q,hit.image);
  const v=verdict(hit),sc=hit.productScore??hit.score,nx=found.results[hi+1]?(found.results[hi+1].productScore??found.results[hi+1].score):0;
  const check=q._specChecks?.[hit.label.group||hit.image.id];
 if(check?.conflict){v.ok=false;v.text='规格不符';v.reason=check.reason}
 const lead='领先幅度不明显';
 body.append(el('h3',l.name),el('div',(check?.conflict?'规格不符':(v.ok?(v.tentative?'⚠ 疑似同款':'✓ 人工确认同款'):'候选（待核对）'))+' · 产品匹配度 '+sc.toFixed(3)+(hit.fused?'（双模型）':'（CLIP）')+' · 视觉 '+hit.visual.toFixed(3)+' · '+hit.views.length+' 个角度 · 领先下一位 '+lead,'sub'));
  const views=el('div',null,'views');
  for(const v2 of hit.views){const thumb=image('/images/'+v2.id,'产品角度');thumb.onclick=()=>compare(q,v2);views.append(thumb)}
  body.append(views);
  body.append(btn('放大对比',()=>compare(q,hit.image)));
  const confirm=btn('确认同款',async()=>{
   confirm.disabled=true;
   try{const list=[...(hit.image.confirmations||[]),{decision:'same',query:q.file.name,querySha256:q.sha256,groupId:hit.label.group||hit.image.id,at:new Date().toISOString()}];await savePatch(hit.image.id,{confirmations:list});section.replaceWith(result(q,forceAll))}catch(e){confirm.disabled=false;status('query-status',e.message,true)}
  });
  const reject=btn('标为不同款',async()=>{
   reject.disabled=true;
   try{const list=[...(hit.image.confirmations||[]),{decision:'different',query:q.file.name,querySha256:q.sha256,groupId:hit.label.group||hit.image.id,at:new Date().toISOString()}];await savePatch(hit.image.id,{confirmations:list});section.replaceWith(result(q,forceAll))}catch(e){reject.disabled=false;status('query-status',e.message,true)}
  });
  body.append(confirm,reject);c.append(im,body);grid.append(c);
 }
 section.append(grid);
 if(!found.results.length)section.append(el('p','暂无可检索底图，请先添加底图并完成检索准备。','empty'));
 else if(!vd.ok)section.append(el('p','目前无法确认同款，仅列为形近候选；证据不足不代表图库里没有该产品。','sub'));
 const actions=el('div',null,'actions');
 if(found.stage!=='all')actions.append(btn('没有同款，查找全部底图',()=>section.replaceWith(result(q,true))));
 if(found.results.length>=topK)actions.append(btn('显示更多结果',()=>{topK=Math.min(topK*2,64);$('top-k').value=String(topK);section.replaceWith(result(q,forceAll))}));
 section.append(actions);
 q._summary={name:q.file.name,cat:(vd.ok&&top?top.label.category:null)||a?.mlCategory||'未判定',
  catSource:vd.ok&&top?'底图库标签':'CLIP 零样本',product:vd.ok&&top?top.label.name:(a?.suggestedName&&a.suggestedName!=='未识别'?a.suggestedName:(a?.referenceName&&!vd.ok?'未识别':(a?.suggestedName||'未识别'))),
  best:top?top.label.name:'—',group:top?top.label.group:'—',score:top?(top.productScore??top.score):0,visual:top?top.visual:0,
  lead:vd.ratio||0,ok:vd.ok,tentative:!!vd.tentative,verdict:vd.text,reason:vd.reason||'',n:found.results.length,
  topId:top?top.image.id:null,specConflict:q._specCheck?.conflict||'',
  querySpecs:q._specCheck?.querySpecs?summarizeSpecs(q._specCheck.querySpecs):''};
 return section;
}
 // 多张对比图的汇总表：逐行列出每张图的识别结论与是否找到同款。
function summaryTable(rows){
 const box=el('div',null,'panel summary-panel');
 box.append(el('h2','批量对照汇总（'+rows.length+' 张对比图）'));
 const t=el('table',null,'summary');const head=el('tr');
 for(const h of ['对比图','识别类别（依据）','名称（近邻参考）','最佳候选分组','读图规格','产品匹配度','领先倍数','是否同款'])head.append(el('th',h));
 t.append(head);
 for(let i=0;i<rows.length;i++){
  const r=rows[i],tr=el('tr');
  tr.append(el('td',(i+1)+'. '+r.name),el('td',r.cat+'（'+r.catSource+'）'),el('td',r.product),el('td',r.group),
 el('td',r.querySpecs||'—'),el('td',r.score.toFixed(3)),el('td','待人工核对'),
 el('td',r.ok?(r.tentative?'⚠ 疑似（需人工核对）':'✓ 确认同款'):(r.specConflict?'✗ 规格不符 '+r.specConflict:'形近候选')));
  t.append(tr);
 }
 box.append(t);
 box.append(el('div','说明：每张对比图按外观相似度排序，相似不代表同款；只有已核验的规格冲突才会自动排除候选。请务必人工核对结构、型号和规格。','sub'));
 return box;
}
async function search(){
 if(working||indexing)return status('query-status','底图正在保存或准备检索，请稍后再查找。');
 working=true;renderQueries();$('results').replaceChildren();
 try{
  await refresh();
  const missing=library.filter(x=>!READY(x)).length;
  if(missing){$('index-retry').classList.remove('hidden');return status('query-status','还有 '+missing+' 张底图未准备好检索，请点击左侧“继续准备底图检索”。',true)}
  const rows=[];
  for(let i=0;i<queries.length;i++){
   const q=queries[i];status('query-status','正在识别产品名称和类目 '+(i+1)+' / '+queries.length);
   if(!q.sha256)q.sha256=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await q.file.arrayBuffer())),x=>x.toString(16).padStart(2,'0')).join('');
   if(!q.analysis)q.analysis=await analyzeQuery(q.url,t=>status('query-status',t));
  status('query-status','正在按组检索同款…');
  let sec=result(q);$('results').append(sec);
  // OCR once per query; inspect every returned group's available views.
  // Unknown or inconsistent view evidence cannot reject the whole group.
  {
   const qs=await readSpecs(q.file,t=>status('query-status',t));
   q._specChecks={};
   const candidates=searchLibrary(q.analysis,library,{forceAll:true,topK:Math.min(64,topK*4),querySha256:q.sha256}).results;
   for(const hit of candidates){
    const viewSpecs=await Promise.all(hit.views.map(v=>librarySpecs(v.filename)));
    const cf=compareGroupSpecs(qs,viewSpecs);
    q._specChecks[hit.label.group||hit.image.id]={querySpecs:qs,libSpecs:viewSpecs.find(Boolean)||null,
 conflict:cf.conflict,reason:cf.reason,review:cf.review,libSource:'读图多角度规格'};
   }
   const fresh=result(q);sec.replaceWith(fresh);sec=fresh;
  }
  if(q._summary)rows.push(q._summary);
  }
  if(rows.length>1)$('results').prepend(summaryTable(rows));
  status('query-status','查找完成，共处理 '+queries.length+' 张图片。每张图的识别类别、识别名称与是否找到同款见汇总表；同款关系仍需你人工确认。');
 }catch(e){status('query-status',e.message,true)}finally{working=false;renderQueries();if(baseQueue.length)drainBase()}
}

/* ---------- 拖拽 / 粘贴 / 图片地址 ---------- */
function bindDrop(id,kind,add){
 const area=$(id);
 area.onfocus=()=>pasteTarget=kind;area.onpointerdown=()=>pasteTarget=kind;
 area.ondragover=e=>{e.preventDefault();area.classList.add('drag')};
 area.ondragleave=()=>area.classList.remove('drag');
 area.ondrop=e=>{e.preventDefault();area.classList.remove('drag');const files=[...e.dataTransfer.files];if(files.length)add(files);else{const url=e.dataTransfer.getData('text/uri-list')||e.dataTransfer.getData('text/plain');if(url)fromURL(url,kind)}};
 // 右键菜单：按用户要求的「加一张已加载的对比图片进来，不必全部上传」；脚本直接读取剪贴板数据。
 // 所以菜单第一项聚焦上传区并提示按 Ctrl+V；另外两项直接打开文件选择器 / 图片地址输入框。
 area.oncontextmenu=e=>{
  e.preventDefault();pasteTarget=kind;area.focus();hideDropMenu();
  const menu=$('drop-menu');menu.dataset.kind=kind;
  menu.style.left=Math.min(e.clientX,innerWidth-210)+'px';menu.style.top=Math.min(e.clientY,innerHeight-150)+'px';
  menu.classList.remove('hidden');
 };
}
function hideDropMenu(){$('drop-menu')?.classList.add('hidden')}
async function fromURL(text,kind){
 const id=kind==='base'?'base-status':'query-status';
 try{
  const url=new URL(text.trim());if(url.protocol!=='https:')throw Error('请使用 https 图片地址');
  status(id,'正在读取图片地址…');
  let r=await fetch('/api/proxy-image?url='+encodeURIComponent(url.href),{signal:AbortSignal.timeout(30000)});
  if(!r.ok)r=await fetch(url.href,{referrerPolicy:'no-referrer',signal:AbortSignal.timeout(25000)});
  if(!r.ok)throw Error('图片地址无法读取（'+r.status+'）');
  const b=await r.blob();if(!b.type.startsWith('image/'))throw Error('地址未返回图片');
  const file=new File([b],url.pathname.split('/').pop()||'粘贴图片.png',{type:b.type});
  (kind==='base'?addBase:addQueries)([file]);
 }catch(e){status(id,'无法读取图片链接，请复制图片本身或下载后拖入。'+e.message,true)}
}

/* ---------- 入库与索引 ---------- */
bindDrop('base-drop','base',addBase);bindDrop('query-drop','query',addQueries);
document.addEventListener('paste',e=>{
 if(e.target.matches('input:not([type=file]),textarea,select'))return;
 const files=[...e.clipboardData.items].filter(x=>x.kind==='file'&&x.type.startsWith('image/')).map(x=>x.getAsFile()).filter(Boolean);
 if(files.length){e.preventDefault();(pasteTarget==='base'?addBase:addQueries)(files)}
 else{const text=e.clipboardData.getData('text/plain');if(/^https:\/\//.test(text.trim())){e.preventDefault();fromURL(text,pasteTarget)}}
});
$('base-pick').onclick=()=>{pasteTarget='base';$('base-files').click()};
$('query-pick').onclick=()=>{pasteTarget='query';$('query-files').click()};
$('drop-menu-paste').onclick=()=>{
 const kind=$('drop-menu').dataset.kind||'query';pasteTarget=kind;hideDropMenu();
 $(kind+'-drop').focus();status(kind+'-status','已准备好接收剪贴板图片，请按 Ctrl+V 粘贴（Chrome 安全策略不允许网页直接读取剪贴板）。');
};
$('drop-menu-pick').onclick=()=>{const kind=$('drop-menu').dataset.kind||'query';hideDropMenu();$(kind+'-pick').click()};
$('drop-menu-url').onclick=()=>{
 const kind=$('drop-menu').dataset.kind||'query';hideDropMenu();
 $(kind+'-drop').parentElement.querySelector('details')?.setAttribute('open','');
 $(kind+'-url').focus();status(kind+'-status','请在展开的图片地址框粘贴 https 链接后点「加入」。');
};
document.addEventListener('pointerdown',e=>{if(!e.target.closest('#drop-menu'))hideDropMenu()});
window.addEventListener('blur',hideDropMenu);
$('base-files').onchange=e=>{addBase([...e.target.files]);e.target.value=''};
$('query-files').onchange=e=>{addQueries([...e.target.files]);e.target.value=''};
$('base-url-add').onclick=()=>fromURL($('base-url').value,'base');
$('query-url-add').onclick=()=>fromURL($('query-url').value,'query');
$('query-run').onclick=search;
$('query-clear').onclick=()=>{if(working)return;queries.forEach(x=>URL.revokeObjectURL(x.url));queries=[];renderQueries();$('results').replaceChildren();status('query-status','已清空对比图片，底图库保留不变。')};
$('top-k').onchange=e=>{topK=Number(e.target.value)||8};
$('library-search').oninput=()=>{page=36;renderLibrary()};
$('category-filter').onchange=()=>{page=36;renderLibrary()};
$('library-more').onclick=()=>{page+=36;renderLibrary()};
$('group-toggle').onclick=e=>{grouped=!grouped;e.target.textContent=grouped?'平铺显示':'按同款分组';renderLibrary()};
$('pending-toggle').onclick=e=>{pendingOnly=!pendingOnly;e.target.textContent=pendingOnly?'显示全部':'只看待确认';renderLibrary()};
$('select-all').onclick=()=>{const rows=grouped?filtered():filtered().slice(0,page);const all=rows.every(x=>selected.has(x.id));for(const x of rows)all?selected.delete(x.id):selected.add(x.id);renderLibrary()};
$('export').onclick=exportCsv;
$('catalog-sync').onclick=syncCatalog;
$('batch-edit').onclick=()=>{if(!selected.size)return status('base-status','请先勾选要标注的底图',true);$('batch-count').textContent='已选中 '+selected.size+' 张底图；留空的字段保持不变。';$('batch-dialog').showModal()};
$('batch-cancel').onclick=()=>$('batch-dialog').close();
$('batch-save').onclick=async()=>{
 const body={};if($('batch-name').value.trim())body.name=$('batch-name').value.trim();if($('batch-category').value)body.category=$('batch-category').value;if($('batch-group').value.trim())body.group=$('batch-group').value.trim();
 if(!Object.keys(body).length)return status('base-status','请至少填写一个要修改的字段',true);
 const ids=[...selected];$('batch-save').disabled=true;let n=0;
 await mapLimit(ids,3,async id=>{await savePatch(id,body);n++;status('base-status','正在批量标注 '+n+' / '+ids.length)});
 $('batch-save').disabled=false;$('batch-dialog').close();selected.clear();renderLibrary();status('base-status','已更新 '+n+' 张底图的产品信息。');
};
$('edit-cancel').onclick=()=>$('edit-dialog').close();
$('edit-save').onclick=async()=>{
 const body={name:$('edit-name').value.trim(),category:$('edit-category').value,group:$('edit-group').value.trim()};
 if(!body.name||!body.group)return status('base-status','产品名称和同款分组不能为空',true);
 try{await savePatch($('edit-dialog').dataset.id,body);$('edit-dialog').close();renderLibrary();status('base-status','产品信息已更新。')}catch(e){status('base-status',e.message,true)}
};
$('viewer-close').onclick=()=>$('viewer').close();
$('viewer-mode').onclick=()=>{overlay=!overlay;renderCompare()};
$('opacity').oninput=e=>{const im=$('viewer-body').querySelector('img.over');if(im)im.style.opacity=e.target.value/100};
$('index-retry').onclick=prepareIndex;
$('trash').ontoggle=async()=>{
 if(!$('trash').open)return;
 try{
  const rows=await api('/api/images?trash=1'),box=$('trash-items');box.replaceChildren();
  for(const x of rows){const row=el('p',effectiveLabel(x).name+' '),restore=btn('恢复底图',async()=>{restore.disabled=true;try{await patch(x.id,{deleted:false});row.remove();await refresh()}catch(e){restore.disabled=false;status('base-status',e.message,true)}});row.append(restore);box.append(row)}
  if(!rows.length)box.append(el('p','暂无移出的图片'));
 }catch(e){status('base-status',e.message,true)}
};

refresh().then(()=>{if(library.some(x=>!READY(x)))prepareIndex()}).catch(e=>status('base-status',e.message,true));

