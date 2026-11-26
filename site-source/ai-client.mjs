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
async function api(p,opt={}){const r=await fetch(p,opt);if(!r.ok){let msg;try{msg=(await r.json()).error}catch{}throw Error(msg||'����ʧ�ܣ�'+r.status+'����������')}return r.json()}
const patch=(id,body)=>api('/api/images/'+id,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
function image(src,alt){const im=el('img');im.src=src;im.alt=alt;im.loading='lazy';return im}
function readyLabel(x){return READY(x)?'�ѱ��� �� �ɼ���':'�ѱ��� �� ����׼������'}

/* ---------- ��ͼ�� ---------- */
function initFilters(){
 const cats=[...new Set([...CATEGORIES,...library.map(x=>effectiveLabel(x).category)])].filter(c=>!LEGACY_CATEGORY[c]);
 for(const id of ['category-filter','edit-category','batch-category']){
  const s=$(id),old=s.value;s.replaceChildren();
  if(id!=='category-filter'&&id==='batch-category')s.append(el('option','���޸�',''));
  if(id==='category-filter')s.append(el('option','ȫ�����',''));
  for(const c of cats){const o=el('option',c,c);s.append(o)}
  s.value=old||'';
 }
}
function filtered(){
 const q=$('library-search').value.trim().toLowerCase(),cat=$('category-filter').value;
 return library.filter(x=>{
  const l=effectiveLabel(x);
  if(cat&&l.category!==cat)return false;
  if(pendingOnly&&!(x.ai?.status==='��ȷ��'||l.category==='Otras Categor��as'||l.name==='��Ʒ��ȷ��'))return false;
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
 badges.append(cat,el('span',READY(x)?'�ɼ���':'��׼��',READY(x)?'badge':'badge warn'));
 if(x.ai?.status==='��ȷ��')badges.append(el('span','��ȷ��','badge warn'));
 body.append(el('div',l.group+(l.view?' �� '+l.view+'��ͼ':''),'sub'));
 body.append(el('div',readyLabel(x),'sub'));
 body.append(btn('�༭��Ϣ',()=>openEdit(x)),btn('�Ƴ���ͼ��',async()=>{remove.disabled=true;try{await patch(x.id,{deleted:true});selected.delete(x.id);await refresh()}catch(e){status('base-status',e.message,true)}finally{remove.disabled=false}}));
 const remove=body.lastChild;
 c.append(pick,im,body);return c;
}
function renderLibrary(){
 const rows=filtered(),groups=groupProducts(library);
 $('library-count').textContent=library.length+' �ŵ�ͼ �� '+groups.length+' ��ͬ����� �� '+library.filter(READY).length+' �ſɼ���';
 const pending=library.filter(x=>x.ai?.status==='��ȷ��'||effectiveLabel(x).category==='Otras Categor��as').length;
 $('library-stat').textContent='��ȷ�� '+pending+' �� �� ��Դ��'+(library.filter(x=>effectiveLabel(x).source==='��ƷĿ¼').length)+' �����Բ�ƷĿ¼ �� '+library.filter(x=>effectiveLabel(x).source==='�˹��༭').length+' ���˹��༭';
 const box=$('library');box.replaceChildren();
 if(!rows.length){box.append(el('div',library.length?'û��ƥ��ĵ�ͼ':'��ͼ��Ϊ�գ��������Ӳ�ƷͼƬ��','empty'));}
 else if(grouped){
  for(const g of groupProducts(rows)){
   const head=el('div',null,'group-head');head.append(el('span',g.name),el('span',g.category+' �� '+g.items.length+' �Ŷ�Ƕ�'));
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
function needsSync(x){const c=catalogFor(x.filename);return isDefaultName(x.name)&&c&&c.name!=='��Ʒ��ȷ��'}

/* ---------- �׿⸲�Ǳ�������Ͱ�����Ŀ/�����Ǿɰ�ʱ�������ף� ---------- */
// ���ڵ׿��õ�����������Ŀ����ʾ/ɸѡʱͳһ��������Ͷ�ٷ�һ����Ŀ
const LEGACY_CATEGORY={'����':'Accesorios para Veh��culos','��𹤾�':'Herramientas','���ӵ���':'Electr��nica, Audio y Video','�Ҿ���Ʒ':'Hogar, Muebles y Jard��n','������Ʒ':'Mascotas','������':'Otras Categor��as'};
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
  if(lab&&lab[0]){x.name=lab[0];x.category=lab[1]||x.category;x.group='��'+x.row+'�� �� '+lab[0];x.labelSource='��ͼ���ǩ';relabelled++;
   // ���Ǳ�ͬ��ȷ��״̬���˹��Ѷ����Ĳ�����ʾ����ȷ�ϡ�������������ʵ����
   if(lab[2]&&x.ai)x.ai.status=lab[2]}
  // ���Ǳ�û���ǵ��ģ����ϴ��������ݣ�Ҳ�Ѿ�������Ŀ��������Ͷ�ٷ��ھ�
  if(LEGACY_CATEGORY[x.category])x.category=LEGACY_CATEGORY[x.category];
  const vec=dino[x.filename];
  if(vec&&x.ai&&!DINO_READY(x)){x.ai.dinoEmbedding=vec;x.ai.dinoModel='onnx-community/dinov2-small';x.ai.dinoPooling='all-tokens-mean-v1';withDino++}
 }
 if(relabelled||withDino)console.log('�׿⸲�Ǳ���Ӧ�ã���ǩ '+relabelled+' ����DINOv2 ���� '+withDino+' ��');
 return list;
}
async function refresh(){library=await applyOverrides(await api('/api/images'));initFilters();renderLibrary()}

/* ---------- �༭ / ���� / ���� ---------- */
function openEdit(x){
 const l=effectiveLabel(x);$('edit-name').value=l.name;$('edit-category').value=l.category;$('edit-group').value=l.group;
 $('edit-dialog').dataset.id=x.id;$('edit-dialog').showModal();
}
async function savePatch(id,body){const updated=await patch(id,body),i=library.findIndex(x=>x.id===id);if(i>=0)library[i]=updated;return updated}
function exportCsv(){
 const head=['�ļ���','��Ʒ����','���','ͬ�����','ʶ��״̬','ʶ��ʽ','��ǩ��Դ','�ɼ���','����ʱ��'];
 const rows=[head];
 for(const x of library){const l=effectiveLabel(x);rows.push([x.filename,l.name,l.category,l.group,x.ai?.status||'δʶ��',x.ai?.method||x.ai?.model||'',l.source,READY(x)?'��':'��',x.created])}
 const csv='\ufeff'+rows.map(r=>r.map(v=>'"'+String(v??'').replace(/"/g,'""')+'"').join(',')).join('\r\n');
 const a=el('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));a.download='��ͼ�����嵥-'+new Date().toISOString().slice(0,10)+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),2000);
}
async function mapLimit(items,limit,fn){
 const out=[];let i=0;
 const workers=Array.from({length:Math.min(limit,items.length)},async()=>{while(i<items.length){const n=i++;try{out[n]=await fn(items[n],n)}catch(e){out[n]={error:e.message}}}});
 await Promise.all(workers);return out;
}
async function syncCatalog(){
 const todo=library.filter(needsSync);if(!todo.length)return status('base-status','û����Ҫ����ĵ�ͼ');
 $('catalog-sync').disabled=true;let done=0;
 await mapLimit(todo,3,async x=>{
  const c=catalogFor(x.filename),body={name:c.name,category:c.category,group:c.group};
  if(READY(x))body.ai={...x.ai,suggestedName:c.name,suggestedCategory:c.category,status:'��ʶ��'};
  await savePatch(x.id,body);done++;status('base-status','���ڻ����ƷĿ¼��ǩ '+done+' / '+todo.length);
 });
 $('catalog-sync').disabled=false;renderLibrary();status('base-status','�ѻ��� '+done+' �ŵ�ͼ����ʵ��Ʒ��������Ŀ��');
}

/* ---------- ʶͼģ�� ---------- */
async function localModelBase(){
 try{const r=await fetch('/models/Xenova/clip-vit-base-patch32/config.json',{signal:AbortSignal.timeout(4000)});if(r.ok){const t=await r.text();if(t.includes('model_type')||t.includes('clip'))return '/models/'}}catch{}
 return null;
}
async function ready(report){
 if(recognizer)return recognizer;if(loading)return loading;
 loading=(async()=>{
  const base=await localModelBase();
  report(base?'���ڴӱ�������ʶͼģ�͡�':'�״μ���ʶͼģ�ͣ�Լ 155 MB���뱣��ҳ��򿪡�');
  const lib=await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1');
  recognizer=await createRecognizer(lib,p=>{if(p.status==='progress')report('���ڼ���ʶͼģ�� '+Math.round(p.progress||0)+'%')},'wasm',base);
  return recognizer;
 })().catch(e=>{loading=null;throw Error('ʶͼģ�ͼ���ʧ�ܣ�������������ԡ�'+e.message)});
 return loading;
}
let dinoEmbedder,dinoLoading;
async function readyDino(report){
 if(dinoEmbedder)return dinoEmbedder;if(dinoLoading)return dinoLoading;
 dinoLoading=(async()=>{
  const base=await localModelBase();if(!base)return null;
  const lib=await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1');
  dinoEmbedder=await createDinoEmbedder(lib,p=>{if(p.status==='progress')report('���ڼ��صڶ��Ӿ�ģ�� '+Math.round(p.progress||0)+'%')},'wasm',base);
  return dinoEmbedder;
 })().catch(e=>{dinoLoading=null;console.warn('DINOv2 ����ʧ�ܣ��˻ص�ģ�ͣ�'+e.message);return null});
 return dinoLoading;
}
async function rawAnalyze(source,report){
 const analysis=await (await ready(report)).analyze(source);
 try{
  const dino=await readyDino(report);
  if(dino){analysis.dinoEmbedding=await dino.embed(source);analysis.dinoModel='onnx-community/dinov2-small';analysis.dinoPooling='all-tokens-mean-v1'}
 }catch(e){console.warn('DINOv2 ����ʧ�ܣ�����ֻ�� CLIP��'+e.message)}
 return analysis;
}
// ƽ̨��֤�ο��⣺�˹�ȷ�Ϲ�����ʵ��Ʒͼ��1688/���Ͷ��֤�����Ӿ�ê�㡣
// CLIP ��������ϸ���ȹ�ҵ����ֻ�� 0.2 �֣���ͬϹ�£�����������ê�����ұȶԣ����о��ò�֤����Ʒ��/��Ŀ��
let referencesLib=null;
async function loadReferences(){
 if(referencesLib!==null)return referencesLib;
 try{const r=await fetch('/references.json',{cache:'no-store'});referencesLib=r.ok?await r.json():null}catch{referencesLib=null}
 return referencesLib;
}
function cosine(a,b){if(a?.length!==b?.length||!a?.length)return -1;let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s}
const REF_MIN=0.82; // ͬһ��Ʒ��ͬ�Ƕȵ��� >0.85���ν��� <0.8
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
 }catch(e){console.warn('�ο���ƥ��ʧ�ܣ�',e.message);return null}
}
async function analyzeQuery(source,report){
 const a=identify(await rawAnalyze(source,report),library);
 const ref=await matchReference(a);
 if(ref){
  a.referenceMatch=ref;
  a.suggestedName=ref.name;a.suggestedCategory=ref.category;a.referenceSub=ref.sub;a.referenceSubZh=ref.subZh;
  a.method='ƽ̨��֤�ο�ͼƥ��';a.status='��ʶ��';a.confident=true;
 }
 return a;
}

/* ---------- ��������� ---------- */
function syncBody(x){const c=catalogFor(x.filename);if(!c||c.name==='��Ʒ��ȷ��'||!isDefaultName(x.name))return {};return {name:c.name,category:c.category,group:c.group}}
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
     status('base-status','��ͼ�ѱ��棬����׼������ '+done+' / '+todo.length);
     if(done%5===0)renderLibrary();return;
    }catch(e){if(attempt===1)throw e}
   }
  });
  const failed=outcome.filter(r=>r&&r.error);
  if(failed.length)throw Error(failed.length+' �ŵ�ͼ׼��ʧ�ܣ�'+failed[0].error);
  status('base-status','��ͼ�ѱ��沢�����ڼ�������������ȷ�ϡ�');
 }catch(e){status('base-status','ԭͼ�ѱ��棻����׼��δ��ɣ�'+e.message,true);$('index-retry').classList.remove('hidden')}
 finally{indexing=false;renderQueries();if(baseQueue.length)drainBase()}
}
function validate(files){
 const good=[],bad=[];
 for(const f of files){if(!/^image\/(jpeg|png|webp|gif|bmp)$/i.test(f.type)){bad.push(f.name+'����֧�ִ˸�ʽ');continue}if(f.size>20*1024*1024){bad.push(f.name+'������ 20 MB');continue}good.push(f)}
 return {good,bad};
}
async function jpeg(file){if(['image/jpeg','image/png','image/webp'].includes(file.type))return file;const bitmap=await createImageBitmap(file),canvas=document.createElement('canvas');canvas.width=bitmap.width;canvas.height=bitmap.height;canvas.getContext('2d').drawImage(bitmap,0,0);bitmap.close();const blob=await new Promise(r=>canvas.toBlob(r,'image/png'));if(!blob)throw Error('ͼƬת��ʧ��');return new File([blob],file.name.replace(/\.[^.]+$/,'')+'.png',{type:'image/png'})}
async function drainBase(){
 if(indexing||working)return;working=true;renderQueries();
 const batch=baseQueue.shift();if(!batch){working=false;return}
 const files=batch.files;let ok=0,failed=[];
 $('base-progress').classList.remove('hidden');$('base-progress').max=files.length;
 try{
  for(let i=0;i<files.length;i++){
   try{
    const f=await jpeg(files[i]),data=new FormData();
    data.append('image',f);data.append('name',f.name.replace(/\.[^.]+$/,''));data.append('category','�ο���ͼ');
    data.append('group',sourceGroup(f.name)||batch.group);
    status('base-status','���ڱ����ͼ '+(i+1)+' / '+files.length);
    library.unshift(await api('/api/images',{method:'POST',body:data}));ok++;
   }catch(e){failed.push(files[i]);status('base-status',e.message,true)}
   $('base-progress').value=i+1;
  }
  renderLibrary();status('base-status','�ѱ��� '+ok+' �ŵ�ͼ'+(failed.length?'��'+failed.length+' ��ʧ�ܣ�������':'������׼��������'),!!failed.length);
  if(failed.length)baseQueue.unshift({files:failed,group:batch.group});
 }finally{
  working=false;$('base-progress').classList.add('hidden');renderQueries();
  if(baseQueue.length)drainBase();else prepareIndex();
 }
}
function addBase(files){
 const {good,bad}=validate(files);if(bad.length)status('base-status',bad.join('��'),true);if(!good.length)return;
 baseQueue.push({files:good,group:'��Ʒ '+new Date().toLocaleString('zh-CN',{hour12:false}).replace(/[^\d]/g,'').slice(0,14)+'-'+Math.random().toString(36).slice(2,6)});
 if(working||indexing)status('base-status','�Ѽ����ͼ������� '+baseQueue.length+' ��');else drainBase();
}
function addQueries(files){
 const {good,bad}=validate(files);if(bad.length)status('query-status',bad.join('��'),true);
 for(const file of good)queries.push({file,url:URL.createObjectURL(file),analysis:null});
 renderQueries();
 if(good.length){status('query-status','�Ѽ��� '+queries.length+' �ŶԱ�ͼƬ���������ʼ���ҡ���');$('query-drop').focus()}
}
// ���ŶԱ�ͼƬ���ϽǵĲ棺�����Ƴ����������ȫ����
function removeQuery(i){
 if(working)return status('query-status','���ڲ����У��ݲ����Ƴ�ͼƬ��',true);
 const q=queries[i];if(!q)return;
 URL.revokeObjectURL(q.url);queries.splice(i,1);renderQueries();
 status('query-status','���Ƴ� 1 �ŶԱ�ͼƬ��ʣ�� '+queries.length+' �š�');
}
function renderQueries(){
 const box=$('query-preview');
 box.replaceChildren(...queries.map((x,i)=>{
  const wrap=el('div',null,'qitem'),im=image(x.url,x.file.name),del=el('button','��','qdel');
  del.title='�Ƴ����ŶԱ�ͼƬ';del.setAttribute('aria-label','�Ƴ����ŶԱ�ͼƬ');
  del.onclick=()=>removeQuery(i);
  wrap.append(im,del);return wrap;
 }));
 $('query-run').disabled=!queries.length||working||indexing;$('query-clear').disabled=working;
}

/* ---------- �Ŵ� / �Ա� ---------- */
function view(src,title){compareState=null;$('viewer-title').textContent=title;$('viewer-body').replaceChildren(Object.assign(image(src,title),{className:'full'}));$('opacity-row').classList.add('hidden');$('viewer-mode').classList.add('hidden');$('viewer').showModal()}
function renderCompare(){
 if(!compareState)return;
 $('viewer-title').textContent='�ԱȲ�Ʒ������ϸ�ڣ��ص�˶��ͺ� / ��� / ������';
 $('viewer-body').replaceChildren();$('viewer-mode').classList.remove('hidden');
 if(!overlay){
  const box=el('div',null,'compare');box.append(image(compareState.q,'�ϴ��ĶԱ�ͼƬ'),image(compareState.c,'ͼ���Ʒ��ͼ'));$('viewer-body').append(box);$('opacity-row').classList.add('hidden');
 }else{
  const wrap=el('div',null,'overlay-wrap'),a=image(compareState.c,'ͼ���Ʒ��ͼ'),b=image(compareState.q,'���ӵĶԱ�ͼƬ');b.className='over';b.style.opacity=$('opacity').value/100;
  wrap.append(a,b);$('viewer-body').append(wrap);$('opacity-row').classList.remove('hidden');
 }
 $('viewer-mode').textContent=overlay?'�л����ŶԱ�':'�л����ӶԱ�';
}
function compare(q,candidate){compareState={q:q.url,c:'/images/'+candidate.id};overlay=false;renderCompare();$('viewer').showModal()}

/* ---------- ��ͼ���Ӳ��� ----------
 * �ο� ebiarian/product-image-description-alignment��MIT�����ؼ�������һ�¼������
 * ��ͬ�ھɰ� tesseract wasm�����ϲ��ȶ����������� ppu-paddle-ocr/web + PP-OCRv6 tiny
 * ֱ������������ϴ�ͼ���ͺų��ִ���ͬʱ��Ϊ���������м�ֻ�����źš�
 * ��ͼ����������� OCR + ����Ŀ�����ɵ� /specs.json����ռ���Ϸ�����Դ��
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
 }catch(e){console.warn('��� OCR �����ã�',e.message);return null}
}
async function librarySpecs(filename){
 const lib=await loadSpecsLib();
 return entryToParsed(lib?.specs?.[filename]);
}

/* ---------- ������� ---------- */
// ͬ���ж���������λ���ȵڶ�λ�ķ��ȡ�����ͬ���Ȼ�������ȣ�
// ����û��ͬ��ʱ��ǰ�����ἷ��һ��ʵ�����ŷ�ͬ��ͼ���ȷ��ȶ�ֻ�� 1.02~1.11 ������ͬ���� 2.4 ������
function verdict(hit){return candidateVerdict(hit)}
// �����ĵ����ͺ��ܾ��� CLIP ����Ŀ�þ���ʵ��� DZ47 ��·���гɡ�����ҽ�ơ���
function specCategory(p){
 const m=(p?.models||[]).join(' '),c=p?.curves||[],text=(p?.text||'')+' '+m;
 if(/DZ47|DZ808|ZBM1|iC65|5SX|NC65|BREAKER|��·��/i.test(text)&&c.some(x=>/^[BCD]\d{1,2}$/.test(x)))
  return {category:'Electr��nica, Audio y Video',categoryZh:'������Ƶ��Ƶ',sub:'Componentes Electr��nicos',subZh:'����Ԫ��',
   name:'С�Ͷ�·�� '+c.filter(x=>/^[BCD]\d{1,2}$/.test(x)).join('/')};
 return null;
}
// �� ʶ�������У���� / ���� / ���ơ�����ͬ���õ�ͼ���˹���ǩ�������ͼ��������� CLIP ��������
function idRows(a,top,vd,q){
 const rows=[];
 const libCat=vd.ok&&top?top.label.category:null;
 const specCat=libCat?null:specCategory(q?._specCheck?.querySpecs);
 const cands=a?.mlCandidates||[],ml=cands[0];
 const ref=a?.referenceMatch;
 const cat=libCat||ref?.category||specCat?.category||ml?.category||a?.mlCategory||null;
 const catZh=libCat?null:(ref?(ref.category==='Hogar, Muebles y Jard��n'?'�ҾӼҾ�԰��':(ref.category==='Accesorios para Veh��culos'?'�������':null)):(specCat?.categoryZh||ml?.categoryZh||a?.mlCategoryZh));
 const conf=ml?.score;
 // �����Ų����ţ�top1 <0.30 ʱ CLIP ��ͬϹ�£�ʵ�����̺����"�����ľ�"0.23���������϶���Ŀ��ֻ����ӽ���ѡ
 const noTrust=!libCat&&!ref&&!specCat&&conf!=null&&conf<0.30;
 const low=!noTrust&&!libCat&&!ref&&!specCat&&conf!=null&&conf<0.35;
 const alt=(low&&cands.length>1?'���������� '+(cands.slice(1,3).map(c=>c.category.split(',')[0]+' '+c.score.toFixed(2)).join(' / ')):'')
  +(noTrust&&cands.length?'����ӽ� '+cands.slice(0,3).map(c=>c.category.split(',')[0]+' '+c.score.toFixed(2)).join(' / ')+'�����Ŷȹ��Ͳ����ţ�':'');
 const basis=libCat?'��ͼ��ͬ�������˹���ǩ':ref?('ƽ̨��֤�ο�ͼƥ�䣨���� '+ref.score.toFixed(3)+'����'+REF_MIN+'����'+ref.source):specCat?('��ͼ��� '+summarizeSpecs(q._specCheck.querySpecs)):
  ('CLIP �������ж�'+(conf!=null?'�����Ŷ� '+conf.toFixed(2)+(noTrust?'�����ͣ������ţ�':low?'��ƫ�ͣ������ο���':''):''));
 rows.push('���'+(noTrust?'δ�ж�':(cat?cat+(catZh?'��'+catZh+'��':''):'δ�ж�'))+' �� ���ݣ�'+basis+alt);
 // ������Ŀ�������ղ��õĴ��ࡹȡ���������Ŀ����
 const subs=(a?.mlSubsByParent&&cat&&a.mlSubsByParent[cat])||a?.mlSubs||[];
 const s1=ref?{sub:ref.sub,subZh:ref.subZh,score:null}:specCat?.sub?{sub:specCat.sub,subZh:specCat.subZh,score:null}:(subs[0]);
 const subNoTrust=!ref&&!specCat&&s1&&s1.score!=null&&s1.score<0.30;
 rows.push('���ࣺ'+(subNoTrust?'δ�ж��������Ų�����'+(s1?'����ӽ� '+s1.sub+' '+s1.score.toFixed(2)+(subs[1]?' / '+subs[1].sub+' '+subs[1].score.toFixed(2):'')+'��':''):(s1?(s1.sub+'��'+s1.subZh+'��'+(s1.score!=null?' �� ���Ŷ� '+s1.score.toFixed(2)+(s1.score<0.35?'��ƫ�ͣ������ο���':''):' �� ���ݶ�ͼ���')):'δ�ж�'))+(!specCat&&!ref&&subs[1]&&!subNoTrust?' �� �������� '+subs[1].sub+'��'+subs[1].subZh+'�� '+subs[1].score.toFixed(2):''));
 // ����ͬ�����ֱ���õ�ͼ��ͬ�������˹���ǩ����ͬ�Ƭ/���ܱ�һ�£������ٸ����ν��ο����ǽ��ۡ�
 const libName=vd.ok&&top?top.label.name:null;
 const name=libName||specCat?.name||(a?.suggestedName&&a.suggestedName!=='δʶ��'?a.suggestedName:(a?.referenceName&&!vd.ok?'δʶ��':(a?.suggestedName||'δʶ��')));
 const nameBasis=libName?'��ͼ��ͬ������ǩ':(specCat?.name?'��ͼ���':(a?.method||'CLIP'));
 rows.push('���ƣ�'+name+(!libName&&!ref&&a?.referenceName&&name!==a.referenceName?' �� �ν��ο���'+a.referenceName+'����ͼ���� '+a.referenceGroup+'������ʶ����ۣ�':'')+' �� �ж���ʽ '+nameBasis);
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
 // OCR �ؼ����Ӳ��������һ�µ��ͺ�/���ͬ��������ͬ��
  if(q._specCheck?.conflict&&top&&q._specCheck.topId===top.image.id){
   vd.ok=false;vd.text='��񲻷���'+q._specCheck.conflict+'��';
   vd.reason=q._specCheck.reason||'��ͼ������ͼ���һ�£�����ͬ���';
  }
  // �� �ȸ�ʶ������������Ͷ���ࣩ+ ���ࣨ������+ ���� + ��ͼ���
  const idBox=el('div',null,'recognition');
  idBox.append(el('div','�� ʶ����','recog-title'));
  for(const line of idRows(a,top,vd,q))idBox.append(el('div',line));
  if(q._specCheck?.querySpecs){
   const ls=q._specCheck.libSpecs;
   idBox.append(el('div','��񣨶�ͼ����'+summarizeSpecs(q._specCheck.querySpecs)
    +(ls?' �� ��ͼ��'+summarizeSpecs(ls)+(q._specCheck.libSource?'��'+q._specCheck.libSource+'��':''):' �� ��ͼδ�����ɶ�����޷��Զ��˶�')));
  }
 // �� �ٸ�ͬ���������
 const rec=el('div',null,'recognition'+(vd.ok?'':' warn'));
 rec.append(el('div','�� ͬ�������'+(found.fused?'CLIP + DINOv2 ˫ģ���ں�':'CLIP ��ģ��')+'��ȫ�� '+found.total+' �ŵ�ͼ����ȶԣ�ͬһ��=ͬһ��Ʒ��Ƕȣ���','recog-title'));
 rec.append(el('div',vd.ok?(vd.tentative?'? ����ͬ�'+vd.text+'��'+vd.reason+'�����һ��������ͬ�����غ˶��ͺ�/���/��������ȷ�ϡ�':'? �ҵ�ͬ�'+vd.text+'��'+vd.reason):('���˶ԣ�'+vd.text+'��'+vd.reason+'����ǰ֤�ݲ�����ȷ�Ͽ����Ƿ����ͬ�')));
 if(q._specCheck?.review)rec.append(el('div',q._specCheck.reason||'���֤�ݴ��˶�','sub'));
 idBox.append(rec);
 section.append(idBox);
 if(a.candidates?.length){
  const line=el('div',null,'candidates');line.append(el('span','��ѡ���ƣ�����˹�ȷ�ϣ���','sub'));
  for(const c of a.candidates){
   line.append(btn(c.name+' '+c.score.toFixed(3),()=>{q.analysis={...q.analysis,suggestedName:c.name,suggestedCategory:c.category,status:'�˹�ȷ��',confident:true,method:'�˹�ѡ��',matchedGroup:c.group};section.replaceWith(result(q,forceAll))},'pill'));
  }
  idBox.append(line);
 }
 const grid=el('div',null,'grid');
 for(let hi=0;hi<found.results.length;hi++){
  const hit=found.results[hi],l=effectiveLabel(hit.image),c=el('article',null,'card'),im=image('/images/'+hit.image.id,l.name),body=el('div',null,'body');
  im.onclick=()=>compare(q,hit.image);
  const v=verdict(hit),sc=hit.productScore??hit.score,nx=found.results[hi+1]?(found.results[hi+1].productScore??found.results[hi+1].score):0;
  const check=q._specChecks?.[hit.label.group||hit.image.id];
  if(check?.conflict){v.ok=false;v.text='��񲻷�';v.reason=check.reason}
  const lead='����ֲ��Ǹ���';
  body.append(el('h3',l.name),el('div',(check?.conflict?'��񲻷�':(v.ok?(v.tentative?'? ����ͬ��':'? �˹�ȷ��ͬ��'):'��ѡ���˶�'))+' �� ��Ʒƥ��� '+sc.toFixed(3)+(hit.fused?'��˫ģ�ͣ�':'��CLIP��')+' �� �Ӿ� '+hit.visual.toFixed(3)+' �� '+hit.views.length+' ���Ƕ� �� ������һλ '+lead,'sub'));
  const views=el('div',null,'views');
  for(const v2 of hit.views){const thumb=image('/images/'+v2.id,'��Ʒ�Ƕ�');thumb.onclick=()=>compare(q,v2);views.append(thumb)}
  body.append(views);
  body.append(btn('�Ŵ�Ա�',()=>compare(q,hit.image)));
  const confirm=btn('ȷ��ͬ��',async()=>{
   confirm.disabled=true;
   try{const list=[...(hit.image.confirmations||[]),{decision:'same',query:q.file.name,querySha256:q.sha256,groupId:hit.label.group||hit.image.id,at:new Date().toISOString()}];await savePatch(hit.image.id,{confirmations:list});section.replaceWith(result(q,forceAll))}catch(e){confirm.disabled=false;status('query-status',e.message,true)}
  });
  const reject=btn('��Ϊ��ͬ��',async()=>{
   reject.disabled=true;
   try{const list=[...(hit.image.confirmations||[]),{decision:'different',query:q.file.name,querySha256:q.sha256,groupId:hit.label.group||hit.image.id,at:new Date().toISOString()}];await savePatch(hit.image.id,{confirmations:list});section.replaceWith(result(q,forceAll))}catch(e){reject.disabled=false;status('query-status',e.message,true)}
  });
  body.append(confirm,reject);c.append(im,body);grid.append(c);
 }
 section.append(grid);
 if(!found.results.length)section.append(el('p','���޿ɼ�����ͼ���������ӵ�ͼ����ɼ���׼����','empty'));
 else if(!vd.ok)section.append(el('p','Ŀǰ�޷�ȷ��ͬ�����Ϊ������ѡ��֤�ݲ��㲻��ʾ��ͼ����û�иò�Ʒ��','sub'));
 const actions=el('div',null,'actions');
 if(found.stage!=='all')actions.append(btn('û��ͬ�����ȫ����ͼ',()=>section.replaceWith(result(q,true))));
 if(found.results.length>=topK)actions.append(btn('��ʾ������',()=>{topK=Math.min(topK*2,64);$('top-k').value=String(topK);section.replaceWith(result(q,forceAll))}));
 section.append(actions);
 q._summary={name:q.file.name,cat:(vd.ok&&top?top.label.category:null)||a?.mlCategory||'δ�ж�',
  catSource:vd.ok&&top?'��ͼ���ǩ':'CLIP ������',product:vd.ok&&top?top.label.name:(a?.suggestedName&&a.suggestedName!=='δʶ��'?a.suggestedName:(a?.referenceName&&!vd.ok?'δʶ��':(a?.suggestedName||'δʶ��'))),
  best:top?top.label.name:'��',group:top?top.label.group:'��',score:top?(top.productScore??top.score):0,visual:top?top.visual:0,
  lead:vd.ratio||0,ok:vd.ok,tentative:!!vd.tentative,verdict:vd.text,reason:vd.reason||'',n:found.results.length,
  topId:top?top.image.id:null,specConflict:q._specCheck?.conflict||'',
  querySpecs:q._specCheck?.querySpecs?summarizeSpecs(q._specCheck.querySpecs):''};
 return section;
}
// ���ŶԱ�ͼ�Ļ��ܱ���һ�ο���ÿ��ͼ��ʶ�������Ƿ��ҵ�ͬ�
function summaryTable(rows){
 const box=el('div',null,'panel summary-panel');
 box.append(el('h2','�������ջ��ܣ�'+rows.length+' �ŶԱ�ͼ��'));
 const t=el('table',null,'summary');const head=el('tr');
 for(const h of ['�Ա�ͼ','ʶ��������ݣ�','���ƣ����ڲο���','��Ѻ�ѡ����','��ͼ���','��Ʒƥ���','����˵��','�Ƿ�ͬ��'])head.append(el('th',h));
 t.append(head);
 for(let i=0;i<rows.length;i++){
  const r=rows[i],tr=el('tr');
  tr.append(el('td',(i+1)+'. '+r.name),el('td',r.cat+'��'+r.catSource+'��'),el('td',r.product),el('td',r.group),
   el('td',r.querySpecs||'��'),el('td',r.score.toFixed(3)),el('td','�Ǹ��ʷ���'),
   el('td',r.ok?(r.tentative?'? ���ƣ����˶ԣ�':'? ��ͬ��'):(r.specConflict?'? ��񲻷� '+r.specConflict:'���˶�')));
  t.append(tr);
 }
 box.append(t);
 box.append(el('div','˵����ÿ�ŶԱ�ͼ��������������ֲ�����ͬ����ʣ�ֻ�й��֤�ݳ��ʱ���ų���ѡ��������˶Խṹ���ͺź͹��','sub'));
 return box;
}
async function search(){
 if(working||indexing)return status('query-status','��ͼ���ڱ����׼�����������Ժ��ٲ��ҡ�');
 working=true;renderQueries();$('results').replaceChildren();
 try{
  await refresh();
  const missing=library.filter(x=>!READY(x)).length;
  if(missing){$('index-retry').classList.remove('hidden');return status('query-status','���� '+missing+' �ŵ�ͼδ׼���ü�����������ࡰ����׼����ͼ��������',true)}
  const rows=[];
  for(let i=0;i<queries.length;i++){
   const q=queries[i];status('query-status','����ʶ���Ʒ���ƺ���Ŀ '+(i+1)+' / '+queries.length);
   if(!q.sha256)q.sha256=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await q.file.arrayBuffer())),x=>x.toString(16).padStart(2,'0')).join('');
   if(!q.analysis)q.analysis=await analyzeQuery(q.url,t=>status('query-status',t));
  status('query-status','���ڰ������ͬ�');
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
     conflict:cf.conflict,reason:cf.reason,review:cf.review,libSource:'��ͼ��Ƕȹ���'};
   }
   const fresh=result(q);sec.replaceWith(fresh);sec=fresh;
  }
  if(q._summary)rows.push(q._summary);
  }
  if(rows.length>1)$('results').prepend(summaryTable(rows));
  status('query-status','������ɣ������� '+queries.length+' ��ͼƬ��ÿ��ͼ��ʶ�����ʶ���������Ƿ��ҵ�ͬ������ܱ���ͬ���ϵ�������˹�ȷ�ϡ�');
 }catch(e){status('query-status',e.message,true)}finally{working=false;renderQueries();if(baseQueue.length)drainBase()}
}

/* ---------- ��ק / ճ�� / ͼƬ��ַ ---------- */
function bindDrop(id,kind,add){
 const area=$(id);
 area.onfocus=()=>pasteTarget=kind;area.onpointerdown=()=>pasteTarget=kind;
 area.ondragover=e=>{e.preventDefault();area.classList.add('drag')};
 area.ondragleave=()=>area.classList.remove('drag');
 area.ondrop=e=>{e.preventDefault();area.classList.remove('drag');const files=[...e.dataTransfer.files];if(files.length)add(files);else{const url=e.dataTransfer.getData('text/uri-list')||e.dataTransfer.getData('text/plain');if(url)fromURL(url,kind)}};
 // �Ҽ��˵����û�Ҫ�ġ��һ��Ѽ�����ͼƬ�Ž��������������ȫ���Բ������ű�ֱ�Ӷ����������ݣ�
 // ���Բ˵���һ��۽��ϴ�������ʾ�� Ctrl+V����������ֱ�Ӵ��ļ�ѡ���� / ͼƬ��ַ�����
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
  const url=new URL(text.trim());if(url.protocol!=='https:')throw Error('��ʹ�� https ͼƬ��ַ');
  status(id,'���ڶ�ȡͼƬ��ַ��');
  let r=await fetch('/api/proxy-image?url='+encodeURIComponent(url.href),{signal:AbortSignal.timeout(30000)});
  if(!r.ok)r=await fetch(url.href,{referrerPolicy:'no-referrer',signal:AbortSignal.timeout(25000)});
  if(!r.ok)throw Error('ͼƬ��ַ�޷���ȡ��'+r.status+'��');
  const b=await r.blob();if(!b.type.startsWith('image/'))throw Error('��ַδ����ͼƬ');
  const file=new File([b],url.pathname.split('/').pop()||'ճ��ͼƬ.png',{type:b.type});
  (kind==='base'?addBase:addQueries)([file]);
 }catch(e){status(id,'�޷���ȡͼƬ���ӣ��븴��ͼƬ���������غ����롣'+e.message,true)}
}

/* ---------- �¼��� ---------- */
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
 $(kind+'-drop').focus();status(kind+'-status','��׼���ý��ռ�����ͼƬ���밴 Ctrl+V ճ����Chrome ��ȫ���Բ�������ҳֱ�Ӷ�ȡ�����壩��');
};
$('drop-menu-pick').onclick=()=>{const kind=$('drop-menu').dataset.kind||'query';hideDropMenu();$(kind+'-pick').click()};
$('drop-menu-url').onclick=()=>{
 const kind=$('drop-menu').dataset.kind||'query';hideDropMenu();
 $(kind+'-drop').parentElement.querySelector('details')?.setAttribute('open','');
 $(kind+'-url').focus();status(kind+'-status','����չ����ͼƬ��ַ��ճ�� https ���Ӻ�㡸���롹��');
};
document.addEventListener('pointerdown',e=>{if(!e.target.closest('#drop-menu'))hideDropMenu()});
window.addEventListener('blur',hideDropMenu);
$('base-files').onchange=e=>{addBase([...e.target.files]);e.target.value=''};
$('query-files').onchange=e=>{addQueries([...e.target.files]);e.target.value=''};
$('base-url-add').onclick=()=>fromURL($('base-url').value,'base');
$('query-url-add').onclick=()=>fromURL($('query-url').value,'query');
$('query-run').onclick=search;
$('query-clear').onclick=()=>{if(working)return;queries.forEach(x=>URL.revokeObjectURL(x.url));queries=[];renderQueries();$('results').replaceChildren();status('query-status','����նԱ�ͼƬ����ͼ�Ᵽ�����䡣')};
$('top-k').onchange=e=>{topK=Number(e.target.value)||8};
$('library-search').oninput=()=>{page=36;renderLibrary()};
$('category-filter').onchange=()=>{page=36;renderLibrary()};
$('library-more').onclick=()=>{page+=36;renderLibrary()};
$('group-toggle').onclick=e=>{grouped=!grouped;e.target.textContent=grouped?'ƽ����ʾ':'��ͬ�����';renderLibrary()};
$('pending-toggle').onclick=e=>{pendingOnly=!pendingOnly;e.target.textContent=pendingOnly?'��ʾȫ��':'ֻ����ȷ��';renderLibrary()};
$('select-all').onclick=()=>{const rows=grouped?filtered():filtered().slice(0,page);const all=rows.every(x=>selected.has(x.id));for(const x of rows)all?selected.delete(x.id):selected.add(x.id);renderLibrary()};
$('export').onclick=exportCsv;
$('catalog-sync').onclick=syncCatalog;
$('batch-edit').onclick=()=>{if(!selected.size)return status('base-status','���ȹ�ѡҪ��ע�ĵ�ͼ',true);$('batch-count').textContent='��ѡ�� '+selected.size+' �ŵ�ͼ�����յ��ֶα��ֲ��䡣';$('batch-dialog').showModal()};
$('batch-cancel').onclick=()=>$('batch-dialog').close();
$('batch-save').onclick=async()=>{
 const body={};if($('batch-name').value.trim())body.name=$('batch-name').value.trim();if($('batch-category').value)body.category=$('batch-category').value;if($('batch-group').value.trim())body.group=$('batch-group').value.trim();
 if(!Object.keys(body).length)return status('base-status','��������дһ��Ҫ�޸ĵ��ֶ�',true);
 const ids=[...selected];$('batch-save').disabled=true;let n=0;
 await mapLimit(ids,3,async id=>{await savePatch(id,body);n++;status('base-status','����������ע '+n+' / '+ids.length)});
 $('batch-save').disabled=false;$('batch-dialog').close();selected.clear();renderLibrary();status('base-status','�Ѹ��� '+n+' �ŵ�ͼ�Ĳ�Ʒ��Ϣ��');
};
$('edit-cancel').onclick=()=>$('edit-dialog').close();
$('edit-save').onclick=async()=>{
 const body={name:$('edit-name').value.trim(),category:$('edit-category').value,group:$('edit-group').value.trim()};
 if(!body.name||!body.group)return status('base-status','��Ʒ���ƺ�ͬ����鲻��Ϊ��',true);
 try{await savePatch($('edit-dialog').dataset.id,body);$('edit-dialog').close();renderLibrary();status('base-status','��Ʒ��Ϣ�Ѹ��¡�')}catch(e){status('base-status',e.message,true)}
};
$('viewer-close').onclick=()=>$('viewer').close();
$('viewer-mode').onclick=()=>{overlay=!overlay;renderCompare()};
$('opacity').oninput=e=>{const im=$('viewer-body').querySelector('img.over');if(im)im.style.opacity=e.target.value/100};
$('index-retry').onclick=prepareIndex;
$('trash').ontoggle=async()=>{
 if(!$('trash').open)return;
 try{
  const rows=await api('/api/images?trash=1'),box=$('trash-items');box.replaceChildren();
  for(const x of rows){const row=el('p',effectiveLabel(x).name+' '),restore=btn('�ָ���ͼ',async()=>{restore.disabled=true;try{await patch(x.id,{deleted:false});row.remove();await refresh()}catch(e){restore.disabled=false;status('base-status',e.message,true)}});row.append(restore);box.append(row)}
  if(!rows.length)box.append(el('p','�����Ƴ���ͼƬ'));
 }catch(e){status('base-status',e.message,true)}
};

refresh().then(()=>{if(library.some(x=>!READY(x)))prepareIndex()}).catch(e=>status('base-status',e.message,true));

