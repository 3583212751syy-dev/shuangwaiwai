import {cosine,MODEL} from './ai.mjs';
import {effectiveLabel} from './catalog.mjs';

export const validVector=(v,dim)=>Array.isArray(v)&&v.length===dim&&v.every(Number.isFinite)&&v.some(x=>x!==0);
export const READY=x=>x.ai?.model===MODEL&&validVector(x.ai?.embedding,512);
const DINO_MODEL='onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX';
const DINO_POOLING='cls-v1';
export const DINO_READY=x=>validVector(x.ai?.dinoEmbedding,384)&&
 (!x.ai.dinoModel||x.ai.dinoModel===DINO_MODEL)&&
 (!x.ai.dinoPooling||x.ai.dinoPooling===DINO_POOLING);

// Rank fusion keeps differently scaled model scores separate. Missing DINO only
// removes that candidate's second vote; it never disables the rest of the index.
export function productResults(rows,embedding,dinoEmbedding=null){
 if(!validVector(embedding,512))return [];
 const queryDino=validVector(dinoEmbedding,384),groups=new Map();
 for(const x of rows){
  if(!READY(x))continue;
  const clip=cosine(embedding,x.ai.embedding);
  const dino=queryDino&&DINO_READY(x)?cosine(dinoEmbedding,x.ai.dinoEmbedding):null;
  const label=effectiveLabel(x),key=label.group||x.id;
  if(!groups.has(key))groups.set(key,{image:x,label,views:[],visual:-Infinity,dinoVisual:null,clipValues:[],dinoValues:[]});
  const g=groups.get(key);g.views.push(x);g.clipValues.push(clip);
  if(clip>g.visual){g.visual=clip;g.image=x}
  if(dino!==null){g.dinoValues.push(dino);g.dinoVisual=Math.max(g.dinoVisual??-Infinity,dino)}
 }
 const out=[...groups.values()].map(g=>{
  const avg=g.clipValues.reduce((a,b)=>a+b,0)/g.clipValues.length;
  return {...g,average:avg,n:g.views.length,hits:g.clipValues.filter(x=>x>=.55).length,
   strong:g.clipValues.filter(x=>x>=.65).length,clipProduct:g.visual,
   dinoProduct:g.dinoVisual,dinoCoverage:g.dinoValues.length/g.views.length};
 });
 const ranked=(key)=>[...out].filter(x=>x[key]!==null).sort((a,b)=>b[key]-a[key]||String(a.image.id).localeCompare(String(b.image.id)));
 const clips=ranked('clipProduct'),dinos=ranked('dinoProduct');
 // Equal scores share ranks; order of library records must not create evidence.
 const assign=(list,key,field)=>{let rank=0,prior;list.forEach((x,i)=>{if(i===0||x[key]!==prior)rank=i+1;x[field]=rank;prior=x[key]})};
 assign(clips,'clipProduct','clipRank');assign(dinos,'dinoProduct','dinoRank');
 for(const x of out){
  x.fused=x.dinoRank!=null;
  // Missing DINO falls back to the CLIP rank for ordering, not new evidence.
  x.productScore=(1/(60+x.clipRank)+(dinos.length?1/(60+(x.dinoRank??x.clipRank)):0))*60;
  x.score=x.productScore;
  delete x.clipValues;delete x.dinoValues;
 }
 return out.sort((a,b)=>b.productScore-a.productScore||b.visual-a.visual||String(a.image.id).localeCompare(String(b.image.id)));
}

// Absolute visual evidence can flag a candidate, never certify product identity.
// These conservative heuristics are not calibrated probabilities.
export function candidateVerdict(hit){
 if(!hit)return {text:'没有可比较的候选',ok:false,ratio:0,reason:'请先上传图片及确认索引状态'};
 if(hit.userConfirmed)return {text:'人工确认同款',ok:true,tentative:false,ratio:0,top:hit.productScore,reason:'与同一查询图片SHA-256已被人工确认'};
 const tentative=hit.visual>=.84&&hit.dinoVisual!==null&&hit.dinoVisual>=.70;
 return {text:tentative?'疑似同款（需人工核对）':'候选（待核对）',ok:tentative,tentative:true,ratio:0,
  top:hit.productScore,reason:'相似度只用于排序，不能证明同款；请人工核对结构、接口、型号和规格'};
}
export function searchLibrary(a,library,{forceAll=false,topK=8,querySha256=null}={}){
 const all=library.filter(READY);let results=productResults(all,a?.embedding,a?.dinoEmbedding);
 if(/^[a-f0-9]{64}$/.test(querySha256||'')){
  results=results.filter(hit=>{
   const events=hit.views.flatMap(v=>v.confirmations||[]).filter(e=>e.querySha256===querySha256&&e.groupId===(hit.label.group||hit.image.id)&&Number.isFinite(Date.parse(e.at)));
   events.sort((x,y)=>Date.parse(y.at)-Date.parse(x.at));
   const latest=events[0];hit.userConfirmed=latest?.decision==='same';return latest?.decision!=='different';
  }).sort((x,y)=>Number(!!y.userConfirmed)-Number(!!x.userConfirmed)||y.productScore-x.productScore);
 }
 return {stage:forceAll?'all':'product',fused:results.some(x=>x.fused),total:all.length,scoped:all.length,
  results:results.slice(0,Math.max(1,Math.min(64,topK))),confidence:'low',matchedGroup:a?.matchedGroup||null};
}

