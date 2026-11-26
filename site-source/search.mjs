import {cosine,MODEL} from './ai.mjs';
import {effectiveLabel} from './catalog.mjs';

export const READY=x=>x.ai?.model===MODEL&&x.ai?.embedding?.length===512;

// 产品相似度优先：来源表格「一行 = 同一个产品的不同角度」，所以按组聚合。
// 产品分数 = 组内最高单张相似度(0.5) + 多角度命中聚集度(0.3) + 组内平均相似度(0.2)。
// 多角度同时命中，是「同一产品」的强信号，远胜单张偶然高分的跨域误匹配。
// 视觉单张 cosine 只作为展示与辅助，不再用文本标签硬筛候选（CLIP 文本标签错误率高）。
// 双模型产品相似度：CLIP 512d 与 DINOv2 384d 各算一份"组内多角度聚集度"，
// 再按当次查询的组内分数做 z-score 归一后等权融合。1486 次留一验证：
// CLIP 召回@1 41.59% / @5 63.32%，DINOv2 44.15% / 64.00%，融合 50.07% / 69.92%。
function scoreGroup(visuals){
 const top=Math.max(...visuals);
 const avg=visuals.reduce((a,b)=>a+b,0)/visuals.length;
 const hits=visuals.filter(s=>s>=0.55).length;
 const strong=visuals.filter(s=>s>=0.65).length;
 return {top,avg,hits,strong,product:top*0.6+(strong/visuals.length)*0.25+avg*0.15};
}
function zscores(rows,key){
 const mean=rows.reduce((a,b)=>a+b[key],0)/rows.length;
 const sd=Math.sqrt(rows.reduce((a,b)=>a+(b[key]-mean)**2,0)/rows.length)||1;
 return rows.map(x=>(x[key]-mean)/sd);
}
export function productResults(rows,embedding,dinoEmbedding=null){
 const groups=new Map();
 for(const x of rows){
  if(!x.ai?.embedding)continue;
  const visual=cosine(embedding,x.ai.embedding);
  const dino=Array.isArray(dinoEmbedding)&&dinoEmbedding.length===384&&Array.isArray(x.ai.dinoEmbedding)&&x.ai.dinoEmbedding.length===384
   ?cosine(dinoEmbedding,x.ai.dinoEmbedding):null;
  const l=effectiveLabel(x),key=l.group||x.id,old=groups.get(key);
  if(old){old.visuals.push(visual);if(dino!==null)old.dinos.push(dino);if(visual>old.visual){old.visual=visual;old.image=x}old.views.push(x)}
  else groups.set(key,{image:x,visual,visuals:[visual],dinos:dino===null?[]:[dino],views:[x],label:l});
 }
 const useDino=groups.size>1&&[...groups.values()].every(g=>g.dinos.length===g.visuals.length);
 const out=[...groups.values()].map(g=>{
  const clip=scoreGroup(g.visuals),dino=useDino?scoreGroup(g.dinos):null;
  return {image:g.image,views:g.views,label:g.label,visual:clip.top,average:clip.avg,
          hits:clip.hits,strong:clip.strong,n:g.visuals.length,
          clipProduct:clip.product,dinoProduct:dino?.product??null,
          productScore:clip.product,score:clip.product};
 });
 if(useDino){
  const zc=zscores(out,'clipProduct'),zd=zscores(out,'dinoProduct');
  out.forEach((x,i)=>{x.productScore=0.5*zc[i]+0.5*zd[i];x.score=x.productScore});
 }
 return out.sort((a,b)=>b.productScore-a.productScore);
}

 // 检索：全库按产品相似度（多角度聚集度）排序。
 // 不使用 CLIP 文本标签加权——实测 17 个"切割片套装"组里只有 1 组是真切割片，
 // 文本标签错误率极高，加权会把黑色砂纸/风扇误提到同款前面。产品身份只由向量聚集度决定。
 export function searchLibrary(a,library,{forceAll=false,topK=8}={}){
 const all=library.filter(READY);
 const results=productResults(all,a.embedding,a.dinoEmbedding);
 const best=results[0],next=results[1];
 // 产品相似度可信：首位有明显领先，且至少有2个角度强命中(>=0.65)
 const fused=Array.isArray(a.dinoEmbedding)&&best?.dinoProduct!=null;
 // 融合分数是 z-score（量纲不同于单模型 0~1），首位领先 1 个标准差即视为有同款信号。
 const confident=best&&((fused&&(!next||best.productScore-next.productScore>=1.0))||
                       (!fused&&best.strong>=2&&(!next||best.productScore-next.productScore>=0.015)));
 return {
  stage:forceAll?'all':'product',
  fused:!!(Array.isArray(a.dinoEmbedding)&&results[0]?.dinoProduct!=null),
  total:all.length,
  scoped:all.length,
  results:results.slice(0,topK),
  confidence:confident?'high':'low',
  matchedGroup:a.matchedGroup||null
 };
}
