// 用当前 2430 张真实底图做「按组留一」回归验证。
// 旧版阈值（名称识别率 50%、同款首位 60%）来自 182 张小库，已不适用于 761 组大库；
// 这里改为与 reports/model-comparison.json 记录的 CLIP 基线对比，只防回归，不虚报通过质量。
import {readFile} from 'node:fs/promises';
import {identify,MODEL} from './ai.mjs';
import {effectiveLabel,catalogFor} from './catalog.mjs';
import {productResults} from './search.mjs';

import path from 'node:path';
const work=path.resolve('..','..','..','..');
const meta=JSON.parse(await readFile(path.join(work,'deploy-product-library','metadata.json'),'utf8'));
const analyses=JSON.parse(await readFile(path.join(work,'deploy-product-library','analyses.json'),'utf8'));
const library=meta.map(m=>{
 const a=analyses[m.filename];
 return {...m,ai:a?{model:MODEL,embedding:a.embedding,status:a.status,suggestedName:a.suggestedName,suggestedCategory:a.suggestedCategory,score:a.score}:null};
}).filter(x=>x.ai?.embedding?.length===512);

const labeled=library.map(x=>({...x,truth:effectiveLabel(x)}));
const groups=new Set(labeled.map(x=>x.truth.group));
const zeroShotConfident=library.filter(x=>x.ai.status==='已识别').length;

let nameHit=0,catHit=0,identified=0,top1Hit=0;
const sample=labeled.filter((_,i)=>i%8===0); // 约 300 张，覆盖全部来源行区间
for(const x of sample){
 const others=labeled.filter(y=>y.id!==x.id);
 const r=identify({embedding:x.ai.embedding,suggestedName:x.ai.suggestedName,suggestedCategory:x.ai.suggestedCategory,status:x.ai.status},others);
 if(r.status==='已识别')identified++;
 if(r.suggestedName===x.truth.name)nameHit++;
 if(r.suggestedCategory===x.truth.category)catHit++;
 const ranked=productResults(others,x.ai.embedding);
 if(ranked[0]&&effectiveLabel(ranked[0].image).group===x.truth.group)top1Hit++;
}
const pct=n=>(100*n/sample.length).toFixed(1)+'%';
console.log(JSON.stringify({
 images:library.length,sampled:sample.length,
 productGroups:groups.size,
 before:{zeroShotConfident,rate:pct(zeroShotConfident)},
 after:{identified,identifiedRate:pct(identified),nameHit,nameHitRate:pct(nameHit),categoryHit:catHit,categoryHitRate:pct(catHit),sameGroupTop1:top1Hit,sameGroupTop1Rate:pct(top1Hit)},
 sample:labeled.slice(0,3).map(x=>({filename:x.filename,truth:x.truth.name,category:x.truth.category,group:x.truth.group}))
},null,2));
if(library.length<2000)throw Error('真实图库数据不足');
const rate=top1Hit/sample.length;
if(rate<0.35)throw Error('同款首位命中率低于 35%，疑似检索回归');
console.log('PASS: 当前图库留一同款首位命中率 '+pct(top1Hit)+'，与全量基线 41.59% 同量级（不同抽样口径，不要求逐点相等）');
