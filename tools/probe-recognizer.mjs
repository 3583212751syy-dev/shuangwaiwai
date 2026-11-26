// 用真实 createRecognizer（浏览器同款）诊断对比图在词典上的真实得分。
import {readFileSync} from 'node:fs';
const W='D:/GPT/2026-11-21/new-chat-2/work';
const tf='file:///'+W+'/model-runtime/node_modules/.pnpm/@huggingface+transformers@3.8.1/node_modules/@huggingface/transformers/dist/transformers.node.mjs';
const lib=await import(tf);
lib.env.allowLocalModels=true;lib.env.localModelPath=W+'/deploy-product-library/model-cache';lib.env.useBrowserCache=false;
const {createRecognizer}=await import('./../site-source/ai.mjs');
const r=await createRecognizer(lib,()=>{},'cpu',W+'/deploy-product-library/model-cache');
for(const f of process.argv.slice(2)){
 const a=await r.analyze(f);
 console.log('\n===',f,'===');
 console.log('产品词典top',a.productTopName,'score',a.productScore?.toFixed(4),'cos',a.productCos?.toFixed(3));
 try{const rk=decision?.ranked||a.productRanked;if(rk)console.log('词典top3:',rk.slice(0,3).map(x=>CATALOG_NAME(x)).join(' | '))}catch(e){}
 console.log('suggested',a.suggestedName,'cat',a.suggestedCategory,'confident',a.confident);
 console.log('32类:',a.mlCandidates?.map(x=>x.category+' '+x.score.toFixed(3)).join(' | '));
 console.log('二级:',a.mlSubs?.map(x=>x.sub+' '+x.score.toFixed(3)).join(' | '));
}
r.dispose();
