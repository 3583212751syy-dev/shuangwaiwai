import {readFile} from 'node:fs/promises';
const W='D:/GPT/2026-11-21/new-chat-2/work';
const meta=JSON.parse(await readFile(W+'/deploy-product-library/metadata.json','utf8'));
const embs=JSON.parse(await readFile('work/embeddings.json','utf8'));
const {pipeline,env}=await import('file:///'+(W+'/model-runtime/node_modules/.pnpm/@huggingface+transformers@3.8.1/node_modules/@huggingface/transformers/dist/transformers.node.mjs'));
env.allowLocalModels=true;env.localModelPath=W+'/deploy-product-library/model-cache';env.useBrowserCache=false;
const clip=await pipeline('image-feature-extraction','Xenova/clip-vit-base-patch32',{device:'cpu',dtype:'q8'});
const out=await clip('C:/Users/Administrator/Desktop/1.webp',{pooling:'mean'});
let q=Array.from(out.data||out[0]?.data||out);const n=Math.hypot(...q);q=q.map(x=>x/n);
const embOf=f=>embs['originals/'+f]?.embedding||embs[f]?.embedding;
const cos=(a,b)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s};
// 按 group 聚合：最多分、平均分、>0.55 的张数
const groups={};
for(const x of meta){const e=embOf(x.filename);if(!e)continue;(groups[x.group]??=[]).push(cos(q,e))}
const scored=Object.entries(groups).map(([g,ss])=>{
 const sorted=[...ss].sort((a,b)=>b-a);
 const top=sorted[0],avg=ss.reduce((a,b)=>a+b,0)/ss.length,hits=ss.filter(s=>s>=0.55).length;
 // 产品相似度主分：最高分 + 组内命中数加成 + 平均分（同款多角度聚集=产品信号）
 const product=top*0.5+Math.min(hits,6)/6*0.3+avg*0.2;
 return {g,top,avg,hits,n:ss.length,product};
}).sort((a,b)=>b.product-a.product);
console.log('=== 产品相似度(聚集度)排序 top15 ===');
for(const x of scored.slice(0,15))console.log(x.product.toFixed(4),'top',x.top.toFixed(3),'avg',x.avg.toFixed(3),'hits',x.hits+'/'+x.n,x.g);
const rank=scored.findIndex(x=>x.g==='第473行 · 切割片套装')+1;
console.log('第473行(同款)产品相似度排名',rank,'/',scored.length);
