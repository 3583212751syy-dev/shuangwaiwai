// 用不同视觉特征口径对 1.webp 排序，检验"第473行 image1540 彩色切割片"能否排到首位。
// baseline=现有 CLIP 512 维向量；后续扩展其它模型。
import {readFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('../',import.meta.url));
const W='D:/GPT/2026-11-21/new-chat-2/work';

const meta=JSON.parse(await readFile(W+'/deploy-product-library/metadata.json','utf8'));
const embs=JSON.parse(await readFile(path.join(root,'work/embeddings.json'),'utf8'));
const cos=(a,b)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s};

// 加载 CLIP 模型给 1.webp 提向量
const base=W+'/deploy-product-library/model-cache';
process.env.NO_COLOR='1';
const {pipeline,env}=await import('file:///'+(W+'/model-runtime/node_modules/.pnpm/@huggingface+transformers@3.8.1/node_modules/@huggingface/transformers/dist/transformers.node.mjs').replace(/\\/g,'/'));
env.allowLocalModels=true;env.localModelPath=base;env.useBrowserCache=false;
const clip=await pipeline('image-feature-extraction','Xenova/clip-vit-base-patch32',{device:'cpu',dtype:'q8'});
const out=await clip('C:/Users/Administrator/Desktop/1.webp',{pooling:'mean'});
let q=Array.from(out.data||out[0]?.data||out);
// 归一化
const n=Math.hypot(...q);q=q.map(x=>x/n);
console.log('query dim',q.length);

const embOf=f=>embs['originals/'+f]?.embedding||embs[f]?.embedding;
const ranked=meta.filter(x=>embOf(x.filename)).map(x=>({
 file:x.filename,group:x.group,name:x.name,sim:cos(q,embOf(x.filename))
})).sort((a,b)=>b.sim-a.sim);
const TARGET='image1540.jpeg';
const rank=ranked.findIndex(x=>x.file===TARGET)+1;
console.log('=== CLIP baseline ===');
console.log('目标 image1540(第473行彩色切割片) 排名',rank,'/',ranked.length);
console.log('top20:');
for(const x of ranked.slice(0,20))console.log(' ',x.sim.toFixed(4),x.file,x.group.split('·').pop().trim());
const tgt=ranked.find(x=>x.file===TARGET);
console.log('目标相似度',tgt?.sim?.toFixed(4));
// 所有真正彩色切割片组
console.log('第473行组其他图排名:',ranked.filter(x=>['image1540.jpeg','image1541.jpeg','image1542.jpeg'].includes(x.file)).map(x=>ranked.indexOf(x)+1));
