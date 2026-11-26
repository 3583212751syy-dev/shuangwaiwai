// 全量对照：CLIP 512d vs DINOv2-small 384d，用同一套"产品相似度"公式排序。
// 查询 = 桌面 1.webp（真值：第473行 彩色切割片）。
import {readFileSync} from 'node:fs';
const W='D:/GPT/2026-11-21/new-chat-2/work';
const tf='file:///'+W+'/model-runtime/node_modules/.pnpm/@huggingface+transformers@3.8.1/node_modules/@huggingface/transformers/dist/transformers.node.mjs';
const meta=JSON.parse(readFileSync(W+'/deploy-product-library/metadata.json','utf8'));
const clipEmb=JSON.parse(readFileSync('work/embeddings.json','utf8'));
const dinoEmb=JSON.parse(readFileSync('work/embeddings-dinov2.json','utf8'));
const cos=(a,b)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s};
function norm(v){const n=Math.hypot(...v)||1;return v.map(x=>x/n)}

// ---- CLIP query
const {pipeline,env:env1}=await import(tf);
env1.allowLocalModels=true;env1.localModelPath=W+'/deploy-product-library/model-cache';env1.useBrowserCache=false;
const clip=await pipeline('image-feature-extraction','Xenova/clip-vit-base-patch32',{device:'cpu',dtype:'q8'});
const co=await clip('C:/Users/Administrator/Desktop/1.webp',{pooling:'mean'});
const qClip=norm(Array.from(co.data||co[0]?.data||co));
console.log('CLIP query embedded',qClip.length);

// ---- DINOv2 query
const {AutoModel,AutoProcessor,env:env2,RawImage}=await import(tf);
env2.allowLocalModels=true;env2.localModelPath=W+'/model-alt';env2.useBrowserCache=false;
const processor=await AutoProcessor.from_pretrained('onnx-community/dinov2-small');
const model=await AutoModel.from_pretrained('onnx-community/dinov2-small',{dtype:'q8',device:'cpu'});
import {execFileSync} from 'node:child_process';
const base=W+'/_cmp_raw';
execFileSync('python',['-c',`from PIL import Image;import sys;im=Image.open(sys.argv[1]).convert('RGB');open('${base}.bin','wb').write(im.tobytes());open('${base}.txt','w').write(f'{im.width} {im.height}')`,'C:/Users/Administrator/Desktop/1.webp']);
const [rw,rh]=readFileSync(base+'.txt','utf8').trim().split(' ').map(Number);
const raw=new RawImage(new Uint8Array(readFileSync(base+'.bin')),rw,rh,3);
const inputs=await processor(raw);
const mo=await model(inputs);
const mv=Array.from(mo.last_hidden_state.data),dim=384,n=mv.length/dim,out=new Array(dim).fill(0);
for(let t=0;t<n;t++)for(let d=0;d<dim;d++)out[d]+=mv[t*dim+d];
const qDino=norm(out.map(x=>x/n));
console.log('DINOv2 query embedded',qDino.length);

// ---- 按来源行分组，产品相似度公式与线上 search.mjs 一致
function rank(embOf,q){
 const groups=new Map();
 for(const x of meta){
  const e=embOf(x.filename);if(!e)continue;
  const s=cos(q,e),g=x.group||x.id,old=groups.get(g);
  if(old){old.ss.push(s);if(s>old.top)old.top=s}
  else groups.set(g,{g,ss:[s],top:s});
 }
 const rows=[...groups.values()].map(o=>{
  const avg=o.ss.reduce((a,b)=>a+b,0)/o.ss.length;
  const strong=o.ss.filter(s=>s>=0.65).length;
  return {g:o.g,n:o.ss.length,top:o.top,avg,strong,product:o.top*0.6+(strong/o.ss.length)*0.25+avg*0.15};
 }).sort((a,b)=>b.product-a.product);
 return rows;
}
const clipRows=rank(f=>clipEmb['originals/'+f]?.embedding||clipEmb[f]?.embedding,qClip);
const dinoRows=rank(f=>dinoEmb[f]||dinoEmb['originals/'+f],qDino);
const TARGET='第473行';
for(const [lab,rows] of [['CLIP',clipRows],['DINOv2',dinoRows]]){
 console.log('\n=== '+lab+' 产品相似度 top10 ===');
 for(const x of rows.slice(0,10))console.log(x.product.toFixed(4),'top',x.top.toFixed(3),'avg',x.avg.toFixed(3),'strong',x.strong+'/'+x.n,x.g);
 const i=rows.findIndex(x=>x.g.startsWith(TARGET));
 console.log(lab+' 第473行排名',i+1,'/',rows.length, i>=0?JSON.stringify(rows[i]):'NOT FOUND');
}
const dinoMap=new Map(dinoRows.map((x,i)=>[x.g,{r:i+1,p:x.product}]));
console.log('\n=== CLIP top10 在 DINOv2 中的排名 ===');
for(const x of clipRows.slice(0,10))console.log(x.g,'DINO rank',dinoMap.get(x.g)?.r,'product',dinoMap.get(x.g)?.p?.toFixed(4));
const clipSet=new Set(clipRows.slice(0,20).map(x=>x.g));
console.log('DINO top20 命中 CLIP top20 数',dinoRows.slice(0,20).filter(x=>clipSet.has(x.g)).length);
