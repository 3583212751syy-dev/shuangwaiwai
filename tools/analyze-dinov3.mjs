// 全量 DINOv3-vits16 特征（2430张），复用 dinov2 同一预处理 raw bin。断点续跑。
// 每张存两种池化：m=全 token 均值（与 dinov2 同口径），c=仅 CLS。评估后择优。
import {readFile,writeFile} from 'node:fs/promises';
import {readFileSync,existsSync} from 'node:fs';
const W='D:/GPT/2026-11-21/new-chat-2/work';
const R='D:/GPT/2026-11-21/new-chat-2/work/git-handoff/projects/product-image-compare';
const OUT=R+'/work/embeddings-dinov3.json';
const TF_CANDS=[
 'file:///'+W+'/model-runtime/node_modules/.pnpm/@huggingface+transformers@3.8.1/node_modules/@huggingface/transformers/dist/transformers.node.mjs',
 'file:///'+W+'/model-runtime/node_modules/@huggingface/transformers/dist/transformers.node.mjs',
];
const tf=TF_CANDS.find(p=>existsSync(new URL(p)));
if(!tf)throw new Error('transformers.node.mjs not found');
const {AutoModel,AutoProcessor,env,RawImage}=await import(tf);
env.allowLocalModels=true;env.localModelPath=W+'/model-alt';env.useBrowserCache=false;
const MID='onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX';
const processor=await AutoProcessor.from_pretrained(MID);
const model=await AutoModel.from_pretrained(MID,{dtype:'q8',device:'cpu'});
console.log('DINOv3 loaded (q8 cpu)');

const idx=JSON.parse(await readFile(W+'/_all_raw_idx.json','utf8'));
const rawBuf=readFileSync(W+'/_all_raw.bin');
const SIZE=256;
let done={};try{done=JSON.parse(await readFile(OUT,'utf8'))}catch{}
const files=Object.keys(idx).filter(f=>!done[f]&&idx[f].len);
console.log('todo',files.length,'existing',Object.keys(done).length);
function pool(vec,mode){
 const dim=384,n=vec.length/dim,start=mode==='c'?0:0,cnt=mode==='c'?1:n;
 const out=new Array(dim).fill(0);
 for(let t=(mode==='c'?0:0);t<(mode==='c'?1:n);t++)for(let d=0;d<dim;d++)out[d]+=vec[t*dim+d];
 const norm=Math.hypot(...out)||1;return out.map(x=>+(x/norm).toFixed(6));
}
let i=0;
for(const f of files){
 try{
  const meta=idx[f];
  const data=new Uint8Array(rawBuf.buffer,rawBuf.byteOffset+meta.off,SIZE*SIZE*3);
  const img=new RawImage(data,SIZE,SIZE,3);
  const inputs=await processor(img);
  const out=await model(inputs);
  const vec=Array.from(out.last_hidden_state.data);
  done[f]={m:pool(vec,'m'),c:pool(vec,'c')};
  i++;
  if(i%200===0){await writeFile(OUT,JSON.stringify(done));console.log('progress',i,'/',files.length)}
 }catch(e){console.error('ERR',f,e.message.slice(0,100))}
}
await writeFile(OUT,JSON.stringify(done));
const s=Object.values(done)[0];
console.log('DONE',Object.keys(done).length,'m-dim',s?.m.length,'c-dim',s?.c.length);
