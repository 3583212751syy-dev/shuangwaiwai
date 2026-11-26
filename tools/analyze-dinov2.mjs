// 全量 DINOv2-small 特征（2430张），读预处理好的 raw 二进制，速度快。断点续跑。
import {readFile,writeFile} from 'node:fs/promises';
import {readFileSync} from 'node:fs';
const W='D:/GPT/2026-11-21/new-chat-2/work';
const R='D:/GPT/2026-11-21/new-chat-2/work/git-handoff/projects/product-image-compare';
const OUT=R+'/work/embeddings-dinov2.json';
const tf='file:///'+W+'/model-runtime/node_modules/.pnpm/@huggingface+transformers@3.8.1/node_modules/@huggingface/transformers/dist/transformers.node.mjs';
const {AutoModel,AutoProcessor,env,RawImage}=await import(tf);
env.allowLocalModels=true;env.localModelPath=W+'/model-alt';env.useBrowserCache=false;
const processor=await AutoProcessor.from_pretrained('onnx-community/dinov2-small');
const model=await AutoModel.from_pretrained('onnx-community/dinov2-small',{dtype:'q8',device:'cpu'});
console.log('DINOv2 loaded');

const idx=JSON.parse(await readFile(W+'/_all_raw_idx.json','utf8'));
const rawBuf=readFileSync(W+'/_all_raw.bin');
const SIZE=256;
let done={};try{done=JSON.parse(await readFile(OUT,'utf8'))}catch{}
const files=Object.keys(idx).filter(f=>!done[f]&&idx[f].len);
console.log('todo',files.length,'existing',Object.keys(done).length);
function pool(vec){
 const dim=384,n=vec.length/dim,out=new Array(dim).fill(0);
 for(let t=0;t<n;t++)for(let d=0;d<dim;d++)out[d]+=vec[t*dim+d];
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
  done[f]=pool(Array.from(out.last_hidden_state.data));
  i++;
  if(i%100===0){await writeFile(OUT,JSON.stringify(done));console.log('progress',i,'/',files.length)}
 }catch(e){console.error('ERR',f,e.message.slice(0,80))}
}
await writeFile(OUT,JSON.stringify(done));
console.log('DONE',Object.keys(done).length,'dim',Object.values(done)[0]?.length);
