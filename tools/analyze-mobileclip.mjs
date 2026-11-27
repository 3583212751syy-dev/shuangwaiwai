// 全量 MobileCLIP-S0 视觉特征（2430张），复用同一预处理 raw bin（256px RGB）。断点续跑。
// 输出 embeddings-mobileclip.json：{file: v(512维归一化)}
import {readFile,writeFile} from 'node:fs/promises';
import {readFileSync,existsSync} from 'node:fs';
const W='D:/GPT/2026-11-21/new-chat-2/work';
const R='D:/GPT/2026-11-21/new-chat-2/work/git-handoff/projects/product-image-compare';
const OUT=R+'/work/embeddings-mobileclip-fp32.json';
const TF_CANDS=['file:///'+W+'/_siglip2/node_modules/@huggingface/transformers/dist/transformers.node.mjs',
 'file:///'+W+'/model-runtime/node_modules/.pnpm/@huggingface+transformers@3.8.1/node_modules/@huggingface/transformers/dist/transformers.node.mjs',
 'file:///'+W+'/model-runtime/node_modules/@huggingface/transformers/dist/transformers.node.mjs',
];
const tf=TF_CANDS.find(p=>existsSync(new URL(p)));
if(!tf)throw new Error('transformers.node.mjs not found');
const {CLIPVisionModelWithProjection,AutoProcessor,env,RawImage}=await import(tf);
env.allowLocalModels=true;env.localModelPath=W+'/model-alt';env.useBrowserCache=false;
const MID='Xenova/mobileclip_s0';
const processor=await AutoProcessor.from_pretrained(MID);
const model=await CLIPVisionModelWithProjection.from_pretrained(MID,{dtype:'fp32',device:'cpu'});
console.log('MobileCLIP-S0 vision loaded (q8 cpu)');
console.log('processor config:', JSON.stringify(processor.image_processor?.config||{}).slice(0,300));

const idx=JSON.parse(await readFile(W+'/_all_raw_idx.json','utf8'));
const rawBuf=readFileSync(W+'/_all_raw.bin');
const SIZE=256;
let done={};try{done=JSON.parse(await readFile(OUT,'utf8'))}catch{}
const files=Object.keys(idx).filter(f=>!done[f]&&idx[f].len);
console.log('todo',files.length,'existing',Object.keys(done).length);
let i=0;
for(const f of files){
 try{
  const meta=idx[f];
  const data=new Uint8Array(rawBuf.buffer,rawBuf.byteOffset+meta.off,SIZE*SIZE*3);
  const img=new RawImage(data,SIZE,SIZE,3);
  const inputs=await processor(img);
  const out=await model(inputs);
  const emb=out.image_embeds||out.embeddings;
  if(!emb)throw Error('no image_embeds: '+Object.keys(out));
  const dim=emb.dims?emb.dims.at(-1):Math.sqrt(emb.data.length);
  const vec=Array.from(emb.data.slice(0,dim));
  const norm=Math.hypot(...vec)||1;
  done[f]={v:vec.map(x=>+(x/norm).toFixed(6))};
  i++;
  if(i%300===0){await writeFile(OUT,JSON.stringify(done));console.log('progress',i,'/',files.length,'dim',dim)}
 }catch(e){console.error('ERR',f,e.message.slice(0,120))}
}
await writeFile(OUT,JSON.stringify(done));
console.log('DONE',Object.keys(done).length,'张');
