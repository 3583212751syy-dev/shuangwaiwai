// 用任意本地图片跑一次真实检索：CLIP + DINOv2 双模型融合，按产品分组输出候选。
import {readFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
const W='D:/GPT/2026-11-21/new-chat-2/work';
const tf='file:///'+W+'/model-runtime/node_modules/.pnpm/@huggingface+transformers@3.8.1/node_modules/@huggingface/transformers/dist/transformers.node.mjs';
const {searchLibrary}=await import('./../site-source/search.mjs');
const norm=v=>{const n=Math.hypot(...v)||1;return v.map(x=>x/n)};
const imgs=process.argv.slice(2);
if(!imgs.length){console.error('用法: node tools/query-image.mjs <图片...>');process.exit(1)}

const {pipeline,env:env1}=await import(tf);
env1.allowLocalModels=true;env1.localModelPath=W+'/deploy-product-library/model-cache';env1.useBrowserCache=false;
const clip=await pipeline('image-feature-extraction','Xenova/clip-vit-base-patch32',{device:'cpu',dtype:'q8'});
const {AutoModel,AutoProcessor,env:env2,RawImage}=await import(tf);
env2.allowLocalModels=true;env2.localModelPath=W+'/deploy-product-library/model-cache';env2.useBrowserCache=false;
const processor=await AutoProcessor.from_pretrained('onnx-community/dinov2-small');
const model=await AutoModel.from_pretrained('onnx-community/dinov2-small',{dtype:'q8',device:'cpu'});
const tmp=W+'/_query_raw';
async function dinoEmbed(p){
 execFileSync('python',['-c',`from PIL import Image;import sys;im=Image.open(sys.argv[1]).convert('RGB');im.thumbnail((518,518));open('${tmp}.bin','wb').write(im.tobytes());open('${tmp}.txt','w').write(f'{im.width} {im.height}')`,p]);
 const [w,h]=readFileSync(tmp+'.txt','utf8').trim().split(' ').map(Number);
 const inputs=await processor(new RawImage(new Uint8Array(readFileSync(tmp+'.bin')),w,h,3));
 const mo=await model(inputs),mv=Array.from(mo.last_hidden_state.data),dim=384,n=mv.length/dim,sum=new Array(dim).fill(0);
 for(let t=0;t<n;t++)for(let d=0;d<dim;d++)sum[d]+=mv[t*dim+d];
 return norm(sum.map(x=>x/n));
}
const lib=await (await fetch('http://127.0.0.1:4322/api/images')).json();
for(const p of imgs){
 const co=await clip(p,{pooling:'mean'});
 const qClip=norm(Array.from(co.data||co[0]?.data||co));
 const qDino=await dinoEmbed(p);
 const r=searchLibrary({embedding:qClip,dinoEmbedding:qDino},lib,{topK:6});
 console.log('\n=== '+p+' ===');
 console.log('fused',r.fused,'confidence',r.confidence);
 for(const x of r.results)console.log('  ',x.productScore.toFixed(3),'clip',x.clipProduct.toFixed(3),'dino',x.dinoProduct?.toFixed(3),'视觉',x.visual.toFixed(3),x.label.name,'|',x.label.group);
}
