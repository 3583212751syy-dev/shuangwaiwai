// 本地端到端验收：站点 API 是否带双模型向量、融合搜索是否把 1.webp 首位判为第473行。
import {readFileSync} from 'node:fs';
const W='D:/GPT/2026-11-21/new-chat-2/work';
const base=process.argv[2]||'http://127.0.0.1:4322';
const {searchLibrary}=await import('./../site-source/search.mjs');
const tf='file:///'+W+'/model-runtime/node_modules/.pnpm/@huggingface+transformers@3.8.1/node_modules/@huggingface/transformers/dist/transformers.node.mjs';

const health=await fetch(base+'/?sharecode=MTCP4KDYCY');
const images=await (await fetch(base+'/api/images')).json();
console.log('HTTP',health.status,'images',images.length,
 'clipReady',images.filter(x=>x.ai?.embedding?.length===512).length,
 'dinoReady',images.filter(x=>x.ai?.dinoEmbedding?.length===384).length);
const dinoCfg=await fetch(base+'/models/onnx-community/dinov2-small/config.json');
console.log('DINO model config',dinoCfg.status);

const {pipeline,env:env1}=await import(tf);
env1.allowLocalModels=true;env1.localModelPath=W+'/deploy-product-library/model-cache';env1.useBrowserCache=false;
const clip=await pipeline('image-feature-extraction','Xenova/clip-vit-base-patch32',{device:'cpu',dtype:'q8'});
const co=await clip('C:/Users/Administrator/Desktop/1.webp',{pooling:'mean'});
const norm=v=>{const n=Math.hypot(...v)||1;return v.map(x=>x/n)};
const qClip=norm(Array.from(co.data||co[0]?.data||co));

const {AutoModel,AutoProcessor,env:env2,RawImage}=await import(tf);
env2.allowLocalModels=true;env2.localModelPath=W+'/deploy-product-library/model-cache';env2.useBrowserCache=false;
const processor=await AutoProcessor.from_pretrained('onnx-community/dinov2-small');
const model=await AutoModel.from_pretrained('onnx-community/dinov2-small',{dtype:'q8',device:'cpu'});
import {execFileSync} from 'node:child_process';
const tmp=W+'/_verify_raw';
execFileSync('python',['-c',`from PIL import Image;import sys;im=Image.open(sys.argv[1]).convert('RGB');open('${tmp}.bin','wb').write(im.tobytes());open('${tmp}.txt','w').write(f'{im.width} {im.height}')`,'C:/Users/Administrator/Desktop/1.webp']);
const [w,h]=readFileSync(tmp+'.txt','utf8').trim().split(' ').map(Number);
const inputs=await processor(new RawImage(new Uint8Array(readFileSync(tmp+'.bin')),w,h,3));
const mo=await model(inputs),mv=Array.from(mo.last_hidden_state.data),dim=384,n=mv.length/dim,sum=new Array(dim).fill(0);
for(let t=0;t<n;t++)for(let d=0;d<dim;d++)sum[d]+=mv[t*dim+d];
const qDino=norm(sum.map(x=>x/n));

const a={embedding:qClip,dinoEmbedding:qDino};
const fused=searchLibrary(a,images,{topK:8});
const clipOnly=searchLibrary({embedding:qClip},images,{topK:8});
console.log('\n融合检索 stage',fused.stage,'fused',fused.fused,'confidence',fused.confidence);
for(const x of fused.results.slice(0,8))console.log(x.productScore.toFixed(4),'clip',x.clipProduct.toFixed(4),'dino',x.dinoProduct?.toFixed(4),x.label.name,x.label.group);
console.log('\nCLIP单模型首位',clipOnly.results[0]?.label.group);
const ok=fused.results[0]?.label.group.includes('第473行');
console.log('\n验收',ok?'PASS':'FAIL','首位必须为第473行彩色切割片');
process.exit(ok?0:1);
