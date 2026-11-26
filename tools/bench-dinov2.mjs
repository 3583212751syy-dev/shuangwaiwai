// 对照试验：DINOv2-small 视觉特征能否把 1.webp 与第473行彩色切割片判为同款。
// 快速版：只算 1.webp + 关键组代表图，比较 CLIP vs DINOv2 的相似度区分度。
import {readFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
const W='D:/GPT/2026-11-21/new-chat-2/work';
const tf='file:///'+W+'/model-runtime/node_modules/.pnpm/@huggingface+transformers@3.8.1/node_modules/@huggingface/transformers/dist/transformers.node.mjs';
const {AutoModel,AutoProcessor,env,RawImage}=await import(tf);
env.allowLocalModels=true;env.localModelPath=W+'/model-alt';env.useBrowserCache=false;
const processor=await AutoProcessor.from_pretrained('onnx-community/dinov2-small');
const model=await AutoModel.from_pretrained('onnx-community/dinov2-small',{dtype:'q8',device:'cpu'});
console.log('DINOv2 loaded');

// 用 PIL 把图片转 raw RGB（python 子进程批量）
function toRaw(p){
 const base='D:/GPT/2026-11-21/new-chat-2/work/_tmp_raw';
 execFileSync('python',['-c',
  `from PIL import Image;import sys;im=Image.open(sys.argv[1]).convert('RGB');open('${base}.bin','wb').write(im.tobytes());open('${base}.txt','w').write(f'{im.width} {im.height}')`,p]);
 const [w,h]=readFileSync(base+'.txt','utf8').trim().split(' ').map(Number);
 return new RawImage(new Uint8Array(readFileSync(base+'.bin')),w,h,3);
}
function pool(vec){
 // last_hidden_state = [CLS, patch...] × 384；取 CLS + 其余 mean 拼接常见；这里用全部 token mean
 const dim=384,n=vec.length/dim,out=new Array(dim).fill(0);
 for(let t=0;t<n;t++)for(let d=0;d<dim;d++)out[d]+=vec[t*dim+d];
 for(let d=0;d<dim;d++)out[d]/=n;
 const norm=Math.hypot(...out);return out.map(x=>x/norm);
}
async function emb(p){const inputs=await processor(toRaw(p));const out=await model(inputs);return pool(Array.from(out.last_hidden_state.data))}
const cos=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);

const orig=W+'/full-extraction/originals/';
const targets=[
 ['QUERY 1.webp','C:/Users/Administrator/Desktop/1.webp'],
 ['第473行 彩色切割片(同款) image1541',orig+'image1541.jpeg'],
 ['第473行 彩色切割片(同款) image1540',orig+'image1540.jpeg'],
 ['第93行 高压软管(误标切割片)',orig+'image341.jpeg'],
 ['第458行 金刚石打磨盘',orig+'image?.jpeg'],
 ['第433行 轮胎内胎',orig+'image?.jpeg'],
].filter(x=>!x[1].includes('?'));
const vecs=[];
for(const [lab,p] of targets){const v=await emb(p);vecs.push({lab,v});console.log('embedded',lab)}
const q=vecs[0].v;
console.log('\n=== DINOv2 相似度(查询=1.webp) ===');
for(const x of vecs.slice(1))console.log(cos(q,x.v).toFixed(4),x.lab);
