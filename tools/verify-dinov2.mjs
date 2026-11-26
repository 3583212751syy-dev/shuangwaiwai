// 对照试验：DINOv2-small ONNX 能否用 transformers.js 跑通，并对 1.webp 提取特征
const W='D:/GPT/2026-11-21/new-chat-2/work';
const tfPath='file:///'+W+'/model-runtime/node_modules/.pnpm/@huggingface+transformers@3.8.1/node_modules/@huggingface/transformers/dist/transformers.node.mjs';
const {pipeline,env}=await import(tfPath);
env.allowLocalModels=true;env.localModelPath=W+'/model-alt';env.useBrowserCache=false;
console.log('loading dinov2...');
const extractor=await pipeline('image-feature-extraction','onnx-community/dinov2-small',{device:'cpu',dtype:'q8'});
console.log('loaded');
const out=await extractor('C:/Users/Administrator/Desktop/1.webp',{pooling:'mean'});
const vec=Array.from(out.data||out[0]?.data||out);
console.log('dim',vec.length,'norm',Math.hypot(...vec).toFixed(4),'sample',vec.slice(0,5).map(x=>x.toFixed(4)));
