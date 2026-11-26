const realFetch=globalThis.fetch;
globalThis.fetch=async(u,...a)=>{console.log('FETCH',String(u).slice(0,150));return realFetch(u,...a)};
const W='D:/GPT/2026-11-21/new-chat-2/work';
const tf='file:///'+W+'/model-runtime/node_modules/.pnpm/@huggingface+transformers@3.8.1/node_modules/@huggingface/transformers/dist/transformers.node.mjs';
const {AutoProcessor,env}=await import(tf);
env.allowLocalModels=true;env.localModelPath=W+'/model-alt';env.useBrowserCache=false;
try{await AutoProcessor.from_pretrained('onnx-community/dinov2-small')}catch(e){console.log('ERR',e.message.slice(0,100))}
