// 真机验证：用本机 transformers.js + 本地 CLIP 模型对一张真实底图推理，比对已有向量。
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createRecognizer,cosine} from './ai.mjs';

// 路径按当前 2430 张图库的实际位置解析（旧 182 张阶段的 model-runtime/analyses.json 已不存在）。
const work=path.resolve('..','..','..','..');
const modulePath=path.join(work,'model-runtime','node_modules',".pnpm",'@huggingface+transformers@3.8.1','node_modules','@huggingface','transformers','dist','transformers.node.mjs');
const modelBase=path.join(work,'model-runtime','model-cache')+path.sep;
const analyses=JSON.parse(await readFile(path.join(work,'deploy-product-library','analyses.json'),'utf8'));
const filename=Object.keys(analyses)[0];
const imagePath=path.join(work,'full-extraction','originals',filename);

const lib=await import(pathToFileURL(modulePath).href);
const recognizer=await createRecognizer(lib,()=>{},'cpu',modelBase);
const a=await recognizer.analyze(imagePath);
assert.equal(a.embedding.length,512);
const expected=analyses[filename].embedding;
const sim=cosine(a.embedding,expected);
console.log(JSON.stringify({model:a.model,filename,zeroShot:a.suggestedName,status:a.status,dim:a.embedding.length,cosineToStoredVector:Number(sim.toFixed(6))},null,2));
assert.ok(sim>.99,'本机模型推理结果与已保存向量不一致');
await recognizer.dispose();
console.log('PASS: 本机 CLIP 模型加载与图片推理正常，向量与线上底图一致');
