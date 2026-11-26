// 为筛选保留的全部原图生成真实 CLIP 512 维特征。断点续跑，逐张落盘。
// 用法：node tools/analyze-all.mjs [limit]
import {readFile,writeFile,readdir} from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const repo=path.resolve('.');
const work=path.resolve('..','..','..');           // new-chat-2/work
const origDir=path.join(work,'full-extraction','originals');
const outFile=path.join(repo,'work','embeddings.json');
const modulePath=path.join(work,'model-runtime','node_modules','.pnpm','@huggingface+transformers@3.8.1','node_modules','@huggingface','transformers','dist','transformers.node.mjs');
const modelBase=path.join(work,'model-runtime','model-cache')+path.sep;
const aiPath=path.join(repo,'site-source','ai.mjs');
const selection=JSON.parse(await readFile(path.join(repo,'work','selection.json'),'utf8'));
const keep=selection.filter(x=>x.decision==='keep');
const limit=process.argv[2]?Number(process.argv[2]):keep.length;

let results={};try{results=JSON.parse(await readFile(outFile,'utf8'))}catch{}
const lib=await import(pathToFileURL(modulePath).href);
const {createRecognizer}=await import(pathToFileURL(aiPath).href);
const engine=await createRecognizer(lib,()=>{},'cpu',modelBase);
console.log('model ready; keep='+keep.length+' done='+Object.keys(results).length);

let n=0,failed=[];
for(const item of keep){
 if(results[item.file])continue;
 if(n>=limit)break;
 try{
  const a=await engine.analyze(path.join(origDir,path.basename(item.file)));
  if(!a.embedding||a.embedding.length!==512)throw Error('bad embedding dim');
  results[item.file]={embedding:a.embedding,suggestedName:a.suggestedName,suggestedCategory:a.suggestedCategory,score:a.score,status:a.status,model:a.model};
  n++;
  if(n%25===0){await writeFile(outFile,JSON.stringify(results));console.log('analyzed '+n+' total '+Object.keys(results).length+'/'+keep.length)}
 }catch(e){failed.push({file:item.file,error:String(e).slice(0,200)})}
}
await writeFile(outFile,JSON.stringify(results));
await writeFile(path.join(repo,'work','embedding-failures.json'),JSON.stringify(failed,null,2));
await engine.dispose();
console.log(JSON.stringify({analyzedThisRun:n,total:Object.keys(results).length,expected:keep.length,failed:failed.length}));
