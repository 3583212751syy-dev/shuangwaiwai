import fs from 'node:fs/promises';
import path from 'node:path';
import * as lib from '@huggingface/transformers';
import {createRecognizer} from '../image-compare/ai.mjs';
const folder='C:\\Users\\Administrator\\Desktop\\包装-产品图',output=path.resolve('work/model-runtime/analyses.json');
lib.env.cacheDir=path.resolve('work/model-runtime/model-cache');
let results={};try{results=JSON.parse(await fs.readFile(output,'utf8'))}catch{}
const engine=await createRecognizer(lib,p=>{if(p.status==='done')console.log('Model ready: '+p.file)});
const files=(await fs.readdir(folder)).filter(x=>x.endsWith('.jpg'));let done=0;
for(const f of files){if(!results[f]){results[f]=await engine.analyze(path.join(folder,f));await fs.writeFile(output,JSON.stringify(results));}done++;if(done%10===0)console.log('Analyzed '+done+' / '+files.length)}
await engine.dispose();console.log(JSON.stringify({analyzed:Object.keys(results).length,confident:Object.values(results).filter(x=>x.confident).length,pending:Object.values(results).filter(x=>!x.confident).length}));
