import {mkdir,copyFile,readFile,writeFile} from 'node:fs/promises';
await mkdir('dist/server',{recursive:true});
const html=await readFile('index.html','utf8');
const server=await readFile('worker/index.js','utf8');
const ai=await readFile('ai.mjs','utf8');
const client=await readFile('ai-client.mjs','utf8');
const search=await readFile('search.mjs','utf8');
const catalog=await readFile('catalog.mjs','utf8');
const specs=await readFile('specs.mjs','utf8');
const ocrWeb=await readFile('ocr-web.js','utf8');
const specsJson=await readFile('specs.json','utf8');
const referencesJson=await readFile('references.json','utf8');
const overrides=await readFile('overrides.json','utf8');
const built=server
 .replace('/*HTML*/',JSON.stringify(html))
 .replace('/*AI_CORE*/',JSON.stringify(ai))
 .replace('/*AI_CLIENT*/',JSON.stringify(client))
 .replace('/*CATALOG*/',JSON.stringify(catalog))
 .replace('/*SEARCH*/',JSON.stringify(search))
 .replace('/*SPECS*/',JSON.stringify(specs))
 .replace('/*OCR_WEB*/',JSON.stringify(ocrWeb))
 .replace('/*SPECS_JSON*/',JSON.stringify(specsJson))
 .replace('/*REFERENCES_JSON*/',JSON.stringify(referencesJson))
 .replace('/*OVERRIDES*/',JSON.stringify(overrides));
if(['/*HTML*/','/*AI_CORE*/','/*AI_CLIENT*/','/*CATALOG*/','/*SEARCH*/','/*SPECS*/','/*OCR_WEB*/','/*SPECS_JSON*/','/*REFERENCES_JSON*/','/*OVERRIDES*/'].some(t=>built.includes(t)))throw Error('构建失败：模板占位符未全部替换');
await writeFile('dist/server/index.js',built);
await mkdir('dist/.openai',{recursive:true});
await copyFile('.openai/hosting.json','dist/.openai/hosting.json');
console.log('BUILD OK: dist/server/index.js ('+built.length+' bytes)');
