// 合并离线 OCR + 逐张目检规格 -> site-source/specs.json（随站点发布）
import {readFileSync,writeFileSync,existsSync} from 'node:fs';
const repo='D:/GPT/2026-11-21/new-chat-2/work/git-handoff/projects/product-image-compare';
const ocrPath=repo+'/reports/specs-ocr.json';
const manual=JSON.parse(readFileSync(repo+'/work/specs-manual.json','utf8')).specs;
const ocr=existsSync(ocrPath)?JSON.parse(readFileSync(ocrPath,'utf8')).specs:{};
const specs={};
for(const [f,v] of Object.entries(ocr)){
 if(v.error)continue;
 specs[f]={curves:v.curves||[],models:v.models||[],counts:v.counts||{},poles:v.poles||[],text:v.text||'',source:'PaddleOCR v6 tiny 离线读图'};
}
let manualCount=0;
for(const [f,v] of Object.entries(manual)){
 if(!v||v._meta)continue;
 const cnt={};for(const c of v.curves||[])cnt[c]=v.count||1;
 specs[f]={curves:v.curves||[],models:v.models||[],counts:cnt,poles:v.poles||[],text:v.evidence||'',name:v.name||'',source:'逐张目检录入'};
 manualCount++;
}
const out={_meta:{engine:'PP-OCRv6 tiny（ppu-paddle-ocr / onnxruntime）',method:'离线批量读图 + 逐张目检纠错；人工录入优先',updatedAt:new Date().toISOString(),total:Object.keys(specs).length,manual:manualCount},specs};
writeFileSync(repo+'/site-source/specs.json',JSON.stringify(out));
console.log('规格库',out._meta.total,'张，其中目检录入',manualCount,'张；文件大小',(JSON.stringify(out).length/1048576).toFixed(2)+'MB');
