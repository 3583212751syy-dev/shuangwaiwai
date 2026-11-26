// 把新图库（筛选保留 + 来源分组 + CLIP 特征）打包成自包含站点目录。
// 输出目录沿用已验证的 local-runner/server.mjs 契约：images/、model-cache/、metadata.json、analyses.json。
import {readFile,writeFile,mkdir,copyFile,readdir,rm} from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';

const repo=path.resolve('.');
const work=path.resolve('..','..','..');                 // new-chat-2/work
const origDir=path.join(work,'full-extraction','originals');
const out=path.join(work,'deploy-product-library');      // 新站点目录
const modelSrc=path.join(work,'model-runtime','model-cache');
const siteSrc=path.join(repo,'site-source');

const selection=JSON.parse(await readFile(path.join(repo,'work','selection.json'),'utf8'));
const groups=JSON.parse(await readFile(path.join(repo,'work','groups.json'),'utf8'));
const emb=JSON.parse(await readFile(path.join(repo,'work','embeddings.json'),'utf8'));
const dinoRaw=JSON.parse(await readFile(path.join(repo,'work','embeddings-dinov3.json'),'utf8'));
const dino={};for(const[f,v]of Object.entries(dinoRaw))dino[f]=v.c; // DINOv3-CLS
const manual=JSON.parse(await readFile(path.join(repo,'work','labels-manual.json'),'utf8')).labels||{};
const keep=selection.filter(x=>x.decision==='keep');

// 不整目录删除：既有站点可能正在使用，原地覆盖同名文件即可，避免误删已发布资源。
await mkdir(path.join(out,'images'),{recursive:true});
await mkdir(path.join(out,'model-cache'),{recursive:true});

// 组 -> 展示名/类目：来源行号做同款 key；类目与品名取该组 CLIP 建议的最高置信者，标注来源
const groupByFile=new Map();
for(const g of groups)for(const it of g.items)groupByFile.set(it.file,g);
const groupMeta=new Map();
for(let gi=0;gi<groups.length;gi++){
 const g=groups[gi],human=manual[gi];
 let best=null;
 for(const it of g.items){const a=emb[it.file];if(a&&(!best||a.score>best.score))best=a}
 const name=human?.name||(best?.suggestedName&&!['无法辨认的产品','仅有包装'].includes(best.suggestedName)?best.suggestedName:'待确认产品');
 const category=human?.category||(best?.suggestedCategory&&best.suggestedCategory!=='待分类'?best.suggestedCategory:'待分类');
 // 人工已定名的组不再挂「待确认」；名字里带「待复核」的 105 组如实保留。
 const confirmed=!!human&&!/待复核/.test(human.name||'');
 groupMeta.set(g,{name,category,status:confirmed?'已确认':'待确认',
  source:human?'人工逐组目检定名':'CLIP 建议 + 来源行号分组，待人工确认'});
}

const meta=[];let copied=0,withEmb=0,missingEmb=0,withDino=0;
for(const s of keep){
 const g=groupByFile.get(s.file);const gm=g?groupMeta.get(g):null;
 const a=emb[s.file];
 const id=crypto.createHash('sha1').update(s.sha256).digest('hex').slice(0,32); // 稳定 id = 内容哈希，去重复引用
 const filename=path.basename(s.file);
  const m={
  id,name:gm?.name||'待确认产品',category:gm?.category||'待分类',
  group:g?('第'+g.row+'行 · '+(gm?.name||'待确认产品')):'未分组',
  filename,size:0,created:new Date().toISOString(),source:'包装-产品图.xlsx（已保存文件）',
  classification:gm?.source||'CLIP 建议 + 来源行号（待人工确认）',
  sha256:s.sha256,no:s.no,row:s.row,cells:s.cells,view:s.cells[0]||''
 };
 if(a&&Array.isArray(a.embedding)&&a.embedding.length===512){
  m.ai={model:a.model||'Xenova/clip-vit-base-patch32',version:'3.8.1',status:gm?.status||'待确认',
        suggestedName:a.suggestedName,suggestedCategory:a.suggestedCategory,
        score:a.score,confident:false,embedding:a.embedding,method:'CLIP 文本分类（初筛建议）',at:new Date().toISOString()};
  withEmb++;
 }else{missingEmb++}
 meta.push(m);
 try{await copyFile(path.join(origDir,filename),path.join(out,'images',filename));copied++}catch(e){console.error('copy fail',filename,e.message)}
}
await writeFile(path.join(out,'metadata.json'),JSON.stringify(meta));

// analyses.json 供 server.mjs 组装 ai 字段
const analyses={};
for(const m of meta)if(m.ai){
 analyses[m.filename]={...m.ai};
 const dv=dino[m.filename];
 if(Array.isArray(dv)&&dv.length===384){analyses[m.filename].dinoEmbedding=dv;withDino++}
}
await writeFile(path.join(out,'analyses.json'),JSON.stringify(analyses));

// 复制模型与内联 worker、server、package
for(const f of await readdir(modelSrc))await copyFile(path.join(modelSrc,f),path.join(out,'model-cache',f)).catch(()=>{});
// 模型在子目录，递归复制
async function copyDir(src,dst){await mkdir(dst,{recursive:true});for(const e of await readdir(src,{withFileTypes:true})){const s=path.join(src,e.name),d=path.join(dst,e.name);if(e.isDirectory())await copyDir(s,d);else await copyFile(s,d)}}
await copyDir(modelSrc,path.join(out,'model-cache'));
// 第二视觉模型：DINOv3-vits16 q8 21MB（CLS 池化，检索主信号），与 CLIP 一起随站点发布。
await copyDir(path.join(work,'model-alt','onnx-community','dinov3-vits16-pretrain-lvd1689m-ONNX'),path.join(out,'model-cache','onnx-community','dinov3-vits16-pretrain-lvd1689m-ONNX'));
await copyFile(path.join(siteSrc,'build.mjs'),path.join(out,'build.mjs')); // 参考
// 使用 site-source 刚重建的 worker，避免误用旧目录里的过期产物。
const distWorker=path.join(siteSrc,'dist','server','index.js');
await copyFile(distWorker,path.join(out,'app-worker.mjs'));
try{await copyFile(path.join(siteSrc,'overrides.json'),path.join(out,'overrides.json'))}catch(e){console.warn('overrides.json 缺失：线上将退回桶内旧标签')}
await copyFile(path.join(repo,'local-runner','server.mjs'),path.join(out,'server.mjs'));
await copyFile(path.join(repo,'local-runner','package.json'),path.join(out,'package.json'));

const reconcile=JSON.parse(await readFile(path.join(repo,'work','reconcile.json'),'utf8'));
const summary={images:meta.length,copied,withEmbedding:withEmb,missingEmbedding:missingEmb,withDinoEmbedding:withDino,groups:groupMeta.size,manualLabels:Object.keys(manual).length,
 excluded:reconcile.excluded_pure_packaging,keep:reconcile.keep_for_library,balance:reconcile.balance,
 source:'已保存工作簿 Sheet1，2442 张原图'};
await writeFile(path.join(out,'pack-summary.json'),JSON.stringify(summary,null,2));
console.log(JSON.stringify(summary,null,2));
