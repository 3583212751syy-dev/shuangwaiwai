// 把真实产品目录回填到底图元数据：解决线上只有文件名、没有品名/类目/同款分组的问题。
import {readFile,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {effectiveLabel, isDefaultName} from './catalog.mjs';

const root=path.dirname(fileURLToPath(import.meta.url));
const work=path.resolve(root,'..');
const src=path.join(work,'model-runtime','metadata-before-ai.json');
const anaFile=path.join(work,'model-runtime','analyses.json');
const outMeta=path.join(work,'deploy-image-match','metadata.json');
const outAna=path.join(work,'deploy-image-match','analyses.json');

const meta=JSON.parse(await readFile(src,'utf8'));
const analyses=JSON.parse(await readFile(anaFile,'utf8'));
let named=0,categorized=0,grouped=0;
for(const m of meta){
 const label=effectiveLabel(m);
 if(label.source==='产品目录'||!isDefaultName(m.name))named++;
 m.name=label.name;
 if(label.category!=='待分类'){m.category=label.category;categorized++}
 m.group=label.group;grouped++;
 m.view=label.view;m.labelSource=label.source;
 const a=analyses[m.filename];
 if(a){
  if(['仅有包装','无法辨认的产品'].includes(a.suggestedName)||isDefaultName(a.suggestedName)){
   a.suggestedName=label.name;a.suggestedCategory=label.category;
   if(!a.status)a.status='已识别';
  }
 }
}
await writeFile(outMeta,JSON.stringify(meta,null,2));
await writeFile(outAna,JSON.stringify(analyses,null,2));
console.log('backfill',meta.length,'named',named,'categorized',categorized,'grouped',grouped);
