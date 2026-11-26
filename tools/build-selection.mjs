// 定稿筛选 + 同款分组 + 数量守恒。
// 输入：full-extraction 的清单 + work/decisions.json（人工目检结论）。
// 输出：work/selection.json（每张图的决策）、work/groups.json（同款多角度分组）、work/reconcile.json（数量守恒）。
import {readFile,writeFile} from 'node:fs/promises';
import path from 'node:path';

const work=path.resolve('..','..','..'); // -> work/ (new-chat-2/work)
const repo=path.resolve('.');            // git-handoff/projects/product-image-compare
const src=path.join(work,'full-extraction');
const decisions=JSON.parse(await readFile(path.join(repo,'work','decisions.json'),'utf8'));
const inv=JSON.parse(await readFile(path.join(src,'inventory.json'),'utf8'));
const occ=JSON.parse(await readFile(path.join(src,'occurrences.json'),'utf8'));

const excluded=new Map(decisions.excluded.map(x=>[String(x.no),x]));
const byFile=new Map();
for(const o of occ){if(!byFile.has(o.file))byFile.set(o.file,[]);byFile.get(o.file).push(o)}

const selected=[];const excludedList=[];let failed=0;
for(const item of inv){
 const no=String(Number(PathStem(item.file)));
 const reason=excluded.get(no);
 const srcCells=byFile.get(item.file)||[];
 const rec={
  file:item.file,sha256:item.sha256,no:Number(no),
  decision:reason?'exclude':'keep',
  reason:reason?reason.reason:'',
  sheet:srcCells[0]?.sheet||null,row:srcCells[0]?.row||null,cells:srcCells.map(c=>c.cell),
  occurrences:item.occurrences,width:item.width,height:item.height
 };
 if(reason)excludedList.push(rec);else selected.push(rec);
}
function PathStem(f){return f.split('/').pop().replace(/\.[^.]+$/,'').replace(/^image/,'')}

// 同款多角度：同一 sheet 的同一 row 视为一组；每组内是不同角度/列。
const groups=new Map();
for(const s of selected){
 const key=(s.sheet||'')+'|'+s.row;
 if(!groups.has(key))groups.set(key,{sheet:s.sheet,row:s.row,items:[]});
 groups.get(key).items.push({file:s.file,no:s.no,cells:s.cells,sha256:s.sha256});
}
const groupList=[...groups.values()].sort((a,b)=>(a.row||0)-(b.row||0));

const reconcile={
 total_originals:inv.length,
 excluded_pure_packaging:excludedList.length,
 keep_for_library:selected.length,
 selected_groups:groupList.length,
 failed:failed,
 equation:{excluded:excludedList.length,keep:selected.length,failed,sum:excludedList.length+selected.length+failed},
 balance:(excludedList.length+selected.length+failed)===inv.length,
 source_scope:'仅已保存工作簿 Sheet1；不能证明覆盖在线原表其它工作表',
 review_status:'保守初筛：仅剔除高置信度纯包装；keep 表示保留原图待确认，类目与同款命名由来源行号分组 + 网页人工确认，CLIP 建议仅供参考'
};
await writeFile(path.join(repo,'work','selection.json'),JSON.stringify([...selected,...excludedList],null,2));
await writeFile(path.join(repo,'work','groups.json'),JSON.stringify(groupList,null,2));
await writeFile(path.join(repo,'work','reconcile.json'),JSON.stringify(reconcile,null,2));
console.log(JSON.stringify(reconcile,null,2));
