/* 修复 PR#1 引入的中文乱码：按代码结构模板对照原版(HEAD~1)恢复正确中文 */
import {execSync} from 'node:child_process';
import {readFileSync,writeFileSync} from 'node:fs';

const files=['site-source/ai-client.mjs','site-source/search.mjs','site-source/specs.mjs','site-source/worker/index.js'];
const orig=f=>execSync(`git show HEAD~1:${f}`,{maxBuffer:1e8}).toString('utf8').split('\n');
const cur=f=>readFileSync(f,'utf8').split('\n');
const bad=s=>/\uFFFD/.test(s);
const skeleton=s=>s.replace(/[^\x00-\x7F]/g,'#');           // 非ASCII全替换为占位
const skLoose=s=>s.replace(/[^\x00-\x7F]/g,'#').replace(/["'`]/g,'"').replace(/\s+/g,' ').trim();

for(const f of files){
  const O=orig(f),C=cur(f),out=[];
  let fixed=0,unfixed=0;
  // 原版骨架索引（含中文模板的行）
  for(let i=0;i<C.length;i++){
    const line=C[i];
    if(!bad(line)){out.push(line);continue}
    const sk=skLoose(line);
    const cands=O.map((o,j)=>[j,skLoose(o)]).filter(([,s])=>s===sk);
    if(cands.length===1){
      const o=O[cands[0][0]];
      // 校验：除中文外其余字符完全一致
      if(skeleton(o).replace(/["'`]/g,'"')===skeleton(line).replace(/["'`]/g,'"')||sk===skLoose(o)){
        out.push(o);fixed++;continue;
      }
    }
    if(cands.length>1){ // 多候选：取上下文最近者
      let best=null,bd=1e9;
      for(const[j]of cands){const d=Math.abs(j-i);if(d<bd){bd=d;best=j}}
      if(best!==null&&sk===skLoose(O[best])){out.push(O[best]);fixed++;continue}
    }
    out.push(line);unfixed++;
    console.log(`  未自动恢复 ${f}:${i+1}`);
  }
  if(fixed||unfixed){writeFileSync(f,out.join('\n'));console.log(`${f}: 恢复 ${fixed} 行，剩 ${unfixed} 行需人工`)}
}
