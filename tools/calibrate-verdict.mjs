// verdict 阈值校准：MobileCLIP 主路 + DINOv3cls 第二路（与线上 search.mjs 组聚合口径一致）。
// 统计留一法里「真值组」与「第一名非真值组」的 visual(mc 组内最大余弦) / dinoVisual 分布，输出建议阈值。
import {readFileSync} from 'node:fs';
const W='D:/GPT/2026-11-21/new-chat-2/work';
const R=W+'/git-handoff/projects/product-image-compare';
const meta=JSON.parse(readFileSync(W+'/deploy-product-library/metadata.json','utf8'));
const mcRaw=JSON.parse(readFileSync(R+'/work/embeddings-mobileclip-fp32.json','utf8'));
const mc={};for(const[f,v]of Object.entries(mcRaw))mc[f]=v.v;
const d3raw=JSON.parse(readFileSync(R+'/work/embeddings-dinov3.json','utf8'));
const clipRaw=JSON.parse(readFileSync(R+'/work/embeddings.json','utf8'));
const clip={};for(const[f,v]of Object.entries(clipRaw)){const key=f.replace(/^originals\//,'');const vec=v&&(v.embedding||(Array.isArray(v)?v:null));if(vec)clip[key]=vec}
const d3c={};for(const[f,v]of Object.entries(d3raw))d3c[f]=v.c;
const cos=(a,b)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s};

const byGroup=new Map();
for(const x of meta){if(!byGroup.has(x.group))byGroup.set(x.group,[]);byGroup.get(x.group).push(x.filename)}
const eligible=[...byGroup.entries()].filter(([,fs])=>fs.length>=2&&fs.length<=12);
const queries=[];
for(const[g,fs]of eligible)for(const f of fs.slice(0,2))queries.push({g,f});
console.log('queries',queries.length);

const gtMc=[],gtDino=[],fpMc=[],fpDino=[],gtCl=[],fpCl=[];
function groupStat(qf,qq,vecMap){
 const groups=new Map();
 for(const x of meta){
  if(x.filename===qf)continue;
  const v=vecMap[x.filename];if(!v)continue;
  const s=cos(qq,v);
  if(!groups.has(x.group))groups.set(x.group,[]);
  groups.get(x.group).push(s);
 }
 const rows=[...groups].map(([g,ss])=>({g,top:Math.max(...ss)})).sort((a,b)=>b.top-a.top);
 return rows;
}
let n=0;
for(const q of queries){
 const qmc=mc[q.f],qd3=d3c[q.f],qcl=clip[q.f];
 if(!qmc||!qd3)continue;
 const mcRows=groupStat(q.f,qmc,mc),d3Rows=groupStat(q.f,qd3,d3c),clRows=groupStat(q.f,qcl,clip);
 const gtMcTop=mcRows.find(r=>r.g===q.g)?.top??null,gtD3Top=d3Rows.find(r=>r.g===q.g)?.top??null;
 const nfMc=mcRows.find(r=>r.g!==q.g),nfD3=d3Rows.find(r=>r.g!==q.g);
 const nfCl=clRows.find(r=>r.g!==q.g);
 if(gtMcTop!=null&&nfCl){gtMc.push(mcRows.find(r=>r.g===q.g)?.top);gtDino.push(gtD3Top);gtCl.push(clRows.find(r=>r.g===q.g)?.top);fpMc.push(nfMc.top);fpDino.push(nfD3.top);fpCl.push(nfCl.top)}
 n++;
}
const pct=(a,p)=>{const s=[...a].sort((x,y)=>x-y);return s[Math.floor(s.length*p)]};
console.log('n=',n);
console.log('真值组 mcTop  P5/P10/P25/P50:',pct(gtMc,.05).toFixed(3),pct(gtMc,.10).toFixed(3),pct(gtMc,.25).toFixed(3),pct(gtMc,.50).toFixed(3));
console.log('非真值 top1 mcTop P50/P75/P90/P95:',pct(fpMc,.50).toFixed(3),pct(fpMc,.75).toFixed(3),pct(fpMc,.90).toFixed(3),pct(fpMc,.95).toFixed(3));
console.log('真值组 dinoTop P5/P10/P25/P50:',pct(gtDino,.05).toFixed(3),pct(gtDino,.10).toFixed(3),pct(gtDino,.25).toFixed(3),pct(gtDino,.50).toFixed(3));
console.log('非真值 top1 dinoTop P50/P75/P90/P95:',pct(fpDino,.50).toFixed(3),pct(fpDino,.75).toFixed(3),pct(fpDino,.90).toFixed(3),pct(fpDino,.95).toFixed(3));
// 阈值网格：tentative = mcTop>=T1 && dinoTop>=T2 → 查"疑似"触发率（真值组应触发）与误触发率（fp top1 不应触发）
console.log('真值组 clipTop P5/P10/P25/P50:',pct(gtCl,.05).toFixed(3),pct(gtCl,.10).toFixed(3),pct(gtCl,.25).toFixed(3),pct(gtCl,.50).toFixed(3));
console.log('非真值 top1 clipTop P50/P75/P90/P95:',pct(fpCl,.50).toFixed(3),pct(fpCl,.75).toFixed(3),pct(fpCl,.90).toFixed(3),pct(fpCl,.95).toFixed(3));
console.log('\nCLIP网格（现verdict阈值附近）');
const C1s=[.80,.84,.88],C2s=[.60,.70,.75];
for(const c1 of C1s){let line='cl>='+c1.toFixed(2)+'   ';for(const c2 of C2s){const hit=gtCl.filter((v,i)=>v>=c1&&gtDino[i]>=c2).length/gtCl.length*100;const fp=fpCl.filter((v,i)=>v>=c1&&fpDino[i]>=c2).length/fpCl.length*100;line+=hit.toFixed(1).padStart(5)+'%/'+fp.toFixed(1).padStart(5)+'% '}console.log(line)}
console.log('\n阈值网格（行=T_mc 列=T_dino，格=真值触发率% / fp误触发率%）');
const T1s=[.55,.60,.65,.70,.75],T2s=[.50,.55,.60,.65,.70];
let header='T_mc\\T_dino '+' '.repeat(3)+T2s.map(t=>'d>='+t.toFixed(2)+'    ').join('');
console.log(header);
for(const t1 of T1s){
 let line='mc>='+t1.toFixed(2)+'   ';
 for(const t2 of T2s){
  const hit=gtMc.filter((v,i)=>v>=t1&&gtDino[i]>=t2).length/gtMc.length*100;
  const fp=fpMc.filter((v,i)=>v>=t1&&fpDino[i]>=t2).length/fpMc.length*100;
  line+=`${hit.toFixed(1).padStart(5)}%/${fp.toFixed(1).padStart(5)}% `;
 }
 console.log(line);
}
