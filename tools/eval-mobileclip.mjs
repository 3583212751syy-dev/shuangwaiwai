// DINOv3 vs DINOv2 vs CLIP 留一法对比 + 融合方案对比。
// 口径与线上/历史报告一致：组聚合 top*0.6 + strong(>=0.65)ratio*0.25 + avg*0.15；排除查询图本身。
// 融合部分用 RRF（与 PR#1 修复后的线上 search.mjs 一致）：每路独立按组排名，倒数排名求和。
import {readFileSync} from 'node:fs';
const W='D:/GPT/2026-11-21/new-chat-2/work';
const R=W+'/git-handoff/projects/product-image-compare';
const meta=JSON.parse(readFileSync(W+'/deploy-product-library/metadata.json','utf8'));
const clip=JSON.parse(readFileSync(R+'/work/embeddings.json','utf8'));
let mcRaw={};try{mcRaw=JSON.parse(readFileSync(R+'/work/embeddings-mobileclip-fp32.json','utf8'))}catch{mcRaw=JSON.parse(readFileSync(R+'/work/embeddings-mobileclip.json','utf8'))}const mc={};for(const[f,v]of Object.entries(mcRaw))mc[f]=v.v;
const dino3raw=JSON.parse(readFileSync(R+'/work/embeddings-dinov3.json','utf8'));
const dino3m={},dino3c={};
for(const[f,v]of Object.entries(dino3raw)){dino3m[f]=v.m;dino3c[f]=v.c}
const cos=(a,b)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s};
const norm=v=>{const n=Math.hypot(...v)||1;return v.map(x=>x/n)};
const get=(lib,f)=>{const e=lib['originals/'+f]||lib[f];const v=e&&(e.embedding||Array.isArray(e)&&e);return v?norm(v):null};

const byGroup=new Map();
for(const x of meta){if(!byGroup.has(x.group))byGroup.set(x.group,[]);byGroup.get(x.group).push(x.filename)}
const eligible=[...byGroup.entries()].filter(([,fs])=>fs.length>=2&&fs.length<=12);
const queries=[];
for(const[g,fs]of eligible)for(const f of fs.slice(0,2))queries.push({g,f});

function groupScores(lib,qf,qq,items){
 const groups=new Map();
 for(const it of items){
  if(it.f===qf)continue;
  const v=get(lib,it.f);if(!v)continue;
  const s=cos(qq,it.v);
  if(!groups.has(it.g))groups.set(it.g,[]);
  groups.get(it.g).push(s);
 }
 return[...groups].map(([g,ss])=>{
  const top=Math.max(...ss),avg=ss.reduce((a,b)=>a+b,0)/ss.length,strong=ss.filter(s=>s>=0.65).length;
  return{g,p:top*0.6+(strong/ss.length)*0.25+avg*0.15};
 }).sort((a,b)=>b.p-a.p);
}
function rrfFuse(ranksArr,k=60){
 const sc=new Map();
 for(const ranks of ranksArr)ranks.forEach((r,i)=>{sc.set(r.g,(sc.get(r.g)||0)+1/(k+i+1))});
 return[...sc].sort((a,b)=>b[1]-a[1]).map(([g,s])=>({g,p:s}));
}
function run(name,fns,limit){
 // fns: [{name, lib, mode:'score'|'rank'}]
 let r1=0,r5=0,r20=0,total=0;const misses=[];
 for(const q of queries.slice(0,limit)){
  const per=[];
  for(const m of fns){
   const qq=get(m.lib,q.f);if(!qq)continue;
   const items=[];
   for(const x of meta){const v=get(m.lib,x.filename);if(v)items.push({f:x.filename,g:x.group,v})}
   const rows=groupScores(m.lib,q.f,qq,items);
   per.push(m.mode==='rank'?rows:rows);
  }
  if(!per.length)continue;
  const fused=fns.length===1?per[0]:rrfFuse(per.map(rows=>rows));
  const rank=fused.findIndex(x=>x.g===q.g)+1;
  total++;if(rank===1)r1++;if(rank>0&&rank<=5)r5++;if(rank>0&&rank<=20)r20++;
  if(rank>5&&misses.length<8)misses.push({q:q.f.slice(0,28),rank,top:fused[0].g});
 }
 console.log(`\n=== ${name} ===`);
 console.log(`queries=${total} R@1=${(r1/total).toFixed(4)} R@5=${(r5/total).toFixed(4)} R@20=${(r20/total).toFixed(4)}`);
 if(misses.length){console.log('miss样例:');for(const x of misses)console.log(' rank'+x.rank,x.q,'->',x.top)}
 return{r1:r1/total,r5:r5/total,r20:r20/total};
}
const limit=Number(process.argv[2]||1486);
const N=(lib,mode='rank')=>({lib,mode});
run('MobileCLIP-S0 alone',[N(mc)],limit);
run('CLIP+MobileCLIP RRF',[N(clip),N(mc)],limit);
run('DINOv3cls+MobileCLIP RRF',[N(dino3c),N(mc)],limit);
run('CLIP alone',[N(clip)],limit);

run('DINOv3-mean alone',[N(dino3m)],limit);
run('DINOv3-cls alone',[N(dino3c)],limit);
run('CLIP+DINOv3cls RRF (现线上)',[N(clip),N(dino3c)],limit);
run('DINOv3-cls alone',[N(dino3c)],limit);
run('CLIP+DINOv3mean (RRF)',[N(clip),N(dino3m)],limit);
run('CLIP+DINOv3cls (RRF)',[N(clip),N(dino3c)],limit);
run('CLIP+DINOv3mean+DINOv3cls',[N(clip),N(dino3m),N(dino3c)],limit);
