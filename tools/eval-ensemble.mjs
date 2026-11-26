// 融合对照：CLIP / DINOv2 / 两者融合 在同一批留一查询上的召回率。
import {readFileSync} from 'node:fs';
const W='D:/GPT/2026-11-21/new-chat-2/work';
const meta=JSON.parse(readFileSync(W+'/deploy-product-library/metadata.json','utf8'));
const libs={clip:JSON.parse(readFileSync('work/embeddings.json','utf8')),dino:JSON.parse(readFileSync('work/embeddings-dinov2.json','utf8'))};
const vecOf=(m,f)=>m==='clip'?(libs.clip['originals/'+f]||libs.clip[f])?.embedding:(libs.dino[f]||libs.dino['originals/'+f]);
const cos=(a,b)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s};
const items={};
for(const m of ['clip','dino']){items[m]=[];for(const x of meta){const v=vecOf(m,x.filename);if(!v)continue;const n=Math.hypot(...v)||1;items[m].push({f:x.filename,g:x.group,v:v.map(t=>t/n)})}}
const byGroup=new Map();for(const x of meta)(byGroup.get(x.group)||byGroup.set(x.group,[]).get(x.group)).push(x.filename);
const queries=[];for(const [g,fs] of byGroup)if(fs.length>=2&&fs.length<=12)for(const f of fs.slice(0,2))queries.push({g,f});
const limit=Number(process.argv[2]||queries.length);

function rowsFor(m,q){
 const v=vecOf(m,q.f);if(!v)return null;
 const n=Math.hypot(...v)||1,qq=v.map(t=>t/n),groups=new Map();
 for(const it of items[m]){if(it.f===q.f)continue;const s=cos(qq,it.v);const a=groups.get(it.g);if(a)a.push(s);else groups.set(it.g,[s])}
 return [...groups].map(([g,ss])=>{const top=Math.max(...ss),avg=ss.reduce((a,b)=>a+b,0)/ss.length,strong=ss.filter(s=>s>=0.65).length;
  return {g,p:top*0.6+(strong/ss.length)*0.25+avg*0.15}});
}
function zscore(rows){const mean=rows.reduce((a,b)=>a+b.p,0)/rows.length;const sd=Math.sqrt(rows.reduce((a,b)=>a+(b.p-mean)**2,0)/rows.length)||1;
 return new Map(rows.map(r=>[r.g,(r.p-mean)/sd]))}
const stat={clip:{r1:0,r5:0},dino:{r1:0,r5:0},fusion:{r1:0,r5:0}};let total=0;
for(const q of queries.slice(0,limit)){
 const rc=rowsFor('clip',q),rd=rowsFor('dino',q);if(!rc||!rd)continue;
 total++;
 for(const [m,rows] of [['clip',rc],['dino',rd]]){const rank=rows.sort((a,b)=>b.p-a.p).findIndex(x=>x.g===q.g)+1;if(rank===1)stat[m].r1++;if(rank<=5)stat[m].r5++}
 const zc=zscore(rc),zd=zscore(rd),gs=new Set([...zc.keys(),...zd.keys()]);
 const fused=[...gs].map(g=>({g,p:0.5*(zc.get(g)||-3)+0.5*(zd.get(g)||-3)})).sort((a,b)=>b.p-a.p);
 const rank=fused.findIndex(x=>x.g===q.g)+1;if(rank===1)stat.fusion.r1++;if(rank<=5)stat.fusion.r5++;
}
console.log('total',total);
for(const m of ['clip','dino','fusion'])console.log(m,'recall@1',(stat[m].r1/total).toFixed(4),'recall@5',(stat[m].r5/total).toFixed(4));
