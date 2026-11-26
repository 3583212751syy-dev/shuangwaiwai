// 留一法召回对比：用底图库自身多角度图片当查询，检验哪个模型更适合"产品相似度优先"判据。
// 规则与线上一致：按 group 聚合 top/avg/strong 得到 productScore；排除查询图本身。
import {readFileSync} from 'node:fs';
const W='D:/GPT/2026-11-21/new-chat-2/work';
const meta=JSON.parse(readFileSync(W+'/deploy-product-library/metadata.json','utf8'));
const clip=JSON.parse(readFileSync('work/embeddings.json','utf8'));
const dino=JSON.parse(readFileSync('work/embeddings-dinov2.json','utf8'));
const vecOf=(lib,f,model)=>{
 if(model==='clip'){const x=lib['originals/'+f]||lib[f];return x?.embedding}
 return lib[f]||lib['originals/'+f];
};
const cos=(a,b)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s};

const byGroup=new Map();
for(const x of meta){(byGroup.get(x.group)||byGroup.set(x.group,[]).get(x.group)).push(x.filename)}
const eligible=[...byGroup.entries()].filter(([,fs])=>fs.length>=2 && fs.length<=12);
// 每组最多取 2 张查询，覆盖多角度
const queries=[];
for(const [g,fs] of eligible)for(const f of fs.slice(0,2))queries.push({g,f});
console.log('groups',byGroup.size,'eligible',eligible.length,'queries',queries.length);

function evaluate(model,limit){
 const lib=model==='clip'?clip:dino;
 // 只保留有向量的图片，预先归一化
 const items=[];
 for(const x of meta){const v=vecOf(lib,x.filename,model);if(!v)continue;
  const n=Math.hypot(...v)||1;items.push({f:x.filename,g:x.group,v:v.map(t=>t/n)})}
 let r1=0,r5=0,margin=0,total=0;const miss=[];const t0=Date.now();
 for(const q of queries.slice(0,limit)){
  const qv=vecOf(lib,q.f,model);if(!qv)continue;
  const qn=Math.hypot(...qv)||1,qq=qv.map(t=>t/qn);
  const groups=new Map();
  for(const it of items){if(it.f===q.f)continue;
   const s=cos(qq,it.v),old=groups.get(it.g);
   if(old){old.push(s)}else groups.set(it.g,[s]);
  }
  const rows=[...groups].map(([g,ss])=>{
   const top=Math.max(...ss),avg=ss.reduce((a,b)=>a+b,0)/ss.length,strong=ss.filter(s=>s>=0.65).length;
   return {g,p:top*0.6+(strong/ss.length)*0.25+avg*0.15}
  }).sort((a,b)=>b.p-a.p);
  const rank=rows.findIndex(x=>x.g===q.g)+1;
  total++;if(rank===1)r1++;if(rank>0&&rank<=5)r5++;
  if(rows.length>1&&rank===1)margin+=rows[0].p-rows[1].p;
  if(rank>5&&miss.length<12)miss.push({q:q.f,trueG:q.g,rank,top:rows[0].g});
 }
 return {model,total,r1,r5,recall1:r1/total,recall5:r5/total,avgMargin:margin/r1,ms:Date.now()-t0,miss};
}
const limit=Number(process.argv[2]||400);
for(const m of ['clip','dino']){
 const r=evaluate(m,limit);
 console.log('\n===',m,'===');
 console.log(JSON.stringify({total:r.total,recall1:+r.recall1.toFixed(4),recall5:+r.recall5.toFixed(4),avgMargin:+r.avgMargin.toFixed(4),ms:r.ms},null,2));
 console.log('miss examples:');for(const x of r.miss)console.log(' ',x.rank,x.q,x.trueG,'->',x.top);
}
