// 诊断：对比图在「产品级提示词词典 / 32大类 / 二级类目」上的真实排序，定位错判根因。
import {readFileSync} from 'node:fs';
const W='D:/GPT/2026-11-21/new-chat-2/work';
const tf='file:///'+W+'/model-runtime/node_modules/.pnpm/@huggingface+transformers@3.8.1/node_modules/@huggingface/transformers/dist/transformers.node.mjs';
const {pipeline,env,RawImage,AutoTokenizer,CLIPTextModelWithProjection}=await import(tf);
env.allowLocalModels=true;env.localModelPath=W+'/deploy-product-library/model-cache';env.useBrowserCache=false;
const {CATALOG,ML_CATEGORIES,ML_CATEGORY_EXAMPLES,ML_SUBCATEGORIES}=await import('./../site-source/ai.mjs');
const clip=await pipeline('image-feature-extraction','Xenova/clip-vit-base-patch32',{device:'cpu',dtype:'q8'});
const tokenizer=await AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch32');
const text=await CLIPTextModelWithProjection.from_pretrained('Xenova/clip-vit-base-patch32',{dtype:'q8',device:'cpu'});
const norm=v=>{const n=Math.hypot(...v)||1;return Array.from(v,x=>x/n)};
const cos=(a,b)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s};
function txtVec(texts){
 const inp=tokenizer(texts,{padding:true,truncation:true});
 const out=text?text(inp):null;
 console.log('DEBUG text out keys',Object.keys(out||{}),'dims',JSON.stringify(out?.text_embeds?.dims));
 const d=out?.text_embeds?.data;if(!d)return null;
 const dim=out.text_embeds.dims.at(-1);
 return texts.map((_,i)=>norm(d.slice(i*dim,(i+1)*dim)));
}
const catVecs=txtVec(CATALOG.map(x=>'a product photo of '+x[2]));
const mlTexts=[],owner=[];
ML_CATEGORIES.forEach((c,i)=>{const ex=ML_CATEGORY_EXAMPLES[c.es];const ts=ex?['a photo of '+ex[0],'a close up product photo of '+ex[0],ex[1]]:[c.prompt];ts.forEach(t=>{mlTexts.push(t);owner.push(i)})});
const mlRaw=txtVec(mlTexts);
const dim=mlRaw?.[0]?.length||512;
const mlSum=ML_CATEGORIES.map(()=>new Array(dim).fill(0)),cnt=new Array(ML_CATEGORIES.length).fill(0);
if(mlRaw)mlRaw.forEach((v,k)=>{const o=owner[k];for(let i=0;i<dim;i++)mlSum[o][i]+=v[i];cnt[o]++});
const mlVecs=mlSum.map((v,i)=>norm(v.map(x=>x/cnt[i])));
const subVecs=txtVec(ML_SUBCATEGORIES.map(c=>c.prompt));
for(const p of process.argv.slice(2)){
 const o=await clip(p,{pooling:'mean'});
 const ev=norm(o.data||o[0]?.data||o);
 console.log('\n===',p,'===');
 const ranked=CATALOG.map((c,i)=>({n:c[0],cat:c[1],s:catVecs?cos(ev,catVecs[i]):0})).sort((a,b)=>b.s-a.s);
 console.log('产品词典 top8:');ranked.slice(0,8).forEach(x=>console.log('  ',x.s.toFixed(3),x.n,'|',x.cat));
 const ml=ML_CATEGORIES.map((c,i)=>({n:c.es,s:mlVecs[i]?cos(ev,mlVecs[i]):0})).sort((a,b)=>b.s-a.s);
 console.log('32大类 top5:');ml.slice(0,5).forEach(x=>console.log('  ',x.s.toFixed(3),x.n));
 const sub=ML_SUBCATEGORIES.map((c,i)=>({n:c.es,zh:c.zh,p:c.parent,s:subVecs?cos(ev,subVecs[i]):0})).sort((a,b)=>b.s-a.s);
 console.log('二级类目 top5:');sub.slice(0,5).forEach(x=>console.log('  ',x.s.toFixed(3),x.n,x.zh,'|',x.p));
}
