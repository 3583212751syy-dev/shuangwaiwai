// 跨域外部查询验证：真实用户查询图 vs 三模型检索排名。
// 查询：1.webp（切割片展示图，真值=底图库第473行组 image1540/1541/1542）+ 3 个查证锚点图（库内无同款，看 top1 是否合理）。
import {readFileSync} from 'node:fs';
const W='D:/GPT/2026-11-21/new-chat-2/work';
const R=W+'/git-handoff/projects/product-image-compare';
const tf='file:///'+W+'/_siglip2/node_modules/@huggingface/transformers/dist/transformers.node.mjs';
const {AutoModel,AutoProcessor,CLIPVisionModelWithProjection,env,RawImage,pipeline}=await import(tf);
env.allowLocalModels=true;env.localModelPath=W+'/deploy-product-library/model-cache';env.useBrowserCache=false;

const meta=JSON.parse(readFileSync(W+'/deploy-product-library/metadata.json','utf8'));
const clipLib=JSON.parse(readFileSync(R+'/work/embeddings.json','utf8'));
const dino2=JSON.parse(readFileSync(R+'/work/embeddings-dinov2.json','utf8'));
const dino3raw=JSON.parse(readFileSync(R+'/work/embeddings-dinov3.json','utf8'));
const cos=(a,b)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s};
const norm=v=>{const n=Math.hypot(...v)||1;return v.map(x=>x/n)};
const getVec=(lib,f)=>{const e=lib['originals/'+f]||lib[f];const v=e&&(e.embedding||e);return v?norm(v):null};

// 真值组定位
const GT_GROUP='第473行 · 4片彩色花纹切割片（待复核）';
const gtEntry=meta.find(m=>m.group===GT_GROUP)||meta.find(m=>m.group.includes('473'));
console.log('真值组:',gtEntry?.group,'| 库内图:',gtEntry?.filename);

// 加载三个查询端模型
const clipPipe=await pipeline('image-feature-extraction','Xenova/clip-vit-base-patch32',{device:'cpu',dtype:'q8'});
env.localModelPath=W+'/model-alt';
const dino2Proc=await AutoProcessor.from_pretrained('onnx-community/dinov2-small');
const dino2Model=await AutoModel.from_pretrained('onnx-community/dinov2-small',{dtype:'q8',device:'cpu'});
const dino3Proc=await AutoProcessor.from_pretrained('onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX');
const dino3Model=await AutoModel.from_pretrained('onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX',{dtype:'q8',device:'cpu'});
const mcRaw=JSON.parse(readFileSync(R+'/work/embeddings-mobileclip-fp32.json','utf8'));const mcLib={};for(const[f,v]of Object.entries(mcRaw))mcLib[f]=v.v;
env.localModelPath=W+'/model-alt';
const mcProc=await AutoProcessor.from_pretrained('Xenova/mobileclip_s0');
const mcModel=await CLIPVisionModelWithProjection.from_pretrained('Xenova/mobileclip_s0',{dtype:'fp32',device:'cpu'});
env.localModelPath=W+'/deploy-product-library/model-cache';
env.localModelPath=W+'/deploy-product-library/model-cache';

async function qvec(path){
 const img=await RawImage.read(path);
 const co=await clipPipe(img,{pooling:'mean'});
 const i2=await dino2Proc(img),o2=await dino2Model(i2);
 const i3=await dino3Proc(img),o3=await dino3Model(i3);
 const pool=v=>{const dim=384,n=v.length/dim,o=new Array(dim).fill(0);for(let t=0;t<n;t++)for(let d=0;d<dim;d++)o[d]+=v[t*dim+d];return o};
 return{clip:norm(Array.from(co.data||co[0]?.data)),
        dino2:norm(pool(Array.from(o2.last_hidden_state.data))),
        dino3cls:norm(Array.from(o3.last_hidden_state.data).slice(0,384)),
        mc:norm(Array.from((await mcModel(await mcProc(img))).image_embeds.data))};
}
function rank(path,model,gt){
 const lib=model==='clip'?clipLib:(model==='dino2'?dino2:null);
 const groups=new Map();
 for(const x of meta){
  let v;
  if(model==='clip'||model==='dino2')v=getVec(lib,x.filename);
  else v=dino3raw[x.filename]?.c;
  if(!v)continue;
  if(!groups.has(x.group))groups.set(x.group,{top:0,n:0,sum:0});
  const s=cos(model==='clip'||model==='dino2'?norm(v):v, null);
 }
 return groups;
}
// 直接对每个模型算组排名（查询向量已归一化；库向量已归一化或现归一化）
async function evaluateQuery(path,label,gtGroupLike){
 const q=await qvec(path);
 console.log('\n查询:',label);
 for(const model of ['mc','dino3cls','clip']){
  const rows=[];
  const groups=new Map();
  for(const x of meta){
   let v=model==='dino3cls'?dino3raw[x.filename]?.c:(model==='mc'?mcLib[x.filename]:getVec(model==='clip'?clipLib:dino2,x.filename));
   if(!v)continue;
   v=norm(v);
   const s=cos(q[model],v);
   if(!groups.has(x.group))groups.set(x.group,[]);
   groups.get(x.group).push(s);
  }
  for(const[g,ss]of groups){
   const top=Math.max(...ss),avg=ss.reduce((a,b)=>a+b,0)/ss.length,strong=ss.filter(s=>s>=0.65).length;
   rows.push({g,p:top*0.6+(strong/ss.length)*0.25+avg*0.15,top});
  }
  rows.sort((a,b)=>b.p-a.p);
  const rank=rows.findIndex(x=>gtGroupLike&&x.g===gtGroupLike)+1;
  console.log(` ${model.padEnd(8)} 真值组排名=${rank||'-'} top1=[${rows[0].g.slice(0,40)}] sim=${rows[0].top.toFixed(3)} ${gtGroupLike?('真值组top='+((rows.find(x=>x.g===gtGroupLike)?.top)??0).toFixed(3)):''}`);
 }
}
await evaluateQuery('C:/Users/Administrator/Desktop/图片实验位置/1.webp','1.webp 切割片展示图（真同款）',GT_GROUP);
await evaluateQuery(W+'/ref_gas_spring.png','气弹簧（库内无同款）',null);
await evaluateQuery(W+'/ref_floor_drain.png','防臭地漏（库内无同款）',null);
await evaluateQuery(R+'/work/ref_welding_blanket.png','防火毯（库内无同款）',null);
console.log('\nDONE');
