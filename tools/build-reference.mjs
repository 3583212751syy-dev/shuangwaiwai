// 平台查证参考库：用已人工确认的真实产品图作视觉锚点。
// 用户上传的对比图与锚点 CLIP 余弦 > 阈值时，直接采用查证过的名称/类目，不靠零样本瞎猜。
import {readFileSync,writeFileSync} from 'node:fs';
const W='D:/GPT/2026-11-21/new-chat-2/work';
const PROJ=W+'/git-handoff/projects/product-image-compare';
const tf='file:///'+W+'/model-runtime/node_modules/.pnpm/@huggingface+transformers@3.8.1/node_modules/@huggingface/transformers/dist/transformers.node.mjs';
const {pipeline,env}=await import(tf);
env.allowLocalModels=true;env.localModelPath=W+'/deploy-product-library/model-cache';env.useBrowserCache=false;
const clip=await pipeline('image-feature-extraction','Xenova/clip-vit-base-patch32',{device:'cpu',dtype:'q8'});
const norm=v=>{const n=Math.hypot(...v)||1;return Array.from(v,x=>x/n)};
// 每条：已查证的真实产品。source 写清平台与查证依据，前端可展示。
const ITEMS=[
 {id:'floor-drain-01',name:'不锈钢防臭地漏',category:'Hogar, Muebles y Jardín',sub:'Baños',subZh:'卫浴',
  image:W+'/ref_floor_drain.png',source:'1688 同款：磁吸不锈钢防臭地漏芯（¥0.23–11.70）；图中为弹跳盖+可拆滤网',
  keywords:['地漏','防臭','drain','floor drain','desagüe','drenaje']},
 {id:'welding-blanket-01',name:'防火焊接毯（2件装）',category:'Herramientas',sub:'Accesorios de Herramientas',subZh:'工具配件',
  image:PROJ+'/work/ref_welding_blanket.png',source:'美客多在售：Manta ignífuga para soldadura 50x50cm 5mm 2 piezas（叠放黑色焊接毯+焊枪喷火商品图）',
  keywords:['防火毯','焊接毯','manta ignífuga','welding blanket','fire blanket','manta para soldadura']},
 {id:'breaker-c16-01',name:'小型断路器 DZ47-63 C16（2只装）',category:'Electrónica, Audio y Video',sub:'Componentes Electrónicos',subZh:'电子元件',
  image:W+'/deploy-product-library/images/image1080.jpeg',source:'美客多/1688 同款：DZ47-63 C16 230/400V 小型断路器，2 只装；壳体印刷 C16',
  keywords:['断路器','空开','C16','DZ47','circuit breaker','interruptor']},
 {id:'gas-spring-01',name:'后备箱气弹簧支撑杆（Seat Ibiza 2003-2009）',category:'Accesorios para Vehículos',sub:'Amortiguadores y Suspensión',subZh:'减震与悬挂',
  image:W+'/ref_gas_spring.png',source:'MercadoLibre：一对 Seat Ibiza 2003-2009 后尾箱减震器 $288；OE 6L6 827 550 A/B；StrutsDepot 612410',
  keywords:['气弹簧','减震','撑杆','gas spring','strut','tailgate','cajuela','amortiguador']}
];
const out={_meta:{builtAt:new Date().toISOString(),method:'人工确认产品图 + CLIP 视觉锚点',model:'Xenova/clip-vit-base-patch32'},items:[]};
for(const it of ITEMS){
 const o=await clip(it.image,{pooling:'mean'});
 const v=norm(o.data||o[0]?.data||o);
 out.items.push({...it,embedding:v,image:undefined});
 console.log('embedded',it.id,it.name,v.length);
}
writeFileSync(PROJ+'/site-source/references.json',JSON.stringify(out));
writeFileSync(W+'/deploy-product-library/references.json',JSON.stringify(out));
writeFileSync(W+'/deploy-image-match/references.json',JSON.stringify(out));
console.log('WROTE references.json',JSON.stringify(out).length,'bytes');
