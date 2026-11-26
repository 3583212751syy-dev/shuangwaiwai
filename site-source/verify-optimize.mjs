import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import app from './dist/server/index.js';
import {catalogFor,effectiveLabel,groupProducts,PRODUCT_RANGES} from './catalog.mjs';
import {identify,prototypes,catalogAnalysis,normalize,MODEL} from './ai.mjs';
import {searchLibrary,READY} from './search.mjs';

/* 1. 产品目录映射：原图编号 -> 真实品名/类目/同款分组 */
assert.equal(PRODUCT_RANGES.length,49);
assert.deepEqual(
  [catalogFor('表格第0056行_B列_原图192.jpg').name,catalogFor('表格第0056行_B列_原图192.jpg').category],
  ['弹性绑带及手套套装','Herramientas']
);
assert.equal(catalogFor('表格第0007行_A列_原图007.jpg').name,'气弹簧支撑杆');
assert.equal(catalogFor('表格第0056行_C列_原图191.jpg').group,catalogFor('表格第0056行_B列_原图192.jpg').group);
assert.equal(catalogFor('photo.jpg'),null);

/* 2. 有效标签：人工编辑优先于目录，目录优先于 CLIP 猜测 */
const raw={name:'表格第0056行_B列_原图192',filename:'表格第0056行_B列_原图192.jpg',category:'待分类',group:'表格第0056行_B列_原图192',ai:{model:MODEL,status:'待确认',suggestedName:'仅有包装',suggestedCategory:'待分类',embedding:Array(512).fill(0)}};
assert.equal(effectiveLabel(raw).name,'弹性绑带及手套套装');
assert.equal(effectiveLabel(raw).source,'产品目录');
assert.equal(effectiveLabel({...raw,name:'我改过的名字'}).name,'我改过的名字');
assert.equal(groupProducts([raw,{...raw,id:'2',filename:'表格第0056行_A列_原图191.jpg'}]).length,1);

/* 3. 近邻识别：用已标注同款分组的向量做原型，新品才落待确认 */
const vec=(a,b=0)=>normalize(Array.from({length:512},(_,i)=>i===0?a:(i===1?b:0)));
const lib=[
 {id:'a',name:'气弹簧支撑杆',category:'汽配',group:'杆1',filename:'x.jpg',ai:{model:MODEL,embedding:vec(1,0),status:'已识别',suggestedName:'气弹簧支撑杆',suggestedCategory:'汽配'}},
 {id:'b',name:'气弹簧支撑杆',category:'汽配',group:'杆1',filename:'y.jpg',ai:{model:MODEL,embedding:vec(.98,.2),status:'已识别',suggestedName:'气弹簧支撑杆',suggestedCategory:'汽配'}},
 {id:'c',name:'金刚石打磨盘',category:'五金工具',group:'盘1',filename:'z.jpg',ai:{model:MODEL,embedding:vec(0,1),status:'已识别',suggestedName:'金刚石打磨盘',suggestedCategory:'五金工具'}}
];
assert.equal(prototypes(lib).length,2);
const near=identify({embedding:vec(1,.05),suggestedName:'无法辨认的产品',suggestedCategory:'待分类',status:'待确认'},lib);
assert.equal(near.status,'已识别');
assert.equal(near.suggestedName,'气弹簧支撑杆');
assert.equal(near.method,'图库近邻');
assert.equal(near.matchedGroup,'杆1');
const far=identify({embedding:normalize(Array.from({length:512},(_,i)=>i===2?1:0)),suggestedName:'无法辨认的产品',suggestedCategory:'待分类',status:'待确认'},lib);
assert.equal(far.status,'待确认');
assert.equal(far.candidates.length,2);
assert.equal(catalogAnalysis(raw,{embedding:vec(1),model:MODEL,status:'待确认'}).suggestedName,'弹性绑带及手套套装');
assert.equal(searchLibrary(near,lib).stage,'product');
assert.equal(READY(lib[0]),true);

/* 4. 服务端：新增模块路由、精简元数据、同款确认、图片代理安全 */
const objects=new Map();
const env={BUCKET:{
 async put(k,b,o){objects.set(k,{b,o})},
 async get(k){const v=objects.get(k);return v?{text:async()=>String(v.b),body:v.b,httpMetadata:v.o?.httpMetadata}:null},
 async list(){return {objects:[...objects.keys()].filter(k=>k.startsWith('meta/')).map(key=>({key})),truncated:false}}
}};
const req=(p,o)=>app.fetch(new Request('https://example.test'+p,o),env);
const built=await readFile('dist/server/index.js','utf8');
assert.ok(built.includes('catalogModule')&&built.includes('effectiveLabel'));
assert.equal((await req('/catalog.mjs')).status,200);
assert.ok((await (await req('/catalog.mjs')).text()).includes('PRODUCT_RANGES'));
assert.ok((await (await req('/search.mjs')).text()).includes('effectiveLabel'));
assert.ok((await (await req('/')).text()).includes('按同款分组'));

const form=new FormData();form.append('image',new Blob(['test'],{type:'image/png'}),'表格第0056行_B列_原图192.jpg');
const item=await (await req('/api/images',{method:'POST',body:form})).json();
const ai={model:MODEL,status:'待确认',embedding:Array(512).fill(0),suggestedName:'弹性绑带及手套套装',suggestedCategory:'五金工具',score:.3,method:'图库近邻'};
assert.equal((await req('/api/images/'+item.id,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({ai,confirmations:[{decision:'same',query:'q.png'},{decision:'bogus'}]})})).status,200);
const slim=await (await req('/api/images?slim=1')).json();
assert.equal(slim[0].ai.embedding,undefined);
assert.equal(slim[0].ai.method,'图库近邻');
const full=await (await req('/api/images')).json();
assert.equal(full[0].ai.embedding.length,512);
assert.equal(full[0].confirmations.length,1);
assert.equal(full[0].confirmations[0].decision,'same');
assert.equal((await req('/api/proxy-image?url=http%3A%2F%2F127.0.0.1%2Fa.png')).status,400);
assert.equal((await req('/api/proxy-image?url=https%3A%2F%2Flocalhost%2Fa.png')).status,400);
assert.equal((await req('/api/proxy-image?url=https%3A%2F%2F192.168.1.1%2Fa.png')).status,400);

console.log('PASS: 49 段产品目录映射, 多角度分组, 人工编辑优先, 图库近邻识别/待确认, 目录标签回填, 分组检索, 模块路由, 精简元数据, 同款确认, 代理 SSRF 防护');
