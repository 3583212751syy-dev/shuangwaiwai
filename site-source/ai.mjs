import { effectiveLabel, catalogFor } from './catalog.mjs';

export const MODEL = 'Xenova/clip-vit-base-patch32';
export const DINO_MODEL = 'onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX';

// 美客多墨西哥站一级类目（官方口径，来自 mercadolibre.com.mx 首页「Mostrar todas las categorías」）。
// es=官方西语类目名，zh=中文，prompt=给 CLIP 做零样本判断的英文描述。
export const ML_CATEGORIES = [
  {es:'Herramientas',zh:'工具',prompt:'a photo of a hand tool, wrench, screwdriver, drill, saw blade, grinding disc, pliers, measuring tool'},
  {es:'Hogar, Muebles y Jardín',zh:'家居家具园艺',prompt:'a photo of a home and garden item, kitchen utensil, storage box, furniture, decoration, textile'},
  {es:'Electrodomésticos',zh:'家用电器',prompt:'a photo of a home appliance, kitchen machine, vacuum, blender, air conditioner, washing machine part'},
  {es:'Electrónica, Audio y Video',zh:'电子音频视频',prompt:'a photo of an electronic device, circuit board, relay, switch, connector, power adapter, cable, bulb'},
  {es:'Accesorios para Vehículos',zh:'汽车配件',prompt:'a photo of an automobile part, car interior trim, brake, tire, car light, car connector'},
  {es:'Deportes y Fitness',zh:'运动健身',prompt:'a photo of a sports item, ball, fitness equipment, bicycle accessory, outdoor gear'},
  {es:'Ropa, Bolsas y Calzado',zh:'服饰箱包鞋类',prompt:'a photo of clothing, shirt, garment, hat, bag accessory, textile apparel'},
  {es:'Mascotas',zh:'宠物',prompt:'a photo of a pet product, dog harness, pet clothing, pet toy, pet bowl'},
  {es:'Bebés',zh:'婴儿用品',prompt:'a photo of a baby item, stroller part, baby bottle, baby clothing'},
  {es:'Juegos y Juguetes',zh:'游戏与玩具',prompt:'a photo of a toy, doll, puzzle, kids play item'},
  {es:'Construcción',zh:'建筑装修',prompt:'a photo of a building material, paint, pipe, tile, hardware for construction'},
  {es:'Industrias y Oficinas',zh:'工业与办公',prompt:'a photo of industrial equipment part, machine component, safety gear, uniform'},
  {es:'Computación',zh:'电脑办公',prompt:'a photo of a computer accessory, keyboard, mouse, printer part, office electronics'},
  {es:'Celulares y Telefonía',zh:'手机通讯',prompt:'a photo of a mobile phone accessory, tablet accessory, phone charger'},
  {es:'Belleza y Cuidado Personal',zh:'美容个护',prompt:'a photo of a health or beauty product, grooming device, personal care item'},
  {es:'Salud y Equipamiento Médico',zh:'健康医疗',prompt:'a photo of medical equipment, therapy massager, mobility aid'},
  {es:'Arte, Papelería y Mercería',zh:'美术文具工艺',prompt:'a photo of stationery, craft supplies, art material, notebook, pen'},
  {es:'Instrumentos Musicales',zh:'乐器',prompt:'a photo of a musical instrument or accessory'},
  {es:'Joyas y Relojes',zh:'珠宝手表',prompt:'a photo of jewelry, watch, clock'},
  {es:'Libros, Revistas y Comics',zh:'图书漫画',prompt:'a photo of a book, magazine, comic'},
  {es:'Música, Películas y Series',zh:'影音娱乐',prompt:'a photo of media disc, movie, series merchandise'},
  {es:'Televisores',zh:'电视',prompt:'a photo of a television set or TV accessory'},
  {es:'Cámaras y Accesorios',zh:'相机配件',prompt:'a photo of a camera, lens, tripod, camera accessory'},
  {es:'Consolas y Videojuegos',zh:'游戏主机',prompt:'a photo of a game console, controller, video game'},
  {es:'Agro',zh:'农业',prompt:'a photo of agricultural machinery part, farming tool'},
  {es:'Alimentos y Bebidas',zh:'食品饮料',prompt:'a photo of food or beverage product'},
  {es:'Antigüedades y Colecciones',zh:'古董收藏',prompt:'a photo of a collectible item, antique'},
  {es:'Productos Sustentables',zh:'可持续产品',prompt:'a photo of an eco friendly sustainable product'},
  {es:'Recuerdos, Cotillón y Fiestas',zh:'派对用品',prompt:'a photo of party decoration, gift bag, festive supplies'},
  {es:'Inmuebles',zh:'房产',prompt:'a photo of a property, house, apartment'},
  {es:'Servicios',zh:'服务',prompt:'a photo of a service offering, repair, maintenance'},
  {es:'Otras Categorías',zh:'其他类别',prompt:'a photo of an unclassified industrial hardware part or machine component'}
];

// 美客多二级类目（西语原文取自 mercadolibre.com.mx/categorias 页面，只收录官网确实列出的节点）。
// 提示词增强：每类给「英文类目名 + 典型商品词」，三模板平均（实测 top1 17.4%→19.1%，top3 32.0%→38.6%）。
export const ML_CATEGORY_EXAMPLES = {
 'Herramientas':['hand tools and hardware','wrench screwdriver pliers drill bit saw blade cutting disc grinding wheel abrasive sandpaper clamp screw bolt nut washer gasket bearing spring hose fitting tape measure utility knife'],
 'Hogar, Muebles y Jardín':['home, furniture and garden products','kitchen utensil storage box furniture decoration textile curtain carpet vase plant pot bathroom accessory'],
 'Electrodomésticos':['home appliances','washing machine refrigerator air conditioner vacuum blender microwave electric fan water dispenser'],
 'Electrónica, Audio y Video':['consumer electronics, audio and video equipment','circuit board relay contactor switch connector power adapter cable led bulb speaker headphone'],
 'Accesorios para Vehículos':['car and motorcycle accessories','brake pad brake disc wheel hub shock absorber mirror bumper wiper car light spark plug car switch auto sensor'],
 'Deportes y Fitness':['sports and fitness equipment','ball dumbbell bicycle helmet yoga mat fishing rod tent sports glove'],
 'Ropa, Bolsas y Calzado':['clothing, bags and shoes','shirt t-shirt jacket trousers uniform hat backpack handbag wallet shoes fabric'],
 'Mascotas':['pet products','dog leash cat litter pet bowl pet toy pet harness'],
 'Bebés':['baby products','baby bottle stroller diaper baby clothing pacifier'],
 'Juegos y Juguetes':['toys and games','doll puzzle building block toy car plush toy'],
 'Construcción':['construction materials','cement tile paint pipe plumbing fitting door lock building hardware'],
 'Industrias y Oficinas':['industrial and office supplies','office paper printer safety helmet work glove warehouse equipment industrial machine part fan blower'],
 'Computación':['computer accessories','keyboard mouse monitor printer laptop stand usb hub'],
 'Celulares y Telefonía':['mobile phone accessories','phone case screen protector phone charger phone holder'],
 'Belleza y Cuidado Personal':['beauty and personal care products','makeup brush hair dryer nail file razor cosmetics'],
 'Salud y Equipamiento Médico':['health and medical equipment','massager thermometer blood pressure monitor medical device'],
 'Arte, Papelería y Mercería':['stationery, art and craft supplies','pen pencil notebook scissors craft bead glue paint brush'],
 'Instrumentos Musicales':['musical instruments and accessories','guitar string guitar pick drum keyboard microphone stand'],
 'Joyas y Relojes':['jewelry and watches','ring necklace earring bracelet wrist watch clock'],
 'Libros, Revistas y Comics':['books, magazines and comics','book magazine comic'],
 'Música, Películas y Series':['music, movies and series','cd dvd vinyl record movie poster'],
 'Televisores':['televisions','tv screen tv stand tv antenna'],
 'Cámaras y Accesorios':['cameras and accessories','camera lens tripod memory card camera bag'],
 'Consolas y Videojuegos':['video game consoles','game console controller video game'],
 'Agro':['agriculture and farming','irrigation sprinkler farming tool seed fertilizer'],
 'Alimentos y Bebidas':['food and beverages','snack drink coffee candy canned food'],
 'Antigüedades y Colecciones':['antiques and collectibles','antique coin stamp collectible figurine'],
 'Productos Sustentables':['sustainable eco friendly products','reusable bag solar panel compost'],
 'Recuerdos, Cotillón y Fiestas':['party supplies and gifts','balloon gift bag party decoration ribbon'],
 'Inmuebles':['real estate property','house apartment land'],
 'Servicios':['services','repair service installation service'],
 'Otras Categorías':['unclassified miscellaneous small parts','small plastic part unidentifiable accessory empty packaging box']
};

// parent 必须等于上面 ML_CATEGORIES 里的某个 es。
export const ML_SUBCATEGORIES = [
  {es:'Herramientas Manuales',zh:'手动工具',parent:'Herramientas',prompt:'a photo of a manual hand tool, wrench, pliers, screwdriver, hex key'},
  {es:'Herramientas Eléctricas',zh:'电动工具',parent:'Herramientas',prompt:'a photo of a power tool, electric drill, rotary tool'},
  {es:'Herramientas Neumáticas',zh:'气动工具',parent:'Herramientas',prompt:'a photo of a pneumatic tool, air tool'},
  {es:'Herramientas de Medición',zh:'测量工具',parent:'Herramientas',prompt:'a photo of a measuring tool, caliper, tape measure'},
  {es:'Accesorios de Herramientas',zh:'工具配件',parent:'Herramientas',prompt:'a photo of a tool accessory, welding equipment protection, fire blanket, tool case, tool belt'},
  {es:'Cajas y Organizadores',zh:'收纳箱',parent:'Herramientas',prompt:'a photo of a toolbox, organizer box, case for tools'},
  {es:'Herramientas de Jardín',zh:'园艺工具',parent:'Herramientas',prompt:'a photo of a garden tool, pruning shears'},
  {es:'Herramientas Industriales',zh:'工业工具',parent:'Industrias y Oficinas',prompt:'a photo of an industrial tool, machine tool accessory'},
  {es:'Componentes Electrónicos',zh:'电子元件',parent:'Electrónica, Audio y Video',prompt:'a photo of electronic components, capacitor, resistor, relay, circuit board'},
  {es:'Cables',zh:'线缆',parent:'Electrónica, Audio y Video',prompt:'a photo of a cable, wire harness, power cord'},
  {es:'Pilas y Cargadores',zh:'电池充电器',parent:'Electrónica, Audio y Video',prompt:'a photo of a battery, charger, power adapter'},
  {es:'Accesorios para Audio y Video',zh:'音视频配件',parent:'Electrónica, Audio y Video',prompt:'a photo of an audio video accessory, plug, connector, socket'},
  {es:'Cocina',zh:'厨房',parent:'Hogar, Muebles y Jardín',prompt:'a photo of a kitchen utensil, pot, pan, cup, bowl, jar'},
  {es:'Baños',zh:'卫浴',parent:'Hogar, Muebles y Jardín',prompt:'a photo of bathroom hardware, faucet, drain, shower part, floor drain, anti odor drain cover'},
  {es:'Amortiguadores y Suspensión',zh:'减震与悬挂',parent:'Accesorios para Vehículos',prompt:'a photo of car shock absorber, gas spring strut, tailgate lift support, suspension damper'},
  {es:'Refacciones de Motor',zh:'发动机配件',parent:'Accesorios para Vehículos',prompt:'a photo of engine part, solenoid valve, sensor, fuse tester, ignition part'},
  {es:'Organización para el Hogar',zh:'家居收纳',parent:'Hogar, Muebles y Jardín',prompt:'a photo of a storage box, organizer, plastic case, drawer'},
  {es:'Iluminación para el Hogar',zh:'家居照明',parent:'Hogar, Muebles y Jardín',prompt:'a photo of a home lamp, bulb, light fixture'},
  {es:'Adornos y Decoración del Hogar',zh:'家居装饰',parent:'Hogar, Muebles y Jardín',prompt:'a photo of home decoration, wall sticker, decorative film, vase'},
  {es:'Textiles de Hogar y Decoración',zh:'家纺布艺',parent:'Hogar, Muebles y Jardín',prompt:'a photo of home textile, fabric, curtain, blanket'},
  {es:'Cuidado del Hogar y Lavanderia',zh:'家务清洁',parent:'Hogar, Muebles y Jardín',prompt:'a photo of a household cleaning tool, mop, brush'},
  {es:'Camisas',zh:'衬衫',parent:'Ropa, Bolsas y Calzado',prompt:'a photo of a shirt, blouse, garment'},
  {es:'Playeras',zh:'T恤',parent:'Ropa, Bolsas y Calzado',prompt:'a photo of a t-shirt, top, jersey'},
  {es:'Indumentaria Laboral y Escolar',zh:'工装校服',parent:'Ropa, Bolsas y Calzado',prompt:'a photo of a work uniform, school uniform'},
  {es:'Ropa Deportiva',zh:'运动服',parent:'Ropa, Bolsas y Calzado',prompt:'a photo of sportswear, athletic clothing'},
  {es:'Calzado',zh:'鞋靴',parent:'Ropa, Bolsas y Calzado',prompt:'a photo of shoes, boots, footwear'},
  {es:'Equipaje y Bolsas',zh:'箱包',parent:'Ropa, Bolsas y Calzado',prompt:'a photo of a bag, backpack, luggage, pouch'},
  {es:'Bicicletas y Ciclismo',zh:'自行车',parent:'Deportes y Fitness',prompt:'a photo of a bicycle part, inner tube, chain, sprocket'},
  {es:'Fútbol',zh:'足球',parent:'Deportes y Fitness',prompt:'a photo of a football, soccer ball, sports ball'},
  {es:'Fitness y Musculación',zh:'健身器材',parent:'Deportes y Fitness',prompt:'a photo of fitness equipment, dumbbell, workout gear'},
  {es:'Uniformes y Ropa de Trabajo',zh:'工作制服',parent:'Industrias y Oficinas',prompt:'a photo of a work uniform, safety vest, industrial clothing'},
  {es:'Seguridad Laboral',zh:'劳动防护',parent:'Industrias y Oficinas',prompt:'a photo of safety equipment, gloves, protective gear'},
  {es:'Materiales de Obra',zh:'建材',parent:'Construcción',prompt:'a photo of construction material, tile, paint, pipe for building'},
  {es:'Plomería',zh:'管道',parent:'Construcción',prompt:'a photo of plumbing part, pipe fitting, hose'},
  {es:'Masajes',zh:'按摩器材',parent:'Salud y Equipamiento Médico',prompt:'a photo of a massager, massage ball, therapy device'},
  {es:'Equipamiento Médico',zh:'医疗设备',parent:'Salud y Equipamiento Médico',prompt:'a photo of medical equipment, stethoscope, gauge'},
  {es:'Juguetes',zh:'玩具',parent:'Juegos y Juguetes',prompt:'a photo of a toy, doll, puzzle'},
  {es:'Instrumentos de Cuerdas',zh:'弦乐器',parent:'Instrumentos Musicales',prompt:'a photo of a string instrument accessory'}
];
export const CATALOG = [
 ['气弹簧支撑杆','Accesorios para Vehículos','car gas spring lift support struts'],['车门内拉手','Accesorios para Vehículos','car interior door handles'],['车门外拉手','Accesorios para Vehículos','car exterior door handles'],['轮胎内胎','Accesorios para Vehículos','rubber tire inner tube'],['轮毂螺母','Accesorios para Vehículos','chrome wheel lug nuts'],['汽车开关','Accesorios para Vehículos','car dashboard rocker push button switch'],['汽车电磁阀','Accesorios para Vehículos','automotive vacuum solenoid valve'],['汽车传感器','Accesorios para Vehículos','automotive metal crankshaft camshaft position sensor'],['橡胶衬套及轴承','Accesorios para Vehículos','automotive rubber suspension bushings and bearings'],['接线插头','Accesorios para Vehículos','automotive wire harness electrical connectors'],['塑料固定卡扣','Accesorios para Vehículos','black plastic car trim fastening clips'],['密封条','Accesorios para Vehículos','self adhesive rubber weather stripping seal'],
 ['防火焊接毯','Herramientas','black welding blanket fire resistant folded pieces, fiberglass fire blanket protecting from torch flame, heat resistant welding mat'],['瓷砖倒角工具','Herramientas','metal blue ceramic tile 45 degree bevel cutting tool'],['金刚石打磨盘','Herramientas','diamond grinding wheel disc'],['小型打磨头套装','Herramientas','rotary tool grinding polishing bits set'],['切割片套装','Herramientas','small rotary cutting saw blade discs set'],['快速夹具','Herramientas','red handle metal toggle clamp'],['金属搭扣','Herramientas','stainless steel toggle latch hasp buckle'],['金属固定座','Herramientas','metal mounting bracket socket plate'],['高压清洗喷杆','Herramientas','pressure washer spray lance wand'],['金属泵体','Herramientas','metal pressure washer water pump'],['塑料衬套','Herramientas','white plastic nylon cylindrical bushings'],['金属燃烧管','Herramientas','metal perforated burner tube'],['弹簧吸盘工具','Herramientas','long spring rod suction cup hand tool'],['胶带','Herramientas','roll of green adhesive tape'],['白色密封环','Herramientas','white rubber silicone sealing rings'],['弹性绑带','Herramientas','elastic bungee cords with hooks and gloves'],['螺丝配件','Herramientas','metal screws and fasteners'],
 ['接线端子套装','Electrónica, Audio y Video','red blue yellow insulated electrical crimp wire terminals'],['交流接触器','Electrónica, Audio y Video','electrical AC magnetic contactor'],['继电器套装','Electrónica, Audio y Video','automotive relay circuit board wiring kit'],['电源转接头','Electrónica, Audio y Video','DC power adapter cable with interchangeable barrel tips'],['灯管支架卡座','Electrónica, Audio y Video','fluorescent tube lamp holder socket'],['工作灯','Electrónica, Audio y Video','long rechargeable LED work inspection light'],['反射灯泡','Electrónica, Audio y Video','red infrared heat lamp reflector bulb'],['插座式电气设备','Electrónica, Audio y Video','white plug in electrical power saver device'],
 ['装饰膜','Hogar, Muebles y Jardín','holographic glitter self adhesive decorative film sheets'],['网状用品','Hogar, Muebles y Jardín','white nylon net mesh rope'],['宠物梳','Mascotas','pet dog cat deshedding grooming rake brush'],['宠物洁齿棒','Mascotas','natural wooden cat silvervine dental chew sticks'],
 // 平台查证补充（2026-11-25）：用户实拍对比图反复被误判的产品，按美客多/1688 在售名称写英文提示词。
 ['防臭地漏','Hogar, Muebles y Jardín','stainless steel anti odor floor drain with teal pop up cover and removable filter basket'],
 ['地漏芯防臭器','Hogar, Muebles y Jardín','magnetic stainless steel floor drain core anti odor insert for bathroom'],
 ['浴室地漏盖','Hogar, Muebles y Jardín','round chrome bathroom floor drain cover grate'],
 ['后备箱气弹簧','Accesorios para Vehículos','pair of black car trunk tailgate gas spring lift support struts with ball sockets'],
 ['汽车尾门撑杆','Accesorios para Vehículos','two black hydraulic tailgate lift struts for hatchback car boot'],
 ['汽车减震器','Accesorios para Vehículos','car suspension shock absorber damper cylinder'],
 ['浪涌保护器','Electrónica, Audio y Video','red surge protective device SPD modular DIN rail 385V 20kA'],
 ['小型断路器','Electrónica, Audio y Video','white blue DZ47-63 miniature circuit breaker C16 230V'],
 ['保险丝测试仪','Electrónica, Audio y Video','automotive fuse tester LED test light probe with wire'],
 ['电磁阀线圈','Accesorios para Vehículos','black automotive solenoid valve with electrical connector'],
 ['仅有包装','Otras Categorías','closed cardboard boxes plastic packaging bags no visible product'],['无法辨认的产品','Otras Categorías','unidentifiable object partially hidden inside packaging']
];

export function normalize(v){const n=Math.hypot(...v);if(!n||!Number.isFinite(n))throw Error('无效的图片特征');return Array.from(v,x=>x/n)}
export function cosine(a,b){if(a?.length!==b?.length||!a?.length)return -1;let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return Math.max(-1,Math.min(1,s))}
export function rank(vector,textVectors){const logits=textVectors.map(v=>cosine(vector,v)*100),max=Math.max(...logits),exp=logits.map(x=>Math.exp(x-max)),sum=exp.reduce((a,b)=>a+b,0);return logits.map((x,i)=>({index:i,score:exp[i]/sum,cosine:x/100})).sort((a,b)=>b.score-a.score)}
// 美客多大类别的零样本判断：返回带官方类目名的候选列表。
export function rankCategories(vector,textVectors){
 return rank(vector,textVectors).map(r=>({...ML_CATEGORIES[r.index],score:r.score,cosine:r.cosine}));
}
export function decision(ranking){const top=ranking[0],second=ranking[1],entry=CATALOG[top.index];const confident=entry[0]!=='仅有包装'&&entry[0]!=='无法辨认的产品'&&entry[1]!=='Otras Categorías'&&top.score>=.80&&top.score-second.score>=.20&&top.cosine>=.22;return {suggestedName:entry[0],suggestedCategory:entry[1],status:confident?'已识别':'待确认',score:top.score,confident}}
export function sourceGroup(filename){const m=filename?.match(/^表格第(\d+)行_/);return m?'表格第'+Number(m[1])+'行（多角度）':null}

// 以图库里「已标注的同款分组」为原型：组内多角度向量取平均，比单张更稳。
export function prototypes(library){
 const groups=new Map();
 for(const x of library){
  const v=x.ai?.embedding;if(!Array.isArray(v)||v.length!==512)continue;
  const l=effectiveLabel(x);
  if(!groups.has(l.group))groups.set(l.group,{group:l.group,name:l.name,category:l.category,sum:new Float64Array(512),vectors:[],count:0});
  const g=groups.get(l.group);for(let i=0;i<512;i++)g.sum[i]+=v[i];g.vectors.push(v);g.count++;
 }
 return [...groups.values()].filter(g=>g.count).map(g=>({group:g.group,name:g.name,category:g.category,size:g.count,vector:normalize(Array.from(g.sum)),vectors:g.vectors}));
}

// 识别分类 = 图库近邻投票优先，CLIP 文本分类兜底；低把握一律「待确认」，由人工核对。
export function identify(analysis,library){
 const protos=prototypes(library);
 const ml=analysis?.mlCandidates?.[0]||null;
 const mlInfo=ml?{mlCategory:ml.category,mlCategoryZh:ml.categoryZh,mlScore:ml.score}:{};
 const sub=analysis?.mlSubs?.[0]||null;
 const subInfo=sub?{mlSub:sub.sub,mlSubZh:sub.subZh,mlSubParent:sub.parent,mlSubScore:sub.score}:{};
 if(!protos.length||!Array.isArray(analysis?.embedding))return {...analysis,...mlInfo,...subInfo,method:'文本分类',identified:false,candidates:[]};
 // 同款多角度差异大，组得分取「与组内最相似角度」而非平均值，避免被平均稀释。
 const scored=protos.map(p=>{
  const best=Math.max(...p.vectors.map(v=>cosine(analysis.embedding,v)));
  const avg=cosine(analysis.embedding,p.vector);
  return {group:p.group,name:p.name,category:p.category,size:p.size,score:Math.max(best,avg),memberBest:best};
 }).sort((a,b)=>b.score-a.score);
 const top=scored[0],second=scored[1];
 const confident=top.score>=.84&&(!second||top.score-second.score>=.012);
 // 参考名称：图库里最接近的同款分组名，用来告诉用户「这图像什么」，不冒充最终结论。
 const reference=top?{referenceName:top.name,referenceGroup:top.group,referenceCategory:top.category,referenceScore:top.score}:{};
 if(confident)return {...analysis,...mlInfo,...subInfo,...reference,suggestedName:top.name,suggestedCategory:top.category,status:'已识别',score:top.score,confident:true,method:'图库近邻',matchedGroup:top.group,candidates:scored.slice(0,3),identified:true};
 // 名称不能用「形近底图分组名」冒充：那只是视觉近邻，不是识别结论。
 // 只有在 CLIP 产品级词典给出可用名称（非占位词）时才作为名称，否则明确写「未识别」。
 const CLIP_NAME_OK=analysis.suggestedName&&!['仅有包装','无法辨认的产品','产品待确认','Otras Categorías','未识别'].includes(analysis.suggestedName)&&analysis.productScore>=0.10;
 const textName=CLIP_NAME_OK?analysis.suggestedName:'未识别';
 const textCategory=analysis.suggestedCategory&&analysis.suggestedCategory!=='Otras Categorías'?analysis.suggestedCategory:null;
 return {...analysis,...mlInfo,...subInfo,...reference,suggestedName:textName,suggestedCategory:textCategory,status:'待确认',confident:false,method:CLIP_NAME_OK?'产品词典匹配':'未识别（仅形近候选）',matchedGroup:null,candidates:scored.slice(0,3),identified:false};
}

// 底图落库时的标签：有产品目录就用真实品名/类目，避免只存文件名。
export function catalogAnalysis(x,analysis){
 const c=catalogFor(x.filename||x.name);
 if(!c||c.name==='产品待确认')return analysis;
 return {...analysis,suggestedName:c.name,suggestedCategory:c.category,status:'已识别',confident:true,method:'产品目录',score:1};
}

// MobileCLIP-S0（fp32，视觉塔必须 fp32：Apple 重参数化结构被 q8 量化摧毁，实测 R@1 从 0.5834 崩到 0.0168）。
// 检索主信号：留一评测 1486 查询 R@1=0.5834/R@20=0.8822，超 DINOv3-cls(0.5599) 与全部融合组合；
// 跨域真实查询 1.webp 真同款 top1 命中，库外查询分数回落。注意其绝对分数无真值/非真值分离力
// （tools/calibrate-verdict.mjs），只用于排序，verdict 阈值已按新分布重校准（search.mjs 0.75/0.70）。
export const MC_MODEL='Xenova/mobileclip_s0';
export async function createMobileClipEmbedder(lib,progress=()=>{},device=typeof window==='undefined'?'cpu':'wasm',modelBase=null){
 const opts={device,progress_callback:progress};
 if(modelBase){lib.env.allowLocalModels=true;lib.env.localModelPath=modelBase;if(typeof window!=='undefined')lib.env.useBrowserCache=true}
 else lib.env.allowLocalModels=false;
 if(lib.env.backends?.onnx?.wasm)lib.env.backends.onnx.wasm.numThreads=1;
 const processor=await lib.AutoProcessor.from_pretrained(MC_MODEL,opts);
 const model=await lib.CLIPVisionModelWithProjection.from_pretrained(MC_MODEL,{dtype:'fp32',device,progress_callback:progress});
 return {async embed(source){
  const image=await lib.RawImage.read(source);
  const inputs=await processor(image),out=await model(inputs);
  return normalize(Array.from(out.image_embeds.data));
 },dispose:()=>model.dispose()};
}

// DINOv3-vits16（CLS 池化）视觉特征：检索主信号。CLIP 保留做品名识别。
// 留一评测（1486 次，tools/eval-dinov3.mjs）：CLIP R@1=0.4159、DINOv2=0.4415、
// CLIP+DINOv2 融合=0.4684，DINOv3-CLS 单模型=0.5599——单模型超过全部融合组合，
// 且跨域真实查询（1.webp 真同款 top1 命中）与库外查询（分数自然回落）验证通过。
export async function createDinoEmbedder(lib,progress=()=>{},device=typeof window==='undefined'?'cpu':'wasm',modelBase=null){
 const opts={dtype:'q8',device,progress_callback:progress};
 if(modelBase){lib.env.allowLocalModels=true;lib.env.localModelPath=modelBase;if(typeof window!=='undefined')lib.env.useBrowserCache=true}
 else lib.env.allowLocalModels=false;
 if(lib.env.backends?.onnx?.wasm)lib.env.backends.onnx.wasm.numThreads=1;
 const processor=await lib.AutoProcessor.from_pretrained(DINO_MODEL,opts);
 const model=await lib.AutoModel.from_pretrained(DINO_MODEL,opts);
 return {async embed(source){
  const image=await lib.RawImage.read(source);
  const inputs=await processor(image),out=await model(inputs);
  // DINOv3 留一评测：CLS 池化 R@1=0.5599 远超全 token 均值(0.4670)与 DINOv2(0.4415)。
  const data=out.last_hidden_state.data,dim=384;
  const v=new Array(dim);for(let d=0;d<dim;d++)v[d]=data[d];
  return normalize(v);
 },dispose:()=>model.dispose()};
}

export async function createRecognizer(lib,progress=()=>{},device=typeof window==='undefined'?'cpu':'wasm',modelBase=null){
 if(modelBase){lib.env.allowLocalModels=true;lib.env.localModelPath=modelBase;if(typeof window!=='undefined')lib.env.useBrowserCache=true}else{lib.env.allowLocalModels=false}
 if(lib.env.backends?.onnx?.wasm)lib.env.backends.onnx.wasm.numThreads=1;
 const opts={dtype:'q8',device,progress_callback:progress};
 const tokenizer=await lib.AutoTokenizer.from_pretrained(MODEL,opts);
 const processor=await lib.AutoProcessor.from_pretrained(MODEL,opts);
 const text=await lib.CLIPTextModelWithProjection.from_pretrained(MODEL,opts);
 const input=await tokenizer(CATALOG.map(x=>'a product photo of '+x[2]),{padding:true,truncation:true});
 const output=await text(input),data=output.text_embeds.data,dim=output.text_embeds.dims.at(-1),textVectors=CATALOG.map((_,i)=>normalize(data.slice(i*dim,(i+1)*dim)));
 // 美客多大类别的文本向量（同一套 CLIP 文本塔，一次编码）：每类三条模板取平均
 const mlTextList=[],mlOwner=[];
 ML_CATEGORIES.forEach((c,i)=>{const v=ML_CATEGORY_EXAMPLES[c.es];
  const ts=v?['a photo of '+v[0],'a close up product photo of '+v[0],v[1]]:[c.prompt];
  for(const t of ts){mlTextList.push(t);mlOwner.push(i)}});
 const mlInput=await tokenizer(mlTextList,{padding:true,truncation:true});
 const mlOutput=await text(mlInput),mlData=mlOutput.text_embeds.data;
 const mlSum=ML_CATEGORIES.map(()=>new Float64Array(dim)),mlCount=new Array(ML_CATEGORIES.length).fill(0);
 mlOwner.forEach((o,k)=>{const v=mlData.slice(k*dim,(k+1)*dim);for(let d=0;d<dim;d++)mlSum[o][d]+=v[d];mlCount[o]++});
 const mlVectors=mlSum.map((s,i)=>normalize(Array.from(s).map(x=>x/mlCount[i])));
 const subInput=await tokenizer(ML_SUBCATEGORIES.map(c=>c.prompt),{padding:true,truncation:true});
 const subOutput=await text(subInput),subData=subOutput.text_embeds.data;
 const subVectors=ML_SUBCATEGORIES.map((_,i)=>normalize(subData.slice(i*dim,(i+1)*dim)));
 await text.dispose();
 const vision=await lib.CLIPVisionModelWithProjection.from_pretrained(MODEL,opts);
 return {async analyze(source){const image=await lib.RawImage.read(source),inputs=await processor(image),out=await vision(inputs);const embedding=normalize(out.image_embeds.data);
  const mlAll=rankCategories(embedding,mlVectors);
  // 二级类目只在判定出的父类目下面挑，避免出现「切割片→工作制服」这种跨类错配
  const kids=ML_SUBCATEGORIES.map((s,i)=>({s,i})).filter(x=>x.s.parent===mlAll[0].es);
  const subAll=rank(embedding,subVectors);
  const subs=(kids.length?subAll.filter(r=>kids.some(k=>k.i===r.index)):subAll).slice(0,3);
  // 按父类目分组存好二级类目，前端最终用哪个大类就用哪一组（同款命中时大类来自库标签）
  const subByParent={};
  for(const r of subAll){const s=ML_SUBCATEGORIES[r.index];(subByParent[s.parent]=subByParent[s.parent]||[]).push({sub:s.es,subZh:s.zh,parent:s.parent,score:r.score})}
  for(const k of Object.keys(subByParent))subByParent[k]=subByParent[k].slice(0,3);
  const productRank=rank(embedding,textVectors),productTop=productRank[0];
  return {...decision(productRank),productScore:productTop.score,productTopName:CATALOG[productTop.index]?.[0]||null,productTopCategory:CATALOG[productTop.index]?.[1]||null,embedding,model:MODEL,
   mlCandidates:mlAll.slice(0,3).map(c=>({category:c.es,categoryZh:c.zh,score:c.score})),
   mlSubsByParent:subByParent,
   mlSubs:subs.map(r=>({sub:ML_SUBCATEGORIES[r.index].es,subZh:ML_SUBCATEGORIES[r.index].zh,parent:ML_SUBCATEGORIES[r.index].parent,score:r.score}))};
 },dispose:()=>vision.dispose()};
}
