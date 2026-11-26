// 把 761 组底图的旧分类（汽配/五金工具/电子电气/家居用品/宠物用品/待分类）
// 重映射到 MercadoLibre 墨西哥站官方一级类目（西语原文，来源 mercadolibre.com.mx 首页类目导航）。
// 顺序：① 明确覆盖表（人工判定的错例）→ ② 具体关键词规则 → ③ 旧大类兜底 → ④ 包装/不可辨认 → Otras Categorías
// 只改 category，不改人工定的产品名称与分组。
import {readFileSync,writeFileSync} from 'node:fs';

export const ML_TOP=[
 'Herramientas','Hogar, Muebles y Jardín','Electrodomésticos','Electrónica, Audio y Video',
 'Accesorios para Vehículos','Deportes y Fitness','Ropa, Bolsas y Calzado','Mascotas','Bebés',
 'Juegos y Juguetes','Construcción','Industrias y Oficinas','Computación','Celulares y Telefonía',
 'Belleza y Cuidado Personal','Salud y Equipamiento Médico','Arte, Papelería y Mercería',
 'Instrumentos Musicales','Joyas y Relojes','Libros, Revistas y Comics','Música, Películas y Series',
 'Televisores','Cámaras y Accesorios','Consolas y Videojuegos','Agro','Alimentos y Bebidas',
 'Antigüedades y Colecciones','Productos Sustentables','Recuerdos, Cotillón y Fiestas',
 'Inmuebles','Servicios','Otras Categorías'
];

// ① 人工判定的错例覆盖（关键词出现在名称里即生效）
const OVERRIDE=[
 ['鼓风机','Industrias y Oficinas'],['离心风机','Industrias y Oficinas'],
 ['钢珠','Herramientas'],['球阀','Herramientas'],['金属网状球','Herramientas'],
 ['划线针','Herramientas'],['标记笔','Herramientas'],['触控笔','Herramientas'],
 ['金属笔','Herramientas'],
 ['橡胶帽','Herramientas'],['胶帽','Herramientas'],['脚套','Accesorios para Vehículos'],
 ['橡胶塞','Herramientas'],
 ['包装小件','Otras Categorías'],['包装盒','Otras Categorías'],['包装袋','Otras Categorías'],
 ['小包装盒','Otras Categorías'],['袋装小配件','Otras Categorías'],['吸塑包装','Otras Categorías'],
 ['标签零件袋','Otras Categorías'],
 ['包装电子元件','Electrónica, Audio y Video'],['包装电子模块','Electrónica, Audio y Video'],
 ['包装电子件','Electrónica, Audio y Video'],
 ['金属小零件','Herramientas'],['小五金','Herramientas'],['紧固件','Herramientas'],
 ['透明塑料片','Herramientas'],['透明塑料条','Herramientas'],['塑料片','Herramientas'],
 ['织带','Herramientas'],['皮质套','Hogar, Muebles y Jardín'],['皮革样片','Hogar, Muebles y Jardín'],
 ['布料样片','Hogar, Muebles y Jardín'],
 ['法兰','Herramientas'],['阀门','Herramientas'],['金属球','Herramientas'],
 ['绿植','Hogar, Muebles y Jardín'],['防水布','Construcción'],
 ['运动胶带','Deportes y Fitness'],['足球绷带','Deportes y Fitness'],
 ['胶水','Herramientas'],['胶粘','Herramientas'],
 ['地漏','Construcción'],['下水器','Construcción'],['下水管','Construcción']
];

// ② 关键词规则（用足够具体的词，避免单字误命中）
const RULES=[
 ['Accesorios para Vehículos',['刹车','制动','轮毂','气弹簧','车门','后视镜','保险杠','仪表台','汽车开关','雨刮','挡泥','车牌','排气管','皮带轮','火花塞','点火线','车载','座套','脚垫','方向盘','车锁','机油','滤清器','大灯','尾灯','转向灯','气嘴','减震','半轴','轮毂螺母','轮胎内胎','内胎','汽车电磁阀','汽车传感器','汽车继电器','车用','摩托车','电动车','保险丝','节气门','化油器','马达','发电机','雨刷','挡风','汽修','洗车']],
 ['Electrónica, Audio y Video',['接线端子','继电器','接触器','电源适配','电源转接','适配器','灯泡','灯管','灯带','LED','开关面板','插座','电路板','线缆','数据线','控制器','变压器','万用表','示波器','工作灯','红外','电热','蜂鸣器','插排','延长线','插座式','电表','仪表盘','解码器','点火器','点火线圈','传感器线','电机','马达电机','充电器','电源线','排插','电气装置']],
 ['Herramientas',['切割片','打磨片','磨片','砂纸','砂带','金刚石','钻头','螺丝刀','扳手','内六角','钳','锉','磨头','夹具','卡尺','卷尺','胶带','切割机','倒角','电钻','手电钻','起子','套筒','卸扣','膨胀螺丝','螺丝','螺栓','螺母','垫片','垫圈','密封圈','密封环','O型圈','卡簧','轴承','弹簧','水管','软管','管件','接头','泵体','喷杆','打气','修表','焊接','电烙铁','磨削','钻孔','工具','撬棒','刮刀','钢丝刷','喷枪','高压清洗','油管','喷嘴','螺丝刀组','棘轮','套筒扳手','千斤顶','拉马','卡钳','玻璃刀','开槽','抹刀','吊带','钻夹头','切割头','砂纸机','角磨机','打蜡','抛光','清洁刷','钢丝球','工业风扇','鼓风机','法兰','阀门','球阀','钢珠','金属球','划线针','标记笔','触控笔','金属笔','橡胶帽','胶帽','橡胶塞','织带','金相','显微镜']],
 ['Industrias y Oficinas',['工业风扇','鼓风机','工厂','工装','安全帽','劳保','工业用','设备配件','工业配件','纺织机械','车间']],
 ['Construcción',['建材','瓷砖','地砖','水泥','建筑材料','工地','电线管','排水管','水暖','装修','防水布','填缝','美缝']],
 ['Deportes y Fitness',['乒乓球','篮球','足球','羽毛球','哑铃','跳绳','瑜伽','健身','球拍','泳镜','自行车','单车','运动护具','滑板','钓鱼','帐篷','登山','跑步']],
 ['Ropa, Bolsas y Calzado',['衬衫','T恤','外套','裤子','长裙','球鞋','鞋袜','围巾','腰带','背包','手袋','钱包','拉链','布套','布料样片','皮革样片','工作服','休闲服','运动服','箱包','布包','衣物','织带布']],
 ['Mascotas',['宠物','狗','猫','喂食','牵引','胸背','宠物衣','猫砂','宠物碗','咬胶','宠物梳','洁齿棒']],
 ['Bebés',['婴儿','宝宝','奶瓶','婴儿车','尿布','安抚奶嘴']],
 ['Juegos y Juguetes',['玩具','积木','拼图','娃娃','模型车','遥控车']],
 ['Salud y Equipamiento Médico',['按摩球','按摩器','理疗','血压计','体温计','雾化','美容仪','医疗设备']],
 ['Computación',['键盘','鼠标','打印机','硬盘','U盘','路由器','显示器','办公耗材']],
 ['Celulares y Telefonía',['手机壳','手机支架','平板支架','充电头']],
 ['Hogar, Muebles y Jardín',['收纳','储物','衣架','垃圾袋','拖把','扫把','餐具','茶壶','花瓶','摆件','墙贴','桌布','窗帘','地毯','沙发','收纳盒','厨房用品','浴室','洗手池','水龙头','下水器','排水器','花盆','园艺','清洁用品','网袋','网套','礼盒','礼袋','泡沫箱','泡沫','滚筒','滚轮','丝带','封箱','纸巾','纸巾盒','洗涤','晾衣','插花','香薰']],
 ['Arte, Papelería y Mercería',['文具','美术','画材','印章','订书机','文件夹','文件袋','名片','圆珠笔','笔记本','作业本','手工']],
 ['Electrodomésticos',['洗衣机','冰箱','空调','吸尘器','电饭煲','豆浆机','破壁机','榨汁机','微波炉','烤箱','电风扇','带线风扇','风扇电机','搅拌机','咖啡机','加湿器','净水器','饮水机','家用电器','取暖器','电暖器']],
 ['Instrumentos Musicales',['琴弦','吉他','音准','节拍器','乐器','琴线']],
 ['Joyas y Relojes',['戒指','项链','耳环','手表','挂钟']],
 ['Alimentos y Bebidas',['食品','饮料','零食','调味','胶水','蜡']],
 ['Agro',['灌溉','种植','大棚','农机']],
 ['Cámaras y Accesorios',['相机','镜头','三脚架','滤镜']],
 ['Otras Categorías',['工业零件','机器部件','不明','无法辨认','小件','小配件','透明塑料条','塑料片','圆柱件','扁平件','块状物','小圆片','小环','珠子','小瓶','滴管','小容器','小零件']]
];
const FALLBACK={'汽配':'Accesorios para Vehículos','五金工具':'Herramientas','电子电气':'Electrónica, Audio y Video',
 '家居用品':'Hogar, Muebles y Jardín','宠物用品':'Mascotas','待分类':'Otras Categorías'};

const p='work/labels-manual.json';
// 始终从 remap 前的备份读取，保证可重复执行（幂等）：直接读已改过的文件会让 FALLBACK 失效
const SRC='work/labels-manual.bak-before-remap.json';
const doc=JSON.parse(readFileSync(SRC,'utf8'));
const L=doc.labels;
const stats={};const changes=[];
for(const [k,v] of Object.entries(L)){
 const raw=v.name||'';
 const name=raw.replace(/（待复核）/g,'').replace(/[（(].*?[)）]/g,'');
 let cat=null;
 for(const [w,c] of OVERRIDE){if(name.includes(w)){cat=c;break}}
 if(!cat)for(const [c,kws] of RULES){if(kws.some(w=>name.includes(w))){cat=c;break}}
 if(!cat&&/包装|盒装|纸盒/.test(name))cat='Otras Categorías';
 if(!cat)cat=FALLBACK[v.category]||'Otras Categorías';
 if(cat!==v.category){changes.push({idx:k,from:v.category,to:cat,name:raw});v.category=cat}
 stats[cat]=(stats[cat]||0)+1;
}
writeFileSync(p,JSON.stringify(doc,null,2));
writeFileSync('reports/category-remap.json',JSON.stringify({
 date:new Date().toISOString().slice(0,10),source:'mercadolibre.com.mx 首页类目导航（官方一级类目）',
 total:Object.keys(L).length,stats,changes},null,2));
console.log('分布');
for(const [c,n] of Object.entries(stats).sort((a,b)=>b[1]-a[1]))console.log('  ',c,n);
console.log('兜底到 Otras Categorías',stats['Otras Categorías']||0,'总改动',changes.length);
