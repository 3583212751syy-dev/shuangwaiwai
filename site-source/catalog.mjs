// 产品底图目录：来自「包装-产品图.xlsx」原图编号区间。
// 文件名形如 表格第0056行_B列_原图192.jpg，其中「原图N」决定产品，「表格第M行」决定同款分组。
// 每张图的 A/B/C 列是同一产品的不同角度，重复图片仍保留。
export const PRODUCT_RANGES = [
  [1, 4, '白色网状用品', 'Otras Categorías'], [6, 6, '蓝色长条配件', 'Otras Categorías'],
  [7, 11, '气弹簧支撑杆', 'Accesorios para Vehículos'], [12, 14, '金属固定座', 'Herramientas'],
  [15, 17, '轮胎内胎', 'Accesorios para Vehículos'], [18, 22, '气弹簧支撑杆', 'Accesorios para Vehículos'],
  [23, 26, '灯管支架卡座', 'Electrónica, Audio y Video'], [27, 27, '黑色固定配件', 'Otras Categorías'],
  [28, 30, '白色塑料衬套', 'Herramientas'], [32, 32, '黑色固定配件', 'Otras Categorías'],
  [33, 38, '金刚石打磨盘', 'Herramientas'], [39, 42, '瓷砖倒角工具', 'Herramientas'],
  [43, 46, '金属燃烧器管', 'Herramientas'], [47, 49, '多规格电源转接头', 'Electrónica, Audio y Video'],
  [50, 52, '金属轮毂螺母', 'Accesorios para Vehículos'], [55, 60, '绿色胶带', 'Herramientas'],
  [61, 63, '红色塑料固定件', 'Herramientas'], [64, 65, '接线端子套装', 'Electrónica, Audio y Video'],
  [67, 72, '交流接触器', 'Electrónica, Audio y Video'], [73, 76, '继电器接线套装', 'Electrónica, Audio y Video'],
  [77, 79, '黑色塑料固定卡扣', 'Accesorios para Vehículos'], [80, 82, '自粘密封条', 'Accesorios para Vehículos'],
  [83, 86, '橡胶衬套与轴承配件', 'Accesorios para Vehículos'], [87, 91, '汽车开关组件', 'Accesorios para Vehículos'],
  [92, 95, '车门内拉手', 'Accesorios para Vehículos'], [96, 99, '快速夹具', 'Herramientas'],
  [100, 102, '小型切割打磨片套装', 'Herramientas'], [103, 106, '黄色液位检查工具', 'Accesorios para Vehículos'],
  [107, 109, '金属长轴配件', 'Herramientas'], [111, 113, '瓷砖倒角工具套装', 'Herramientas'],
  [115, 119, '金属泵体组件', 'Herramientas'], [120, 124, '弹簧吸盘拉杆工具', 'Herramientas'],
  [126, 129, '小型打磨头套装', 'Herramientas'], [130, 133, '汽车电磁阀', 'Accesorios para Vehículos'],
  [134, 138, '高压清洗机喷杆', 'Herramientas'], [139, 142, '镭射装饰膜', 'Hogar, Muebles y Jardín'],
  [143, 146, '汽车接线插头', 'Accesorios para Vehículos'], [147, 149, '镭射装饰膜', 'Hogar, Muebles y Jardín'],
  [150, 152, '气弹簧支撑杆', 'Accesorios para Vehículos'], [153, 157, '充电式长条工作灯', 'Electrónica, Audio y Video'],
  [158, 161, '气弹簧支撑杆', 'Accesorios para Vehículos'], [163, 165, '瓷砖倒角工具套装', 'Herramientas'],
  [166, 169, '汽车金属传感器', 'Accesorios para Vehículos'], [171, 173, '插座式电气设备', 'Electrónica, Audio y Video'],
  [174, 177, '红色反射灯泡', 'Electrónica, Audio y Video'], [178, 181, '白色密封环', 'Herramientas'],
  [182, 184, '金属搭扣', 'Herramientas'], [185, 190, '红色接线端子', 'Electrónica, Audio y Video'],
  [191, 192, '弹性绑带及手套套装', 'Herramientas']
];

// 类目口径：MercadoLibre 墨西哥站官方一级类目（西语原文，来源 mercadolibre.com.mx 首页类目导航）。
export const CATEGORIES = [
 'Herramientas','Hogar, Muebles y Jardín','Electrodomésticos','Electrónica, Audio y Video',
 'Accesorios para Vehículos','Deportes y Fitness','Ropa, Bolsas y Calzado','Mascotas','Bebés',
 'Juegos y Juguetes','Construcción','Industrias y Oficinas','Computación','Celulares y Telefonía',
 'Belleza y Cuidado Personal','Salud y Equipamiento Médico','Arte, Papelería y Mercería',
 'Instrumentos Musicales','Joyas y Relojes','Libros, Revistas y Comics','Música, Películas y Series',
 'Televisores','Cámaras y Accesorios','Consolas y Videojuegos','Agro','Alimentos y Bebidas',
 'Antigüedades y Colecciones','Productos Sustentables','Recuerdos, Cotillón y Fiestas',
 'Inmuebles','Servicios','Otras Categorías'
];

// 默认文件名 = 尚未人工命名。人工编辑过的名称/分组永远优先。
export function isDefaultName(name) {
  return !name || /^表格第\d+行_/.test(name) || /^原图\d+$/.test(name);
}

export function catalogFor(filename) {
  if (!filename) return null;
  const n = Number((String(filename).match(/原图(\d+)/) || [])[1]);
  if (!Number.isFinite(n)) return null;
  const row = (String(filename).match(/第(\d+)行/) || [])[1];
  const col = (String(filename).match(/行_(.+?)列/) || [])[1] || '';
  const hit = PRODUCT_RANGES.find(r => n >= r[0] && n <= r[1]);
  const name = hit ? hit[2] : '产品待确认';
  const category = hit ? hit[3] : 'Otras Categorías';
  const rowNo = row ? Number(row) : null;
  return {
    number: n,
    row: rowNo,
    view: col,
    name,
    category,
    group: hit ? name + ' · 表格第' + (rowNo ?? n) + '行' : '表格第' + (rowNo ?? n) + '行'
  };
}

// 有效标签：人工编辑 > 产品目录 > CLIP 建议。解决线上底图只有文件名、没有产品信息的问题。
export function effectiveLabel(x) {
  const c = catalogFor(x.filename || x.name);
  const manualName = !isDefaultName(x.name) ? x.name : null;
  const manualGroup = x.group && !isDefaultName(x.group) && x.group !== x.name ? x.group : null;
  const fallbackName = manualName || c?.name || (!['无法辨认的产品', '仅有包装'].includes(x.ai?.suggestedName) ? x.ai?.suggestedName : null) || '产品待确认';
  const fallbackCategory = (x.category && x.category !== 'Otras Categorías') ? x.category
    : (c?.category && c.category !== 'Otras Categorías') ? c.category
    : (x.ai?.suggestedCategory && x.ai.suggestedCategory !== 'Otras Categorías') ? x.ai.suggestedCategory
    : 'Otras Categorías';
  return {
    name: fallbackName,
    category: fallbackCategory,
    group: manualGroup || c?.group || x.group || x.id,
    view: c?.view || '',
    source: manualName ? '人工编辑' : c ? '产品目录' : x.ai ? 'CLIP 识别' : '未标注'
  };
}

export function groupProducts(library) {
  const groups = new Map();
  for (const x of library) {
    const l = effectiveLabel(x);
    if (!groups.has(l.group)) groups.set(l.group, { group: l.group, name: l.name, category: l.category, items: [] });
    groups.get(l.group).items.push({ ...x, label: l });
  }
  return [...groups.values()];
}
