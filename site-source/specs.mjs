/* 读图规格抽取与对比（浏览器 / Node 通用）
 * 目标：解决「外观一样但 C16≠C20、2 只≠1 只」这类 CLIP/DINO 无法区分的细粒度差异。
 * 思路来自 ebiarian/product-image-description-alignment（MIT）：关键特征不一致即否决；
 * 本模块额外保留同一型号在图中出现的次数，用来表示同画面里的同款件数。
 */

// 断路器脱扣曲线/电流规格：C16、C20、B10、D63。要求独立 token，避免从 CCC16X 之类误取。
const CURVE=/\b([BCD])(\d{1,2})\b/g;
// 常见型号：DZ47-63、DZ47-63、iC65N、5SX5、ZB1-63、DZ47s
const MODEL=/\b([A-Z]{1,4}\d{0,3}[A-Z]?[- ]?\d{1,4}[A-Z]?)\b/g;
// 极数 / 中文数量描述，只做提示不做硬否决
const POLE=/\b([1-4]P)\b/g;

export function parseSpecText(text){
 const raw=String(text||'').toUpperCase().replace(/\u00a0/g,' ');
 const counts=Object.create(null),tokens=new Set();
 const add=(t)=>{if(!t)return;t=String(t).replace(/\s+/g,'');if(!t||t.length>14)return;counts[t]=(counts[t]||0)+1;tokens.add(t)};
 for(const m of raw.matchAll(CURVE))add(m[0]);
 for(const m of raw.matchAll(MODEL)){
  const t=m[1].replace(/\s/g,'');
  // 单独的 C16/D63 由 CURVE 负责；过滤纯数字与过长串
  if(!/^[BCD]\d{1,2}$/.test(t)&&/[A-Z]/.test(t)&&/\d/.test(t))add(t);
 }
 const curves=[...tokens].filter(t=>/^[BCD]\d{1,2}$/.test(t))
  .sort((a,b)=>(counts[b]-counts[a])||a.localeCompare(b));
 const models=[...tokens].filter(t=>!/^[BCD]\d{1,2}$/.test(t)&&![1,2,3,4].includes(Number(t.slice(0,1))))
  .sort((a,b)=>(counts[b]-counts[a])||a.localeCompare(b));
 const poles=[...new Set([...raw.matchAll(POLE)].map(m=>m[1]))];
 return {tokens:[...tokens],counts,curves,models,poles,
  // 同一规格读到的次数（上限 6，避免 OCR 重复噪声被当成件数）
  curveCount:t=>Math.min(6,counts[t]||0)};
}

export function summarize(parsed){
 if(!parsed)return '';
 const parts=[];
 for(const c of parsed.curves.slice(0,3))parts.push(c+(parsed.counts[c]>1?' ×'+Math.min(6,parsed.counts[c]):''));
 for(const m of parsed.models.slice(0,3))parts.push(m);
 if(parsed.poles.length)parts.push(parsed.poles.slice(0,2).join('/'));
 return parts.join(' · ')||'未读到规格';
}

/* 型号/规格硬对比。保守：只在两边都读到明确规格时才否决。
 * 返回 {conflict,level,reason,details}
 * level: 'spec' 明确规格冲突 | 'count' 数量冲突 | null
 */
export function compareSpecs(query,library,opts={}){
 const out={conflict:null,level:null,reason:'',details:{query:summarize(query),library:summarize(library)}};
 if(!query||!library)return out;
 const qc=query.curves||[],lc=library.curves||[];
 const qm=query.models||[],lm=library.models||[];
 // ① 脱扣曲线/电流规格：C16 vs C20 是最可靠的否决信号
 if(qc.length&&lc.length){
  const shared=qc.filter(x=>lc.includes(x));
  if(!shared.length){
   out.conflict=qc[0]+' ≠ '+lc[0];out.level='spec';
   out.reason='读图规格冲突：对比图是 '+out.details.query+'，底图是 '+out.details.library+'。型号/电流规格不同，按非同款处理';
   return out;
  }
  // 有共同型号时再看件数：两个独立 C16 vs 一个 C20 已在上面否决；
  // 两个 C16 vs 一个 C16 才是数量冲突。
  const common=shared[0],qn=Math.min(6,query.counts[common]||0),ln=Math.min(6,library.counts[common]||0);
  if(qn>=2&&ln>=1&&qn!==ln&&opts.countCheck!==false){
   out.conflict=common+' ×'+qn+' ≠ ×'+ln;out.level='count';
   out.reason='读图数量冲突：同一规格 '+common+' 在对比图出现 '+qn+' 次、底图出现 '+ln+' 次；单只与多只组合不是同一件商品';
  }
 }
 // ②  alphanumeric 型号：要求两边都读到至少 2 个不同型号才下结论，避免 OCR 单字误杀
 if(!out.conflict&&qm.length>=2&&lm.length>=2){
  const hit=qm.find(x=>lm.includes(x));
  if(!hit){
   out.conflict=qm[0]+' ≠ '+lm[0];out.level='spec';
   out.reason='读图型号冲突：对比图 '+out.details.query+'，底图 '+out.details.library+'；关键型号不一致，按非同款处理';
  }
 }
 return out;
}
