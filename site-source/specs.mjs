/* 读图规格抽取与对比（浏览器 / Node 通用）
 * 目标：识别「外观一样但 C16≠C20、2 只≠1 只」这类 CLIP/DINO 无法区分的细粒度差异。
 * 思路来自 ebiarian/product-image-description-alignment（MIT）：关键特征不一致即否决；
 * 本模块只把「人工核验过」的规格差异用于否决，未经核验的 OCR 差异仅提示人工复核。
 */

// 断路器脱扣曲线/电流规格（C16、C20、B10、D63）。要求独立 token，避免从 CCC16X 之类误取。
const CURVE=/\b([BCD])(\d{1,2})\b/g;
// 常见型号：DZ47-63、iC65N、5SX5、ZB1-63、DZ47s
const MODEL=/\b([A-Z]{1,4}\d{0,3}[A-Z]?[- ]?\d{1,4}[A-Z]?)\b/g;
// 极数 / 相数（1P-4P），只做提示不做硬否决
const POLE=/\b([1-4]P)\b/g;

export function parseSpecText(text,{verified=false,quantity=null}={}){
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
 return {tokens:[...tokens],counts,curves,models,poles,verified:verified===true,quantity:Number.isInteger(quantity)&&quantity>0?quantity:null,
  // 同一规格出现的次数（上限 6，防止 OCR 重复噪声被当成件数）
  curveCount:t=>Math.min(6,counts[t]||0)};
}

export function summarize(parsed){
 if(!parsed)return '';
 const parts=[];
 for(const c of parsed.curves.slice(0,3))parts.push(c);
 for(const m of parsed.models.slice(0,3))parts.push(m);
 if(parsed.poles.length)parts.push(parsed.poles.slice(0,2).join('/'));
 return parts.join(' · ')||'未读到规格';
}

// Only manually verified, unambiguous fields can exclude a candidate.
// OCR text alone does not supply confidence/geometry and remains review evidence.
export function compareSpecs(query,library,opts={}){
 const out={conflict:null,level:null,reason:'',review:false,
  details:{query:summarize(query),library:summarize(library)}};
 if(!query||!library)return {...out,review:true,reason:'缺少可核对的规格证据'};
 const trusted=query.verified===true&&library.verified===true;
 const qc=query.curves||[],lc=library.curves||[],qm=query.models||[],lm=library.models||[];
 const mismatch=(a,b)=>a.length&&b.length&&!a.some(x=>b.includes(x));
 let difference=null,level='spec';
 if(mismatch(qc,lc))difference=qc.join('/')+' ≠ '+lc.join('/');
 else if(mismatch(qm,lm))difference=qm.join('/')+' ≠ '+lm.join('/');
 else if(opts.countCheck!==false&&Number.isInteger(query.quantity)&&query.quantity>0&&
  Number.isInteger(library.quantity)&&library.quantity>0&&query.quantity!==library.quantity){
  difference=query.quantity+'只装 ≠ '+library.quantity+'只装';level='count';
 }
 if(!difference)return out;
 const unambiguous=level==='count'||(mismatch(qc,lc)?qc.length===1&&lc.length===1:qm.length===1&&lm.length===1);
 if(trusted&&unambiguous)return {...out,conflict:difference,level,reason:'已核验规格冲突：'+difference};
 return {...out,review:true,reason:'规格可能不同，请核对原图：'+difference};
}
export function compareGroupSpecs(query,views){
 const checks=(views||[]).map(v=>compareSpecs(query,v));
 if(!checks.length)return {conflict:null,review:true,reason:'该候选缺少规格记录'};
 // Unknown, ambiguous or agreeing views prevent an automatic group-wide rejection.
 if(checks.every(c=>c.conflict))return checks[0];
 return {conflict:null,review:checks.some(c=>c.review||c.conflict),
  reason:checks.some(c=>c.review||c.conflict)?'部分角度规格未知或不一致，组内候选转人工复核':checks[0].reason};
}
