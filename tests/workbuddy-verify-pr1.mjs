/* WorkBuddy 独立验证：codex PR#1 修复后的五场景行为 + 旧数据兼容（2026-11-26） */
import {parseSpecText,compareSpecs,compareGroupSpecs} from '../site-source/specs.mjs';
import {productResults,searchLibrary,candidateVerdict} from '../site-source/search.mjs';
import {readFileSync} from 'node:fs';
let pass=0,fail=0;
const ok=(name,cond)=>{cond?(pass++,console.log('  PASS',name)):(fail++,console.log('  FAIL',name))};

const model='Xenova/clip-vit-base-patch32';
const vec=(n,s=1)=>[s,Math.sqrt(1-s*s),...Array(n-2).fill(0)];
const row=(id,clip=.9,dino=.8,group=id)=>({id,name:id,group,category:'Herramientas',ai:{model,embedding:vec(512,clip),...(dino===null?{}:{dinoEmbedding:vec(384,dino)})}});
const q={embedding:vec(512),dinoEmbedding:vec(384)};

console.log('【场景1】C16 vs C20 未验证 → 应只提示复核，不硬否决');
const r1=compareSpecs(parseSpecText('C16'),parseSpecText('C20'));
ok('conflict 为空',r1.conflict===null); ok('标记 review',r1.review===true);

console.log('【场景2】C16 vs C20 双侧 verified → 仍可硬排除');
const r2=compareSpecs(parseSpecText('C16',{verified:true}),parseSpecText('C20',{verified:true}));
ok('conflict 非空且 level=spec',!!r2.conflict&&r2.level==='spec');

console.log('【场景3】C16×3 vs C16 → 重复 OCR 不再当 3 件装');
const r3=compareSpecs(parseSpecText('C16 C16 C16'),parseSpecText('C16'));
ok('不误判件数冲突',r3.conflict===null);

console.log('【场景4】真同款 DZ47-63 不误伤（多 token 部分读出）');
const r4=compareSpecs(parseSpecText('DZ47-63 C16 230V 400V'),parseSpecText('DZ47-63 C16'));
ok('无冲突',r4.conflict===null);

console.log('【场景5】一条缺 DINO 不关闭全库 + 强 CLIP 候选不被埋');
const r5=productResults([row('a',.99),row('b',.3,null)],null,q.embedding,q.dinoEmbedding);
ok('首位为 a（RRF 排名正确）',r5[0].image.id==='a');
ok('a 双模型齐 → fused=true',r5[0].fused===true);
ok('b 缺 DINO → 标记 fused=false（CLIP 回退）',r5[1].fused===false);
const r5b=productResults([row('a',.99,null),row('b',.3,.3)],null,q.embedding,q.dinoEmbedding);
ok('强 CLIP 纯模型候选不被弱双模型埋掉',r5b[0].image.id==='a');

console.log('【场景6】旧 overrides.json 记录兼容（无 confirmations/dino 字段）');
const overrides=JSON.parse(readFileSync('../site-source/overrides.json','utf8'));
const sample=Array.isArray(overrides)?overrides[0]:overrides[Object.keys(overrides)[0]];
console.log('  旧记录样例字段:',Object.keys(sample).join(','));
const legacy=JSON.parse(JSON.stringify(row('leg',.9)));
legacy.name=sample.name||legacy.name;
const r6=searchLibrary(q,[legacy],{});
ok('旧记录正常检索',r6.results.length===1);
const v6=candidateVerdict(r6.results[0]);
ok('强证据只到疑似同款且强制人工核对（tentative）',v6.ok===true&&v6.tentative===true);
ok('人工未确认前不算确认同款',v6.text.includes('人工')===true);
const legacy2=JSON.parse(JSON.stringify(legacy));legacy2.userConfirmed=true;
ok('人工确认后 tentative=false',candidateVerdict({...r6.results[0],userConfirmed:true}).tentative===false);

console.log('【场景7】人工否定过滤 + 其它图片不受影响');
const r7=searchLibrary(q,[row('a',.9,.8,'g'),row('b',.8,.7,'g2')],{querySha256:'a'.repeat(64)});
ok('否定前两结果',r7.results.length===2);
r7.results[0].image.confirmations=[{decision:'different',querySha256:'a'.repeat(64),groupId:'g',at:'2026-11-26T01:00:00Z'}];
const r7b=searchLibrary(q,[r7.results[0].image,r7.results[1].image],{querySha256:'a'.repeat(64)});
ok('否定组被过滤',r7b.results.length===1&&r7b.results[0].image.id==='b');
ok('其它组保留',r7b.results[0].image.id==='b');

console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
process.exit(fail?1:0);
