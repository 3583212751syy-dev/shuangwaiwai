import assert from 'node:assert/strict';
import {searchLibrary,productResults} from './search.mjs';
import {MODEL} from './ai.mjs';
const vector=(x,y)=>[x,y,...Array(510).fill(0)];
const record=(id,name,category,embedding,group=id)=>({id,name,category,group,ai:{model:MODEL,suggestedName:name,suggestedCategory:category,embedding}});
const query={suggestedName:'支撑杆',suggestedCategory:'汽配',embedding:vector(1,0)};
// a+b 同组「杆1」两个角度都高分；c/d 单张。产品相似度应让同组多角度胜出。
const a=record('a','支撑杆','汽配',vector(1,0),'杆1'),b=record('b','支撑杆','汽配',vector(.95,.312),'杆1'),c=record('c','开关','汽配',vector(.7,.714)),d=record('d','胶带','五金工具',vector(.9,.436));
let r=searchLibrary(query,[a,b,c,d]);
assert.equal(r.stage,'product');
assert.equal(r.results[0].views.length,2);          // 同组多角度聚合成一个产品
assert.equal(r.results[0].image.id,'a');            // 多角度组胜出
assert.ok(r.results.some(x=>x.image.id==='d'));     // 全库参与，不再按文本标签硬筛
assert.ok(r.confidence==='high');                    // 多角度聚集=高置信
r=searchLibrary(query,[a,b,c,d],{forceAll:true});assert.equal(r.stage,'all');
r=searchLibrary(query,[c,d]);assert.equal(r.results.length,2);
assert.equal(searchLibrary(query,[]).results.length,0);
assert.equal(productResults([a,b],query.embedding).length,1);
// 文本标签错误时不影响排序：错误标签 d 不应因名称被提到前面
r=searchLibrary({...query,suggestedName:'胶带',suggestedCategory:'五金工具'},[a,b,c,d]);
assert.equal(r.results[0].image.id,'a');
console.log('PASS: product-similarity-first search, multi-angle grouping, no text-label hard filter, high confidence on clusters, empty library');
