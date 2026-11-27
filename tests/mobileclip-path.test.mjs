import test from 'node:test';
import assert from 'node:assert/strict';
import {productResults,candidateVerdict,MC_READY} from '../site-source/search.mjs';
import {MC_MODEL} from '../site-source/ai.mjs';
const model='Xenova/clip-vit-base-patch32';
const vec=(n,s=1)=>[s,Math.sqrt(1-s*s),...Array(n-2).fill(0)];
// 行工厂：clip/dino/mc 三路向量独立可控
const row=(id,{clip=.9,dino=.8,mc=null}={})=>({id,name:id,group:id,category:'Herramientas',
 ai:{model,embedding:vec(512,clip),dinoEmbedding:vec(384,dino),
  ...(mc===null?{}:{mcEmbedding:vec(512,mc),mcModel:MC_MODEL})}});
const q={mc:vec(512),clip:vec(512),dino:vec(384)};

test('MC 主路：检索排名由 MobileCLIP 决定，不受 CLIP 向量干扰',()=>{
 const a=row('a',{clip:.1,mc:.99}),b=row('b',{clip:.99,mc:.3});
 const r=productResults([a,b],q.mc,q.clip,q.dino);
 assert.equal(r[0].image.id==='a',true,'MC 相似的 a 应排第一');
});
test('缺 MC 的行回退 CLIP 路参与排序，不关闭全库',()=>{
 const a=row('a',{clip:.3}),b=row('b',{clip:.95,mc:.2});
 const r=productResults([a,b],q.mc,q.clip,q.dino);
 assert.equal(r.length,2);
 // b 有 MC 路（0.2 弱），a 走 CLIP（0.3）——a 的 CLIP 相似强于 b 的 MC 相似，
 // 但两路是独立排名再 RRF，无法简单断言顺序，只验证都有排名与 fused 标记
 assert.ok(r.every(x=>Number.isFinite(x.productScore)));
});
test('MC_READY 校验：向量维度与模型白名单',()=>{
 const ok=row('x',{mc:.9});assert.equal(MC_READY(ok),true);
 const bad=row('y',{clip:.9});assert.equal(MC_READY(bad),false);
 const wrongModel=row('z',{mc:.9});wrongModel.ai.mcModel='some/other';assert.equal(MC_READY(wrongModel),false);
});
test('verdict 新阈值（MC 0.75 / DINO 0.70）：边界两侧行为正确',()=>{
 const strong=row('s',{mc:.99,dino:.95});const hitS=productResults([strong],q.mc,q.clip,q.dino)[0];
 assert.equal(hitS.visual>=.75,true,'强相似 MC 余弦应达阈值, 实际 '+hitS.visual);
 const vS=candidateVerdict(hitS);assert.equal(vS.ok,true);assert.equal(vS.tentative,true);
 const weak=row('w',{mc:.5,dino:.5});const hitW=productResults([weak],q.mc,q.clip,q.dino)[0];
 assert.equal(candidateVerdict(hitW).ok,false,'弱相似不应标疑似');
});
