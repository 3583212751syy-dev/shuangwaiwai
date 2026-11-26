import assert from 'node:assert/strict';
import {searchLibrary} from './image-compare/search.mjs';
async function run(c){const headers={'OAI-Sites-Authorization':'Bearer '+c.token};async function read(p){const r=await fetch(c.url+p,{headers,signal:AbortSignal.timeout(45000)});assert.equal(r.status,200);return r}
 const html=await (await read('/')).text();assert.ok(html.includes('添加底图'));assert.ok(html.includes('上传图片找同款'));assert.ok(!html.includes('查看待确认'));
 const script=await (await read('/ai-client.mjs')).text();assert.ok(script.includes("addEventListener('paste'"));assert.ok(script.includes('area.ondrop'));assert.ok(script.includes('searchLibrary'));
 assert.ok((await (await read('/search.mjs')).text()).includes('forceAll'));
 const all=await (await read('/api/images')).json();assert.equal(all.length,182);assert.equal(all.filter(x=>x.ai?.embedding?.length===512).length,182);
 for(const x of [all[0],all[50],all[100]]){const result=searchLibrary(x.ai,all);assert.ok(result.results.some(r=>r.image.id===x.id||r.views.some(v=>v.id===x.id)));assert.ok(result.results[0].score>.99)}
 console.log(JSON.stringify({publishedUI:'simple two-entry',savedImages:all.length,productGroups:new Set(all.map(x=>x.group)).size,featureSearch:'passed',pasteAndDropHandlers:'present'}));
}
if(process.stdin.isTTY)process.stdin.setRawMode(true);process.stdin.setEncoding('utf8');console.log('Ready for verification credentials (hidden).');let input='';process.stdin.on('data',x=>{input+=x;if(/[\r\n]/.test(input)){process.stdin.removeAllListeners('data');if(process.stdin.isTTY)process.stdin.setRawMode(false);process.stdin.pause();run(JSON.parse(input.trim())).catch(e=>{console.error(e.message);process.exitCode=1})}});
