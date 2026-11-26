import fs from 'node:fs/promises';
import path from 'node:path';
const folder='C:\\Users\\Administrator\\Desktop\\包装-产品图';
const checkpoint=path.resolve('work/gallery-import-results.json');
const ranges=[
 [1,4,'白色网状用品','待分类'],[6,6,'蓝色长条配件','待分类'],[7,11,'气弹簧支撑杆','汽配'],[12,14,'金属固定座','五金工具'],[15,17,'轮胎内胎','汽配'],[18,22,'气弹簧支撑杆','汽配'],[23,26,'灯管支架卡座','电子电气'],[27,27,'黑色固定配件','待分类'],[28,30,'白色塑料衬套','五金工具'],[32,32,'黑色固定配件','待分类'],[33,38,'金刚石打磨盘','五金工具'],[39,42,'瓷砖倒角工具','五金工具'],[43,46,'金属燃烧器管','五金工具'],[47,49,'多规格电源转接头','电子电气'],[50,52,'金属轮毂螺母','汽配'],[55,60,'绿色胶带','五金工具'],[61,63,'红色塑料固定件','五金工具'],[64,65,'接线端子套装','电子电气'],[67,72,'交流接触器','电子电气'],[73,76,'继电器接线套装','电子电气'],[77,79,'黑色塑料固定卡扣','汽配'],[80,82,'自粘密封条','汽配'],[83,86,'橡胶衬套与轴承配件','汽配'],[87,91,'汽车开关组件','汽配'],[92,95,'车门内拉手','汽配'],[96,99,'快速夹具','五金工具'],[100,102,'小型切割打磨片套装','五金工具'],[103,106,'黄色液位检查工具','汽配'],[107,109,'金属长轴配件','五金工具'],[111,113,'瓷砖倒角工具套装','五金工具'],[115,119,'金属泵体组件','五金工具'],[120,124,'弹簧吸盘拉杆工具','五金工具'],[126,129,'小型打磨头套装','五金工具'],[130,133,'汽车电磁阀','汽配'],[134,138,'高压清洗机喷杆','五金工具'],[139,142,'镭射装饰膜','家居用品'],[143,146,'汽车接线插头','汽配'],[147,149,'镭射装饰膜','家居用品'],[150,152,'气弹簧支撑杆','汽配'],[153,157,'充电式长条工作灯','电子电气'],[158,161,'气弹簧支撑杆','汽配'],[163,165,'瓷砖倒角工具套装','五金工具'],[166,169,'汽车金属传感器','汽配'],[171,173,'插座式电气设备','电子电气'],[174,177,'红色反射灯泡','电子电气'],[178,181,'白色密封环','五金工具'],[182,184,'金属搭扣','五金工具'],[185,190,'红色接线端子','电子电气'],[191,192,'弹性绑带及手套套装','五金工具']];
function classify(n){const r=ranges.find(r=>n>=r[0]&&n<=r[1]);return r?[r[2],r[3]]:['产品待确认','待分类']}
async function main(cfg){
 const headers={'OAI-Sites-Authorization':'Bearer '+cfg.token};
 async function api(p,opt={}){const r=await fetch(cfg.url+p,{...opt,headers:{...headers,...opt.headers},signal:AbortSignal.timeout(60000)});if(!r.ok)throw Error('HTTP '+r.status+' '+(await r.text()).slice(0,150));return r.json()}
 const existing=await api('/api/images');console.log('Existing images: '+existing.length);
 let results=[];try{results=JSON.parse(await fs.readFile(checkpoint,'utf8'))}catch{}
 const files=(await fs.readdir(folder)).filter(f=>f.endsWith('.jpg')).sort();
 const known=new Map(existing.map(x=>[x.filename,x]));let imported=0,skipped=0;
 for(const f of files){
  // Resume this import by exact source filename; all repeated images with different source filenames are retained.
  if(known.has(f)){skipped++;continue}
  const n=Number(f.match(/原图(\d+)/)[1]),row=f.match(/第(\d+)行/)[1],col=f.match(/行_(.+)列/)[1];
  const [name,category]=classify(n),form=new FormData();form.append('image',new Blob([await fs.readFile(path.join(folder,f))],{type:'image/jpeg'}),f);
  form.append('name',name+' · '+col+'视图');form.append('category',category);form.append('group',name+' · 表格第'+Number(row)+'行');
  const result=await api('/api/images',{method:'POST',body:form});known.set(f,result);results.push(result);await fs.writeFile(checkpoint,JSON.stringify(results,null,2));imported++;
  if(imported%10===0)console.log('Imported '+imported+' / '+files.length);
 }
 const final=await api('/api/images');const missing=files.filter(f=>!final.some(x=>x.filename===f));
 const sample=final.find(x=>files.includes(x.filename));const response=await fetch(cfg.url+'/images/'+sample.id,{headers,signal:AbortSignal.timeout(30000)});
 if(!response.ok)throw Error('Saved image read failed: '+response.status);
 const bytes=Buffer.from(await response.arrayBuffer());const original=await fs.readFile(path.join(folder,sample.filename));if(!bytes.equals(original))throw Error('Image byte verification failed');
 const summary={source:files.length,imported,resumed:skipped,total:final.length,missing:missing.length,groups:new Set(final.filter(x=>files.includes(x.filename)).map(x=>x.group)).size,verifiedOriginalBytes:true};
 await fs.writeFile('work/gallery-import-summary.json',JSON.stringify(summary,null,2));console.log(JSON.stringify(summary));if(missing.length)process.exitCode=1;
}
if(process.stdin.isTTY)process.stdin.setRawMode(true);process.stdin.setEncoding('utf8');console.log('Ready for import credentials on stdin (hidden).');let input='';process.stdin.on('data',chunk=>{input+=chunk;if(input.includes('\n')||input.includes('\r')){process.stdin.removeAllListeners('data');if(process.stdin.isTTY)process.stdin.setRawMode(false);process.stdin.pause();main(JSON.parse(input.trim())).catch(e=>{console.error(e.message);process.exitCode=1})}});
