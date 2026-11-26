/* 离线批量读底图规格：PaddleOCR v6 tiny（Apache-2.0，MIT 包 ppu-paddle-ocr）
 * 输出 specs-ocr.json：filename -> {text,tokens,counts,curves,models}
 * 线上只带结果 JSON（几百 KB），不再跑 wasm OCR。
 */
import {readdirSync,readFileSync,writeFileSync,existsSync} from 'node:fs';
import path from 'node:path';
import {PaddleOcrService} from '../../../../_ocr_ppu/node_modules/ppu-paddle-ocr/index.js';
import {parseSpecText} from '../site-source/specs.mjs';

const W='D:/GPT/2026-11-21/new-chat-2/work';
const IMG=path.join(W,'deploy-product-library/images');
const M=path.join(W,'_ocr_ppu/models/');
const OUT=path.join(W,'git-handoff/projects/product-image-compare/reports/specs-ocr.json');
const svc=new PaddleOcrService({model:{
 detection:M+'PP-OCRv6_tiny_det.ort',
 recognition:M+'PP-OCRv6_tiny_rec.ort',
 charactersDictionary:M+'ppocrv6_tiny_dict.txt'}});
await svc.initialize();
const files=readdirSync(IMG).filter(f=>/\.(jpe?g|png|webp)$/i.test(f));
const done=existsSync(OUT)?JSON.parse(readFileSync(OUT,'utf8')):{_meta:{engine:'PP-OCRv6 tiny',total:files.length},specs:{}};
let n=0;
for(const f of files){
 if(done.specs[f])continue;
 try{
  const buf=readFileSync(path.join(IMG,f));
  const r=await svc.recognize(new Uint8Array(buf).buffer,{flatten:true});
  const p=parseSpecText(r.text||'');
  done.specs[f]={text:String(r.text||'').slice(0,600),tokens:p.tokens,counts:p.counts,curves:p.curves,models:p.models,poles:p.poles};
 }catch(e){done.specs[f]={error:String(e.message||e).slice(0,120)}}
 n++;
 if(n%50===0){writeFileSync(OUT,JSON.stringify(done));console.log('OCR',n,'/',files.length)}
}
await svc.destroy();
writeFileSync(OUT,JSON.stringify(done));
const withSpec=Object.values(done.specs).filter(x=>x.curves?.length||x.models?.length).length;
console.log('DONE',Object.keys(done.specs).length,'withSpec',withSpec,'->',OUT);
