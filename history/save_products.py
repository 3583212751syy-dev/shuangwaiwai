from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as E
import csv,json,shutil,hashlib
p=Path(__file__).parent
out=p/'export'/'包装-产品图';out.mkdir(parents=True,exist_ok=True)
excluded={5:'外层包装遮挡产品',31:'只有封闭纸盒',53:'只有包装材料',54:'只有包装材料',66:'包装标签遮挡实际产品',110:'盒内主要产品被手套和包装遮挡',114:'只有封闭纸箱',125:'标签遮挡实际产品',162:'盒内主要产品被手套和包装遮挡',170:'只有外包装，产品不可见'}
with ZipFile(p/'product-source.xlsx') as z:
 root=E.fromstring(z.read('xl/drawings/drawing1.xml'))
 rels={x.get('Id'):x.get('Target') for x in E.fromstring(z.read('xl/drawings/_rels/drawing1.xml.rels'))}
 rows=[]
 for a in root:
  blip=a.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip')
  if blip is None:continue
  rid=blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed');target=rels[rid]
  orig=Path(target).name;n=int(''.join(filter(str.isdigit,Path(orig).stem)))
  row=int(a.find('{*}from/{*}row').text)+1;col=int(a.find('{*}from/{*}col').text)
  letter='';c=col+1
  while c:c,t=divmod(c-1,26);letter=chr(65+t)+letter
  name=f'表格第{row:04d}行_{letter}列_原图{n:03d}.jpg'
  decision='跳过' if n in excluded else '保留'
  rows.append({'原始图片':orig,'单元格':f'{letter}{row}','处理':decision,'原因':excluded.get(n,'可辨认产品外形'),'保存文件':name if decision=='保留' else ''})
  if decision=='保留':
   data=z.read('xl/media/'+orig);(out/name).write_bytes(data)
   assert hashlib.sha256((out/name).read_bytes()).digest()==hashlib.sha256(data).digest()
 with (out/'图片筛选清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary={'总原图':len(rows),'已保存':sum(x['处理']=='保留' for x in rows),'已跳过':sum(x['处理']=='跳过' for x in rows),'folder':str(out)}
 (p/'export-summary.json').write_text(json.dumps(summary,ensure_ascii=False),encoding='utf8')
 print(json.dumps(summary,ensure_ascii=True))
