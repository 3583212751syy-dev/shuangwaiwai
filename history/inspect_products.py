from pathlib import Path
from zipfile import ZipFile
from PIL import Image,ImageOps,ImageDraw
import json,xml.etree.ElementTree as E
p=Path(__file__).parent
out=p/'product-review';out.mkdir(exist_ok=True)
with ZipFile(p/'product-source.xlsx') as z:
 media=sorted([n for n in z.namelist() if n.startswith('xl/media/') and not n.endswith('/')],key=lambda n:int(''.join(filter(str.isdigit,Path(n).stem)) or 0))
 for n in media:(out/Path(n).name).write_bytes(z.read(n))
 print('images',len(media))
 print('sheet',z.read('xl/workbook.xml').decode()[:2000])
 root=E.fromstring(z.read('xl/drawings/drawing1.xml'))
 print('drawing',E.tostring(root,encoding='unicode')[:1800])
 for batch in range(0,len(media),48):
  canvas=Image.new('RGB',(1200,1400),'#ffffff');d=ImageDraw.Draw(canvas)
  for i,n in enumerate(media[batch:batch+48]):
   im=Image.open(out/Path(n).name).convert('RGB');im.thumbnail((190,146))
   x=(i%6)*200;y=(i//6)*175
   canvas.paste(im,(x+(200-im.width)//2,y));d.text((x+8,y+148),Path(n).name,fill='black')
  canvas.save(out/f'review-{batch//48+1}.jpg')
 (out/'media.json').write_text(json.dumps(media),encoding='utf8')
