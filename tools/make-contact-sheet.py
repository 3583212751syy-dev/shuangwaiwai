"""把提取出的原图做成带编号的联系表，便于逐张目检筛选。
原图只读、不重编码、不剪裁；联系表是独立缩略图输出。
Usage: python make-contact-sheet.py <originals_dir> <inventory.json> <out_dir> [cols] [rows] [cell]
"""
import json, sys, math
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont

originals=Path(sys.argv[1]); inventory=json.loads(Path(sys.argv[2]).read_text(encoding='utf8'))
out=Path(sys.argv[3]); out.mkdir(parents=True,exist_ok=True)
cols=int(sys.argv[4]) if len(sys.argv)>4 else 8
rows=int(sys.argv[5]) if len(sys.argv)>5 else 8
cell=int(sys.argv[6]) if len(sys.argv)>6 else 240
label=30; per=cols*rows; cw=cell; ch=cell+label
try: font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',17)
except: font=ImageFont.load_default()

items=sorted(inventory,key=lambda x:int(''.join(c for c in Path(x['file']).stem if c.isdigit()) or 0))
# 按 image 后的数字排序，保证编号稳定
items.sort(key=lambda x:int(Path(x['file']).stem.replace('image','') or 0))
pages=math.ceil(len(items)/per)
for page in range(pages):
    batch=items[page*per:(page+1)*per]
    sheet=Image.new('RGB',(cols*cw,rows*ch),(245,245,245))
    d=ImageDraw.Draw(sheet)
    for i,item in enumerate(batch):
        name=Path(item['file']).name
        try:
            with Image.open(originals/name) as im:
                im=ImageOps.exif_transpose(im).convert('RGB')
                thumb=ImageOps.contain(im,(cw-8,cell-8))
            x=(i%cols)*cw+(cw-thumb.width)//2; y=(i//cols)*ch+(cell-thumb.height)//2
            sheet.paste(thumb,(x,y))
        except Exception as e:
            d.text(((i%cols)*cw+8,(i//cols)*ch+20),'DECODE FAIL',fill=(200,0,0),font=font)
        d.rectangle(((i%cols)*cw,(i//cols)*ch+cell,((i+1)%cols)*cw if (i+1)%cols else cols*cw,(i//cols+1)*ch),fill=(30,40,35))
        no=int(name.replace('image','').replace('.jpeg','').replace('.jpg','').replace('.png','') or 0)
        d.text(((i%cols)*cw+6,(i//cols)*ch+cell+5),f'{i+1+page*per:04d} #{no}',fill=(255,255,255),font=font)
    dest=out/f'contact-{page+1:03d}.jpg'
    sheet.save(dest,quality=88)
    if page in (0,pages//2,pages-1):print('wrote',dest,'items',len(batch))
print(json.dumps({'pages':pages,'per_page':per,'cols':cols,'rows':rows,'cell':cell,'total':len(items)},ensure_ascii=False))
