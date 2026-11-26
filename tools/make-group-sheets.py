"""生成分组联系表：每组(来源行)一格，选一张代表图 + 当前建议名/类别，供逐组目检定名。
原图只读，不重编码；缩略图仅为目检用。"""
import json, os, sys
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
work = os.path.join(repo, 'work')
groups = json.load(open(os.path.join(work, 'groups.json'), encoding='utf-8'))
meta = json.load(open(r'D:/GPT/2026-11-21/new-chat-2/work/deploy-product-library/metadata.json', encoding='utf-8'))
orig = r'D:/GPT/2026-11-21/new-chat-2/work/full-extraction/originals'
out = os.path.join(work, 'group-sheets')
os.makedirs(out, exist_ok=True)

by_file = {m['filename']: m for m in meta}
# 每组按图片面积(宽*高)选最大者作代表，面积相同取文件名小者
def rep_of(g):
    best = None
    for it in g['items']:
        f = os.path.basename(it['file'])
        m = by_file.get(f)
        try:
            im = Image.open(os.path.join(orig, f))
            area = im.size[0] * im.size[1]
        except Exception:
            area = 0
        key = (area, -int(f[5:-5]) if f.startswith('image') and f[5:-5].isdigit() else 0)
        if best is None or key > best[0]:
            best = (key, it, m, f)
    return best[1:] if best else (None, None, None)

COLS, ROWS = 8, 6          # 48 组/页
CELL_W, CELL_H = 210, 200
THUMB = 150
font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 12)
fontb = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 14)

total = len(groups)
pages = (total + COLS * ROWS - 1) // (COLS * ROWS)
print(f'groups={total} pages={pages}')

for pg in range(pages):
    sheet = Image.new('RGB', (COLS * CELL_W, ROWS * CELL_H), 'white')
    d = ImageDraw.Draw(sheet)
    for i in range(COLS * ROWS):
        idx = pg * COLS * ROWS + i
        if idx >= total:
            break
        g = groups[idx]
        it, m, f = rep_of(g)
        col, row = i % COLS, i // COLS
        x, y = col * CELL_W, row * CELL_H
        d.rectangle([x + 2, y + 2, x + CELL_W - 2, y + CELL_H - 2], outline='#ccc')
        label = f'#{idx} 第{g["row"]}行 x{len(g["items"])}'
        d.text((x + 6, y + 4), label, fill='#333', font=fontb)
        if f:
            try:
                im = Image.open(os.path.join(orig, f)).convert('RGB')
                im.thumbnail((THUMB, THUMB - 6))
                px = x + (CELL_W - im.width) // 2
                py = y + 24 + (THUMB - im.height) // 2
                sheet.paste(im, (px, py))
            except Exception as e:
                d.text((x + 8, y + 60), f'ERR {e}', fill='red', font=font)
            name = (m or {}).get('name', '')
            cat = (m or {}).get('category', '')
            d.text((x + 6, y + CELL_H - 42), f'{name} | {cat}', fill='#0a5', font=font)
            d.text((x + 6, y + CELL_H - 22), f, fill='#888', font=font)
    p = os.path.join(out, f'groups-{pg + 1:03d}.jpg')
    sheet.save(p, quality=88)
print('done', out)
