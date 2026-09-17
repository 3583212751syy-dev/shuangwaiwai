"""sheet_mk: 通用 2xN 对比图生成器（可命令行复用）。

用法:
  python src/sheet_mk.py out.jpg "标题" crop:x0,y0,x1,y1 \
      "标签1=路径1" "标签2=路径2" ...

crop 传 `full` 表示整图缩放；否则 x0,y0,x1,y1 表示 1:1 裁切框。
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(sz):
    for p in (r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc",
              r"C:\Windows\Fonts\arialbd.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def main():
    out = Path(sys.argv[1])
    title = sys.argv[2]
    crop = sys.argv[3]
    items = []
    for a in sys.argv[4:]:
        lab, _, p = a.partition("=")
        items.append((lab, Path(p)))
    n = len(items)
    cols = 2 if n <= 4 else 3
    rows = (n + cols - 1) // cols

    ims = [Image.open(p).convert("RGB") for _, p in items]
    if crop == "full":
        cell = 560
    else:
        x0, y0, x1, y1 = [int(v) for v in crop.split(",")]
        cell = x1 - x0
        ims = [im.crop((x0, y0, min(x1, im.width), min(y1, im.height)))
               for im in ims]
        for im in ims:
            if im.size != (cell, cell):
                im = im.resize((cell, cell), Image.LANCZOS)

    pad, lab = 10, 40
    sheet = Image.new("RGB", (cols * cell + pad * (cols + 1),
                              rows * (cell + lab) + pad * (rows + 1) + 30),
                      (24, 24, 26))
    dr = ImageDraw.Draw(sheet)
    dr.text((pad, 4), title, fill=(255, 210, 90), font=font(22))
    for i, (lb, _) in enumerate(items):
        im = ims[i]
        if im.size != (cell, cell):
            im = im.resize((cell, cell), Image.LANCZOS)
        x = pad + (i % cols) * (cell + pad)
        y = 30 + pad + (i // cols) * (cell + lab + pad)
        sheet.paste(im, (x, y))
        dr.text((x + 4, y + cell + 6), lb, fill=(240, 240, 240), font=font(24))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(out), quality=94)
    print(f"[sheet] {out} {sheet.size}")


if __name__ == "__main__":
    main()
