"""v350c: p4 阈值策略对比图（原图 / 全局旧版 / 局部自适应）。
出两张：① 树区 1:1 放大 2x2；② 整图 2x2。用于肉眼判定"是否还发毛"。
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
D = ROOT / "jobs" / "v350_p4local"
OUT = ROOT / "jobs" / "v350_p4local" / "sheet"
OUT.mkdir(parents=True, exist_ok=True)

PANELS = [
    ("原图(矢量线稿)", SRC),
    ("G0 全局阈值(旧交付)", D / "G0_global(旧交付).jpg"),
    ("L1 块48 无预平滑", D / "L1_b48_lo50_hi180_s0.0.jpg"),
    ("L4 块64 预平滑0.8", D / "L4_b64_lo50_hi200_s0.8.jpg"),
]


def font(sz):
    for p in (r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc",
              r"C:\Windows\Fonts\arialbd.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def grid(items, cell, path, title):
    cols, rows = 2, 2
    pad, lab = 10, 40
    sheet = Image.new("RGB", (cols * cell + pad * (cols + 1),
                              rows * (cell + lab) + pad * (rows + 1)), (24, 24, 26))
    dr = ImageDraw.Draw(sheet)
    f = font(24)
    for i, (t, p) in enumerate(items):
        im = Image.open(p).convert("RGB")
        if im.size != (cell, cell):
            im = im.resize((cell, cell), Image.LANCZOS)
        x = pad + (i % cols) * (cell + pad)
        y = pad + (i // cols) * (cell + lab + pad)
        sheet.paste(im, (x, y))
        dr.text((x + 4, y + cell + 6), t, fill=(240, 240, 240), font=f)
    dr.text((pad, 4), title, fill=(255, 210, 90), font=font(22))
    sheet.save(str(path), quality=94)
    print(f"[sheet] {path} {sheet.size}")


def main():
    crop = (90, 230, 830, 970)          # 740x740 大树区
    items = []
    for t, p in PANELS:
        im = Image.open(p).convert("RGB")
        c = im.crop(crop) if im.size != (740, 740) else im
        tmp = D / f"_c_{p.stem}.jpg"
        c.save(str(tmp), quality=96)
        items.append((t, tmp))
    grid(items, 740, OUT / "S1_树区1to1.jpg", "p4 树区 1:1 对比（740x740 裁切）")

    full = [(t, p) for t, p in PANELS]
    grid(full, 621, OUT / "S2_整图.jpg", "p4 整图对比（1242x1754 缩小）")


if __name__ == "__main__":
    main()
