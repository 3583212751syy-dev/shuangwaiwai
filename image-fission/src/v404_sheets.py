# -*- coding: utf-8 -*-
"""v404_sheets.py —— p6「单词裂变」词库的四张对比拼版（可重复生成）

背景：第 25 轮那四张拼版是**临时脚本一次性拼的**，没入库；第 26 轮发现底板脏导致
全部 10 版重出后，拼版也得重做 → 顺手固化成脚本。

产出（jobs/v403_wordbank/）：
  · S_词库10版总览.jpg      —— ORIG + 10 个词的全幅缩略（看整体观感）
  · S_词库1比1细看.jpg      —— 原图尺度的标题带横条逐版排（看拼写/描边/尖刺）
  · S_字标拼写校对.jpg      —— 纯字标（不带主体）逐字校对拼写
用法：python src/v404_sheets.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "jobs" / "v403_wordbank"
GLY = PROJ / "jobs" / "_probe" / "elemgen_mid"
ORIG = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")

# 与 v403_words.BANK 同序
BANK = [
    ("RAVEN", "渡鸦"), ("VORGRAVEN", "前墓"), ("MOURNGRAVE", "哀悼之墓"),
    ("SKARVALD", "残垣王"), ("IRONVEIL", "铁幕"), ("GRAVETIDE", "墓潮"),
    ("STORMHELM", "风暴舵"), ("ASHREAVER", "灰烬劫掠者"), ("DUSKBANE", "暮祸"),
    ("THORNMOURN", "荆棘悲悼"),
]
BG = (22, 22, 22)
LAB = (255, 214, 110)


def _fit(im: Image.Image, w: int) -> Image.Image:
    return im.resize((w, max(1, int(round(im.height * w / im.width)))), Image.LANCZOS)


def _grid(tiles, tw, cols, out: Path, cap=26, gap=10):
    """tiles: [(label, Image)] → 等宽网格拼版。"""
    s = [(l, _fit(im, tw)) for l, im in tiles]
    th = max(x[1].height for x in s)
    rows = (len(s) + cols - 1) // cols
    sh = Image.new("RGB", (tw * cols + gap * (cols + 1), (th + cap) * rows + gap), BG)
    d = ImageDraw.Draw(sh)
    for i, (l, im) in enumerate(s):
        r, c = divmod(i, cols)
        x = gap + c * (tw + gap)
        y = gap + r * (th + cap)
        sh.paste(im, (x, y + cap))
        d.text((x + 4, y + 6), l, fill=LAB)
    sh.save(out, quality=93)
    print(f"-> {out.name}  {sh.size}")


def main() -> None:
    o = Image.open(ORIG).convert("RGB")

    # ① 10 版总览（ORIG + 10 词，全幅）
    tiles = [("ORIG 原图 MRCHOSR", o)]
    for w, sense in BANK:
        p = OUT / f"p6_word_{w}.jpg"
        if p.exists():
            tiles.append((f"{w} · {sense}", Image.open(p).convert("RGB")))
    _grid(tiles, 520, 6, OUT / "S_词库10版总览.jpg", cap=24, gap=8)

    # ② 1:1 细看（原图尺度标题带，横条并排）
    band = (0, 40, o.width, 1160)
    tiles = [("ORIG 原标题（原图尺度）", o.crop(band))]
    for w, sense in BANK:
        p = OUT / f"p6_word_{w}.jpg"
        if p.exists():
            tiles.append((f"{w} · {sense}", Image.open(p).convert("RGB").crop(band)))
    _grid(tiles, 1040, 2, OUT / "S_词库1比1细看.jpg", cap=26, gap=8)

    # ③ 字标拼写校对（纯字标）
    tiles = [("ORIG 原标题字标", o.crop((0, 130, o.width, 1100)))]
    for w, _s in BANK:
        p = GLY / f"w_{w}.png"
        if p.exists():
            tiles.append((w, Image.open(p).convert("RGB")))
    _grid(tiles, 620, 3, OUT / "S_字标拼写校对.jpg", cap=24, gap=8)


if __name__ == "__main__":
    main()
