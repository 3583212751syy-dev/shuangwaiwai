# -*- coding: utf-8 -*-
"""v403_words.py —— p6 标题「单词裂变」：把词库里的每个词都贴一版出来对比

用户要求
--------
「字体裂变，我要的是把文本单词都裂变」
「文本裂变按照之前把单词都裂变成别的意思的那一个代码」

做法
----
沿用 `src/text_fission.py` 的词库机制（一个词 → 一张变体图，词义各不相同），
但走的是 p6 真正在用的那条**金属尖刺字标**路线：
  1. 用 `make_v329.py pinterest6` 跑一次（P6_WORD_SRC=空字标）→ 得到
     「主体已定稿 + 旧标题已擦净」的底板（省掉重复擦字，10 个词只擦一次）；
  2. 对词库里每个词的字标，按**与原图标题完全相同的排版几何**贴回：
     横向满幅 3513px + 墨迹顶对齐 y=88（原图 MRCHOSR 白墨 x[12,3524] y[88,1800]）；
  3. 输出全分辨率变体 + 一张「原标题 ↔ 各候选词」的标题带对比图。

用法
----
    python src/v403_words.py                 # 生成全部词库变体
    python src/v403_words.py --words RAVEN,IRONVEIL
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJ = Path(__file__).resolve().parent.parent
VIS = PROJ / "jobs" / "_probe"
GLYPH_DIR = VIS / "elemgen_mid"
OUT = PROJ / "jobs" / "v403_wordbank"
OUT.mkdir(parents=True, exist_ok=True)
BASE = OUT / "_base_notitle.jpg"

TGT_W = 3513.0      # 原图标题白墨全幅宽
TOP_Y = 88.0        # 原图标题白墨顶
TEXT_RGB = np.array([238, 240, 248], np.float32)
ORIG = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")

# ---------------------------------------------------------------- 词库
# 每个词都是**另一个意思**（黑金属命名风），不是同一个词的拼写变体。
BANK = [
    # (词, 词义, 字标文件)
    ("RAVEN",      "渡鸦（原字标 n1）",             "w_RAVEN.png"),
    ("VORGRAVEN",  "前墓（原字标 n5）",             "w_VORGRAVEN.png"),
    ("MOURNGRAVE", "哀悼之墓（原字标 n4）",         "w_MOURNGRAVE.png"),
    ("SKARVALD",   "残垣王（原字标 n6）",           "w_SKARVALD.png"),
    ("IRONVEIL",   "铁幕",                          "w_IRONVEIL.png"),
    ("GRAVETIDE",  "墓潮",                          "w_GRAVETIDE.png"),
    ("STORMHELM",  "风暴舵",                        "w_STORMHELM.png"),
    ("ASHREAVER",  "灰烬劫掠者",                    "w_ASHREAVER.png"),
    ("DUSKBANE",   "暮祸",                          "w_DUSKBANE.png"),
    ("THORNMOURN", "荆棘悲悼",                      "w_THORNMOURN.png"),
]


def glyph_to_alpha(path: Path, W: int, H: int):
    """把字标 PNG 转成「全幅贴合 + 顶对齐」的 alpha 层（复制 make_v329 的几何）。

    ⚠️ 极性自动判定（v403 新增）：SDXL+Harrlogos 有时吐**黑字浅底**（如 n1/IRONVEIL），
    有时吐**白字黑底**（如 n4/n5/n6）。旧代码硬取「白色笔画」，遇到黑字浅底会把
    整块背景当笔画 → 成品糊成一团。这里按**图缘中位亮度**判极性：图缘亮 ⇒ 字是暗的
    ⇒ 先反相，再走同一套阈值。
    """
    g = Image.open(path).convert("L")
    ga = np.asarray(g, np.float32) / 255.0
    bw = max(2, int(round(min(g.size) * 0.04)))          # 边框带宽度
    border = np.concatenate([ga[:bw].ravel(), ga[-bw:].ravel(),
                             ga[:, :bw].ravel(), ga[:, -bw:].ravel()])
    inverted = float(np.median(border)) > 0.5
    if inverted:                                         # 底亮 ⇒ 字暗 ⇒ 反相
        ga = 1.0 - ga
    # 反相过的字标（黑字浅底）：浅底常带**渐变/晕影**，反相后残值会以极低 alpha
    # 漏进成品 → 标题后面浮出一块淡灰矩形（肉眼可见）。故用更严的阈值把中灰切掉。
    thr, span = (0.55, 0.35) if inverted else (0.42, 0.30)
    ga = np.clip((ga - thr) / span, 0.0, 1.0)            # 只取白色笔画
    inkm = ga > 0.35
    ys, xs = np.where(inkm)
    if len(ys) == 0:
        raise SystemExit(f"字标无墨迹：{path}")
    by0, by1 = int(ys.min()), int(ys.max()) + 1
    bx0, bx1 = int(xs.min()), int(xs.max()) + 1
    rws = inkm.sum(1)
    dense = np.where(rws > rws.max() * 0.25)[0]          # 字身带（去掉发丝尖刺）
    y0 = max(by0, int(dense.min()) - 55)
    y1 = min(by1, int(dense.max()) + 55)
    sub = ga[y0:y1, bx0:bx1]
    sh, sw = sub.shape
    sc = float(TGT_W) / float(sw)                        # 按原图横向满幅定标
    tw, th = int(round(sw * sc)), int(round(sh * sc))
    gly = Image.fromarray((sub * 255).astype(np.uint8), "L").resize((tw, th), Image.LANCZOS)
    aa = np.asarray(gly, np.float32) / 255.0
    _ri = np.where((sub > 0.35).any(1))[0]
    _ink_top = int(_ri.min()) if len(_ri) else 0
    cy = int(round(TOP_Y - _ink_top * sc))
    cy = max(0, min(cy, H - th))
    a = np.zeros((H, W), np.float32)
    cx0 = (W - tw) // 2
    x1 = min(W, cx0 + tw)
    a[cy:cy + th, cx0:x1] = aa[:, :x1 - cx0]
    return a, (cx0, cy, tw, th)


def make_base() -> None:
    """用空字标跑一次管线，拿到「主体定稿 + 旧标题已擦净」的底板。"""
    if BASE.exists():
        print(f"[base] 已存在 {BASE.name}")
        return
    blank = GLYPH_DIR / "w_BLANK.png"
    if not blank.exists():
        Image.new("RGB", (1024, 1024), (0, 0, 0)).save(blank)
    env = {"P6_WORD_SRC": "w_BLANK.png"}
    print("[base] 跑管线生成无标题底板 ...")
    r = subprocess.run([sys.executable, "make_v329.py", "pinterest6"],
                       cwd=str(PROJ), env={**__import__("os").environ, **env},
                       capture_output=True, text=True)
    print("   ", (r.stdout or "").strip().splitlines()[-1:] or r.stderr[-200:])
    src = PROJ / "jobs" / "router_out_v329" / "pinterest6_variant.jpg"
    if not src.exists():
        raise SystemExit("底板生成失败：找不到 pinterest6_variant.jpg")
    shutil.copy2(src, BASE)


def build_sheet(rows: list[tuple[str, str, Path]]) -> None:
    """标题带对比图：原标题 + 各候选词。"""
    o = Image.open(ORIG).convert("RGB")
    box = (0, 0, o.width, 1900)
    tw = 560
    tiles = []
    c = o.crop(box)
    tiles.append(("ORIG 原标题 MRCHOSR", c.resize((tw, int(c.height * tw / c.width)), Image.LANCZOS)))
    for word, sense, p in rows:
        im = Image.open(p).convert("RGB").crop(box)
        tiles.append((f"{word} · {sense}", im.resize((tw, int(im.height * tw / im.width)), Image.LANCZOS)))
    t = tiles[0][1]
    cols = 4
    rows_n = (len(tiles) + cols - 1) // cols
    gap = 10
    sheet = Image.new("RGB", (tw * cols + gap * (cols + 1),
                              (t.height + 30) * rows_n + gap), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    for i, (lab, im) in enumerate(tiles):
        r, cc = divmod(i, cols)
        x = gap + cc * (tw + gap)
        y = gap + r * (t.height + 30)
        sheet.paste(im, (x, y + 22))
        d.text((x + 4, y + 5), lab, fill=(255, 220, 120))
    sheet.save(OUT / "S_词库对比_标题带.jpg", quality=93)
    print("sheet ->", OUT / "S_词库对比_标题带.jpg", sheet.size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", default="", help="只做这些词（逗号分隔）")
    ap.add_argument("--no-base", action="store_true", help="不重建底板")
    a = ap.parse_args()

    want = [w.strip().upper() for w in a.words.split(",") if w.strip()]
    bank = [b for b in BANK if (not want or b[0] in want)]

    if not a.no_base:
        make_base()

    base = Image.open(BASE).convert("RGB")
    W, H = base.size
    bar = np.asarray(base, np.float32)

    rows = []
    for word, sense, gf in bank:
        gp = GLYPH_DIR / gf
        if not gp.exists():
            print(f"  ! 缺字标 {gf}（跳过 {word}）")
            continue
        al, geo = glyph_to_alpha(gp, W, H)
        o = bar * (1 - al[..., None]) + TEXT_RGB[None, None, :] * al[..., None]
        out = Image.fromarray(np.clip(o, 0, 255).astype(np.uint8), "RGB")
        p = OUT / f"p6_word_{word}.jpg"
        out.save(p, quality=93)
        rows.append((word, sense, p))
        print(f"  ok {word:12s} {sense:12s} slab={geo[2]}x{geo[3]}@x{geo[0]} y{geo[1]}")

    if rows:
        build_sheet(rows)
    (OUT / "bank.json").write_text(
        json.dumps([{"word": w, "sense": s} for w, s, _ in rows], ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"[done] {len(rows)} 版 -> {OUT}")


if __name__ == "__main__":
    main()
