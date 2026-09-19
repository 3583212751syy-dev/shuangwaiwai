# -*- coding: utf-8 -*-
"""v407_deliver.py —— 第 27 轮 p6 整套交付（主体重生成 + 新排版文本 + 对比图）

流程
----
1. **重建底板**：用指定主体重生成品跑一次 `make_v329.py pinterest6`
   （`P6_WORD_SRC` 指向不存在的字标 ⇒ 跳过贴字分支 ⇒ 得到"新主体 + 原标题已 LaMa 擦净"的底板）。
   注意：擦字走 Big-LaMa（🔴5/🔴6），**不是色块遮盖**。
2. 对词库每个词：按 🔴43 生成的加高字标 → 极性判定 + 阈值切背景 →
   按**原图密集字身带顶 y=359**落位（🔴44）→ 白色墨迹 alpha 叠加合成。
3. 出对比拼版：标题带 / 左半 1:1 / 右半 1:1 / 整幅。

用法：python src/v407_deliver.py --subject jobs/v407_subject2/p6_v390_s2024_cn0.2_snap.jpg
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from v407_compose import glyph_alpha, compose, GLY, ORIG      # noqa: E402

OUT = ROOT / "jobs" / "v407_title"
BASE = OUT / "_base_v407.jpg"
BANK = ["VORGRAVEN", "RAVEN", "MOURNGRAVE", "SKARVALD", "IRONVEIL", "GRAVETIDE",
        "STORMHELM", "ASHREAVER", "DUSKBANE", "THORNMOURN"]
DESK = Path("E:/Desktop/v407_第27轮交付")


def build_base(subject: Path, force: bool = False) -> Path:
    if BASE.exists() and not force:
        print(f"[base] 复用 {BASE.name}")
        return BASE
    env = {**os.environ,
           "P6_REBIRTH": str(subject),
           "P6_WORD_SRC": "__no_such_glyph__.png"}
    print("[base] 跑 make_v329 pinterest6（新主体 + LaMa 擦字）...")
    r = subprocess.run([sys.executable, "make_v329.py", "pinterest6"],
                       cwd=str(ROOT), env=env, capture_output=True, text=True)
    tail = (r.stdout or "").strip().splitlines()[-3:]
    print("   ", " | ".join(tail))
    src = ROOT / "jobs" / "router_out_v329" / "pinterest6_variant.jpg"
    if not src.exists():
        raise SystemExit("底板生成失败")
    shutil.copy2(src, BASE)
    return BASE


def band(im: Image.Image, w: int = 1500) -> Image.Image:
    c = im.crop((0, 0, im.width, 1560))
    return c.resize((w, int(1560 * w / im.width)), Image.LANCZOS)


def half(im: Image.Image, box, w: int = 520) -> Image.Image:
    c = im.crop(box)
    return c.resize((w, int(c.height * w / c.width)), Image.LANCZOS)


def stack(tiles, w, cap=22, gap=10, out: Path = None, cols=1):
    s = [(l, im if im.width == w else im.resize((w, int(im.height * w / im.width)), Image.LANCZOS))
         for l, im in tiles]
    th = max(x[1].height for x in s)
    rows = (len(s) + cols - 1) // cols
    sh = Image.new("RGB", (w * cols + gap * (cols + 1), (th + cap) * rows + gap), (22, 22, 22))
    d = ImageDraw.Draw(sh)
    for i, (l, im) in enumerate(s):
        r, c = divmod(i, cols)
        x = gap + c * (w + gap)
        y = gap + r * (th + cap)
        sh.paste(im, (x, y + cap))
        d.text((x + 4, y + 4), l, fill=(255, 214, 110))
    if out:
        sh.save(out, quality=93)
    return sh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--words", default=",".join(BANK))
    ap.add_argument("--force-base", action="store_true")
    ap.add_argument("--desk", default=None, help="交付目录（默认 v407_第27轮交付）")
    ap.add_argument("--base-name", default=None, help="底板缓存文件名（换主体时务必换名）")
    ap.add_argument("--tag", default="v407", help="图上标注的版本名")
    a = ap.parse_args()

    global DESK, BASE
    if a.desk:
        DESK = Path(a.desk)
    if a.base_name:
        BASE = OUT / a.base_name
    tag = a.tag
    subj = Path(a.subject)
    if not subj.exists():
        raise SystemExit(f"缺主体文件 {subj}")
    base = build_base(subj, a.force_base)
    bimg = Image.open(base).convert("RGB")
    W, H = bimg.size
    print(f"[base] {W}x{H}")

    words = [w.strip().upper() for w in a.words.split(",") if w.strip()]
    finals = {}
    for wd in words:
        gp = GLY / f"w_{wd}.png"
        if not gp.exists():
            print(f"  ! 缺字标 w_{wd}.png")
            continue
        al, geo = glyph_alpha(gp, W, H)
        out = compose(bimg, al)
        dst = OUT / f"final_{wd}.jpg"
        out.save(dst, quality=93)
        finals[wd] = out
        print(f"  ok {wd:12s} bg={geo['bg_med']:.2f} slab={geo['slab']} pos={geo['pos']}")

    if not finals:
        raise SystemExit("没有可交付的词")

    orig = Image.open(ORIG).convert("RGB")
    # 交付目录
    if DESK.exists():
        shutil.rmtree(DESK)
    (DESK / "10_p6词库_新排版").mkdir(parents=True, exist_ok=True)
    (DESK / "20_主体对比").mkdir(parents=True, exist_ok=True)

    # ① 标题带逐版对比
    tiles = [("ORIG MRCHOSR", orig)] + [(f"{w}（新排版）", finals[w]) for w in finals]
    stack([(l, band(im)) for l, im in tiles], 1500, out=DESK / "00_标题带_逐版对比.jpg")
    # ② VORGRAVEN 1:1 左右半
    v = finals.get("VORGRAVEN", list(finals.values())[0])
    stack([("ORIG 左半", half(orig, (0, 0, 1772, 1500))),
           (f"{tag} 左半", half(v, (0, 0, 1772, 1500))),
           ("ORIG 右半", half(orig, (1771, 0, 3543, 1500))),
           (f"{tag} 右半", half(v, (1771, 0, 3543, 1500)))],
          520, cols=2, out=DESK / "01_VORGRAVEN_1比1_左右半.jpg", gap=12)
    # ③ 整幅
    f = lambda im: im.resize((520, int(im.height * 520 / im.width)), Image.LANCZOS)
    stack([("ORIG", f(orig)), (tag, f(v))], 520, cols=2,
          out=DESK / f"02_整幅_原图_vs_{tag}.jpg")
    # ④ 主体对比
    stack([("ORIG 主体", half(orig, (700, 1150, 2900, 4400), 540)),
           (f"{tag} 主体（裂变）", half(Image.open(subj).convert("RGB"), (700, 1150, 2900, 4400), 540))],
          540, cols=2, out=DESK / "20_主体对比" / "主体_原图_vs_新.jpg", gap=12)
    stack([("ORIG 颅骨", half(orig, (900, 2900, 2700, 4500), 520)),
           (f"{tag} 颅骨", half(v, (900, 2900, 2700, 4500), 520))],
          520, cols=2, out=DESK / "20_主体对比" / "颅骨_原图_vs_新.jpg", gap=12)

    # 全分辨率
    for wd, im in finals.items():
        im.save(DESK / "10_p6词库_新排版" / f"p6_{wd}.jpg", quality=94)
    shutil.copy2(subj, DESK / "20_主体对比" / f"主体重生_{subj.stem}.jpg")
    print(f"[done] {len(finals)} 版 → {DESK}")


if __name__ == "__main__":
    main()
