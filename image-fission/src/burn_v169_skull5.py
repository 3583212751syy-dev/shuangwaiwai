#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v169 burn skull_5 — 一次性产两版
- 复用 v164 skull_5 干净底图（无失真无意义不明小元素）
- A 版「文字裂变」：烧 BONE / BLOOM / ASH（与图元素隐喻同构：骨/玫瑰/燃尽）
- B 版「原图文字保留」：烧 TRUE / NEVER / DIES（保留原 logo 美术形态）
- 字体：PirataOne（直笔哥特衬线） + 黑边描红 + 上下血滴装饰 + USM 锐化
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path("E:/Desktop/双接口/image-fission")
DESK = Path("E:/Desktop/双接口/image-fission/outputs")
BASE = ROOT / "jobs" / "smoke_v164" / "v164_skull_5.jpg"
FONT_PATH = ROOT / "fonts" / "PirataOne-Regular.ttf"

# 配色（与原图血红呼应）
RED_MAIN = (192, 28, 40)         # 主红
RED_DARK = (110, 12, 18)         # 暗血红（描边）
RED_BLOOD = (148, 18, 26)        # 血滴色（深红）
WHITE_HILITE = (245, 215, 200)   # 高光米白（避免纯白刺眼）

VERSIONS = [
    {
        "tag": "A",
        "words": ("BONE", "BLOOM", "ASH"),
        "out": DESK / "image-fission-v169a-skull_5-BONE-BLOOM-ASH.jpg",
    },
    {
        "tag": "B",
        "words": ("TRUE", "NEVER", "DIES"),
        "out": DESK / "image-fission-v169b-skull_5-original-text.jpg",
    },
]


def burn_text(base_im: Image.Image, words):
    """在底图上烧三行红色血滴哥特衬线 logo（true/never/dies 风格）。"""
    W, H = base_im.size
    out_im = base_im.copy()

    # 三行 y 锚点（百分比），避开主体骷髅
    # 顶 / 中下 / 底
    row_anchors = [0.10, 0.58, 0.89]
    # 单行 logo 高度上限
    row_max_h = int(H * 0.10)

    # 字号：取 W 的 ~14% 作为目标宽度，按最大词长反算字号
    max_word_len = max(len(w) for w in words)
    target_width = int(W * 0.74)
    font_size = int(target_width / max_word_len * 0.65)
    font_size = min(font_size, row_max_h * 2)

    font_main = ImageFont.truetype(str(FONT_PATH), font_size)

    for word, y_anchor in zip(words, row_anchors):
        # 测 bbox
        bbox = font_main.getbbox(word)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # logo 层（透明）
        logo_w = int(W * 0.95)
        logo_h = int(th * 1.7)  # 上下血滴留余
        logo = Image.new("RGBA", (logo_w, logo_h), (0, 0, 0, 0))
        ld = ImageDraw.Draw(logo)

        # 中心位置
        cx = logo_w // 2
        cy = logo_h // 2

        # ---------- 1. 黑色厚外描边（多层偏移）----------
        stroke = max(2, int(font_size * 0.07))
        for dx in range(-stroke, stroke + 1):
            for dy in range(-stroke, stroke + 1):
                if dx * dx + dy * dy <= stroke * stroke:
                    ld.text((cx - tw // 2 - bbox[0] + dx,
                             cy - th // 2 - bbox[1] + dy),
                            word, font=font_main, fill=(*RED_DARK, 255))

        # ---------- 2. 主红色填充 ----------
        ld.text((cx - tw // 2 - bbox[0], cy - th // 2 - bbox[1]),
                word, font=font_main, fill=(*RED_MAIN, 255))

        # ---------- 3. 高光（左上一抹米白提亮）----------
        for dx, dy in [(-2, -2), (-1, -1)]:
            ld.text((cx - tw // 2 - bbox[0] + dx,
                     cy - th // 2 - bbox[1] + dy),
                    word, font=font_main, fill=(*WHITE_HILITE, 90))

        # ---------- 4. 上下血滴装饰（death metal 特征）----------
        # 沿文字顶部和底部各打一排小血滴
        n_drops = len(word) * 3  # 每字 3 滴
        letter_top_y = cy - th // 2 - bbox[1]
        letter_bot_y = letter_top_y + th
        letter_left = cx - tw // 2 - bbox[0]
        letter_right = letter_left + tw
        drop_h = int(font_size * 0.20)

        for i in range(n_drops):
            t = (i + 0.5) / n_drops
            x = letter_left + t * tw
            # 上血滴
            tri = [(x - drop_h * 0.35, letter_top_y),
                   (x + drop_h * 0.35, letter_top_y),
                   (x, letter_top_y - drop_h)]
            ld.polygon(tri, fill=(*RED_BLOOD, 255))
            # 下血滴
            tri2 = [(x - drop_h * 0.35, letter_bot_y),
                    (x + drop_h * 0.35, letter_bot_y),
                    (x, letter_bot_y + drop_h)]
            ld.polygon(tri2, fill=(*RED_BLOOD, 255))

        # ---------- 5. USM 锐化（仅 logo 层）----------
        logo = logo.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2))

        # ---------- 6. 合成到画布 ----------
        paste_x = (W - logo_w) // 2
        paste_y = int(H * y_anchor) - logo_h // 2
        out_im.paste(logo, (paste_x, paste_y), logo)

    # 全图轻锐化
    out_im = out_im.filter(ImageFilter.UnsharpMask(radius=1.5, percent=90, threshold=3))
    return out_im


def make_compare(images_with_labels, out_path):
    """拼多版对照图（原图 | v164 | v169A | v169B）。"""
    H_show = 1500
    gap = 24

    def fit(im):
        w = int(im.width * H_show / im.height)
        return im.resize((w, H_show), Image.LANCZOS)

    panels = []
    for img, label in images_with_labels:
        p = fit(img)
        panels.append((p, label))

    total_w = sum(p.width for p, _ in panels) + gap * (len(panels) - 1)
    header_h = 70
    canvas = Image.new("RGB", (total_w, H_show + header_h), (24, 24, 24))
    d = ImageDraw.Draw(canvas)
    x = 0
    for p, label in panels:
        canvas.paste(p, (x, header_h))
        d.text((x + 14, 22), label, fill=(255, 235, 200))
        x += p.width + gap
    canvas.save(out_path, "JPEG", quality=86, optimize=True)
    print(f"[compare] {out_path.name} {out_path.stat().st_size/1024/1024:.2f}MB")


def main():
    print(f"[load] {BASE}")
    base = Image.open(BASE).convert("RGB")
    print(f"[size] {base.size[0]}x{base.size[1]}")

    # 找原图
    orig_path = ROOT / "ComfyUI" / "input" / "pinterest_skull_5.jpg"
    orig = Image.open(orig_path).convert("RGB")
    print(f"[orig] {orig_path.name} {orig.size[0]}x{orig.size[1]}")

    # v164 干净底图（同 base）
    v164 = base.copy()

    results = {"A": None, "B": None}
    for ver in VERSIONS:
        print(f"\n[burn {ver['tag']}] words={ver['words']}")
        out_im = burn_text(base, ver["words"])
        out_im.save(ver["out"], "JPEG", quality=92, optimize=True)
        print(f"[ok] {ver['out'].name} {ver['out'].stat().st_size/1024/1024:.2f}MB")
        results[ver["tag"]] = out_im

    # 四联拼图：原图 | v164干净无字 | v169A文字裂变 | v169B原图文字保留
    compare_out = DESK / "image-fission-v169-skull_5-4way-compare.jpg"
    make_compare([
        (orig, "原图 TRUE/NEVER/DIES (Bon Jovi, 侵权)"),
        (v164, "v164 干净底图 (v147 裂变禁词版)"),
        (results["A"], "v169A 文字裂变: BONE/BLOOM/ASH"),
        (results["B"], "v169B 原图文字保留: TRUE/NEVER/DIES"),
    ], compare_out)


if __name__ == "__main__":
    main()