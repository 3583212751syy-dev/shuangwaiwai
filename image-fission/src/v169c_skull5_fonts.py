#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v169c: 用 git 下载的 3 个哥特/滴血字体重烧 skull_5 原图文字 TRUE/NEVER/DIES，对比选最接近原图的"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path("E:/Desktop/双接口/image-fission")
DESK = Path("E:/Desktop/双接口/image-fission/outputs")
BASE = ROOT / "jobs" / "smoke_v164" / "v164_skull_5.jpg"

RED_MAIN = (192, 28, 40)
RED_DARK = (110, 12, 18)
RED_BLOOD = (148, 18, 26)
WHITE_HILITE = (245, 215, 200)

FONTS = {
    "Nosifer": ROOT / "fonts" / "Nosifer-Regular.ttf",
    "VampiroOne": ROOT / "fonts" / "VampiroOne-Regular.ttf",
    "PirataOne": ROOT / "fonts" / "PirataOne-Regular.ttf",
}

WORDS = ("TRUE", "NEVER", "DIES")
ROW_ANCHORS = [0.10, 0.58, 0.89]


def burn_text(base_im, words, font_path, font_size_ratio=0.62):
    W, H = base_im.size
    out_im = base_im.copy()
    row_max_h = int(H * 0.10)
    max_word_len = max(len(w) for w in words)
    target_width = int(W * 0.74)
    font_size = int(target_width / max_word_len * font_size_ratio)
    font_size = min(font_size, int(row_max_h * 2))
    font_main = ImageFont.truetype(str(font_path), font_size)

    for word, y_anchor in zip(words, ROW_ANCHORS):
        bbox = font_main.getbbox(word)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        logo_w = int(W * 0.95)
        logo_h = int(th * 1.7)
        logo = Image.new("RGBA", (logo_w, logo_h), (0, 0, 0, 0))
        ld = ImageDraw.Draw(logo)
        cx = logo_w // 2
        cy = logo_h // 2
        stroke = max(2, int(font_size * 0.07))
        for dx in range(-stroke, stroke + 1):
            for dy in range(-stroke, stroke + 1):
                if dx * dx + dy * dy <= stroke * stroke:
                    ld.text((cx - tw // 2 - bbox[0] + dx, cy - th // 2 - bbox[1] + dy),
                            word, font=font_main, fill=(*RED_DARK, 255))
        ld.text((cx - tw // 2 - bbox[0], cy - th // 2 - bbox[1]),
                word, font=font_main, fill=(*RED_MAIN, 255))
        for dx, dy in [(-2, -2), (-1, -1)]:
            ld.text((cx - tw // 2 - bbox[0] + dx, cy - th // 2 - bbox[1] + dy),
                    word, font=font_main, fill=(*WHITE_HILITE, 90))
        # 上下血滴
        n_drops = len(word) * 3
        letter_top_y = cy - th // 2 - bbox[1]
        letter_bot_y = letter_top_y + th
        letter_left = cx - tw // 2 - bbox[0]
        letter_right = letter_left + tw
        drop_h = int(font_size * 0.20)
        for i in range(n_drops):
            t = (i + 0.5) / n_drops
            x = letter_left + t * tw
            ld.polygon([(x - drop_h * 0.35, letter_top_y), (x + drop_h * 0.35, letter_top_y),
                        (x, letter_top_y - drop_h)], fill=(*RED_BLOOD, 255))
            ld.polygon([(x - drop_h * 0.35, letter_bot_y), (x + drop_h * 0.35, letter_bot_y),
                        (x, letter_bot_y + drop_h)], fill=(*RED_BLOOD, 255))
        logo = logo.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2))
        paste_x = (W - logo_w) // 2
        paste_y = int(H * y_anchor) - logo_h // 2
        out_im.paste(logo, (paste_x, paste_y), logo)
    out_im = out_im.filter(ImageFilter.UnsharpMask(radius=1.5, percent=90, threshold=3))
    return out_im


def make_compare(images_labels, out_path):
    H_show = 1450
    gap = 20

    def fit(im):
        w = int(im.width * H_show / im.height)
        return im.resize((w, H_show), Image.LANCZOS)

    panels = [(fit(im), lb) for im, lb in images_labels]
    total_w = sum(p.width for p, _ in panels) + gap * (len(panels) - 1)
    header_h = 70
    canvas = Image.new("RGB", (total_w, H_show + header_h), (24, 24, 24))
    d = ImageDraw.Draw(canvas)
    x = 0
    for p, lb in panels:
        canvas.paste(p, (x, header_h))
        d.text((x + 14, 22), lb, fill=(255, 235, 200))
        x += p.width + gap
    canvas.save(out_path, "JPEG", quality=86, optimize=True)
    print(f"[compare] {out_path.name}")


def main():
    base = Image.open(BASE).convert("RGB")
    orig = Image.open(ROOT / "ComfyUI" / "input" / "pinterest_skull_5.jpg").convert("RGB")
    print(f"[size] {base.size}")

    results = {}
    for name, fp in FONTS.items():
        out = DESK / f"image-fission-v169c-skull_5-{name}.jpg"
        im = burn_text(base, WORDS, fp)
        im.save(out, "JPEG", quality=92, optimize=True)
        print(f"[ok] {out.name} {out.stat().st_size/1024/1024:.2f}MB")
        results[name] = im

    # 四联：原图 | Nosifer | VampiroOne | Creepster
    make_compare([
        (orig, "原图 TRUE/NEVER/DIES (参考)"),
        (results["Nosifer"], "v169c Nosifer (滴血哥特)"),
        (results["VampiroOne"], "v169c VampiroOne (哥特粗)"),
        (results["PirataOne"], "v169c PirataOne (现状)"),
    ], DESK / "image-fission-v169c-skull_5-4font-compare.jpg")


if __name__ == "__main__":
    main()
