"""
v198b — 在 v197 迷彩底色上，徽章外圈上方烧金色横幅 MOONCREST
方案：放弃沿弧排字，改用矩形横幅（标准徽章外圈上方设计）
"""
import os, math
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

PROJECT = Path(__file__).resolve().parents[1]
FONTS = PROJECT / "fonts"
JOB = PROJECT / "jobs" / "smoke_v198b"
JOB.mkdir(parents=True, exist_ok=True)

V197 = PROJECT / "jobs" / "smoke_v197" / "v197_bat_logo.jpg"
OUT = JOB / "v198b_bat_logo_burned.jpg"

GOLD = (190, 145, 60)
DARK_PURPLE = (38, 20, 50)
BLACK = (15, 10, 5)


def main():
    if not V197.exists():
        raise SystemExit(f"missing v197 image: {V197}")
    img_pil = Image.open(V197).convert("RGB")
    W, H = img_pil.size
    print(f"[load] {V197} {W}x{H}")

    # === 徽章外圈上方金色横幅 ===
    # 徽章中心 (2050, 1850)，外圈椭圆顶部约 y=900
    # 横幅放在徽章外圈上方、顶部飘带下方
    banner_x1 = int(W * 0.27)
    banner_x2 = int(W * 0.73)
    banner_y1 = int(H * 0.20)
    banner_y2 = int(H * 0.245)
    banner_h = banner_y2 - banner_y1
    banner_w = banner_x2 - banner_x1
    banner_cx = (banner_x1 + banner_x2) // 2
    banner_cy = (banner_y1 + banner_y2) // 2

    print(f"[banner] ({banner_x1},{banner_y1})-({banner_x2},{banner_y2}) {banner_w}x{banner_h}")

    draw = ImageDraw.Draw(img_pil)
    # 金色填充横幅
    draw.rectangle([banner_x1, banner_y1, banner_x2, banner_y2], fill=GOLD)
    # 深色细描边
    border = 4
    draw.rectangle([banner_x1, banner_y1, banner_x2, banner_y2], outline=BLACK, width=border)

    # 横幅上 MOONCREST 黑字
    font_size = int(banner_h * 0.70)
    font_size = max(60, min(font_size, 200))
    font = ImageFont.truetype(str(FONTS / "PirataOne-Regular.ttf"), font_size)
    text = "MOONCREST"
    bb = draw.textbbox((0, 0), text, font=font)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]
    tx = banner_cx - tw // 2
    ty = banner_cy - th // 2 - bb[1]
    draw.text((tx, ty), text, font=font, fill=BLACK)

    # === 徽章中央 CURSE 横字（深紫 + 黑描边，放在徽章下沿外侧）===
    # 徽章下沿约 y=H*0.50=2050，CURSE 放在 y=H*0.55=2250 避免被徽章下沿切
    curse_font_size = int(H * 0.085)
    font_curse = ImageFont.truetype(str(FONTS / "PirataOne-Regular.ttf"), curse_font_size)
    text_curse = "CURSE"
    bb = draw.textbbox((0, 0), text_curse, font=font_curse)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]
    curse_x = (W - tw) // 2
    curse_y = int(H * 0.55) - bb[1]
    # 粗黑色描边（8 offset + thick stroke）
    for dx, dy in [(-3, -2), (3, -2), (-3, 2), (3, 2), (0, -4), (0, 4), (-4, 0), (4, 0)]:
        draw.text((curse_x + dx, curse_y + dy), text_curse, font=font_curse, fill=BLACK)
    draw.text((curse_x, curse_y), text_curse, font=font_curse, fill=DARK_PURPLE)

    # === 徽章下方 EST. MMXXVI（金色 Rye + 黑描边）===
    bot_font_size = int(H * 0.030)
    font_bot = ImageFont.truetype(str(FONTS / "Rye-Regular.ttf"), bot_font_size)
    text_bot = "EST. MMXXVI"
    bb = draw.textbbox((0, 0), text_bot, font=font_bot)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]
    bot_x = (W - tw) // 2
    bot_y = int(H * 0.69) - bb[1]
    # 黑描边
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        draw.text((bot_x + dx, bot_y + dy), text_bot, font=font_bot, fill=BLACK)
    draw.text((bot_x, bot_y), text_bot, font=font_bot, fill=GOLD)

    # === 顶部飘带 EST. MMXXVI（金色 Rye）===
    top_font_size = int(H * 0.038)
    font_top = ImageFont.truetype(str(FONTS / "Rye-Regular.ttf"), top_font_size)
    text_top = "EST. MMXXVI"
    bb = draw.textbbox((0, 0), text_top, font=font_top)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]
    top_x = (W - tw) // 2
    top_y = int(H * 0.085) - bb[1]
    # 黑描边
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        draw.text((top_x + dx, top_y + dy), text_top, font=font_top, fill=BLACK)
    draw.text((top_x, top_y), text_top, font=font_top, fill=GOLD)

    # === USM 锐化 ===
    img_pil = img_pil.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2))

    img_pil.save(OUT, quality=95)
    print(f"\n[done] {OUT}")
    print(f"size: {img_pil.size}")


if __name__ == "__main__":
    main()