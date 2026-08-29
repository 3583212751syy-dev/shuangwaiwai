#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v161b final 烧字 (eagle_2)
=========================
把 Stage A 4x 当底图 + PIL 烧 DOMINION (PirataOne 哥特) 在底部黑底处。
避开上方小鹰+主鹰+主骷髅+底部 3 骷髅的密集区。
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
import shutil

JOBS = Path(r"E:\Desktop\双接口\image-fission\jobs")
OUT = Path(r"C:\Users\lenovo\WorkBuddy\2026-08-24-16-39-13\outputs\v161b")
OUT.mkdir(parents=True, exist_ok=True)

STAGE_A = JOBS / "smoke_v161" / "stageA_eagle_2.jpg"
FONT_PATH = r"E:/Desktop/双接口/image-fission/fonts/PirataOne-Regular.ttf"

WORD = "DOMINION"
FONT_SIZE_RATIO = 0.055    # 比 v127 略小 (0.075→0.055)，给密集图留呼吸
Y_RATIO = 0.93            # 极底部，3 骷髅正下方
SPACING = 4
SHADOW_OFF = 5
SHADOW_ALPHA = 170
FILL = (245, 230, 215, 255)        # 米色（v127 同款）
STROKE = (20, 12, 8, 255)          # 暗描边
STROKE_W = 4
FUSION = 0.14                       # 文字层向底层 overlay 拾取纹理（v127 同款）


def soft_light(base, blend):
    res = np.empty_like(base)
    m = blend <= 0.5
    res[m] = base[m] - (1 - 2 * blend[m]) * base[m] * (1 - base[m])
    nm = ~m
    res[nm] = base[nm] + (2 * blend[nm] - 1) * (np.sqrt(base[nm]) - base[nm])
    return np.clip(res, 0, 1)


def main():
    print(f"底图: {STAGE_A}", flush=True)
    im = Image.open(STAGE_A).convert("RGBA")
    w, h = im.size
    print(f"  尺寸 {w}x{h}", flush=True)
    font_size = int(h * FONT_SIZE_RATIO)
    font = ImageFont.truetype(FONT_PATH, font_size)
    print(f"  字体 PirataOne 字号 {font_size}", flush=True)

    shadow = Image.new("RGBA", im.size, (0, 0, 0, 0))
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    draw = ImageDraw.Draw(layer)
    widths = [font.getbbox(ch)[2] - font.getbbox(ch)[0] for ch in WORD]
    total_w = sum(widths) + SPACING * (len(WORD) - 1)
    x = (w - total_w) // 2
    y = int(h * Y_RATIO)
    cur_x = x
    for ch in WORD:
        cw = font.getbbox(ch)[2] - font.getbbox(ch)[0]
        sd.text((cur_x + SHADOW_OFF, y + SHADOW_OFF), ch, font=font,
                fill=(0, 0, 0, SHADOW_ALPHA),
                stroke_width=max(1, STROKE_W // 2),
                stroke_fill=(0, 0, 0, min(SHADOW_ALPHA + 40, 255)))
        draw.text((cur_x, y), ch, font=font,
                  fill=FILL, stroke_width=STROKE_W,
                  stroke_fill=STROKE)
        cur_x += cw + SPACING

    shadow = shadow.filter(ImageFilter.GaussianBlur(3))
    out = Image.alpha_composite(im, shadow)
    out = Image.alpha_composite(out, layer)

    # 文字区向底层图 overlay 拾取纹理（防"贴图感"）
    arr = np.array(out).astype(np.float32) / 255.0
    lay = np.array(layer).astype(np.float32) / 255.0
    base_arr = np.array(im).astype(np.float32) / 255.0
    mask = lay[:, :, 3:4]
    fused = soft_light(base_arr[:, :, :3], lay[:, :, :3])
    rgb = arr[:, :, :3] * (1 - FUSION * mask) + fused * (FUSION * mask)
    alpha = arr[:, :, 3:4]
    arr = np.concatenate([np.clip(rgb, 0, 1), alpha], axis=2)
    out = Image.fromarray((arr * 255).astype(np.uint8), "RGBA")
    out = out.convert("RGB")

    out_path = OUT / "eagle_2_final.jpg"
    out.save(out_path, "JPEG", quality=92, optimize=True)
    print(f"  保存 {out_path} {out_path.stat().st_size/1024/1024:.1f}MB", flush=True)

    # 同步保存到桌面给用户直观访问
    desktop = Path(r"E:\Desktop")
    desk_path = desktop / "image-fission-v161b-eagle_2.jpg"
    shutil.copy(out_path, desk_path)
    print(f"  桌面副本 {desk_path}", flush=True)

    # 拼图：原图 | Stage A | Final (with text)
    from PIL import ImageDraw as ID
    cells = [str(STAGE_A.parent / "stageA_eagle_2_work.png"),  # work-res (more comparable)
             str(STAGE_A),
             str(out_path)]
    labels = ["Stage A work (912x1216)", "Stage A 4x (3584x4864)", "v161b Final + DOMINION (4x)"]
    ch, pad, lh = 900, 16, 50
    def rp(p):
        im=Image.open(p).convert("RGB"); iw,ih=im.size
        s=min(ch/ih, (ch*0.42)/iw); nw,nh=int(iw*s),int(ih*s)
        c=Image.new("RGB",(nw,ch),(10,10,12)); c.paste(im,((nw-c.width)//2,0)); return c
    cells_im=[rp(p) for p in cells]
    gw=sum(c.width for c in cells_im)+pad*(len(cells_im)+1)
    big=Image.new("RGB",(gw,ch+lh+pad),(12,12,14)); d=ID.Draw(big)
    try: f=ImageFont.truetype(r"C:/Windows/Fonts/msyhbd.ttc",24)
    except: f=ImageFont.load_default()
    x_off=pad
    for i,c in enumerate(cells_im):
        big.paste(c,(x_off,lh)); d.text((x_off+6,12),labels[i],fill=(225,225,225),font=f); x_off+=c.width+pad
    cmp_path = OUT / "compare_eagle_2_v161b_final.png"
    big.save(cmp_path,"PNG",optimize=True)
    print(f"  拼图 {cmp_path}", flush=True)


if __name__ == "__main__":
    main()
