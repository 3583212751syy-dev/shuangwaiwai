"""v128 PIL 烧字 (复用 v127 烧字逻辑, 路径指向 v128 + outputs/v128)."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
import os, sys

FONTS = {
    "metal_mania": r"E:/Desktop/双接口/image-fission/fonts/MetalMania-Regular.ttf",
    "pirata_one":  r"E:/Desktop/双接口/image-fission/fonts/PirataOne-Regular.ttf",
    "rye":         r"E:/Desktop/双接口/image-fission/fonts/Rye-Regular.ttf",
}

JOBS = Path(r"E:\Desktop\双接口\image-fission\jobs")
# 强制指定 v128 目录（跳过被杀的 rerun）
SRC = JOBS / "smoke_v128_1787814169"
OUT = Path(r"C:\Users\lenovo\WorkBuddy\2026-08-24-16-39-13\outputs\v128")
OUT.mkdir(parents=True, exist_ok=True)
CROP = 8
FUSION = 0.14

BURNS = [
    {"src": f"{SRC}/v128_clean_eagle_2.png", "name": "eagle_2",
     "word": "DOMINION", "font": "pirata_one", "size_ratio": 0.075,
     "fill": (245, 230, 215, 255), "stroke": (20, 12, 8, 255),
     "stroke_w": 4, "y_ratio": 0.13, "spacing": 4,
     "shadow_off": 5, "shadow_alpha": 150},
    {"src": f"{SRC}/v128_clean_denim_3.png", "name": "denim_3",
     "word": "UPCY", "font": "rye", "size_ratio": 0.085,
     "fill": (50, 60, 90, 255), "stroke": (255, 255, 255, 255),
     "stroke_w": 5, "y_ratio": 0.10, "spacing": 8,
     "shadow_off": 6, "shadow_alpha": 150},
    {"src": f"{SRC}/v128_clean_skull_5.png", "name": "skull_5",
     "word": "VENOM", "font": "pirata_one", "size_ratio": 0.085,
     "fill": (215, 30, 40, 255), "stroke": (8, 8, 12, 255),
     "stroke_w": 4, "y_ratio": 0.13, "spacing": 5,
     "shadow_off": 5, "shadow_alpha": 150},
    {"src": f"{SRC}/v128_clean_metal_6.png", "name": "metal_6",
     "word": "MRCHGSR", "font": "metal_mania", "size_ratio": 0.09,
     "fill": (250, 250, 250, 255), "stroke": (10, 10, 10, 255),
     "stroke_w": 4, "y_ratio": 0.06, "spacing": 3,
     "shadow_off": 5, "shadow_alpha": 150},
]


def soft_light(base, blend):
    res = np.empty_like(base)
    m = blend <= 0.5
    res[m] = base[m] - (1 - 2 * blend[m]) * base[m] * (1 - base[m])
    nm = ~m
    res[nm] = base[nm] + (2 * blend[nm] - 1) * (np.sqrt(base[nm]) - base[nm])
    return np.clip(res, 0, 1)


def burn_fused(cfg):
    src = Path(cfg["src"])
    if not src.exists():
        # 兼容 jpg / png
        alt = src.with_suffix(".jpg")
        if alt.exists():
            src = alt
        else:
            print(f"  SKIP {cfg['name']}: {src} 不存在", flush=True); return
    im = Image.open(src).convert("RGBA")
    w, h = im.size
    font_size = int(h * cfg["size_ratio"])
    font = ImageFont.truetype(FONTS[cfg["font"]], font_size)
    spacing = cfg.get("spacing", 6)
    line_height = int(font_size * 1.3)

    text = cfg["word"]
    lines = text.split("\n")
    total_h = line_height * len(lines) + (len(lines) - 1) * int(font_size * 0.2)

    shadow = Image.new("RGBA", im.size, (0, 0, 0, 0))
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    draw = ImageDraw.Draw(layer)
    sh_off = cfg.get("shadow_off", 6)
    sh_alpha = cfg.get("shadow_alpha", 150)
    start_y = int(h * cfg["y_ratio"])

    for i, line in enumerate(lines):
        widths = [font.getbbox(ch)[2] - font.getbbox(ch)[0] for ch in line]
        total_w = sum(widths) + spacing * (len(line) - 1)
        x = (w - total_w) // 2
        y = start_y + i * line_height
        cur_x = x
        for ch in line:
            cw = font.getbbox(ch)[2] - font.getbbox(ch)[0]
            sd.text((cur_x + sh_off, y + sh_off), ch, font=font,
                    fill=(0, 0, 0, sh_alpha),
                    stroke_width=max(1, cfg["stroke_w"] // 2),
                    stroke_fill=(0, 0, 0, min(sh_alpha + 40, 255)))
            draw.text((cur_x, y), ch, font=font,
                      fill=cfg["fill"], stroke_width=cfg["stroke_w"],
                      stroke_fill=cfg["stroke"])
            cur_x += cw + spacing

    shadow = shadow.filter(ImageFilter.GaussianBlur(3))

    out = Image.alpha_composite(im, shadow)
    out = Image.alpha_composite(out, layer)

    arr = np.array(out).astype(np.float32) / 255.0
    lay = np.array(layer).astype(np.float32) / 255.0
    base = np.array(im).astype(np.float32) / 255.0
    mask = lay[:, :, 3:4]
    fused = soft_light(base[:, :, :3], lay[:, :, :3])
    rgb = arr[:, :, :3] * (1 - FUSION * mask) + fused * (FUSION * mask)
    alpha = arr[:, :, 3:4]
    arr = np.concatenate([np.clip(rgb, 0, 1), alpha], axis=2)
    out = Image.fromarray((arr * 255).astype(np.uint8), "RGBA")

    out = out.convert("RGB").crop((CROP, CROP, w - CROP, h - CROP))
    out.save(f"{OUT}/{cfg['name']}_final.jpg", "JPEG", quality=92)
    print(f"  OK {cfg['name']} word={text!r} font={cfg['font']} size={out.size}", flush=True)


def no_text(name):
    base = SRC / f"v128_clean_{name}.png"
    if not base.exists(): base = SRC / f"v128_clean_{name}.jpg"
    if not base.exists(): base = SRC / f"v127_clean_{name}.jpg"
    if not base.exists():
        print(f"  SKIP {name}: 找不到干净图"); return
    im = Image.open(base).convert("RGB")
    w, h = im.size
    im = im.crop((CROP, CROP, w - CROP, h - CROP))
    im.save(f"{OUT}/{name}_final.jpg", "JPEG", quality=92)
    print(f"  OK {name} (无字) size={im.size}", flush=True)


def main():
    print(f"源目录: {SRC}", flush=True)
    print(f"输出目录: {OUT}", flush=True)
    no_text("illust_1")
    no_text("camo_4")
    for cfg in BURNS:
        burn_fused(cfg)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
