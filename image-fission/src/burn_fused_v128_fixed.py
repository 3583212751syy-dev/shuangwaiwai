"""v128 fix: 检测 AI 残留乱字 → 黑底遮罩 → 重烧干净文字.
应对 eagle_2 中间徽章/底部横幅还有 AI 生成的乱字、skull_5 的 TRUE NEVER DIES 等.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
import sys

OUT_FINAL = Path(r"C:\Users\lenovo\WorkBuddy\2026-08-24-16-39-13\outputs\v128")
CLEAN = Path(r"E:\Desktop\双接口\image-fission\jobs\smoke_v128_1787814169")
FONTS = {
    "metal_mania": r"E:/Desktop/双接口/image-fission/fonts/MetalMania-Regular.ttf",
    "pirata_one":  r"E:/Desktop/双接口/image-fission/fonts/PirataOne-Regular.ttf",
    "rye":         r"E:/Desktop/双接口/image-fission/fonts/Rye-Regular.ttf",
}
FUSION = 0.14
CROP = 8

# 每张的: 文字内容 + 字体 + 在整图中的相对位置 (y_ratio) + 字号比例 + mask 区域 (x0,y0,x1,y1 都按相对比例)
# mask 区域决定先涂黑哪里 (清掉 AI 残留的乱字)
SPECS = {
    "eagle_2":  {"word": "DOMINION", "font": "pirata_one", "size_ratio": 0.075, "y_ratio": 0.06,
                 "fill": (245, 230, 215, 255), "stroke": (20, 12, 8, 255), "stroke_w": 4, "spacing": 4,
                 "shadow_off": 5, "shadow_alpha": 150,
                 # 中央文字带 (KSKBL / LILXZY 5 / 0) + 底部横幅，精确遮盖
                 "masks": [(0.35, 0.53, 0.63, 0.64),  # 中部 AI 乱字带
                           (0.15, 0.80, 0.85, 0.93)]}, # 底横幅
    "skull_5":  {"word": "VENOM", "font": "pirata_one", "size_ratio": 0.085, "y_ratio": 0.06,
                 "fill": (215, 30, 40, 255), "stroke": (8, 8, 12, 255), "stroke_w": 4, "spacing": 5,
                 "shadow_off": 5, "shadow_alpha": 150,
                 # TOP + BOTTOM 的 TRUE NEVER DIES 区域
                 "masks": [(0.18, 0.06, 0.82, 0.20),  # 顶 TRUE
                           (0.10, 0.78, 0.90, 0.97)]}, # 底 NEVER DIES
    "denim_3":  {"word": "UPCY", "font": "rye", "size_ratio": 0.10, "y_ratio": 0.10,
                 "fill": (248, 240, 225, 255), "stroke": (25, 35, 70, 255), "stroke_w": 7, "spacing": 10,
                 "shadow_off": 6, "shadow_alpha": 160,
                 "masks": []}, # 高对比奶白字 + 深蓝描边，牛仔底上可读
    "metal_6":  {"word": "MRCHGSR", "font": "metal_mania", "size_ratio": 0.115, "y_ratio": 0.06,
                 "fill": (252, 252, 252, 255), "stroke": (8, 8, 8, 255), "stroke_w": 6, "spacing": 3,
                 "shadow_off": 5, "shadow_alpha": 160,
                 "masks": [(0.53, 0.05, 0.79, 0.21)]}, # 右侧 AI 残留 "681" 遮盖
}


def soft_light(base, blend):
    res = np.empty_like(base)
    m = blend <= 0.5
    res[m] = base[m] - (1 - 2 * blend[m]) * base[m] * (1 - base[m])
    nm = ~m
    res[nm] = base[nm] + (2 * blend[nm] - 1) * (np.sqrt(base[nm]) - base[nm])
    return np.clip(res, 0, 1)


def mask_and_burn(key, spec):
    src = CLEAN / f"v128_clean_{key}.png"
    if not src.exists():
        print(f"  SKIP {key}: {src} 不存在"); return
    im = Image.open(src).convert("RGBA")
    w, h = im.size

    # 1) mask: 在指定区域画半透明黑色覆盖 (模糊边缘以融入)
    if spec["masks"]:
        mask_layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
        md = ImageDraw.Draw(mask_layer)
        for (x0r, y0r, x1r, y1r) in spec["masks"]:
            x0, y0 = int(x0r * w), int(y0r * h)
            x1, y1 = int(x1r * w), int(y1r * h)
            # 一半透明黑 + 模糊边缘以融入画面 (避免硬边)
            md.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 255))
        mask_layer = mask_layer.filter(ImageFilter.GaussianBlur(10))
        im = Image.alpha_composite(im, mask_layer)
        print(f"  masked {len(spec['masks'])} region(s)", end="")

    # 2) text burn (同 v127 算法)
    font_size = int(h * spec["size_ratio"])
    font = ImageFont.truetype(FONTS[spec["font"]], font_size)
    spacing = spec.get("spacing", 6)
    line_height = int(font_size * 1.3)
    text = spec["word"]
    lines = text.split("\n")

    shadow = Image.new("RGBA", im.size, (0, 0, 0, 0))
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    draw = ImageDraw.Draw(layer)
    sh_off = spec.get("shadow_off", 6)
    sh_alpha = spec.get("shadow_alpha", 150)
    start_y = int(h * spec["y_ratio"])

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
                    stroke_width=max(1, spec["stroke_w"] // 2),
                    stroke_fill=(0, 0, 0, min(sh_alpha + 40, 255)))
            draw.text((cur_x, y), ch, font=font,
                      fill=spec["fill"], stroke_width=spec["stroke_w"],
                      stroke_fill=spec["stroke"])
            cur_x += cw + spacing

    shadow = shadow.filter(ImageFilter.GaussianBlur(3))
    out = Image.alpha_composite(im, shadow)
    out = Image.alpha_composite(out, layer)

    # text blend with base texture
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
    out.save(OUT_FINAL / f"{key}_final.jpg", "JPEG", quality=92)
    print(f"  OK {key} word={text!r}", flush=True)


def main():
    print("=== v128 fix: mask AI 残留乱字 → 重烧 ===", flush=True)
    for key in ["eagle_2", "skull_5", "denim_3", "metal_6"]:
        spec = SPECS.get(key)
        if spec:
            mask_and_burn(key, spec)
    # illust_1 + camo_4 无字，直接从前一版本复制（已是终态）
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
