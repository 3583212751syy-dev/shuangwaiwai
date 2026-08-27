"""v128 v3: 按用户要求「修改原图文本成不侵权，不侵权则换风格字体」.
- eagle_2: JACKIE DIANNIES(中)+底部词 → 同类型不侵权 JAKOVI DELMARA / IONARA VEX
- denim_3: UPOY(顶) → 同类型不侵权 UPRA
- skull_5: TRUE NEVER DIES → 不侵权，保留原词只换符合裂变风格的哥特字体
- metal_6: W → 不侵权，保留原词只换金属风字体
位置全部按原图 OCR 区间，软遮罩(边缘采样色)清 AI 残留。
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
FUSION = 0.12
CROP = 8

# 按原图 OCR 实际文字位置定义 mask 区域 + 烧字参数
SPECS = {
    "eagle_2": {
        # 原图文字：中(0.55-0.61 "JACKIE DIANNIES") + 下(0.85-0.92 疑似 IONA DUJEL)
        "masks": [
            (0.28, 0.50, 0.72, 0.64),  # 中部文字带 (JACKIE DIANNIES 区域)
            (0.15, 0.82, 0.85, 0.94),  # 底部文字带
            (0.35, 0.42, 0.65, 0.50),  # 中徽章额外AI字（柔化，不烧字）
        ],
        "burns": [
            # 中部双词 → 同类型不侵权 JAKOVI DELMARA（分两行贴合原双词结构）
            {"word": "JAKOVI\nDELMARA", "font": "pirata_one", "size_ratio": 0.052, "y_ratio": 0.525,
             "fill": (245, 230, 215, 255), "stroke": (20, 12, 8, 255), "stroke_w": 3, "spacing": 3},
            # 底部 → 同类型不侵权 IONARA VEX
            {"word": "IONARA VEX", "font": "pirata_one", "size_ratio": 0.05, "y_ratio": 0.86,
             "fill": (245, 230, 215, 255), "stroke": (20, 12, 8, 255), "stroke_w": 3, "spacing": 2},
        ],
    },
    "denim_3": {
        # 原图文字：上(0.07-0.28 "UPOY" 满宽)
        "masks": [(0.05, 0.05, 0.95, 0.30)],
        "burns": [
            # UPOY → 同类型不侵权 UPRA
            {"word": "UPRA", "font": "rye", "size_ratio": 0.14, "y_ratio": 0.09,
             "fill": (248, 240, 225, 255), "stroke": (25, 35, 70, 255), "stroke_w": 7, "spacing": 12},
        ],
    },
    "skull_5": {
        # 原图文字：上(0.08-0.21 "TRUE") + 下(0.70-0.93 "NEVER DIES")
        # 不侵权 → 保留原词，只换符合裂变风格的哥特字体
        "masks": [
            (0.15, 0.05, 0.85, 0.22),
            (0.10, 0.68, 0.90, 0.95),
        ],
        "burns": [
            {"word": "TRUE", "font": "pirata_one", "size_ratio": 0.085, "y_ratio": 0.08,
             "fill": (215, 30, 40, 255), "stroke": (8, 8, 12, 255), "stroke_w": 4, "spacing": 5},
            {"word": "NEVER DIES", "font": "pirata_one", "size_ratio": 0.07, "y_ratio": 0.78,
             "fill": (215, 30, 40, 255), "stroke": (8, 8, 12, 255), "stroke_w": 3, "spacing": 4},
        ],
    },
    "metal_6": {
        # 原图文字：上(0.08-0.22 logo "W") 单字母不侵权 → 保留，换金属风字体
        "masks": [
            (0.10, 0.05, 0.90, 0.25),
            (0.53, 0.05, 0.79, 0.21),  # 右侧 681 残留
        ],
        "burns": [
            {"word": "W", "font": "metal_mania", "size_ratio": 0.17, "y_ratio": 0.055,
             "fill": (252, 252, 252, 255), "stroke": (8, 8, 8, 255), "stroke_w": 6, "spacing": 3},
        ],
    },
}


def soft_light(base, blend):
    res = np.empty_like(base)
    m = blend <= 0.5
    res[m] = base[m] - (1 - 2 * blend[m]) * base[m] * (1 - base[m])
    nm = ~m
    res[nm] = base[nm] + (2 * blend[nm] - 1) * (np.sqrt(base[nm]) - base[nm])
    return np.clip(res, 0, 1)


def sample_edge_color(im, x0, y0, x1, y1, edge=20):
    """从 mask 区域的边缘采样平均颜色，用于软遮罩."""
    arr = np.array(im)
    edges = []
    if y0 - edge >= 0:
        edges.append(arr[y0-edge:y0, x0:x1])
    if y1 + edge <= im.size[1]:
        edges.append(arr[y1:y1+edge, x0:x1])
    if x0 - edge >= 0:
        edges.append(arr[y0:y1, x0-edge:x0])
    if x1 + edge <= im.size[0]:
        edges.append(arr[y0:y1, x1:x1+edge])
    if not edges:
        return (0, 0, 0, 255)
    pixels = np.concatenate([e.reshape(-1, min(e.shape[-1], 4)) for e in edges])
    avg = pixels.mean(axis=0)
    if avg.shape[0] < 4:
        return (int(avg[0]), int(avg[1]), int(avg[2]), 255)
    return (int(avg[0]), int(avg[1]), int(avg[2]), int(avg[3]))


def soft_mask_layer(im, x0r, y0r, x1r, y1r, blur=20, alpha=200):
    """用边缘采样色做软遮罩层."""
    w, h = im.size
    x0, y0 = int(x0r * w), int(y0r * h)
    x1, y1 = int(x1r * w), int(y1r * h)
    color = sample_edge_color(im, x0, y0, x1, y1, edge=20)
    mask_layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    md = ImageDraw.Draw(mask_layer)
    md.rectangle([x0, y0, x1, y1], fill=(color[0], color[1], color[2], alpha))
    mask_layer = mask_layer.filter(ImageFilter.GaussianBlur(blur))
    return mask_layer


def burn_text_layer(im, spec):
    """在 im 上烧一个文字层，返回合成后的图."""
    w, h = im.size
    font_size = int(h * spec["size_ratio"])
    font = ImageFont.truetype(FONTS[spec["font"]], font_size)
    spacing = spec.get("spacing", 6)
    line_height = int(font_size * 1.35)
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

    arr = np.array(out).astype(np.float32) / 255.0
    lay = np.array(layer).astype(np.float32) / 255.0
    base = np.array(im).astype(np.float32) / 255.0
    mask = lay[:, :, 3:4]
    fused = soft_light(base[:, :, :3], lay[:, :, :3])
    rgb = arr[:, :, :3] * (1 - FUSION * mask) + fused * (FUSION * mask)
    alpha = arr[:, :, 3:4]
    arr = np.concatenate([np.clip(rgb, 0, 1), alpha], axis=2)
    return Image.fromarray((arr * 255).astype(np.uint8), "RGBA")


def process(key):
    src = CLEAN / f"v128_clean_{key}.png"
    if not src.exists():
        print(f"  SKIP {key}: {src} not found", flush=True)
        return
    im = Image.open(src).convert("RGBA")
    spec = SPECS.get(key)
    if not spec:
        print(f"  SKIP {key}: no spec", flush=True)
        return

    # 1) 软遮罩（边缘采样色，不用黑方块）
    for mask_spec in spec.get("masks", []):
        mask_layer = soft_mask_layer(im, *mask_spec, blur=14, alpha=245)
        im = Image.alpha_composite(im, mask_layer)
    print(f"  {key} masked {len(spec.get('masks', []))} region(s)", end="", flush=True)

    # 2) 烧字（按原图位置/原风格）
    for burn_spec in spec.get("burns", []):
        im = burn_text_layer(im, burn_spec)
    print(f"  burned {len(spec.get('burns', []))} word(s)", flush=True)

    # 保存
    w, h = im.size
    out = im.convert("RGB").crop((CROP, CROP, w - CROP, h - CROP))
    out.save(OUT_FINAL / f"{key}_final.jpg", "JPEG", quality=92)


def main():
    print("=== v128 v3: 改原图文本成不侵权(同类型) / 不侵权原词换风格 ===", flush=True)
    for key in ["eagle_2", "skull_5", "denim_3", "metal_6"]:
        process(key)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
