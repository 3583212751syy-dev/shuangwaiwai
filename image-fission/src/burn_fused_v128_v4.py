"""v128 v4: 按用户纠正「改原图文本成跟图片主题相关的其他单词」.
- 侵权/需裂变文本 → 改成与该图视觉内容相关的单词（不再用无关人名/神权词）
- eagle_2(双鹰王座/火焰): EAGLE SOVEREIGN(中) + FLAME RAPTOR(底)
- denim_3(牛仔): DENIM
- skull_5(骷髅): SKULL(上) + NEVER DIES(下,保留结构换首词)
- metal_6(死亡金属): METAL
位置按原图 OCR 区间，软遮罩(边缘采样色)清 AI 残留。
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
import sys

OUT_FINAL = Path(r"C:\Users\lenovo\WorkBuddy\2026-08-24-16-39-13\outputs\v128")
CLEAN = Path(r"E:\Desktop\双接口\image-fission\jobs\smoke_v128_1787822084")
FONTS = {
    "metal_mania": r"E:/Desktop/双接口/image-fission/fonts/MetalMania-Regular.ttf",
    "pirata_one":  r"E:/Desktop/双接口/image-fission/fonts/PirataOne-Regular.ttf",
    "rye":         r"E:/Desktop/双接口/image-fission/fonts/Rye-Regular.ttf",
}
FUSION = 0.12
CROP = 8

SPECS = {
    "eagle_2": {
        "masks": [
            (0.28, 0.50, 0.72, 0.64),
            (0.15, 0.82, 0.85, 0.94),
            (0.35, 0.42, 0.65, 0.50),
        ],
        "burns": [
            {"word": "EAGLE\nSOVEREIGN", "font": "pirata_one", "size_ratio": 0.052, "y_ratio": 0.525,
             "fill": (245, 230, 215, 255), "stroke": (20, 12, 8, 255), "stroke_w": 3, "spacing": 3},
            {"word": "FLAME RAPTOR", "font": "pirata_one", "size_ratio": 0.045, "y_ratio": 0.865,
             "fill": (245, 230, 215, 255), "stroke": (20, 12, 8, 255), "stroke_w": 3, "spacing": 2},
        ],
    },
    "denim_3": {
        "masks": [(0.05, 0.05, 0.95, 0.30)],
        "burns": [
            {"word": "DENIM", "font": "rye", "size_ratio": 0.14, "y_ratio": 0.09,
             "fill": (248, 240, 225, 255), "stroke": (25, 35, 70, 255), "stroke_w": 7, "spacing": 12},
        ],
    },
    "skull_5": {
        "masks": [
            (0.15, 0.05, 0.85, 0.22),
            (0.10, 0.68, 0.90, 0.95),
        ],
        "burns": [
            {"word": "SKULL", "font": "pirata_one", "size_ratio": 0.085, "y_ratio": 0.08,
             "fill": (215, 30, 40, 255), "stroke": (8, 8, 12, 255), "stroke_w": 4, "spacing": 5},
            {"word": "NEVER DIES", "font": "pirata_one", "size_ratio": 0.07, "y_ratio": 0.78,
             "fill": (215, 30, 40, 255), "stroke": (8, 8, 12, 255), "stroke_w": 3, "spacing": 4},
        ],
    },
    "metal_6": {
        "masks": [
            (0.10, 0.05, 0.90, 0.25),
            (0.53, 0.05, 0.79, 0.21),
        ],
        "burns": [
            {"word": "METAL", "font": "metal_mania", "size_ratio": 0.15, "y_ratio": 0.055,
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


def soft_mask_layer(im, x0r, y0r, x1r, y1r, blur=14, alpha=245):
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
    for mask_spec in spec.get("masks", []):
        mask_layer = soft_mask_layer(im, *mask_spec)
        im = Image.alpha_composite(im, mask_layer)
    print(f"  {key} masked {len(spec.get('masks', []))} region(s)", end="", flush=True)
    for burn_spec in spec.get("burns", []):
        im = burn_text_layer(im, burn_spec)
    print(f"  burned {len(spec.get('burns', []))} word(s)", flush=True)
    w, h = im.size
    out = im.convert("RGB").crop((CROP, CROP, w - CROP, h - CROP))
    out.save(OUT_FINAL / f"{key}_final.jpg", "JPEG", quality=92)


def main():
    print("=== v128 v4: 改原图文本为跟图片主题相关的单词 ===", flush=True)
    for key in ["eagle_2", "skull_5", "denim_3", "metal_6"]:
        process(key)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
