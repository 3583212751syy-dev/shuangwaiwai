"""
text_fission.py — 文本裂变（只动文字，不新增图像内容）

规则（用户 2026-08-27 确认）：
1. 输入 = 桌面「图裂变测试图」里的原始 jpg，绝不用裂变后的图当输入。
2. 不新增图像内容：画面主体/构图/配色保持原样，只动文字。
3. 文本规则：不侵权的原词保留；明显侵权/品牌词替换成跟图片相关的其他单词。
4. 文本不能乱：不用 AI 造字，先 inpaint 清掉旧字，再用 PIL 按原图风格叠回正确单词。

用法：python src/text_fission.py
输出：jobs/text_fission_<ts>/ 下每张图多个文本变体 + gallery.html
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SRC = Path(r"E:/Desktop/图裂变测试图")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "jobs" / f"text_fission_{int(__import__('time').time())}"

FONTS = {
    "impact":  r"C:/Windows/Fonts/impact.ttf",
    "oldengl": r"C:/Windows/Fonts/OLDENGL.TTF",   # 哥特黑体
    "stencil": r"C:/Windows/Fonts/STENCIL.TTF",   # 军风模板
    "rock":    r"C:/Windows/Fonts/ROCK.TTF",      # 摇滚
    "gothicb": r"C:/Windows/Fonts/GOTHICB.TTF",
}

# 每张图的文本配置。
# regions: 每个文字区域 {bbox, original, infringe, font, fill, stroke, stroke_w, banks}
#   infringe=True  -> 该区域是侵权/品牌词，banks 只放替换词（不含原词）
#   infringe=False -> 保留原词，banks 第一个为原词，其余为相关词（用于裂变变体）
# 若某区域 banks 有多词，则每个词生成一张变体图（文本裂变不一样）。
CONFIG = {
    "pinterest_denim_3.jpg": {
        "regions": [{
            "bbox": (0, 90, 736, 372),
            "original": "DENIM", "infringe": False, "font": "impact",
            "fill": (245, 245, 250, 255), "stroke": (40, 55, 95, 255), "stroke_w": 8,
            "banks": ["DENIM", "JEANS", "INDIGO", "STITCH", "PATCH"],
        }]
    },
    "pinterest_skull_5.jpg": {
        "regions": [
            {
                "bbox": (170, 95, 560, 275),
                "original": "TRUE", "infringe": False, "font": "oldengl",
                "fill": (225, 30, 40, 255), "stroke": (10, 10, 15, 255), "stroke_w": 7,
                "banks": ["TRUE", "REAPER", "RAVEN", "BONES", "ASHES"],
            },
            {
                "bbox": (110, 900, 620, 1210),
                "original": "NEVER DIES", "infringe": False, "font": "oldengl",
                "fill": (235, 235, 240, 255), "stroke": (10, 10, 15, 255), "stroke_w": 6,
                "banks": ["NEVER DIES"],   # 底部标语通用，不裂变
            },
        ]
    },
    # eagle_2 含 JACK DANIELS 品牌字（侵权）-> 全部替换为相关词
    "pinterest_eagle_2.jpg": {
        "regions": [{
            "bbox": (340, 700, 610, 790),
            "original": "JACK DANIELS", "infringe": True, "font": "impact",
            "fill": (235, 200, 120, 255), "stroke": (20, 15, 10, 255), "stroke_w": 6,
            "banks": ["RAPTOR", "EMPIRE", "CROWN", "STORM", "CLAWS"],
        }]
    },
}


def inpaint_region(img_bgr, bbox, dilate=18):
    """清掉 bbox 区域文字，用周围纹理填充（TELEA）。"""
    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
    h, w = img_bgr.shape[:2]
    x2, y2 = min(w, x2), min(h, y2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
    mask = cv2.dilate(mask, k, iterations=1)
    try:
        return cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)
    except Exception:
        return img_bgr


def fit_font(font_path, text, max_w, max_h):
    """找到能放进 (max_w,max_h) 的最大字号。"""
    size = 12
    font = ImageFont.truetype(font_path, size)
    while True:
        bbox = font.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw > max_w or th > max_h or size > 600:
            break
        size += 4
        font = ImageFont.truetype(font_path, size)
    # 回退一号
    size = max(12, size - 4)
    return ImageFont.truetype(font_path, size)


def render_text(img_rgba, region, word):
    """在 region 内居中渲染 word，返回合成后的 RGBA。"""
    x1, y1, x2, y2 = region["bbox"]
    region_w, region_h = x2 - x1, y2 - y1
    font = fit_font(FONTS[region["font"]], word, int(region_w * 0.92), int(region_h * 0.82))
    draw = ImageDraw.Draw(img_rgba)
    bbox = font.getbbox(word)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cx = x1 + (region_w - tw) // 2
    cy = y1 + (region_h - th) // 2 - bbox[1]
    draw.text((cx, cy), word, font=font,
              fill=region["fill"], stroke_width=region["stroke_w"],
              stroke_fill=region["stroke"])
    return img_rgba


def process_image(fname):
    cfg = CONFIG[fname]
    img_pil = Image.open(SRC / fname).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    # 先对所有区域 inpaint（清旧字）
    cleaned = img_bgr.copy()
    for r in cfg["regions"]:
        cleaned = inpaint_region(cleaned, r["bbox"])
    base_rgba = Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)).convert("RGBA")

    # 每个区域取 banks 中每个词 -> 组合成变体
    # 简单策略：以第一个区域为裂变轴，其余区域固定取 banks[0]
    axis = cfg["regions"][0]
    axis_words = axis["banks"]
    fixed_regions = cfg["regions"][1:]
    results = []
    for w in axis_words:
        layer = base_rgba.copy()
        # 固定区域用 banks[0]
        for fr in fixed_regions:
            layer = render_text(layer, fr, fr["banks"][0])
        # 裂变轴区域用当前词
        layer = render_text(layer, axis, w)
        out_name = f"{Path(fname).stem}__{w.replace(' ', '_')}.png"
        out_path = OUT / out_name
        layer.convert("RGB").save(out_path, "PNG")
        results.append((w, out_path))
        print(f"  OK {out_name}  (text={w})")
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== text_fission -> {OUT} ===")
    all_results = {}
    for fname in CONFIG:
        print(f"\n--- {fname} ---")
        all_results[fname] = process_image(fname)
    # 生成画廊
    gallery = OUT / "gallery.html"
    html = ["<html><head><meta charset='utf-8'><style>",
            "body{font-family:sans-serif;background:#111;color:#eee;margin:0;padding:20px}",
            ".img{display:inline-block;margin:8px;vertical-align:top}",
            "img{height:360px;border:1px solid #444;border-radius:6px}",
            "p{text-align:center;font-size:13px;margin:4px}",
            "h2{color:#7cf}", "</style></head><body>"]
    for fname, res in all_results.items():
        html.append(f"<h2>{fname}</h2>")
        html.append("<div>")
        for w, p in res:
            rel = Path(p).name
            html.append(f"<div class='img'><img src='{rel}'><p>{w}</p></div>")
        html.append("</div>")
    html.append("</body></html>")
    gallery.write_text("\n".join(html), encoding="utf-8")
    print(f"\n=== gallery: {gallery} ===")


if __name__ == "__main__":
    main()
