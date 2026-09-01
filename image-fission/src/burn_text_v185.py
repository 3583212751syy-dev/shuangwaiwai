"""
v185 侵权文字 PIL 烧字脚本
------------------------------
对 3 张含侵权文字的裂变图（fireball_skull / bat_logo / camo_armed）做后期处理：
  1. 在裂变结果图上自动定位文字（优先 OCR 生成图，落空则回退到"源图坐标按缩放比换算"）
  2. OpenCV Telea inpaint 擦除原文字区域
  3. PIL 按 1:1 字体/位置/美术形态烧入"不侵权替换词"
输出：jobs/smoke_v185/v185_{id}_burned.jpg

依赖：easyocr / opencv-python / Pillow / numpy（venv 已装）
运行：python src/burn_text_v185.py
"""
import os, json, cv2, numpy as np
from PIL import Image, ImageFont, ImageDraw

def load_bgr(path):
    """用 PIL 读图（cv2 对该批 JPEG 解码失败），转 BGR numpy 供 cv2 处理"""
    return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOB = os.path.join(ROOT, "jobs", "smoke_v185")
FONT_DIR = os.path.join(ROOT, "fonts")
SRC_INPUT = os.path.join(ROOT, "ComfyUI", "input")  # 源图副本 test_*.jpg

# 源图 OCR 坐标（jobs/smoke_v185/_ocr_src.json，由探针生成）
OCR_SRC = json.load(open(os.path.join(JOB, "_ocr_src.json"), encoding="utf-8"))

# 每张图的替换配置
#   src_key   : _ocr_src.json 的 key
#   src_file  : ComfyUI/input 里的源图副本（用于回退坐标 & 缩放比）
#   lines     : 要烧入的替换词（按行，与源图行数对应）
#   font      : 匹配的字体
#   color     : 文字颜色 RGB
#   src_box   : 源图坐标下要擦除+烧字的区域 [x1,y1,x2,y2]（回退用）
CONFIG = {
    "fireball_skull": {
        "src_key": "fireball_skull",
        "src_file": "test_581f43423ef2d71d4447c0f634411138.jpg",
        "lines": ["FIRE", "SKULL"],
        "font": "Creepster-Regular.ttf",
        "color": (245, 238, 230),
        "src_box": [879, 500, 1164, 1530],
    },
    "bat_logo": {
        "src_key": "bat_logo",
        "src_file": "test_6978fabda2cc99629fa9e81f802762d3.jpg",
        "lines": ["BARAKA"],
        "font": "PirataOne-Regular.ttf",
        "color": (245, 235, 210),
        "src_box": [281, 946, 1284, 1340],
    },
    "camo_armed": {
        "src_key": "camo_armed",
        "src_file": "test_b78e60de8dfdf44acda99395326a7298.jpg",
        "lines": ["WARRIOR", "PRIDE"],
        "font": "Rye-Regular.ttf",
        "color": (28, 26, 22),
        "src_box": [446, 529, 1157, 779],
    },
}


def fit_font(text, font_path, max_w, max_h):
    """返回能放下 text 的最大字号字体对象"""
    size = max(8, int(max_h))
    while size > 6:
        f = ImageFont.truetype(font_path, size)
        w = f.getlength(text)
        if w <= max_w * 0.96:
            return f
        size -= 2
    return ImageFont.truetype(font_path, 6)


def burn_lines(img_bgr, box, lines, font_path, color):
    """在 box 区域内逐行居中烧字（先 inpaint 擦除）"""
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = img_bgr.shape[:2]
    # 擦除：扩大掩膜
    mx = max(6, (x2 - x1) // 24)
    my = max(6, (y2 - y1) // 24)
    ax1, ay1, ax2, ay2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (ax1, ay1), (ax2, ay2), 255, -1)
    img_bgr = cv2.inpaint(img_bgr, mask, 6, cv2.INPAINT_TELEA)

    # 转 PIL 烧字（cv2.imwrite 在该环境失效，最终用 PIL 存盘）
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    n = len(lines)
    row_h = (y2 - y1) / n
    for i, line in enumerate(lines):
        ry1 = y1 + i * row_h
        ry2 = y1 + (i + 1) * row_h
        f = fit_font(line, font_path, (x2 - x1), (ry2 - ry1) * 0.92)
        lw = f.getlength(line)
        lx = x1 + ((x2 - x1) - lw) / 2
        ly = ry1 + ((ry2 - ry1) - (f.size * 1.0)) / 2
        draw.text((lx, ly), line, font=f, fill=color)
    return img_pil


def locate_box(gen_path, cfg):
    """优先 OCR 生成图定位文字；落空则回退缩放源图坐标"""
    gen = load_bgr(gen_path)
    gh, gw = gen.shape[:2]
    src_dims = OCR_SRC[cfg["src_key"]]["dims"]
    sw, sh = src_dims
    sx, sy = gw / sw, gh / sh  # 缩放比（等比，取 x）
    sb = cfg["src_box"]
    scaled = [sb[0] * sx, sb[1] * sy, sb[2] * sx, sb[3] * sy]

    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=True, verbose=False)
        res = reader.readtext(gen, detail=1, paragraph=False)
        # 取落在 scaled 框（外扩 25%）内的检测框，求并集
        ex = (scaled[2] - scaled[0]) * 0.25
        ey = (scaled[3] - scaled[1]) * 0.25
        bx1, by1, bx2, by2 = scaled[0] - ex, scaled[1] - ey, scaled[2] + ex, scaled[3] + ey
        found = []
        for (bbox, text, conf) in res:
            xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                found.append((min(xs), min(ys), max(xs), max(ys)))
        if found:
            xs = [p[0] for p in found] + [p[2] for p in found]
            ys = [p[1] for p in found] + [p[3] for p in found]
            return [min(xs), min(ys), max(xs), max(ys)], "ocr_gen"
    except Exception as e:
        print(f"  [warn] OCR 失败，回退缩放坐标: {e}")
    return scaled, "scaled_src"


def main():
    from PIL import __version__ as _pv
    print(f"Pillow {_pv}")
    for cid, cfg in CONFIG.items():
        gen_path = os.path.join(JOB, f"v185_{cid}.jpg")
        if not os.path.exists(gen_path):
            print(f"[skip] {cid} 生成图不存在: {gen_path}")
            continue
        font_path = os.path.join(FONT_DIR, cfg["font"])
        box, method = locate_box(gen_path, cfg)
        print(f"=== {cid} 定位={method} box={[round(v) for v in box]} ===")
        gen = load_bgr(gen_path)
        out = burn_lines(gen, box, cfg["lines"], font_path, cfg["color"])
        out_path = os.path.join(JOB, f"v185_{cid}_burned.jpg")
        out.save(out_path, "JPEG", quality=95)
        print(f"  烧字完成 -> {out_path}  lines={cfg['lines']}")


if __name__ == "__main__":
    main()
