"""
v195 交付拼图 + 量化自检（基于已生成的 6 张结果）。
- 对每张：原图 | 裂变 双列拼接，标注 id / 配色交集 / 结构相似度 / OCR 残留字
- 输出单张 pair 拼图 jobs/smoke_v195/compare_v195_<id>.jpg
- 输出总览 2列(原|裂) x 3行 拼图 jobs/smoke_v195/_6up_compare_v195.jpg
"""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
IN = PROJECT / "ComfyUI" / "input"
JOB = PROJECT / "jobs" / "smoke_v195"
JOB.mkdir(parents=True, exist_ok=True)

REFS = [
    ("camo_classic",     "test_5784eab326634d17573b469e91cdc565.jpg"),
    ("floral_bw",        "test_Pinterest_1.jpg"),
    ("denim_patch",      "test_Pinterest_3.jpg"),
    ("palm_camo",        "test_Pinterest_4.jpg"),
    ("skull_snake_rose", "test_Pinterest_5.jpg"),
    ("eagle_skull_metal","test_Pinterest_6.jpg"),
]

def load_bgr(path):
    return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)

def hist_intersection(src, dst, bins=32):
    inter = 0.0
    for c in range(3):
        hs = cv2.calcHist([src], [c], None, [bins], [0, 256])
        hd = cv2.calcHist([dst], [c], None, [bins], [0, 256])
        cv2.normalize(hs, hs); cv2.normalize(hd, hd)
        inter += cv2.compareHist(hs, hd, cv2.HISTCMP_INTERSECT)
    return inter / 3.0

def structural_diff(src, dst):
    h, w = min(src.shape[0], dst.shape[0]), min(src.shape[1], dst.shape[1])
    s = cv2.resize(src, (w, h)); d = cv2.resize(dst, (w, h))
    mse = ((s.astype(np.float32) - d.astype(np.float32)) ** 2).mean()
    return float(np.clip(1 - mse / (255.0**2), 0, 1))

def ocr_text(path):
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        res = reader.readtext(str(path))
        return [t for _, t, _ in res] if res else []
    except Exception as e:
        return [f"OCR_NA:{e}"]

def font(size=28):
    for p in [r"C:/Windows/Fonts/msyhbd.ttc", r"C:/Windows/Fonts/arial.ttf"]:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()

def label_img(img, title, sub):
    img = img.copy()
    d = ImageDraw.Draw(img)
    f1, f2 = font(34), font(24)
    d.text((14, 12), title, fill=(255, 80, 80), font=f1)
    d.text((14, 56), sub, fill=(255, 255, 255), font=f2)
    return img

def fit(path, target_h=1024):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    nw = max(1, int(w * target_h / h))
    return im.resize((nw, target_h), Image.LANCZOS)

rows = []
summary = []
for cid, refname in REFS:
    ref_path = IN / refname
    fis_path = JOB / f"v195_{cid}.jpg"
    if not fis_path.exists():
        print(f"[skip] {cid} missing fission"); continue
    ref_bgr = load_bgr(ref_path)
    fis_bgr = load_bgr(fis_path)
    hi = hist_intersection(ref_bgr, fis_bgr)
    sd = structural_diff(ref_bgr, fis_bgr)
    txts = ocr_text(fis_path)
    ocr_str = "无残留字" if not txts or (len(txts)==1 and str(txts[0]).startswith("OCR_NA")) else "残留: " + "/".join(txts)
    print(f"{cid}: 配色交集={hi:.3f} 结构相似={sd:.3f} | {ocr_str}", flush=True)
    summary.append((cid, hi, sd, ocr_str))

    left = fit(ref_path)
    right = fit(fis_path)
    # 统一高度后拼左右
    H = max(left.height, right.height); W = left.width + right.width + 40
    canvas = Image.new("RGB", (W, H), (20, 20, 20))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + 40, 0))
    canvas = label_img(canvas, cid, f"配色交集 {hi:.3f} | 结构 {sd:.3f}")
    pair_path = JOB / f"compare_v195_{cid}.jpg"
    canvas.save(str(pair_path), quality=92)
    rows.append((cid, canvas))

# 6-up 总览：每行 [原图|裂变]，3 行
def thumb(im, th=560):
    w, h = im.size
    return im.resize((max(1, int(w*th/h)), th), Image.LANCZOS)

cols = []
for cid, refname in REFS:
    ref_path = IN / refname
    fis_path = JOB / f"v195_{cid}.jpg"
    if not (JOB / f"v195_{cid}.jpg").exists():
        continue
    t_ref = thumb(Image.open(ref_path).convert("RGB"))
    t_fis = thumb(Image.open(fis_path).convert("RGB"))
    # 加白边便于区分
    def border(im, color):
        c = Image.new("RGB", (im.width+6, im.height+6), color); c.paste(im,(3,3)); return c
    cols.append((border(t_ref,(80,160,255)), border(t_fis,(255,120,120))))

row_imgs = []
for i in range(0, len(cols), 2):
    pair = cols[i:i+2]
    # 每对上下堆叠或左右？这里做 2 列：左原 右裂
    pass

# 直接做 2 列（原 | 裂）x 3 行 的网格
grid_rows = []
for i in range(0, len(cols), 2):
    r = cols[i:i+2]
    # 每行：orig0, fis0, orig1, fis1
    cells = []
    for (o, f) in r:
        cells.append(o); cells.append(f)
    # 统一这一行高度
    maxh = max(c.height for c in cells)
    line = Image.new("RGB", (sum(c.width for c in cells)+10*(len(cells)-1), maxh), (10,10,10))
    x = 0
    for c in cells:
        line.paste(c, (x, (maxh-c.height)//2)); x += c.width + 10
    grid_rows.append(line)

total = Image.new("RGB", (max(g.width for g in grid_rows), sum(g.height for g in grid_rows)+ (len(grid_rows))*10), (10,10,10))
y = 0
for g in grid_rows:
    total.paste(g, (0, y)); y += g.height + 10
# 顶栏标题
hdr = Image.new("RGB", (total.width, 60), (30,30,30))
d = ImageDraw.Draw(hdr)
d.text((14, 14), "v195 6张裂变对照  [左列=原图 | 右列=裂变]  x3行", fill=(255,210,90), font=font(30))
out = Image.new("RGB", (total.width, hdr.height+total.height), (10,10,10))
out.paste(hdr, (0,0)); out.paste(total, (0,hdr.height))
out.save(str(JOB / "_6up_compare_v195.jpg"), quality=90)
print("saved _6up_compare_v195.jpg", flush=True)

print("\n=== SUMMARY ===")
for cid, hi, sd, ocr_str in summary:
    print(f"  {cid:18s} 配色={hi:.3f} 结构={sd:.3f} | {ocr_str}")
