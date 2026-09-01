"""v178 对照：原图 vs 裂变图（正确基线，不是裂变比裂变）
1. denim:  ORIGINAL | v178 fission
2. camo:   ORIGINAL | v174 fission
3. 全 6 张：每行 = 原图 | 裂变，2 行 × 6 列
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent / "outputs" / "final"
GALLERY = HERE.parent / "web_gallery" / "img"
OUT = HERE.parent / "outputs" / "v178"
OUT.mkdir(parents=True, exist_ok=True)

BG = (14, 15, 18)
GAP = 12
LABEL_H = 46
H = 760


def fit(im, h):
    w, ih = im.size
    return im.resize((int(w * h / ih), h), Image.LANCZOS)


def get_font(sz=20):
    for p in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/arial.ttf"]:
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def two_up(orig_path, fiss_path, out_path, title_orig, title_fiss):
    a = fit(Image.open(orig_path).convert("RGB"), H)
    b = fit(Image.open(fiss_path).convert("RGB"), H)
    W = a.width + b.width + GAP
    canvas = Image.new("RGB", (W, H + LABEL_H), BG)
    canvas.paste(a, (0, LABEL_H))
    canvas.paste(b, (a.width + GAP, LABEL_H))
    d = ImageDraw.Draw(canvas)
    f = get_font(20)
    d.text((8, 8), title_orig, fill=(230, 234, 240), font=f)
    d.text((a.width + GAP + 8, 8), title_fiss, fill=(255, 106, 61), font=f)
    canvas.save(out_path, "JPEG", quality=90, optimize=True)
    print(f"  ✓ {out_path} ({out_path.stat().st_size/1024/1024:.2f}MB)")


def orig_vs_fiss_6up(out_path):
    ids = ["eagle_2", "metal_6", "camo_4", "denim_3", "illust_1", "skull_5"]
    cell_h = 600
    cell_w = 560
    W = cell_w * 6 + GAP * 5
    H_total = (cell_h + LABEL_H + 30) * 2 + GAP
    canvas = Image.new("RGB", (W, H_total), BG)
    d = ImageDraw.Draw(canvas)
    f = get_font(19)
    f2 = get_font(12)
    for i, rid in enumerate(ids):
        col = i
        # 原图行
        oy = 0
        ox = col * (cell_w + GAP)
        oimg = fit(Image.open(GALLERY / f"orig_{rid}.jpg").convert("RGB"), cell_h)
        oimg = oimg.resize((cell_w, int(oimg.height * cell_w / oimg.width)), Image.LANCZOS)
        oimg = oimg.crop((0, 0, cell_w, min(cell_h, oimg.height)))
        canvas.paste(oimg, (ox, oy + LABEL_H + 26))
        d.text((ox + 6, oy + 6), f"原图 {rid}", fill=(230, 234, 240), font=f)
        # 裂变行
        fy = cell_h + LABEL_H + 30 + GAP
        fimg = fit(Image.open(FINAL / f"{rid}.jpg").convert("RGB"), cell_h)
        fimg = fimg.resize((cell_w, int(fimg.height * cell_w / fimg.width)), Image.LANCZOS)
        fimg = fimg.crop((0, 0, cell_w, min(cell_h, fimg.height)))
        canvas.paste(fimg, (ox, fy + LABEL_H + 26))
        d.text((ox + 6, fy + 6), f"裂变 {rid}", fill=(255, 106, 61), font=f)
        d.text((ox + 6, fy + 28), "v178" if rid in ("camo_4", "denim_3") else "v147基线", fill=(120, 180, 255), font=f2)
    canvas.save(out_path, "JPEG", quality=88, optimize=True)
    print(f"  ✓ {out_path} ({out_path.stat().st_size/1024/1024:.2f}MB)")


# 1) denim 原图 vs v178
two_up(GALLERY / "orig_denim_3.jpg", FINAL / "denim_3.jpg",
       OUT / "compare_denim_ORIG_vs_v178.jpg",
       "原图 denim_3（碎牛仔拼贴蝴蝶）", "v178 裂变（LORA=0 + INTACT 干净蝴蝶）")

# 2) camo 原图 vs v174
two_up(GALLERY / "orig_camo_4.jpg", FINAL / "camo_4.jpg",
       OUT / "compare_camo_ORIG_vs_v174.jpg",
       "原图 camo_4（圆润迷彩 + 棕榈）", "v174 裂变（LORA=0 圆润 + v147 配色）")

# 3) 全 6 张 原图|裂变
orig_vs_fiss_6up(OUT / "orig_vs_fiss_6up.jpg")

print("\nDONE: 原图 vs 裂变 对照已生成")
