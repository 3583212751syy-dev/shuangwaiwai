"""v178 对照图构建：
1. camo  before/after:  v177(块状+棕褐) | v174(圆润+深绿棕沙) ←final
2. denim before/after:  v177(破布拼贴) | v178(完整干净蝴蝶) ←final
3. _6up_compare.jpg     全 6 张终版
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent / "outputs" / "final"
GALLERY = HERE.parent / "web_gallery" / "img"
OUT = HERE.parent / "outputs" / "v178"
OUT.mkdir(parents=True, exist_ok=True)

BG = (14, 15, 18)
H = 720  # 对照图每张高
GAP = 16
LABEL_H = 56


def fit(im, h):
    w, ih = im.size
    nw = int(w * h / ih)
    return im.resize((nw, h), Image.LANCZOS)


def get_font(sz=22):
    for p in ["arial.ttf", "C:/Windows/Fonts/arial.ttf",
              "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyh.ttf"]:
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def two_up(left_path, right_path, out_path, left_label, right_label,
           right_tag="v178 final", left_tag="v177 旧版"):
    a = fit(Image.open(left_path).convert("RGB"), H)
    b = fit(Image.open(right_path).convert("RGB"), H)
    W = a.width + b.width + GAP
    canvas = Image.new("RGB", (W, H + LABEL_H), BG)
    canvas.paste(a, (0, LABEL_H))
    canvas.paste(b, (a.width + GAP, LABEL_H))
    d = ImageDraw.Draw(canvas)
    f = get_font(20)
    f2 = get_font(13)
    d.text((8, 6), left_label, fill=(154, 160, 171), font=f)
    d.text((8, 32), left_tag, fill=(120, 120, 130), font=f2)
    d.text((a.width + GAP + 8, 6), right_label, fill=(255, 106, 61), font=f)
    d.text((a.width + GAP + 8, 32), right_tag, fill=(120, 180, 255), font=f2)
    canvas.save(out_path, "JPEG", quality=90, optimize=True)
    print(f"  ✓ {out_path} ({out_path.stat().st_size/1024/1024:.2f}MB)")


def six_up(out_path):
    ids = ["eagle_2", "metal_6", "camo_4", "denim_3", "illust_1", "skull_5"]
    labels = {
        "eagle_2": "EAGLE",
        "metal_6": "METAL",
        "camo_4": "CAMO",
        "denim_3": "DENIM",
        "illust_1": "ILLUST",
        "skull_5": "SKULL",
    }
    imgs = [Image.open(FINAL / f"{i}.jpg").convert("RGB") for i in ids]
    # 3 行 × 2 列
    cell_w = 600
    rows = 3
    cell_h = H
    W = cell_w * 2 + GAP
    H_total = (cell_h + LABEL_H) * rows + GAP * (rows - 1)
    canvas = Image.new("RGB", (W, H_total), BG)
    f = get_font(22)
    d = ImageDraw.Draw(canvas)
    for idx, (rid, im) in enumerate(zip(ids, imgs)):
        r, c = divmod(idx, 2)
        x = c * (cell_w + GAP)
        y = r * (cell_h + LABEL_H + GAP)
        thumb = fit(im, cell_h)
        if thumb.width > cell_w:
            thumb = fit(im, cell_h)
            scale = cell_w / thumb.width
            thumb = thumb.resize((cell_w, int(thumb.height * scale)), Image.LANCZOS)
        # 居中
        tx = x + (cell_w - thumb.width) // 2
        canvas.paste(thumb, (tx, y + LABEL_H))
        d.text((x + 12, y + 12), f"{labels[rid]}", fill=(255, 106, 61), font=f)
        d.text((x + 12, y + 40), rid, fill=(120, 120, 130), font=get_font(13))
    canvas.save(out_path, "JPEG", quality=88, optimize=True)
    print(f"  ✓ {out_path} ({out_path.stat().st_size/1024/1024:.2f}MB)")


# === 1) camo before/after ===
# 左 v177（旧 final/ 已被覆盖）→ 从历史 v177 找；v177 final 已被 v178 覆盖，
# 所以用 v177_outputs 里的 camo_4_raw 或 ct 版。camo v177 的 final 是 CT s=1.30 版。
# v177 输出在 outputs/v177/，文件名 camo_4_ct_s1.30.png
v177_camo = OUT / "_v177_camo_legacy.png"
v177_src = HERE.parent / "outputs" / "v177" / "camo_4_ct_s1.30.png"
if v177_src.exists():
    Image.open(v177_src).convert("RGB").save(v177_camo)
else:
    # fallback: 用 web_gallery 当前 fiss_camo_4 是 v178 final，不能用
    # 用原图临时占位
    Image.open(GALLERY / "orig_camo_4.jpg").convert("RGB").save(v177_camo)

two_up(v177_camo,
       FINAL / "camo_4.jpg",
       OUT / "compare_camo_v177_vs_v178.jpg",
       "迷彩 旧版：硬多边形 + 棕褐单色（丢深绿）",
       "迷彩 v178：圆润有机色块 + 深绿/棕/沙齐全")

# === 2) denim before/after ===
v177_denim = OUT / "_v177_denim_legacy.png"
v177_src_d = HERE.parent / "outputs" / "v177" / "denim_3_raw.png"
if v177_src_d.exists():
    Image.open(v177_src_d).convert("RGB").save(v177_denim)
else:
    Image.open(GALLERY / "orig_denim_3.jpg").convert("RGB").save(v177_denim)

two_up(v177_denim,
       FINAL / "denim_3.jpg",
       OUT / "compare_denim_v177_vs_v178.jpg",
       "蝴蝶 旧版：破布拼贴（提示自己写了 PATCHES + FRAYED）",
       "蝴蝶 v178：完整干净对称蝴蝶（提示重写 INTACT/CLEAN）")

# === 3) 6-up 全集 ===
six_up(FINAL / "_6up_compare.jpg")

print("\nALL compare images built.")
