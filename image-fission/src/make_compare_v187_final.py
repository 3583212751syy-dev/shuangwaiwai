"""
make_compare_v187_final.py — 最终交付对照拼图
- 18 张原图 vs 裂变对照（v185 12 张 + v186 5 张 + 1 张 = 18 行）
- 4 张烧字对照（fireball/bat_logo/camo_armed 烧字前后 + dark_knight 如果涉及）
"""

import os
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = r"E:/Desktop/双接口/image-fission/jobs"

# 18 张：id → 路径
COMPARE = [
    # v185 已完成的 12 张（除 6 张失败的 + 1 张违规大麻）
    ("hiphop_baby",      "v185", "input/test_0c4719b1bdd76d452559fc4586a6a3cd.jpg"),
    ("orange_paisley",   "v186", "input/test_13c8b7bf8dae757e6c2d4b3d6a860f9d.jpg"),  # v186
    ("racing",           "v186", "input/test_4a20ce8e280a44056c80fe6e83b676d3.jpg"),  # v186
    ("dark_knight",      "v186", "input/test_66156da5bae4b2fc1c4e82f78cfa8a55.jpg"),  # v186
    ("fireball_skull",   "v185", "input/test_581f43423ef2d71d4447c0f634411138.jpg"),
    ("bat_logo",         "v186", "input/test_6978fabda2cc99629fa9e81f802762d3.jpg"),  # v186
    ("lava_dino",        "v185", "input/test_754bd0928c6b51ccdb66f161aa411f2f.jpg"),
    ("limitless_splash", "v185", "input/test_793cf2eb0c1dd7603cd043b163ab4935.jpg"),
    ("orange_pines",     "v185", "input/test_7b445250f19d58a07b612500bb43be1d.jpg"),
    ("corset_goth",      "v185", "input/test_7c79f3bb333c1680e6399d04347ade6c.jpg"),
    ("tactical_vest",    "v185", "input/test_a4be80676f0f4e9e79d71a6eafa5a7ea.jpg"),
    ("x_lion",           "v185", "input/test_b70cf19ba9c6c45c4e022c1d6260e9c4.jpg"),
    ("mech_gear",        "v185", "input/test_d77d9a9d22aaf6f1bb1f20c34d21cefa.jpg"),
    ("checker_vortex",   "v185", "input/test_e887bc80eaf6f68ee30bb1b6b18ad607.jpg"),
    ("camo_armed",       "v186", "input/test_b78e60de8dfdf44acda99395326a7298.jpg"),  # v186
    ("ace_card",         "v185", "input/test_d2c8e4ebd8c79e87a26c5ea6e74fdf28.jpg"),
    ("paris_stripes",    "v185", "input/test_eedce8d9d99aa15f1b1a7c6f57fa0f6b.jpg"),
    ("eagle_skull_chain","v185", "input/test_efee1a99ce69acf5ed40fb9b39aa0a89.jpg"),
]

# 烧字对照（4 张：fireball_v185 + bat_logo_v186 + camo_armed_v186）
BURN_COMPARE = [
    ("fireball_skull", "v185", "input/test_581f43423ef2d71d4447c0f634411138.jpg",
     "v185_fireball_skull_burned.jpg", "SKULLFIRE"),
    ("bat_logo",       "v186", "input/test_6978fabda2cc99629fa9e81f802762d3.jpg",
     "v186_bat_logo_burned.jpg", "BATANO + MSOUL"),
    ("camo_armed",     "v186", "input/test_b78e60de8dfdf44acda99395326a7298.jpg",
     "v186_camo_armed_burned.jpg", "WE HONOR THE BRAVE"),
]


def thumb(src, w=520, h=666):
    """缩小到统一宽高 (520x666 接近 4:5 比例)"""
    try:
        im = Image.open(src).convert("RGB")
        im.thumbnail((w, h), Image.LANCZOS)
        return im
    except Exception as e:
        print(f"[warn] load fail {src}: {e}")
        return Image.new("RGB", (w, h), (200, 200, 200))


def grid(rows, cols, gap=8, header_h=36, label_h=24):
    """构造 Nx2 或 Nx3 拼图"""
    W = cols * 520 + (cols + 1) * gap
    H = len(rows) * (666 + header_h + label_h) + (len(rows) + 1) * gap
    out = Image.new("RGB", (W, H), (245, 30, 30))  # 红边警示
    draw = ImageDraw.Draw(out)
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            x = gap + ci * (520 + gap)
            y = gap + ri * (666 + header_h + label_h) + header_h
            im = thumb(cell["img"])
            out.paste(im, (x, y))
            # label
            lx = x + 6
            ly = y - header_h + 4
            draw.text((lx, ly), cell.get("label", ""), fill=(255, 255, 255))
    return out


def main_compare():
    """18 行 × 2 列：源图 | 裂变"""
    rows = []
    for (rid, ver, src_rel) in COMPARE:
        src_full = os.path.join(OUT_DIR, "..", "ComfyUI", src_rel).replace("/", "\\")
        src_full = os.path.normpath(src_full)
        gen_full = os.path.join(OUT_DIR, "smoke_v185", f"v185_{rid}.jpg")
        if ver == "v186":
            gen_full = os.path.join(OUT_DIR, "smoke_v186", f"v186_{rid}.jpg")
        rows.append([
            {"label": f"[原] {rid}", "img": src_full},
            {"label": f"[{ver}] {rid}", "img": gen_full},
        ])
    img = grid(rows, cols=2)
    img.save(os.path.join(OUT_DIR, "smoke_v185", "_compare_final_18.jpg"), quality=90)
    print(f"saved _compare_final_18.jpg  size={img.size}")


def burn_compare():
    """3 行 × 3 列：源图 | 裂变 | 烧字"""
    rows = []
    for (rid, ver, src_rel, burned_rel, label) in BURN_COMPARE:
        src_full = os.path.normpath(os.path.join(OUT_DIR, "..", "ComfyUI", src_rel))
        gen_full = os.path.join(OUT_DIR, "smoke_v185", f"v185_{rid}.jpg")
        if ver == "v186":
            gen_full = os.path.join(OUT_DIR, "smoke_v186", f"v186_{rid}.jpg")
        burned_full = os.path.join(OUT_DIR, "smoke_v186" if ver == "v186" else "smoke_v185", burned_rel)
        rows.append([
            {"label": f"[原] {rid}", "img": src_full},
            {"label": f"[{ver}] {rid}", "img": gen_full},
            {"label": f"[烧字] {label}", "img": burned_full},
        ])
    img = grid(rows, cols=3)
    img.save(os.path.join(OUT_DIR, "smoke_v185", "_compare_burned_final_3.jpg"), quality=90)
    print(f"saved _compare_burned_final_3.jpg  size={img.size}")


if __name__ == "__main__":
    main_compare()
    burn_compare()