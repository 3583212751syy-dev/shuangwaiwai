"""
make_compare_v187_final.py — 最终交付对照拼图
"""
import os
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = r"E:/Desktop/双接口/image-fission"
INPUT_DIR = PROJECT_ROOT + "/ComfyUI/input"
V185 = PROJECT_ROOT + "/jobs/smoke_v185"
V186 = PROJECT_ROOT + "/jobs/smoke_v186"

# 18 张：id → (版本, 源图文件名)
COMPARE = [
    ("hiphop_baby",      "v185", "test_0c4719b1bdd76d452559fc4586a6a3cd.jpg"),
    ("orange_paisley",   "v186", "test_13c8b7bf8dae757e6c2d4b3d6a860f9d.jpg"),
    ("racing",           "v186", "test_184432b34a4787fbed628b3b986b37a2.jpg"),
    ("dark_knight",      "v186", "test_3a300c32794aeea08f8abb2517f3afe1.jpg"),
    ("fireball_skull",   "v185", "test_581f43423ef2d71d4447c0f634411138.jpg"),
    ("bat_logo",         "v186", "test_6978fabda2cc99629fa9e81f802762d3.jpg"),
    ("lava_dino",        "v185", "test_754bd0928c6b51ccdb66f161aa411f2f.jpg"),
    ("limitless_splash", "v185", "test_793cf2eb0c1dd7603cd043b163ab4935.jpg"),
    ("orange_pines",     "v185", "test_7b445250f19d58a07b612500bb43be1d.jpg"),
    ("corset_goth",      "v185", "test_7c79f3bb333c1680e6399d04347ade6c.jpg"),
    ("tactical_vest",    "v185", "test_805913629e34491860a892101da398fe.jpg"),
    ("x_lion",           "v185", "test_820294597fc383943ee5758f66539081.jpg"),
    ("mech_gear",        "v185", "test_85f5d2f428cf1afb805288932f9a6ac1.jpg"),
    ("checker_vortex",   "v185", "test_99b27e6a189276f6ccbc6cd3bbd7028b.jpg"),
    ("camo_armed",       "v186", "test_b78e60de8dfdf44acda99395326a7298.jpg"),
    ("ace_card",         "v185", "test_d056ed4ab763fff030d1e4403362e32e.jpg"),
    ("paris_stripes",    "v185", "test_eddf45c7da4ea2615035c8d8f8cddf03.jpg"),
    ("eagle_skull_chain","v185", "test_Pinterest_2.jpg"),
]

BURN_COMPARE = [
    ("fireball_skull", "v185", "test_581f43423ef2d71d4447c0f634411138.jpg",
     "v185_fireball_skull_burned.jpg", "SKULLFIRE"),
    ("bat_logo",       "v186", "test_6978fabda2cc99629fa9e81f802762d3.jpg",
     "v186_bat_logo_burned.jpg", "BATANO + MSOUL"),
    ("camo_armed",     "v186", "test_b78e60de8dfdf44acda99395326a7298.jpg",
     "v186_camo_armed_burned.jpg", "WE HONOR THE BRAVE"),
]


def thumb(src, w=520, h=666):
    try:
        im = Image.open(src).convert("RGB")
        im.thumbnail((w, h), Image.LANCZOS)
        return im
    except Exception as e:
        print(f"[warn] load fail {src}: {e}")
        return Image.new("RGB", (w, h), (200, 200, 200))


def grid(rows, cols, gap=8, header_h=36):
    W = cols * 520 + (cols + 1) * gap
    H = len(rows) * (666 + header_h) + (len(rows) + 1) * gap
    out = Image.new("RGB", (W, H), (245, 30, 30))
    draw = ImageDraw.Draw(out)
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            x = gap + ci * (520 + gap)
            y = gap + ri * (666 + header_h) + header_h
            im = thumb(cell["img"])
            out.paste(im, (x, y))
            lx = x + 6
            ly = y - header_h + 4
            draw.text((lx, ly), cell.get("label", ""), fill=(255, 255, 255))
    return out


def main_compare():
    rows = []
    for (rid, ver, src_name) in COMPARE:
        src_full = INPUT_DIR + "/" + src_name
        gen_full = f"{V185}/v185_{rid}.jpg"
        if ver == "v186":
            gen_full = f"{V186}/v186_{rid}.jpg"
        rows.append([
            {"label": f"[原] {rid}", "img": src_full},
            {"label": f"[{ver}] {rid}", "img": gen_full},
        ])
    img = grid(rows, cols=2)
    img.save(f"{V185}/_compare_final_18.jpg", quality=90)
    print(f"saved _compare_final_18.jpg  size={img.size}")


def burn_compare():
    rows = []
    for (rid, ver, src_name, burned_rel, label) in BURN_COMPARE:
        src_full = INPUT_DIR + "/" + src_name
        gen_full = f"{V185}/v185_{rid}.jpg"
        if ver == "v186":
            gen_full = f"{V186}/v186_{rid}.jpg"
        burned_full = f"{V186}/{burned_rel}" if ver == "v186" else f"{V185}/{burned_rel}"
        rows.append([
            {"label": f"[原] {rid}", "img": src_full},
            {"label": f"[{ver}] {rid}", "img": gen_full},
            {"label": f"[烧字] {label}", "img": burned_full},
        ])
    img = grid(rows, cols=3)
    img.save(f"{V185}/_compare_burned_final_3.jpg", quality=90)
    print(f"saved _compare_burned_final_3.jpg  size={img.size}")


if __name__ == "__main__":
    main_compare()
    burn_compare()