"""
v307_pipeline.py — 同元素 AI 重生成（裂变）管线
处理两张图：FIREBALL(金色骷髅头) / ARMED FORCES(灰迷彩+狗牌)
策略：主体 AI 重绘 + 原图版式框架/文字同元素保留
- FIREBALL: 原图框架保留，AI 骷髅替换左侧主体区
- ARMED:    AI 迷彩+狗牌作底，原图纯黑字原位覆盖
用法: python v307_pipeline.py
"""
import numpy as np
from PIL import Image, ImageFilter

SRC_FIREBALL = r"E:\迁移\Documents\My Pictures\Saved Pictures\歪歪.library\images\MTI8C5CX8FFVT.info\581f43423ef2d71d4447c0f634411138.jpg"
SRC_ARMED    = r"E:\迁移\Documents\My Pictures\Saved Pictures\歪歪.library\images\MTI8BUIRMMKHF.info\b78e60de8dfdf44acda99395326a7298.jpg"
GEN_SKULL    = r"E:\Desktop\双接口\image-fission\jobs\v307\Majestic_golden_skull_emblem___2026-09-08T02-44-18.png"
GEN_CAMO     = r"E:\Desktop\双接口\image-fission\jobs\v307\Military_camouflage_pattern_te_2026-09-08T02-44-51.png"
OUT_FIRE     = r"E:\Desktop\双接口\image-fission\jobs\v307\v307_fireball_final.png"
OUT_ARMED    = r"E:\Desktop\双接口\image-fission\jobs\v307\v307_armed_final.png"

def feather_mask(mask, r=6):
    m = (mask * 255).astype(np.uint8)
    im = Image.fromarray(m).filter(ImageFilter.GaussianBlur(r))
    return np.clip(np.asarray(im, dtype=np.float32) / 255.0, 0, 1)

# ---------- FIREBALL ----------
def process_fireball():
    src = np.asarray(Image.open(SRC_FIREBALL).convert("RGB"), dtype=np.float32)
    h, w = src.shape[:2]
    gen = Image.open(GEN_SKULL).convert("RGB")
    ga = np.asarray(gen, dtype=np.float32)
    gh = ga.mean(2)
    # 骷髅 mask: 非纯黑(>25)即骷髅；眼窝黑(<25)透明→原黑底透出
    skull = (gh > 25)
    # 左侧替换区（避开右侧金字 x>850）
    rx0, ry0, rx1, ry1 = 80, 200, 850, 1480
    rw, rh = rx1-rx0, ry1-ry0
    # AI 骷髅保持比例 fit into (rw,rh)
    gh_, gw_ = gen.size[1], gen.size[0]
    scale = min(rw/gw_, rh/gh_)
    nw, nh = int(gw_*scale), int(gh_*scale)
    gen_r = gen.resize((nw, nh))
    skull_r = np.asarray(gen_r, dtype=np.float32).mean(2) > 25
    # 居中放入 rect
    ox = rx0 + (rw-nw)//2
    oy = ry0 + (rh-nh)//2
    fm = feather_mask(skull_r, 7)
    # 贴回
    region = src[oy:oy+nh, ox:ox+nw].copy()
    gen_a = np.asarray(gen_r, dtype=np.float32)
    for c in range(3):
        region[:,:,c] = region[:,:,c]*(1-fm) + gen_a[:,:,c]*fm
    out = src.copy()
    out[oy:oy+nh, ox:ox+nw] = region
    Image.fromarray(out.astype(np.uint8)).save(OUT_FIRE)
    print(f"[FIREBALL] skull placed at x{ox}-{ox+nw} y{oy}-{oy+nh}, scale={scale:.3f}")

# ---------- ARMED ----------
def process_armed():
    src = np.asarray(Image.open(SRC_ARMED).convert("RGB"), dtype=np.float32)
    h, w = src.shape[:2]
    gen = Image.open(GEN_CAMO).convert("RGB").resize((w, h))
    base = np.asarray(gen, dtype=np.float32)
    gs = src.mean(2)
    # 原图纯黑字 (g<40)，含中央 ARMED FORCES + 顶部 WE SUPPORT THE
    text_mask = (gs < 40)
    out = base.copy()
    out[text_mask] = src[text_mask]
    Image.fromarray(out.astype(np.uint8)).save(OUT_ARMED)
    print(f"[ARMED] AI camo+dogtag base, black text overlaid: {int(text_mask.sum())} px")

if __name__ == "__main__":
    process_fireball()
    process_armed()
    print("DONE")
