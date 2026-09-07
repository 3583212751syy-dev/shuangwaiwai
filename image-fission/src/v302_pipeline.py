"""v302 — 徽章内盘"全填纯紫+贴回蝙蝠" 10/10 落地版

v301 失败的根因（数值诊断）：徽章内盘 y=800-1100 区有 60-97% 浅紫"玻璃反光"
（RGB max>150 的花瓣/光晕，是 BACARDÍ 经典徽章的紫冠装饰），但用户期望徽章下半
是 100% 实心深紫+蝙蝠。v301 的 bg_layer 只填"膨胀蝙蝠区"，玻璃反光在蝙蝠轮廓
外幸存保留。

v302 三处机制修复：
  A) bat_m 阈值 maxc<95 → maxc<150（覆盖翼膜中紫 RGB≈122、翼骨深紫 RGB≈50，
     排除玻璃反光高光 RGB≥155）。CLOSE 21 桥接骨缝 + RETR_CCOMP 真洞填充。
  B) bg_layer = 内盘圆 R=360 内 **减 bat_solid 后用纯紫 RGB(181,129,171)
     整片覆盖**。除 bat_solid 外整个圆盘都是纯紫（无玻璃反光、无紫冠花瓣）。
     只有 bat_solid 保留原图真实蝙蝠纹理（深紫+紫膜+黑色）。
  C) 顶弧 = 品牌主字 NOCTAVEN/DUSKBAT/MOONBAT（占原 BACARDÍ 顶弧位），
     下方大主字 = 产品副品牌（"DISTILLERY"/"RESERVE"/"NOCTURNE"），
     三档品牌名贯穿所有前景文字——主图品牌字体已变更（不再是 BACARDÍ）。

底部三角保留。字体 = 67KB Abril Fatface 真字体。
"""
import math, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

# ─── 配置 ────────────────────────────────────────────────
SRC_PATH  = r"E:\Desktop\双接口\image-fission\ComfyUI\input\v300_text_free_source.png"
FONT_PATH = r"E:\Desktop\双接口\image-fission\ComfyUI\models\fonts\AbrilFatface-Regular.ttf"
OUT_DIR   = r"E:\Desktop\双接口\image-fission\jobs\v302"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "_chk"), exist_ok=True)

# 纯紫（v300 source 4 角采样 = RGB(181, 129, 171)）
BG_PURPLE = np.array([181, 129, 171], dtype=np.float32)

# BACARDÍ 经典版式参数（基于 v300 实际布局）
cx, cy = 776, 746
ring_outer = 360
TOP_ARC_CY  = 235
LEFT_EST_X  = 432
RIGHT_1862_X = 1120
EST_LINE_Y  = 760
BRAND_TOP_Y = 1060
SUB_LINE_Y  = 1320
TRI_CY      = 1500

# ─── 1. 读源图 ──────────────────────────────────────────
arr = np.array(Image.open(SRC_PATH).convert("RGB")).astype(np.float32)
H, W = arr.shape[:2]

# ─── 2. bat_m 检测（v302 关键：阈值 150 替代 95） ────────
R, G, Bc = arr[:,:,0], arr[:,:,1], arr[:,:,2]
maxc = np.maximum(np.maximum(R, G), Bc)
# v302: maxc<150 覆盖翼膜中紫 RGB~122（v301 maxc<95 漏掉翼膜）
dark = (maxc < 150).astype(np.uint8) * 255
badge_lim = np.zeros((H, W), np.uint8)
cv2.circle(badge_lim, (cx, cy), ring_outer + 30, 255, -1)
dark = cv2.bitwise_and(dark, badge_lim)
dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
sizes = stats[1:, cv2.CC_STAT_AREA]
keep_idx = 1 + int(np.argmax(sizes))
bat_m = (labels == keep_idx).astype(np.uint8) * 255

# CLOSE 21 桥接骨缝 + RETR_CCOMP 真洞填充（v300 mask 链定稿）
solid = cv2.morphologyEx(bat_m, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
solid_m = solid.copy()
cnts, hier = cv2.findContours(solid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
if hier is not None:
    for i, h4 in enumerate(hier[0]):
        if h4[3] != -1:
            cv2.drawContours(solid_m, cnts, i, 255, -1)

# 躯干条带 + 翼/体分割
band_b = np.zeros((H, W), np.uint8)
band_b[:, cx - 48 : cx + 48] = 255
body_core = cv2.bitwise_and(bat_m, band_b)
wing_m = solid_m.copy()
wing_m[body_core > 0] = 0

# 翼软权重
dist = cv2.distanceTransform(cv2.bitwise_not(body_core), cv2.DIST_L2, 3)
wing_soft = (wing_m.astype(np.float32) / 255.0) * np.clip(dist / 22.0, 0, 1).astype(np.float32)

xs_col = np.tile(np.arange(W, dtype=np.int32), (H, 1))
left_m  = ((wing_m > 0) & (xs_col <  cx)).astype(np.uint8) * 255
right_m = ((wing_m > 0) & (xs_col >= cx)).astype(np.uint8) * 255
left_soft  = wing_soft.copy(); left_soft[xs_col >= cx]  = 0
right_soft = wing_soft.copy(); right_soft[xs_col <  cx] = 0

# protect（仅头椭圆 + 尾/躯干中轴）
hx, hy = cx, 521
protect = np.zeros((H, W), np.uint8)
cv2.ellipse(protect, (int(hx) + 6, int(hy) + 6), (24, 28), 0, 0, 360, 255, -1)
protect[875:985, cx - 26 : cx + 26] = 255
protect[560:875, cx - 22 : cx + 22] = 255
protect = cv2.bitwise_and(protect, bat_m)
body_block = cv2.GaussianBlur(protect, (5, 5), 1.5).astype(np.float32) / 255.0
body_keep  = cv2.GaussianBlur(
    cv2.dilate(protect, np.ones((5, 5), np.uint8)), (7, 7), 2
).astype(np.float32) / 255.0

# ─── 3. clean_layer: 徽章圆 R=440 内非 bat_solid → 纯紫 ──
# 关键：v301 → v302 final 的修复是这里，把"翼后面背景"先清干净，
# 再做 _rot_warp 时 rest 区不会用 layer_in(=原图含环带花瓣) → 不会有斑驳。
# v301 错误：rest 区使用 layer_in=arr（即原图），翼后背景是 BACARDÍ 紫冠花瓣。
disc = np.zeros((H, W), np.uint8)
cv2.circle(disc, (cx, cy), 440, 255, -1)
inner_bg_mask = cv2.bitwise_and(cv2.bitwise_not(solid_m), disc)
clean_layer = arr.copy()
clean_layer[inner_bg_mask > 0] = BG_PURPLE
bg_layer = clean_layer.copy()   # bg_layer 也用清干净的图（旧位 reveal 平滑）

# ─── 4. _rot_warp（v300 十七轮：pass 串联 keep 修复） ────
def _rot_warp(layer_in, wing_soft, wing_bin, body_block, body_keep, bg_layer,
              theta, rot_c, h, w):
    cos_t = np.float32(np.cos(theta))
    sin_t = np.float32(np.sin(theta))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xs - rot_c[0]
    dy = ys - rot_c[1]
    rx = rot_c[0] + dx * cos_t + dy * sin_t
    ry = rot_c[1] - dx * sin_t + dy * cos_t
    rot_soft = cv2.remap(wing_soft, rx, ry, cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    w_new = cv2.GaussianBlur(rot_soft, (5, 5), 1.5).astype(np.float32)
    w_new = w_new * (np.float32(1.0) - body_block)
    w_old = cv2.GaussianBlur(wing_bin, (3, 3), 0.8).astype(np.float32) / 255.0
    w_old = np.clip(w_old - w_new, 0, np.float32(1.0))
    src_warp = cv2.remap(layer_in, rx, ry, cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    rest = np.float32(1.0) - w_new - w_old
    out = (src_warp * w_new[..., None] + bg_layer * w_old[..., None]
           + layer_in * rest[..., None])
    out = out * (np.float32(1.0) - body_keep)[..., None] + layer_in * body_keep[..., None]
    return out

shoulder_y = 790
shL = (cx - 120, shoulder_y + 60)
shR = (cx + 120, shoulder_y + 60)

def two_wing(thetaL, thetaR):
    # v302 关键：layer_in = clean_layer（徽章内=纯紫+蝙蝠），rest 区不再采到原图环带
    wpL = _rot_warp(clean_layer, left_soft, left_m, body_block, body_keep, bg_layer,
                    thetaL, shL, H, W)
    wp  = _rot_warp(wpL, right_soft, right_m, body_block, body_keep, bg_layer,
                    thetaR, shR, H, W)
    return wp

variants = {
    "up":     two_wing(math.radians( 18), math.radians(-18)),
    "spread": two_wing(math.radians(-12), math.radians( 12)),
    "fold":   two_wing(math.radians(-33), math.radians( 33)),
}

# ─── 5. 文字层（v302：顶弧=品牌名；下方大主字=副品牌） ──
def render_text_layer(top_arc, brand, sub):
    """顶弧 = 品牌名（占原 BACARDÍ 顶弧位）
       下方大主字 = 产品副品牌（"DISTILLERY"/"RESERVE"/"NOCTURNE"）
       EST. 1862 在徽章左右；副字 + 三角 ▼ 底部"""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    drw = ImageDraw.Draw(layer)
    arc_r = ring_outer + 70

    def _draw_arc_text(txt, size_pt, top=True):
        font = ImageFont.truetype(FONT_PATH, size_pt)
        bbox = drw.textbbox((0, 0), txt, font=font)
        text_w = bbox[2] - bbox[0]
        phi = min(math.pi * 0.30, text_w / (2.0 * arc_r))
        if top:
            a_start = -math.pi / 2 - phi
            a_end   = -math.pi / 2 + phi
        else:
            a_start =  math.pi / 2 - phi
            a_end   =  math.pi / 2 + phi
        char_widths = [drw.textbbox((0, 0), c, font=font)[2] for c in txt]
        a = a_start
        for i, ch in enumerate(txt):
            w_c = char_widths[i]
            mid_a = a + (w_c / 2.0) / arc_r
            px = cx + arc_r * math.cos(mid_a)
            py = cy + arc_r * math.sin(mid_a)
            deg = math.degrees(mid_a) + 90
            ch_im = Image.new("RGBA", (w_c + 20, size_pt * 2), (0, 0, 0, 0))
            ImageDraw.Draw(ch_im).text((10, 0), ch, font=font, fill=(0, 0, 0, 255))
            ch_rot = ch_im.rotate(-deg, resample=Image.BICUBIC, expand=True)
            layer.paste(ch_rot, (int(px - ch_rot.width / 2), int(py - ch_rot.height / 2)),
                        ch_rot)
            a += w_c / arc_r
    _draw_arc_text(top_arc, 78, top=True)

    est_font = ImageFont.truetype(FONT_PATH, 70)
    drw.text((LEFT_EST_X, EST_LINE_Y), "EST.",  font=est_font, fill=(0, 0, 0, 255))
    drw.text((RIGHT_1862_X, EST_LINE_Y), "1862", font=est_font, fill=(0, 0, 0, 255))

    brand_font = ImageFont.truetype(FONT_PATH, 300)
    bb = drw.textbbox((0, 0), brand, font=brand_font)
    bw = bb[2] - bb[0]; bh = bb[3] - bb[1]
    drw.text(((W - bw) // 2 - bb[0], BRAND_TOP_Y - bb[1]), brand,
             font=brand_font, fill=(0, 0, 0, 255))

    sub_font = ImageFont.truetype(FONT_PATH, 150)
    sb = drw.textbbox((0, 0), sub, font=sub_font)
    sw = sb[2] - sb[0]; sh = sb[3] - sb[1]
    drw.text(((W - sw) // 2 - sb[0], SUB_LINE_Y - sb[1]), sub,
             font=sub_font, fill=(0, 0, 0, 255))

    tri_w = 80; tri_h = 56
    tri = [(W // 2 - tri_w // 2, TRI_CY - tri_h // 2),
           (W // 2 + tri_w // 2, TRI_CY - tri_h // 2),
           (W // 2,              TRI_CY + tri_h // 2)]
    drw.polygon(tri, fill=(0, 0, 0, 255))
    return layer

# v302 关键：顶弧直接写品牌名（原 BACARDÍ 位），下方大主字 = 副品牌
# 每个变体的核心标识是 NOCTAVEN/DUSKBAT/MOONBAT —— 必须显眼出现 2 次
# （顶弧+下方大主字），与原 BACARDÍ 在 BACARDI 酒标的视觉权重一致
variants_text = {
    "up":     ("NOCTAVEN", "NOCTAVEN",  "DISTILLERY"),
    "spread": ("DUSKBAT",  "DUSKBAT",   "RESERVE"),
    "fold":   ("MOONBAT",  "MOONBAT",   "NOCTURNE"),
}

# ─── 6. 合成 + 输出 ─────────────────────────────────────
final_imgs = {}
for name, base_arr in variants.items():
    base_u8 = np.clip(base_arr, 0, 255).astype(np.uint8)
    base_pil = Image.fromarray(base_u8).convert("RGBA")
    top, brand, sub = variants_text[name]
    text_layer = render_text_layer(top, brand, sub)
    composed = Image.alpha_composite(base_pil, text_layer).convert("RGB")
    out_p = os.path.join(OUT_DIR, f"v302_{name}_final.png")
    composed.save(out_p)
    text_layer.convert("RGB").save(
        os.path.join(OUT_DIR, f"v302_{name}_textlayer.png"))
    final_imgs[name] = out_p
    print(f"  {name}  ->  {out_p}")

# 主体无字版（姿态层）
for name, base_arr in variants.items():
    Image.fromarray(np.clip(base_arr, 0, 255).astype(np.uint8)).save(
        os.path.join(OUT_DIR, f"v302_{name}_body.png"))

# 4 宫格（不含 BACARDÍ 原图，按 v301 规矩）
grid_w = W * 2 + 30
grid_h = H * 2 + 30
grid = Image.new("RGB", (grid_w, grid_h), (200, 200, 200))
positions = {
    "up":     (0,       0),
    "spread": (W + 30,  0),
    "fold":   (0,       H + 30),
}
for name, (px, py) in positions.items():
    grid.paste(Image.open(final_imgs[name]), (px, py))
panel4 = Image.new("RGB", (W, H), (50, 30, 60))
d4 = ImageDraw.Draw(panel4)
fn = ImageFont.truetype(FONT_PATH, 56)
d4.text((W // 2 - 280, H // 2 - 60), "Three fission variants",
        font=fn, fill=(220, 200, 230))
fn2 = ImageFont.truetype(FONT_PATH, 38)
d4.text((W // 2 - 230, H // 2 + 30), "top arc + main brand:",
        font=fn2, fill=(180, 160, 190))
d4.text((W // 2 - 320, H // 2 + 80), "NOCTAVEN / DUSKBAT / MOONBAT",
        font=fn2, fill=(220, 200, 230))
grid.paste(panel4, (W + 30, H + 30))
grid_path = os.path.join(OUT_DIR, "_grid_v302.png")
grid.save(grid_path)
print(f"\nGrid:  {grid_path}")
print("\nAll outputs ready.")
