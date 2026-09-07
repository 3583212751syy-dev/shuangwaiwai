"""v303 — 蝙蝠主体真正"分块裂变"版

v302 被用户痛批："只是整张翅膀刚体旋转，没有改变蝙蝠角度/腿长/头尾角度"。
v303 将蝙蝠拆成 6 个独立刚体部位：
  - 头（head）
  - 躯干（torso，保留/轻微倾斜）
  - 左翼（left_wing）
  - 右翼（right_wing）
  - 尾巴（tail）
  - 左腿 / 右腿（legs）
每个部位单独旋转+缩放+平移，再软边拼回 clean_layer（纯紫+蝙蝠）。

三档姿态：
  up     — 头仰、双翼高扬内收、尾巴下垂、腿收拢
  spread — 头正视、双翼水平大展、尾巴平直、腿微张
  fold   — 头俯、双翼下垂外展、尾巴上翘、腿张开
"""
import math, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

# ─── 配置 ────────────────────────────────────────────────
SRC_PATH  = r"E:\Desktop\双接口\image-fission\ComfyUI\input\v300_text_free_source.png"
FONT_PATH = r"E:\Desktop\双接口\image-fission\ComfyUI\models\fonts\AbrilFatface-Regular.ttf"
OUT_DIR   = r"E:\Desktop\双接口\image-fission\jobs\v303"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "_chk"), exist_ok=True)

BG_PURPLE = np.array([181, 129, 171], dtype=np.float32)
cx, cy = 776, 746
ring_outer = 360
TOP_ARC_CY  = 235
LEFT_EST_X  = 432
RIGHT_1862_X = 1120
EST_LINE_Y  = 760
BRAND_TOP_Y = 1060
SUB_LINE_Y  = 1320
TRI_CY      = 1500

# ─── 1. 源图 + bat_m（沿用 v302） ────────────────────────
arr = np.array(Image.open(SRC_PATH).convert("RGB")).astype(np.float32)
H, W = arr.shape[:2]
R, G, Bc = arr[:,:,0], arr[:,:,1], arr[:,:,2]
maxc = np.maximum(np.maximum(R, G), Bc)
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

solid = cv2.morphologyEx(bat_m, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
solid_m = solid.copy()
cnts, hier = cv2.findContours(solid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
if hier is not None:
    for i, h4 in enumerate(hier[0]):
        if h4[3] != -1:
            cv2.drawContours(solid_m, cnts, i, 255, -1)

# clean_layer: 内盘 R=440 非 bat_solid → 纯紫
disc = np.zeros((H, W), np.uint8)
cv2.circle(disc, (cx, cy), 440, 255, -1)
inner_bg_mask = cv2.bitwise_and(cv2.bitwise_not(solid_m), disc)
clean_layer = arr.copy()
clean_layer[inner_bg_mask > 0] = BG_PURPLE

# ─── 2. 蝙蝠部位分割 ────────────────────────────────────
# 基于几何的 mask 分割，不需要 AI，直接按 v300 已知蝙蝠结构切
hx, hy = cx, 521          # 头中心
neck_y = hy + 35          # 颈
shoulder_y = 790          # 翼肩
hip_y = 875               # 尾根

# 躯干：y<800 向两翼延伸 70px 覆盖翼根；y>=800 收窄到 cx±40，避免包含紫冠花瓣
band_b = np.zeros((H, W), np.uint8)
band_b[:800, cx - 70 : cx + 70] = 255
band_b[800:, cx - 40 : cx + 40] = 255
torso_m = cv2.bitwise_and(band_b, solid_m)
torso_m[:hy + 25, :] = 0       # 让头 mask 来接管顶部
torso_m[hip_y:, :] = 0         # 让尾巴 mask 来接管底部

# 头：椭圆，与 solid_m 相交；向下延长覆盖颈部，和躯干重叠
head_m = np.zeros((H, W), np.uint8)
cv2.ellipse(head_m, (cx + 6, hy + 16), (34, 50), 0, 0, 360, 255, -1)
head_m = cv2.bitwise_and(head_m, solid_m)

# 尾巴：尾根以下中心；向上延长和躯干重叠
tail_m = np.zeros((H, W), np.uint8)
tail_m[hip_y - 20 : 985, cx - 35 : cx + 35] = 255
tail_m = cv2.bitwise_and(tail_m, solid_m)

# 身体核心 = 头 + 躯干 + 尾巴（整体轻微旋转，内部无缝）
body_core_m = cv2.bitwise_or(head_m, torso_m)
body_core_m = cv2.bitwise_or(body_core_m, tail_m)

# 腿：躯干下方两侧的小突起；只保留深色腿骨（maxc<80），避免把紫色花瓣底座误判为腿
left_leg_m = np.zeros((H, W), np.uint8)
left_leg_m[800:960, cx - 90 : cx - 35] = 255
left_leg_m = cv2.bitwise_and(left_leg_m, solid_m)
left_leg_m = cv2.bitwise_and(left_leg_m, (maxc < 80).astype(np.uint8) * 255)
right_leg_m = np.zeros((H, W), np.uint8)
right_leg_m[800:960, cx + 35 : cx + 90] = 255
right_leg_m = cv2.bitwise_and(right_leg_m, solid_m)
right_leg_m = cv2.bitwise_and(right_leg_m, (maxc < 80).astype(np.uint8) * 255)

# 删除原 BACARDÍ 蝙蝠 logo 下部"紫冠花瓣底座"：
# y>760 区域只保留 body_core 和深色腿，其余全部从 solid_m 删除
lower = np.zeros_like(solid_m)
lower[760:, :] = solid_m[760:, :]
preserve_lower = cv2.bitwise_or(body_core_m, left_leg_m)
preserve_lower = cv2.bitwise_or(preserve_lower, right_leg_m)
crown = cv2.bitwise_and(lower, cv2.bitwise_not(preserve_lower))
solid_m[crown > 0] = 0

# 重新计算 body_core / leg（solid_m 已变）
body_core_m = cv2.bitwise_and(body_core_m, solid_m)
left_leg_m = cv2.bitwise_and(left_leg_m, solid_m)
right_leg_m = cv2.bitwise_and(right_leg_m, solid_m)

# 翼 = solid - (身体核心 + 腿)
exclude = cv2.bitwise_or(body_core_m, left_leg_m)
exclude = cv2.bitwise_or(exclude, right_leg_m)
wings_m = cv2.bitwise_and(solid_m, cv2.bitwise_not(exclude))

xs_col = np.tile(np.arange(W, dtype=np.int32), (H, 1))
left_wing_m  = cv2.bitwise_and(wings_m, (xs_col <  cx).astype(np.uint8) * 255)
right_wing_m = cv2.bitwise_and(wings_m, (xs_col >= cx).astype(np.uint8) * 255)

PARTS = ["body_core", "left_leg", "right_leg",
         "left_wing", "right_wing"]
masks = {
    "body_core": body_core_m,
    "left_leg": left_leg_m,
    "right_leg": right_leg_m,
    "left_wing": left_wing_m,
    "right_wing": right_wing_m,
}

# 调试：保存分割图
os.makedirs(os.path.join(OUT_DIR, "_parts"), exist_ok=True)
for name, m in masks.items():
    Image.fromarray(m).save(os.path.join(OUT_DIR, f"_parts/part_{name}.png"))

# ─── 3. 部位变换工具 ────────────────────────────────────
def part_affine_bgra(src_bgra, mask, angle_deg, scale, anchor, center_out=None):
    """对单部位做旋转+缩放，保持 anchor 不动。返回变换后的 BGRA 图。
    src_bgra: HxWx4 uint8, alpha=mask"""
    if center_out is None:
        center_out = anchor
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    # 绕 anchor 旋转 + 缩放；再平移到 center_out
    M = np.array([
        [scale * c, -scale * s, center_out[0] - anchor[0] * scale * c + anchor[1] * scale * s],
        [scale * s,  scale * c, center_out[1] - anchor[0] * scale * s - anchor[1] * scale * c]
    ], dtype=np.float64)
    out = cv2.warpAffine(src_bgra, M, (W, H),
                         flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(0, 0, 0, 0))
    return out

def compose_bat(layer_in, params):
    """将各部位按 params 独立变换后软边拼回。
    关键：从纯紫背景开始，只贴回变换后的部位，彻底杜绝'双蝙蝠'残影。"""
    # 起始画布：纯紫背景
    out = np.full_like(layer_in, BG_PURPLE)

    # 身体核心 = 头+躯干+尾巴，整体轻微旋转/倾斜，内部无缝
    p = params["body_core"]
    src_bgra = np.dstack([layer_in, masks["body_core"]])
    t = part_affine_bgra(src_bgra.astype(np.uint8), masks["body_core"],
                         p.get("rot", 0), p.get("scale", 1.0),
                         (cx, (hy + hip_y) // 2),
                         center_out=(cx + p.get("dx", 0), (hy + hip_y) // 2 + p.get("dy", 0)))
    alpha = t[:,:,3].astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (7, 7), 2)
    for c in range(3):
        out[:,:,c] = t[:,:,c] * alpha + out[:,:,c] * (1 - alpha)

    # 处理翅膀（内收/外展/缩放），羽化半径 5 防紫边
    for side, anchor in [("left", (cx - 120, shoulder_y + 60)),
                         ("right", (cx + 120, shoulder_y + 60))]:
        name = f"{side}_wing"
        p = params[name]
        src_bgra = np.dstack([layer_in, masks[name]])
        t = part_affine_bgra(src_bgra.astype(np.uint8), masks[name],
                             p["rot"], p["scale"], anchor)
        alpha = t[:,:,3].astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (5, 5), 1.5)
        for c in range(3):
            out[:,:,c] = t[:,:,c] * alpha + out[:,:,c] * (1 - alpha)

    # 处理腿
    for side, anchor in [("left", (cx - 55, 880)), ("right", (cx + 55, 880))]:
        name = f"{side}_leg"
        p = params[name]
        src_bgra = np.dstack([layer_in, masks[name]])
        t = part_affine_bgra(src_bgra.astype(np.uint8), masks[name],
                             p["rot"], p.get("scale", 1.0), anchor)
        alpha = t[:,:,3].astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (5, 5), 1.5)
        for c in range(3):
            out[:,:,c] = t[:,:,c] * alpha + out[:,:,c] * (1 - alpha)

    return out

# ─── 4. 三档姿态参数 ────────────────────────────────────
poses = {
    "up": {
        "body_core": {"rot": -3, "dy": -6, "dx": 0, "scale": 1.0},
        "left_leg":  {"rot": 18, "scale": 1.0},
        "right_leg": {"rot": -18, "scale": 1.0},
        "left_wing": {"rot": 22, "scale": 0.92},
        "right_wing":{"rot": -22, "scale": 0.92},
    },
    "spread": {
        "body_core": {"rot": 0, "dy": 0, "dx": 0, "scale": 1.0},
        "left_leg":  {"rot": -8, "scale": 1.05},
        "right_leg": {"rot": 8, "scale": 1.05},
        "left_wing": {"rot": -18, "scale": 1.08},
        "right_wing":{"rot": 18, "scale": 1.08},
    },
    "fold": {
        "body_core": {"rot": 3, "dy": 6, "dx": 0, "scale": 1.0},
        "left_leg":  {"rot": -22, "scale": 1.1},
        "right_leg": {"rot": 22, "scale": 1.1},
        "left_wing": {"rot": -18, "scale": 0.98},
        "right_wing":{"rot": 18, "scale": 0.98},
    },
}

variants = {}
for name, p in poses.items():
    out = compose_bat(clean_layer, p)
    variants[name] = out
    print(f"  composed {name}")

# ─── 5. 文字层（同 v302） ────────────────────────────────
def render_text_layer(top_arc, brand, sub):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    drw = ImageDraw.Draw(layer)
    arc_r = ring_outer + 70
    def _draw_arc_text(txt, size_pt, top=True):
        font = ImageFont.truetype(FONT_PATH, size_pt)
        bbox = drw.textbbox((0, 0), txt, font=font)
        text_w = bbox[2] - bbox[0]
        phi = min(math.pi * 0.30, text_w / (2.0 * arc_r))
        if top:
            a_start = -math.pi / 2 - phi; a_end = -math.pi / 2 + phi
        else:
            a_start =  math.pi / 2 - phi; a_end =  math.pi / 2 + phi
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

    tri_w, tri_h = 80, 56
    tri = [(W // 2 - tri_w // 2, TRI_CY - tri_h // 2),
           (W // 2 + tri_w // 2, TRI_CY - tri_h // 2),
           (W // 2,              TRI_CY + tri_h // 2)]
    drw.polygon(tri, fill=(0, 0, 0, 255))
    return layer

variants_text = {
    "up":     ("NOCTAVEN", "NOCTAVEN",  "DISTILLERY"),
    "spread": ("DUSKBAT",  "DUSKBAT",   "RESERVE"),
    "fold":   ("MOONBAT",  "MOONBAT",   "NOCTURNE"),
}

final_imgs = {}
for name, base_arr in variants.items():
    base_u8 = np.clip(base_arr, 0, 255).astype(np.uint8)
    base_pil = Image.fromarray(base_u8).convert("RGBA")
    top, brand, sub = variants_text[name]
    text_layer = render_text_layer(top, brand, sub)
    composed = Image.alpha_composite(base_pil, text_layer).convert("RGB")
    out_p = os.path.join(OUT_DIR, f"v303_{name}_final.png")
    composed.save(out_p)
    text_layer.convert("RGB").save(
        os.path.join(OUT_DIR, f"v303_{name}_textlayer.png"))
    final_imgs[name] = out_p
    print(f"  {name} saved")

for name, base_arr in variants.items():
    Image.fromarray(np.clip(base_arr, 0, 255).astype(np.uint8)).save(
        os.path.join(OUT_DIR, f"v303_{name}_body.png"))

# 4 宫格
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
d4.text((W // 2 - 230, H // 2 + 30), "head/tail/legs/wings",
        font=fn2, fill=(180, 160, 190))
d4.text((W // 2 - 320, H // 2 + 80), "NOCTAVEN / DUSKBAT / MOONBAT",
        font=fn2, fill=(220, 200, 230))
grid.paste(panel4, (W + 30, H + 30))
grid_path = os.path.join(OUT_DIR, "_grid_v303.png")
grid.save(grid_path)
print(f"\nGrid: {grid_path}")
print("All outputs ready.")
