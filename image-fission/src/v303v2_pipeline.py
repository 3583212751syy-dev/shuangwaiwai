"""v303v2 — 在 v302 刚体旋转成熟方案上增加 头/尾 独立旋转

v303（分块拼贴）已证伪：几何矩形/椭圆 mask 分割有机形状会毁容
（躯干柱状化、头消失、尾成刺）。

v303v2 原则：所有部位变化都用 v302 的 _rot_warp 软权重方案
（源空间权重随内容旋转 + bg_layer reveal + 串联 keep），自然无缝：
  - 翅膀：v302 two_wing 三档（+18/-12/-33）
  - 头：小角度旋转（-8/0/+8），旋转中心颈部，软权重 mask
  - 尾：小角度旋转（-12/0/+15），旋转中心尾根，软权重 mask
  - 腿：不单独动（腿部小、动了破坏风险大、视觉收益小）
  - clean_layer：徽章圆 R=440 非蝙蝠 → 纯紫（v302 定稿）
"""
import math, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

# ─── 配置 ────────────────────────────────────────────────
SRC_PATH  = r"E:\Desktop\双接口\image-fission\ComfyUI\input\v300_text_free_source.png"
FONT_PATH = r"E:\Desktop\双接口\image-fission\ComfyUI\models\fonts\AbrilFatface-Regular.ttf"
OUT_DIR   = r"E:\Desktop\双接口\image-fission\jobs\v303v2"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "_chk"), exist_ok=True)

BG_PURPLE = np.array([181, 129, 171], dtype=np.float32)
cx, cy = 776, 746
ring_outer = 360
LEFT_EST_X  = 432
RIGHT_1862_X = 1120
EST_LINE_Y  = 760
BRAND_TOP_Y = 1060
SUB_LINE_Y  = 1320
TRI_CY      = 1500

# ─── 1. 源图 + bat_m（v302 定稿链） ──────────────────────
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

# ★ 删除紫冠花瓣底座（v302 遗漏的装饰残留）：
# y>760 躯干条带(|x-cx|<=90)内只保留黑色躯干/尾/腿（maxc<80），
# 紫色花瓣（maxc 80-150）全部删除；条带外是翅膀区，solid_m 全保留
crown_zone = np.zeros((H, W), np.uint8)
crown_zone[760:, :] = 255
dark_in_band = (maxc < 80).astype(np.uint8) * 255
band90 = np.zeros((H, W), np.uint8)
band90[760:, cx - 90 : cx + 90] = 255
keep_black = cv2.bitwise_and(band90, dark_in_band)      # 条带内黑色躯干/尾/腿
wing_zone = np.zeros((H, W), np.uint8)
wing_zone[760:, :cx - 90] = 255
wing_zone[760:, cx + 90 :] = 255
keep_wing = cv2.bitwise_and(wing_zone, solid_m)          # 条带外翅膀区
keep_all = cv2.bitwise_or(keep_black, keep_wing)
crown = cv2.bitwise_and(crown_zone, cv2.bitwise_not(keep_all))
solid_m[crown > 0] = 0

# clean_layer: 徽章圆 R=440 内非 bat_solid → 纯紫（v302 定稿）
disc = np.zeros((H, W), np.uint8)
cv2.circle(disc, (cx, cy), 440, 255, -1)
inner_bg_mask = cv2.bitwise_and(cv2.bitwise_not(solid_m), disc)
clean_layer = arr.copy()
clean_layer[inner_bg_mask > 0] = BG_PURPLE

# ─── 2. 部位 mask（软权重式，v302 思路） ─────────────────
hx, hy = cx, 521           # 头中心
neck_y = hy + 48           # 颈部旋转中心
shoulder_y = 790
hip_y = 862                # 尾根旋转中心

band_b = np.zeros((H, W), np.uint8)
band_b[:, cx - 48 : cx + 48] = 255
body_core = cv2.bitwise_and(bat_m, band_b)
wing_m = solid_m.copy()
wing_m[body_core > 0] = 0

# 翼软权重（距身体 0~22px 渐变，v302 定稿）
dist = cv2.distanceTransform(cv2.bitwise_not(body_core), cv2.DIST_L2, 3)
wing_soft = (wing_m.astype(np.float32) / 255.0) * np.clip(dist / 22.0, 0, 1).astype(np.float32)
xs_col = np.tile(np.arange(W, dtype=np.int32), (H, 1))
left_m  = ((wing_m > 0) & (xs_col <  cx)).astype(np.uint8) * 255
right_m = ((wing_m > 0) & (xs_col >= cx)).astype(np.uint8) * 255
left_soft  = wing_soft.copy(); left_soft[xs_col >= cx]  = 0
right_soft = wing_soft.copy(); right_soft[xs_col <  cx] = 0

# 头 mask：bat_m 顶部椭圆区（软权重 = 距边界 0~12px 渐变）
head_zone = np.zeros((H, W), np.uint8)
cv2.ellipse(head_zone, (cx + 4, hy + 10), (38, 52), 0, 0, 360, 255, -1)
head_m = cv2.bitwise_and(head_zone, bat_m)
head_dist = cv2.distanceTransform(cv2.bitwise_not(head_m), cv2.DIST_L2, 3)
head_soft_full = (head_m.astype(np.float32) / 255.0) * np.clip(head_dist / 12.0, 0, 1).astype(np.float32)

# 尾 mask：bat_m 下部窄条（软权重同思路）
tail_zone = np.zeros((H, W), np.uint8)
tail_zone[hip_y - 10 : 980, cx - 30 : cx + 30] = 255
tail_m = cv2.bitwise_and(tail_zone, bat_m)
tail_dist = cv2.distanceTransform(cv2.bitwise_not(tail_m), cv2.DIST_L2, 3)
tail_soft_full = (tail_m.astype(np.float32) / 255.0) * np.clip(tail_dist / 10.0, 0, 1).astype(np.float32)

# protect（头/尾强制原样的兜底区不用——头尾本身就是要动的部位；
# 身体核心 protect 仍保留，防翼旋转擦到躯干）
protect = np.zeros((H, W), np.uint8)
protect[hy + 60 : 875, cx - 22 : cx + 22] = 255     # 躯干中轴
protect = cv2.bitwise_and(protect, bat_m)
body_block = cv2.GaussianBlur(protect, (5, 5), 1.5).astype(np.float32) / 255.0
body_keep  = cv2.GaussianBlur(
    cv2.dilate(protect, np.ones((5, 5), np.uint8)), (7, 7), 2
).astype(np.float32) / 255.0

# ─── 3. _rot_warp（v300 十七轮定稿：串联 keep 修复版） ───
def _rot_warp(layer_in, part_soft, part_bin, body_block, body_keep, bg_layer,
              theta, rot_c, h, w, center_shift=(0, 0)):
    cos_t = np.float32(np.cos(theta))
    sin_t = np.float32(np.sin(theta))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xs - rot_c[0]
    dy = ys - rot_c[1]
    rx = rot_c[0] + dx * cos_t + dy * sin_t
    ry = rot_c[1] - dx * sin_t + dy * cos_t
    # 旋转后整体平移（头/尾的 dy 位移）
    rx = rx - np.float32(center_shift[0])
    ry = ry - np.float32(center_shift[1])
    rot_soft = cv2.remap(part_soft, rx, ry, cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    w_new = cv2.GaussianBlur(rot_soft, (5, 5), 1.5).astype(np.float32)
    w_new = w_new * (np.float32(1.0) - body_block)
    w_old = cv2.GaussianBlur(part_bin, (3, 3), 0.8).astype(np.float32) / 255.0
    w_old = np.clip(w_old - w_new, 0, np.float32(1.0))
    src_warp = cv2.remap(layer_in, rx, ry, cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    rest = np.float32(1.0) - w_new - w_old
    out = (src_warp * w_new[..., None] + bg_layer * w_old[..., None]
           + layer_in * rest[..., None])
    out = out * (np.float32(1.0) - body_keep)[..., None] + layer_in * body_keep[..., None]
    return out, w_new

shL = (cx - 120, shoulder_y + 60)
shR = (cx + 120, shoulder_y + 60)

def make_variant(wingL, wingR, head_rot, head_dy, tail_rot):
    """翅膀双旋 + 头旋转/位移 + 尾旋转。串联执行，每步 keep 新区防互擦。"""
    # step1: 左翼
    out, keepL = _rot_warp(clean_layer, left_soft, left_m, body_block, body_keep, clean_layer,
                           wingL, shL, H, W)
    # step2: 右翼（keep 左翼新区）
    keepL_b = cv2.GaussianBlur(keepL, (9, 9), 2).astype(np.float32)
    out, keepR = _rot_warp(out, right_soft, right_m, body_block, body_keep, clean_layer,
                           wingR, shR, H, W)
    keep_all = np.maximum(keepL_b, cv2.GaussianBlur(keepR, (9, 9), 2).astype(np.float32))
    # step3: 头（旋转 + 上下位移）
    out, keepH = _rot_warp(out, head_soft_full, head_m, np.zeros_like(body_block),
                           np.zeros_like(body_keep), clean_layer,
                           math.radians(head_rot), (cx, neck_y), H, W,
                           center_shift=(0, head_dy))
    keep_all = np.maximum(keep_all, cv2.GaussianBlur(keepH, (9, 9), 2).astype(np.float32))
    # step4: 尾（旋转）
    out, keepT = _rot_warp(out, tail_soft_full, tail_m, np.zeros_like(body_block),
                           np.zeros_like(body_keep), clean_layer,
                           math.radians(tail_rot), (cx, hip_y), H, W)
    return out

poses = {
    "up":     dict(wingL=math.radians( 18), wingR=math.radians(-18),
                   head_rot=-8, head_dy=-8,  tail_rot=-14),
    "spread": dict(wingL=math.radians(-12), wingR=math.radians( 12),
                   head_rot= 0, head_dy= 0,  tail_rot=  0),
    "fold":   dict(wingL=math.radians(-33), wingR=math.radians( 33),
                   head_rot= 8, head_dy= 8,  tail_rot= 16),
}

variants = {}
for name, p in poses.items():
    variants[name] = make_variant(**p)
    print(f"  composed {name}")

# ─── 4. 文字层（v302 定稿） ─────────────────────────────
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
    bw = bb[2] - bb[0]
    drw.text(((W - bw) // 2 - bb[0], BRAND_TOP_Y - bb[1]), brand,
             font=brand_font, fill=(0, 0, 0, 255))
    sub_font = ImageFont.truetype(FONT_PATH, 150)
    sb = drw.textbbox((0, 0), sub, font=sub_font)
    sw = sb[2] - sb[0]
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
    out_p = os.path.join(OUT_DIR, f"v303v2_{name}_final.png")
    composed.save(out_p)
    final_imgs[name] = out_p
    Image.fromarray(np.clip(base_arr, 0, 255).astype(np.uint8)).save(
        os.path.join(OUT_DIR, f"v303v2_{name}_body.png"))
    print(f"  saved {out_p}")

grid_w = W * 2 + 30
grid_h = H * 2 + 30
grid = Image.new("RGB", (grid_w, grid_h), (200, 200, 200))
positions = {"up": (0, 0), "spread": (W + 30, 0), "fold": (0, H + 30)}
for name, (px, py) in positions.items():
    grid.paste(Image.open(final_imgs[name]), (px, py))
panel4 = Image.new("RGB", (W, H), (50, 30, 60))
d4 = ImageDraw.Draw(panel4)
fn = ImageFont.truetype(FONT_PATH, 56)
d4.text((W // 2 - 280, H // 2 - 60), "Three fission variants",
        font=fn, fill=(220, 200, 230))
fn2 = ImageFont.truetype(FONT_PATH, 38)
d4.text((W // 2 - 300, H // 2 + 30), "wings + head + tail pose",
        font=fn2, fill=(180, 160, 190))
d4.text((W // 2 - 320, H // 2 + 80), "NOCTAVEN / DUSKBAT / MOONBAT",
        font=fn2, fill=(220, 200, 230))
grid.paste(panel4, (W + 30, H + 30))
grid_path = os.path.join(OUT_DIR, "_grid_v303v2.png")
grid.save(grid_path)
print(f"\nGrid: {grid_path}")
print("All outputs ready.")
