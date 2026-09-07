"""v301 — 10/10 路径（半自动 / C 路线）
保留 v300 已修的 mask 链 + _rot_warp pass 串联 keep 修复，只改三件事：
  1) bg_layer 纯紫 RGB(181,129,171) 填洞（彻底告别 TELEA 拖拽斑驳）
  2) 用 AbrilFatface 67KB 真字体渲染文字层（Pillow 原位写，非 SDXL 重画）
  3) 旧版"BACARDÍ MCKHEART"原图不放进宫格

底图：v300_text_free_source.png（纯紫背景，徽章原图无字版）
姿态：up +18° / spread -12° / fold -33°（v300 十七轮定档）
"""
import math, sys, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

# ─── 配置 ────────────────────────────────────────────────
SRC_PATH    = r"E:\Desktop\双接口\image-fission\ComfyUI\input\v300_text_free_source.png"
FONT_PATH   = r"E:\Desktop\双接口\image-fission\ComfyUI\models\fonts\AbrilFatface-Regular.ttf"
OUT_DIR     = r"E:\Desktop\双接口\image-fission\jobs\v301"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "_chk"), exist_ok=True)

# 纯紫背景（v300 source 4 角采样 = RGB(181, 129, 171)）
BG_PURPLE = np.array([181, 129, 171], dtype=np.float32)

# BACARDÍ 经典版式参数（基于 v300 实际布局）
cx, cy = 776, 746              # 徽章中心（与 v300 一致）
ring_outer = 360                # 徽章外圈半径
# 文字锚点（按 v300_up_final.png 实际位置测量）
TOP_ARC_CY  = 235                # 顶弧绕环竖直锚点
LEFT_EST_X  = 432                # "EST." 横坐标
RIGHT_1862_X = 1120              # "1862" 横坐标
EST_LINE_Y  = 760                # EST. 1862 行 y
BRAND_TOP_Y = 1060               # 品牌名顶部 y
SUB_LINE_Y  = 1320               # 副字基线 y
TRI_CY      = 1500               # 三角竖直中心

# ─── 1. mask 链（沿用 v300 定稿） ────────────────────────
arr = np.array(Image.open(SRC_PATH).convert("RGB")).astype(np.float32)
H, W = arr.shape[:2]
R, G, Bc = arr[:,:,0], arr[:,:,1], arr[:,:,2]
maxc = np.maximum(np.maximum(R, G), Bc)
dark = (maxc < 95).astype(np.uint8) * 255
badge_lim = np.zeros((H, W), np.uint8)
cv2.circle(badge_lim, (cx, cy), ring_outer + 30, 255, -1)
dark = cv2.bitwise_and(dark, badge_lim)
dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
sizes = stats[1:, cv2.CC_STAT_AREA]
keep_idx = 1 + int(np.argmax(sizes))
bat_m = (labels == keep_idx).astype(np.uint8) * 255

# CLOSE 21 桥接骨缝 + RETR_CCOMP 真洞填充
solid = cv2.morphologyEx(bat_m, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
solid_m = solid.copy()
cnts, hier = cv2.findContours(solid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
if hier is not None:
    for i, h4 in enumerate(hier[0]):
        if h4[3] != -1:
            cv2.drawContours(solid_m, cnts, i, 255, -1)

# 躯干条带：身体+翼的分割线
band_b = np.zeros((H, W), np.uint8)
band_b[:, cx - 48 : cx + 48] = 255
body_core = cv2.bitwise_and(bat_m, band_b)
wing_m = solid_m.copy()
wing_m[body_core > 0] = 0

# 翼软权重（距身体距离 0~22px 渐变）
dist = cv2.distanceTransform(cv2.bitwise_not(body_core), cv2.DIST_L2, 3)
wing_soft = (wing_m.astype(np.float32) / 255.0) * np.clip(dist / 22.0, 0, 1).astype(np.float32)

# 左右半翼 mask + 软权重
xs_col = np.tile(np.arange(W, dtype=np.int32), (H, 1))
left_m  = ((wing_m > 0) & (xs_col <  cx)).astype(np.uint8) * 255
right_m = ((wing_m > 0) & (xs_col >= cx)).astype(np.uint8) * 255
left_soft  = wing_soft.copy(); left_soft[xs_col >= cx]  = 0
right_soft = wing_soft.copy(); right_soft[xs_col <  cx] = 0

# 身体 protect（仅头部椭圆 + 尾/躯干中轴 — v300 十七轮定稿）
hx, hy = cx, 521
protect = np.zeros((H, W), np.uint8)
cv2.ellipse(protect, (int(hx) + 6, int(hy) + 6), (24, 28), 0, 0, 360, 255, -1)
protect[875:985, cx - 26 : cx + 26] = 255           # 尾
protect[560:875, cx - 22 : cx + 22] = 255           # 躯干中轴
protect = cv2.bitwise_and(protect, bat_m)
body_block = cv2.GaussianBlur(protect, (5, 5), 1.5).astype(np.float32) / 255.0
body_keep  = cv2.GaussianBlur(
    cv2.dilate(protect, np.ones((5, 5), np.uint8)), (7, 7), 2
).astype(np.float32) / 255.0

# ─── 2. bg_layer 修复（v301 关键） ─────────────────────
# 旧版：TELEA inpaint 1/2 尺度 → 紫冠图案拖成白噪声（v300 十六轮主凶）
# v301：直接填纯紫 RGB(181,129,171)，不靠任何 inpaint/SDXL
dig = cv2.dilate(bat_m, np.ones((9, 9), np.uint8), iterations=1)
dig[cv2.dilate(body_core, np.ones((3, 3), np.uint8)) > 0] = 0
bg_layer = arr.copy()
bg_layer[dig > 0] = BG_PURPLE                              # 关键：纯色填洞
# 但徽章外的盘内紫冠是"装饰"，不是 bg — 保留原像素
# 实际上 v300 source 整图背景已是纯紫，所以全填纯紫正确

# ─── 3. _rot_warp（v300 十七轮：pass 串联 keep 修复版） ─
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
    wpL = _rot_warp(arr, left_soft, left_m, body_block, body_keep, bg_layer,
                    thetaL, shL, H, W)
    wp  = _rot_warp(wpL, right_soft, right_m, body_block, body_keep, bg_layer,
                    thetaR, shR, H, W)
    return wp

variants = {
    "up":     two_wing(math.radians( 18), math.radians(-18)),   # 轻扬 V
    "spread": two_wing(math.radians(-12), math.radians( 12)),   # 微垂平展
    "fold":   two_wing(math.radians(-33), math.radians( 33)),   # 大幅垂翼
}

# ─── 4. 文字层（Pillow + Abril Fatface 67KB 真字体） ─────
def render_text_layer(name_top, brand, sub):
    """返回 RGBA PIL.Image（透明底，仅文字 + 三角），黑色 #000。
    文字按 BACARDÍ 经典版式：顶弧绕环 + EST. 1862 + 品牌大写 + 副字 + 三角。"""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    drw = ImageDraw.Draw(layer)
    # 顶弧文字（沿以徽章中心为圆心的弧线排版，半径 = 圆环外圈 + 50）
    arc_r = ring_outer + 70
    def _draw_arc_text(txt, size_pt, base_y, top=True):
        font = ImageFont.truetype(FONT_PATH, size_pt)
        # 字符宽度测量
        bbox = drw.textbbox((0, 0), txt, font=font)
        text_w = bbox[2] - bbox[0]
        # 弧线参数：以 (cx, cy) 为圆心，弧度覆盖文本宽度的 1/弧长比例
        # 弧长 ≈ text_w → 半角 phi = text_w / (2 * arc_r)
        phi = min(math.pi * 0.30, text_w / (2.0 * arc_r))   # 最大 54°，防文字塌在两侧
        steps = max(len(txt) * 4, 60)
        # 顶弧：字符朝向圆心（旋转 -90° ~ -90°-phi）
        if top:
            a_start = -math.pi / 2 - phi
            a_end   = -math.pi / 2 + phi
        else:
            a_start =  math.pi / 2 - phi
            a_end   =  math.pi / 2 + phi
        # 测量每个字符宽度，累进角度
        char_widths = [drw.textbbox((0, 0), c, font=font)[2] for c in txt]
        total_w = sum(char_widths)
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
    _draw_arc_text(name_top, 78, TOP_ARC_CY, top=True)
    # "EST." 在徽章左侧
    est_font = ImageFont.truetype(FONT_PATH, 58)
    drw.text((LEFT_EST_X, EST_LINE_Y), "EST.", font=est_font, fill=(0, 0, 0, 255))
    # "1862" 在徽章右侧
    drw.text((RIGHT_1862_X, EST_LINE_Y), "1862", font=est_font, fill=(0, 0, 0, 255))
    # 品牌名（中央，大写，粗黑）
    brand_font = ImageFont.truetype(FONT_PATH, 260)
    bb = drw.textbbox((0, 0), brand, font=brand_font)
    bw = bb[2] - bb[0]; bh = bb[3] - bb[1]
    drw.text(((W - bw) // 2 - bb[0], BRAND_TOP_Y - bb[1]), brand,
             font=brand_font, fill=(0, 0, 0, 255))
    # 副字（中央，下一行）
    sub_font = ImageFont.truetype(FONT_PATH, 130)
    sb = drw.textbbox((0, 0), sub, font=sub_font)
    sw = sb[2] - sb[0]; sh = sb[3] - sb[1]
    drw.text(((W - sw) // 2 - sb[0], SUB_LINE_Y - sb[1]), sub,
             font=sub_font, fill=(0, 0, 0, 255))
    # 三角 ▼（实心，等腰）
    tri_w = 56
    tri_h = 40
    tri = [(W // 2 - tri_w // 2, TRI_CY - tri_h // 2),
           (W // 2 + tri_w // 2, TRI_CY - tri_h // 2),
           (W // 2,              TRI_CY + tri_h // 2)]
    drw.polygon(tri, fill=(0, 0, 0, 255))
    return layer

variants_text = {
    "up":     ("SHADOW OF THE WING", "NOCTAVEN",  "DISTILLERY"),
    "spread": ("WINGS OF TWILIGHT",  "DUSKBAT",   "RESERVE"),
    "fold":   ("GUARDIAN OF THE DARK","MOONBAT", "NOCTURNE"),
}

# ─── 5. 合成 + 输出 ──────────────────────────────────────
final_imgs = {}
for name, base_arr in variants.items():
    base_u8 = np.clip(base_arr, 0, 255).astype(np.uint8)
    base_pil = Image.fromarray(base_u8).convert("RGBA")
    top, brand, sub = variants_text[name]
    text_layer = render_text_layer(top, brand, sub)
    composed = Image.alpha_composite(base_pil, text_layer).convert("RGB")
    out_p = os.path.join(OUT_DIR, f"v301_{name}_final.png")
    composed.save(out_p)
    # 单独存文字层（供用户检查）
    text_layer.convert("RGB").save(
        os.path.join(OUT_DIR, f"v301_{name}_textlayer.png"))
    final_imgs[name] = out_p
    print(f"  {name}  ->  {out_p}")

# 4 宫格（不含原图）
grid_w = W * 2 + 30
grid_h = H * 2 + 30
grid = Image.new("RGB", (grid_w, grid_h), (200, 200, 200))
positions = {
    "up":     (0,       0),
    "spread": (W + 30,  0),
    "fold":   (0,       H + 30),
}
# 第 4 格放一个"原图 BACARDÍ 留作参照"标签 + 提示
from PIL import ImageDraw as ID
for name, (px, py) in positions.items():
    grid.paste(Image.open(final_imgs[name]), (px, py))
# 第 4 格：放黑底白字"原图 BACARDÍ 不入库；裂变三档见另三格"
panel4 = Image.new("RGB", (W, H), (50, 30, 60))
d4 = ID.Draw(panel4)
fn = ImageFont.truetype(FONT_PATH, 56)
d4.text((W // 2 - 350, H // 2 - 60), "BACARDI original kept aside",
        font=fn, fill=(220, 200, 230))
d4.text((W // 2 - 320, H // 2 + 20), "for reference only",
        font=fn, fill=(220, 200, 230))
fn2 = ImageFont.truetype(FONT_PATH, 38)
d4.text((W // 2 - 230, H // 2 + 110), "Three fission variants ->",
        font=fn2, fill=(180, 160, 190))
grid.paste(panel4, (W + 30, H + 30))
grid_path = os.path.join(OUT_DIR, "_grid_v301.png")
grid.save(grid_path)
print(f"\nGrid:  {grid_path}")

# 主体无字版（半自动 C 路线的"姿态大框"层）
for name, base_arr in variants.items():
    np.clip(base_arr, 0, 255).astype(np.uint8)
    Image.fromarray(np.clip(base_arr, 0, 255).astype(np.uint8)).save(
        os.path.join(OUT_DIR, f"v301_{name}_body.png"))

print("\nAll outputs ready.")
