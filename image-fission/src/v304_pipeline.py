"""v304 — 严格黑蝙蝠 mask 徽章裂变（紫冠/花瓣/紫盘彻底清除版）

v303v2 遗留问题根因（本轮实测定位）：
  1. bat_m 用 maxc<150 → 紫盘(RGB 93,21,133 maxc≈130-140) 整个混入 bat_m
  2. CLOSE(21) 把花瓣(maxc 175-215)与翅膀接触点桥接进 solid_m
  3. 翅膀旋转时把紫盘/花瓣一起拖走 → 顶弧涂抹、弧线残影、矩形切割残缺

v304 方案：
  - bat_strict = maxc<75（纯黑蝙蝠：头/翼/躯干/尾/脚），CC 最大连通域
    → 紫盘/花瓣/玻璃字/圆环 全部排除在 mask 外
  - CLOSE(25)+CCOMP 填孔 → 封住翼内紫纹/脸部紫纹（封闭孔洞）
  - ∩ dilate(bat_core,4) → 限制花瓣/紫盘接触点外溢 ≤4px
  - clean_layer：徽章圆 R=440 内非 solid → 纯紫（紫冠装饰/中央彩色 logo 按永久规则清除）
  - 翅膀/头/尾旋转全部在 纯紫平地 上进行 → 姿态变化最大化可见
"""
import math, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

# ─── 配置 ────────────────────────────────────────────────
SRC_PATH  = r"E:\Desktop\双接口\image-fission\ComfyUI\input\v300_text_free_source.png"
FONT_PATH = r"E:\Desktop\双接口\image-fission\ComfyUI\models\fonts\AbrilFatface-Regular.ttf"
OUT_DIR   = r"E:\Desktop\双接口\image-fission\jobs\v304"
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

# ─── 1. 严格黑蝙蝠 mask（v304 核心改动） ─────────────────
arr = np.array(Image.open(SRC_PATH).convert("RGB")).astype(np.float32)
H, W = arr.shape[:2]
R, G, Bc = arr[:,:,0], arr[:,:,1], arr[:,:,2]
maxc = np.maximum(np.maximum(R, G), Bc)

badge_lim = np.zeros((H, W), np.uint8)
cv2.circle(badge_lim, (cx, cy), ring_outer + 30, 255, -1)

# 纯黑蝙蝠（紫盘 maxc 130-140 / 花瓣 maxc 175-215 / 玻璃 181 全部排除）
dark_strict = ((maxc < 75).astype(np.uint8) * 255)
dark_strict = cv2.bitwise_and(dark_strict, badge_lim)
dark_strict = cv2.morphologyEx(dark_strict, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

n, labels, stats, _ = cv2.connectedComponentsWithStats(dark_strict, connectivity=8)
sizes = stats[1:, cv2.CC_STAT_AREA]
keep_idx = 1 + int(np.argmax(sizes))
bx, by, bw, bh, _area = stats[keep_idx]
print(f"  bat_core(strict) bbox: x{bx}..{bx+bw} y{by}..{by+bh} area={_area}")
bat_core = (labels == keep_idx).astype(np.uint8) * 255

# ★ 回收被 maxc<75 剪掉的部分：翼骨细梢/抗锯齿边缘
#   （贴近黑核 3px 内的 75-150 过渡带；花瓣 175-215 排除；紫盘只进 ≤3px 描边）
#   注：耳朵实测本来就短（尖端 y≈523），maxc<75 已覆盖，无需头椭圆回收
#   ——头椭圆回收会把紫盘大块带进来（头顶"礼帽"伪影，已证伪）
ext_edge = ((maxc >= 75) & (maxc < 150)).astype(np.uint8) * 255
ext_edge = cv2.bitwise_and(ext_edge, cv2.dilate(bat_core, np.ones((7, 7), np.uint8)))
cand = cv2.bitwise_or(bat_core, ext_edge)
n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
sizes2 = stats2[1:, cv2.CC_STAT_AREA]
keep2 = 1 + int(np.argmax(sizes2))
bx, by, bw, bh, _area = stats2[keep2]
print(f"  bat_core(merged) bbox: x{bx}..{bx+bw} y{by}..{by+bh} area={_area}")
bat_core = (labels2 == keep2).astype(np.uint8) * 255

# 封闭孔洞（翼内紫纹/脸部紫纹被黑包住 → 填回）
# ★ 尺寸限制：只填 小面积(≤2500px) 或 窄(短边≤30px) 的孔——
#   翼内紫纹(长条窄)和脸部细节能填回；翅膀与躯干之间的紫盘竖条(~45px 宽)
#   是大孔，不填 → 留给 clean_layer 擦成纯紫（否则随翼旋转成紫板块，已证伪）
solid = cv2.morphologyEx(bat_core, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
solid_m = solid.copy()
cnts, hier = cv2.findContours(solid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
if hier is not None:
    for i, h4 in enumerate(hier[0]):
        if h4[3] != -1:
            area = cv2.contourArea(cnts[i])
            if area <= 2500:
                cv2.drawContours(solid_m, cnts, i, 255, -1)
                continue
            _rect = cv2.minAreaRect(cnts[i])
            _short = min(_rect[1])
            if _short <= 22:
                cv2.drawContours(solid_m, cnts, i, 255, -1)
# 限制接触点外溢：solid 边界最多超出 bat_core 4px（花瓣/紫盘接触桥 ≤4px 残余）
halo_lim = cv2.dilate(bat_core, np.ones((9, 9), np.uint8))
solid_m = cv2.bitwise_and(solid_m, halo_lim)

# clean_layer: 徽章圆 R=440 内非 bat_solid → 纯紫（紫盘/花瓣/玻璃字/圆环全清）
disc = np.zeros((H, W), np.uint8)
cv2.circle(disc, (cx, cy), 440, 255, -1)
inner_bg_mask = cv2.bitwise_and(cv2.bitwise_not(solid_m), disc)
clean_layer = arr.copy()
clean_layer[inner_bg_mask > 0] = BG_PURPLE

# ★ 外皮亮色清除：solid_m−bat_core 外皮环上的花瓣/玻璃亮色(maxc>150)
#   一律改写成精确 BG 紫 → 随翼旋转也隐形（BG on BG）
#   （真翼缘抗锯齿的亮端本来就趋近 BG 色，改写无缝）
skin = cv2.subtract(solid_m, bat_core)
bright_skin = cv2.bitwise_and(skin, (maxc > 150).astype(np.uint8) * 255)
clean_layer[bright_skin > 0] = BG_PURPLE

# ★ 翼躯间紫盘竖条清除（携带侧）：CLOSE(21) 把条带两侧桥接后，残余的
#   紫盘像素(75-165)会随翼 mask 旋转成"淡紫板块"（已证伪）。
#   不改 mask（避免伤翼骨），只把条带区内的非黑非亮像素改写成精确 BG
#   → 翼携带的是隐形像素；翼骨(maxc<75)与亮粉纹(≥165)保留
strip = np.zeros((H, W), np.uint8)
cv2.rectangle(strip, (cx - 92, 540), (cx - 30, 845), 255, -1)
cv2.rectangle(strip, (cx + 30, 540), (cx + 92, 845), 255, -1)
mid = ((maxc >= 75) & (maxc < 165)).astype(np.uint8) * 255
strip_mid = cv2.bitwise_and(cv2.bitwise_and(solid_m, strip), mid)
clean_layer[strip_mid > 0] = BG_PURPLE

# ★ 厚亮色板块清除：玻璃裂纹暗线会把 翼底↔花瓣/玻璃 板块桥进 bat_core
#   （板块随翼旋转 → 淡粉色涂抹）。特征=厚(≥5px)亮色区块；
#   翼缘粉细纹/脚部白高光(≤4px 细线)腐蚀后消失 → 保留
bright_in_solid = cv2.bitwise_and((maxc > 150).astype(np.uint8) * 255, solid_m)
lower_zone = np.zeros((H, W), np.uint8)
lower_zone[690:, :] = 255
bright_lower = cv2.bitwise_and(bright_in_solid, lower_zone)
slab = cv2.erode(bright_lower, np.ones((5, 5), np.uint8))
slab = cv2.dilate(slab, np.ones((11, 11), np.uint8))
slab = cv2.bitwise_and(slab, solid_m)
if slab.any():
    bat_core[slab > 0] = 0
    solid_m[slab > 0] = 0
    clean_layer[slab > 0] = BG_PURPLE
    print(f"  slab removed px={int((slab>0).sum())}")

# ─── 2. 部位 mask（全部基于纯黑蝙蝠） ────────────────────
hx, hy = cx, 521           # 头中心
neck_y = hy + 48           # 颈部旋转中心
shoulder_y = 790
hip_y = 862                # 尾根旋转中心

band_b = np.zeros((H, W), np.uint8)
band_b[:, cx - 48 : cx + 48] = 255
# 脚爪必须静止（属于躯干）：脚爪椭圆并入 body_core，否则脚会跟着翅膀转
feet_zone = np.zeros((H, W), np.uint8)
cv2.ellipse(feet_zone, (cx, 795), (58, 48), 0, 0, 360, 255, -1)
band_b = cv2.bitwise_or(band_b, feet_zone)
body_core = cv2.bitwise_and(bat_core, band_b)
wing_m = solid_m.copy()
wing_m[body_core > 0] = 0

# ★ reveal 层：翅膀旋走后露出的背景 = 纯紫（不是 clean_layer 原样）
#   clean_layer 在实心区保留了原图（紫盘渐变带/花瓣残留），
#   翅膀一转走就露出这些原像素 → 淡紫涂抹。reveal 层把 wing_m 区全部填 BG。
reveal_layer = clean_layer.copy()
reveal_layer[wing_m > 0] = BG_PURPLE

# debug 转储
Image.fromarray(np.clip(clean_layer, 0, 255).astype(np.uint8)).save(
    os.path.join(OUT_DIR, "_chk", "dbg_clean_layer.png"))
Image.fromarray(np.clip(reveal_layer, 0, 255).astype(np.uint8)).save(
    os.path.join(OUT_DIR, "_chk", "dbg_reveal_layer.png"))
_wing_vis = arr.astype(np.uint8).copy()
_wing_vis[wing_m > 0] = (_wing_vis[wing_m > 0] * 0.5 + np.array([0, 0, 160]) * 0.5).astype(np.uint8)
Image.fromarray(_wing_vis[cy - 460 : cy + 460, cx - 460 : cx + 460]).save(
    os.path.join(OUT_DIR, "_chk", "dbg_wingm.png"))
_core_vis = arr.astype(np.uint8).copy()
_core_vis[bat_core > 0] = (_core_vis[bat_core > 0] * 0.5 + np.array([0, 160, 0]) * 0.5).astype(np.uint8)
Image.fromarray(_core_vis[cy - 460 : cy + 460, cx - 460 : cx + 460]).save(
    os.path.join(OUT_DIR, "_chk", "dbg_batcore.png"))

# 翼软权重（v305: 22px→4px——宽渐变带被大角度旋转拉成烟熏雾（fold 实测），
# 4px 仅作抗撕裂羽化，翼根贴 pivot 位移小）
dist = cv2.distanceTransform(cv2.bitwise_not(body_core), cv2.DIST_L2, 3)
wing_soft = (wing_m.astype(np.float32) / 255.0) * np.clip(dist / 4.0, 0, 1).astype(np.float32)
xs_col = np.tile(np.arange(W, dtype=np.int32), (H, 1))
left_m  = ((wing_m > 0) & (xs_col <  cx)).astype(np.uint8) * 255
right_m = ((wing_m > 0) & (xs_col >= cx)).astype(np.uint8) * 255
left_soft  = wing_soft.copy(); left_soft[xs_col >= cx]  = 0
right_soft = wing_soft.copy(); right_soft[xs_col <  cx] = 0

# 头 mask：顶部椭圆区（含封闭找回的脸部紫纹）
head_zone = np.zeros((H, W), np.uint8)
cv2.ellipse(head_zone, (cx + 4, hy + 10), (38, 52), 0, 0, 360, 255, -1)
head_m = cv2.bitwise_and(head_zone, solid_m)

# ★ v305e: 耳朵并入 head_m——耳尖是紫色的(maxc 87-139)，在 bat_core(strict<75)
# 和 solid_m 之外；头旋转时耳尖留在旧位=紫色残块（z7 实测）。用连通性从大椭圆
# 回收：耳与头黑体像素连通；紫盘同色域但不与头连通→不会混入
#（"礼帽伪影"当年是无连通性约束的整椭圆回收，已证伪路线）
head_zone_big = np.zeros((H, W), np.uint8)
cv2.ellipse(head_zone_big, (cx + 4, 545), (42, 68), 0, 0, 360, 255, -1)
ear_cand = cv2.bitwise_and(head_zone_big, (maxc < 160).astype(np.uint8) * 255)
seed = cv2.bitwise_and(bat_core, head_zone_big)
cc_in = cv2.bitwise_or(ear_cand, seed)
n3, l3, s3, _ = cv2.connectedComponentsWithStats(cc_in, 8)
main_lbl = np.bincount(l3[seed > 0]).argmax()
ear_ext = (l3 == main_lbl).astype(np.uint8) * 255
ear_ext = cv2.subtract(ear_ext, bat_core)
ear_ext = cv2.subtract(ear_ext, head_m)
ear_lim = cv2.distanceTransform(cv2.bitwise_not(seed), cv2.DIST_L2, 3)
ear_ext = cv2.bitwise_and(ear_ext, (ear_lim <= 14).astype(np.uint8) * 255)
if ear_ext.any():
    print(f"  ear_ext px={int((ear_ext>0).sum())}")
head_m = cv2.bitwise_or(head_m, ear_ext)
# v305d 关键修复：此前 distanceTransform(¬head_m) 方向反了——mask 内全 0
# → head_soft≡0 → 头/尾旋转自 v302 起从未生效（no-op），v305b 换 reveal_head
# 后反而把 head_m 咬掉一块。距离变换必须用 mask 本身（mask 内→边界距离）。
head_dist = cv2.distanceTransform(head_m, cv2.DIST_L2, 3)
head_soft_full = (head_m.astype(np.float32) / 255.0) * np.clip(head_dist / 2.5, 0, 1).astype(np.float32)

# 尾 mask：下部窄条
tail_zone = np.zeros((H, W), np.uint8)
tail_zone[hip_y - 10 : 980, cx - 30 : cx + 30] = 255
tail_m = cv2.bitwise_and(tail_zone, solid_m)
tail_dist = cv2.distanceTransform(tail_m, cv2.DIST_L2, 3)   # v305d: 同 head，方向修复
tail_soft_full = (tail_m.astype(np.float32) / 255.0) * np.clip(tail_dist / 2.5, 0, 1).astype(np.float32)

# 躯干中轴 protect（防翼旋转擦到躯干）
protect = np.zeros((H, W), np.uint8)
protect[hy + 60 : 875, cx - 22 : cx + 22] = 255
protect = cv2.bitwise_and(protect, bat_core)
body_block = cv2.GaussianBlur(protect, (5, 5), 1.5).astype(np.float32) / 255.0
body_keep  = cv2.GaussianBlur(
    cv2.dilate(protect, np.ones((5, 5), np.uint8)), (7, 7), 2
).astype(np.float32) / 255.0

# v305 关键修复：head/tail 旋转步骤的清场背景。
# 此前传 clean_layer——它在 solid_m 内保留原像素 → 旧头/旧尾位置的
# w_old 清场区显示的是原头/原尾 → 头 +8°/尾 +16° 位移后旧位残留错位
# 暴露 = 头顶紫块（旧耳冠纹）+ 尾部重影（fold 实测，spread 头尾不动
# 所以从未暴露）。旧位必须露出"该部件移走后的纯紫地"。
reveal_head = clean_layer.copy()
reveal_head[head_m > 0] = BG_PURPLE
reveal_tail = clean_layer.copy()
reveal_tail[tail_m > 0] = BG_PURPLE

# mask 转储（残留定位用）
for nm, m in [("solid", solid_m), ("wing", wing_m), ("body", body_core),
              ("head", head_m), ("tail", tail_m)]:
    Image.fromarray(m).save(os.path.join(OUT_DIR, "_chk", f"mask_{nm}.png"))

# ─── 3. _rot_warp（v300 十七轮定稿：串联 keep 修复版） ───
def _rot_warp(layer_in, part_soft, part_bin, body_block, body_keep, bg_layer,
              theta, rot_c, h, w, center_shift=(0, 0), dbg_maps=None):
    cos_t = np.float32(np.cos(theta))
    sin_t = np.float32(np.sin(theta))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xs - rot_c[0]
    dy = ys - rot_c[1]
    rx = rot_c[0] + dx * cos_t + dy * sin_t
    ry = rot_c[1] - dx * sin_t + dy * cos_t
    rx = rx - np.float32(center_shift[0])
    ry = ry - np.float32(center_shift[1])
    rot_soft = cv2.remap(part_soft, rx, ry, cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    w_new = cv2.GaussianBlur(rot_soft, (3, 3), 1.0).astype(np.float32)
    w_new = w_new * (np.float32(1.0) - body_block)
    w_old = cv2.GaussianBlur(part_bin, (3, 3), 0.8).astype(np.float32) / 255.0
    w_old = np.clip(w_old - w_new, 0, np.float32(1.0))
    if dbg_maps is not None:
        dbg_maps['w_new'] = w_new.copy()
        dbg_maps['w_old'] = w_old.copy()
        dbg_maps['rx'] = rx.copy()
        dbg_maps['ry'] = ry.copy()
        dbg_maps['rot_soft'] = rot_soft.copy()
        dbg_maps['layer_in'] = layer_in      # 引用（调用方作用域内有效）
        dbg_maps['bg_layer'] = bg_layer      # 引用
    src_warp = cv2.remap(layer_in, rx, ry, cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    if dbg_maps is not None:
        dbg_maps['src_warp'] = np.clip(src_warp, 0, 255).astype(np.uint8)
    # v305b: 旧件区硬覆盖——w_old 3px 羽化在 mask 边界行权重<1，缺的权重
    # 透出 clean_layer 边界过渡浅像素 =「旧翼弧细线/白点」真根因（ring 清
    # solid 外无效：那些像素本就是 BG）。old_hard 在 mask±3px 内=1 彻底盖住，
    # 外缘由 edge_band（blur21 大支撑）续接；新翼位由 1-w_new 封顶保护。
    old_hard = cv2.GaussianBlur(
        cv2.dilate(part_bin, np.ones((5, 5), np.uint8)), (7, 7), 1.5
    ).astype(np.float32) / 255.0
    edge_band = cv2.GaussianBlur(part_bin, (21, 21), 5.0).astype(np.float32) / 255.0
    w_bg = np.clip(np.maximum(np.maximum(w_old, old_hard), edge_band),
                   np.float32(0.0), np.float32(1.0) - w_new)
    out = (src_warp * w_new[..., None] + bg_layer * w_bg[..., None]
           + layer_in * (np.float32(1.0) - w_new - w_bg)[..., None])
    out = out * (np.float32(1.0) - body_keep)[..., None] + layer_in * body_keep[..., None]
    return out, w_new

shL = (cx - 120, shoulder_y + 60)
shR = (cx + 120, shoulder_y + 60)

# ─── 翼根 ghost 诊断面板（fold 翼根 pale 云定位） ─────────
# 输出三项贡献分解：cnew=w_new 带入的非BG内容 / crest=rest 区透出的
# layer_in 原像素 / cold=w_old 清除未净的残留。云状伪影出现在哪个
# 面板 = 哪一项是根因。
GHOST_BOX = (cx - 230, 500, cx + 10, 800)   # x0,y0,x1,y1：左翼根+旧翼上弧区

def _save_dbg_maps(tag, step, dm, out):
    x0, y0, x1, y1 = GHOST_BOX
    w_new, w_old = dm['w_new'], dm['w_old']
    rest = np.clip(np.float32(1.0) - w_new - w_old, 0, np.float32(1.0))
    bgc = BG_PURPLE.reshape(1, 1, 3).astype(np.float32)
    d_new = np.abs(dm['src_warp'].astype(np.float32) - bgc).max(axis=2) / np.float32(255)
    d_in  = np.abs(dm['layer_in'] - bgc).max(axis=2) / np.float32(255)
    d_old = np.abs(dm['bg_layer'] - bgc).max(axis=2) / np.float32(255)
    panels = {
        "out":   np.clip(out, 0, 255).astype(np.uint8),
        "wnew":  (np.clip(w_new, 0, 1) * 255).astype(np.uint8),
        "wold":  (np.clip(w_old, 0, 1) * 255).astype(np.uint8),
        "cover": ((np.float32(1.0) - rest) * 255).astype(np.uint8),
        "cnew":  (np.clip(w_new * d_new, 0, 1) * 255).astype(np.uint8),
        "crest": (np.clip(rest * d_in, 0, 1) * 255).astype(np.uint8),
        "cold":  (np.clip(w_old * d_old, 0, 1) * 255).astype(np.uint8),
    }
    for k, m in panels.items():
        Image.fromarray(m[y0:y1, x0:x1]).save(
            os.path.join(OUT_DIR, "_chk", f"dbg_{tag}_{step}_{k}.png"))
    Image.fromarray(panels["wnew"]).save(
        os.path.join(OUT_DIR, "_chk", f"dbg_{tag}_{step}_wnew_full.png"))
    Image.fromarray(panels["wold"]).save(
        os.path.join(OUT_DIR, "_chk", f"dbg_{tag}_{step}_wold_full.png"))
    print(f"  [dbg] {tag} {step} ghost panels saved")

def make_variant(wingL, wingR, head_rot, head_dy, tail_rot, dbg=None):
    """翅膀双旋 + 头旋转/位移 + 尾旋转。串联执行，每步 keep 新区防互擦。"""
    dbg1 = {} if dbg else None
    dbg2 = {} if dbg else None
    out, keepL = _rot_warp(clean_layer, left_soft, left_m, body_block, body_keep, reveal_layer,
                           wingL, shL, H, W, dbg_maps=dbg1)
    keepL_b = cv2.GaussianBlur(keepL, (9, 9), 2).astype(np.float32)
    if dbg:
        _save_dbg_maps(dbg, "step1L", dbg1, out)
        Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(
            os.path.join(OUT_DIR, "_chk", f"dbg_{dbg}_step1.png"))
    out, keepR = _rot_warp(out, right_soft, right_m, body_block, body_keep, reveal_layer,
                           wingR, shR, H, W, dbg_maps=dbg2)
    if dbg:
        _save_dbg_maps(dbg, "step2R", dbg2, out)
        Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(
            os.path.join(OUT_DIR, "_chk", f"dbg_{dbg}_step2.png"))
    keep_all = np.maximum(keepL_b, cv2.GaussianBlur(keepR, (9, 9), 2).astype(np.float32))
    # v305c: 恒等变换直接跳过（rot=0 且无位移时任何 warp 都是对自身边缘的侵蚀）
    if abs(head_rot) < 0.001 and head_dy == 0:
        keepH = np.zeros((H, W), np.float32)
    else:
        pre = out.copy()
        out, keepH = _rot_warp(out, head_soft_full, head_m, np.zeros_like(body_block),
                               np.zeros_like(body_keep), reveal_head,
                               math.radians(head_rot), (cx, neck_y), H, W,
                               center_shift=(0, head_dy))
        z = head_m > 0
        dd = np.abs(out - pre).max(axis=2)
        print(f"  [probe head] zone px={int(z.sum())} changed>10={int((dd[z]>10).sum())} "
              f"max={dd[z].max():.0f}")
    keep_all = np.maximum(keep_all, cv2.GaussianBlur(keepH, (9, 9), 2).astype(np.float32))
    if abs(tail_rot) < 0.001:
        keepT = np.zeros((H, W), np.float32)
    else:
        out, keepT = _rot_warp(out, tail_soft_full, tail_m, np.zeros_like(body_block),
                               np.zeros_like(body_keep), reveal_tail,
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
    variants[name] = make_variant(**p, dbg=name)
    print(f"  composed {name}")

# ─── 3.5 v305 残留自检（交付红线：先扫后报） ────────────
ys_g, xs_g = np.mgrid[0:H, 0:W]
disc_chk = ((xs_g - cx) ** 2 + (ys_g - cy) ** 2) < 450 ** 2
for name in poses:
    body_u8 = np.clip(variants[name], 0, 255).astype(np.uint8)
    d = body_u8.astype(np.int16) - BG_PURPLE.astype(np.int16)
    bright = (d.min(axis=2) > 18) & disc_chk            # 比 BG 亮=亮残留
    haze = (d.max(axis=2) < -18) & (d.max(axis=2) > -70) & disc_chk  # 半透明灰雾
    muted = ((np.abs(d).max(axis=2) > 25) & ~bright & ~haze
             & (d.max(axis=2) >= -70) & disc_chk)       # 中性浊化（白斑/紫条候选）
    print(f"[residue {name}] bright px={int(bright.sum())} haze px={int(haze.sum())} "
          f"muted px={int(muted.sum())}")
    vis = body_u8.copy()
    vis[haze] = (vis[haze] * 0.4 + np.array([0, 200, 0]) * 0.6).astype(np.uint8)
    vis[muted] = (vis[muted] * 0.4 + np.array([255, 200, 0]) * 0.6).astype(np.uint8)
    vis[bright] = (vis[bright] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
    bx0, by0, bx1, by1 = cx - 460, cy - 470, cx + 460, cy + 450
    Image.fromarray(vis[by0:by1, bx0:bx1]).resize(
        (int((bx1 - bx0) * 1.4), int((by1 - by0) * 1.4)), Image.LANCZOS
    ).save(os.path.join(OUT_DIR, "_chk", f"residue_{name}.png"))
    nb, lab_b, sb, _ = cv2.connectedComponentsWithStats(
        (muted.astype(np.uint8) * 255), 8)
    order = np.argsort(-sb[1:, 4])[:8]
    for oi in order:
        i = 1 + int(oi)
        x, y, w, h, a = sb[i]
        if a < 25:
            continue
        m = lab_b == i
        print(f"   muted CC bbox=({x},{y} {w}x{h}) area={a} "
              f"meanRGB={body_u8[m].mean(axis=0).round(0)}")

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
    out_p = os.path.join(OUT_DIR, f"v304_{name}_final.png")
    composed.save(out_p)
    final_imgs[name] = out_p
    Image.fromarray(np.clip(base_arr, 0, 255).astype(np.uint8)).save(
        os.path.join(OUT_DIR, f"v304_{name}_body.png"))
    print(f"  saved {out_p}")

# ─── 5. 徽章特写 + 三姿态对比条（肉眼验证关键输出） ──────
BX0, BY0, BX1, BY1 = cx - 460, cy - 470, cx + 460, cy + 450
ZOOM = 1.6
zw, zh = int((BX1 - BX0) * ZOOM), int((BY1 - BY0) * ZOOM)
for name in poses:
    body_img = Image.open(os.path.join(OUT_DIR, f"v304_{name}_body.png"))
    zoom = body_img.crop((BX0, BY0, BX1, BY1)).resize((zw, zh), Image.LANCZOS)
    zoom.save(os.path.join(OUT_DIR, "_chk", f"v304_{name}_badge_zoom.png"))

comp_h = zh
comp_w = zw * 3 + 40
comp = Image.new("RGB", (comp_w, comp_h), (30, 30, 30))
for i, name in enumerate(["up", "spread", "fold"]):
    z = Image.open(os.path.join(OUT_DIR, "_chk", f"v304_{name}_badge_zoom.png"))
    comp.paste(z, (i * (zw + 20), 0))
comp.save(os.path.join(OUT_DIR, "_chk", "v304_pose_compare.png"))
print("  saved badge zooms + pose compare")

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
grid_path = os.path.join(OUT_DIR, "_grid_v304.png")
grid.save(grid_path)
print(f"\nGrid: {grid_path}")
print("All outputs ready.")
