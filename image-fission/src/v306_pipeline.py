"""v306 — 空徽章底板 + 蝙蝠按姿态重建（圆环保留 / 翼展缩放 / 腋膜补膜 / 腿爪伸展）

用户指令：
  1. 恢复圆环与圆环文字版式（v301 布局：环保留 + 顶弧字 + EST./1862 侧翼）
  2. 主体不再刚体挪移：按原图风格重新生成——翅膀伸展长度、打开角度、
     腿部伸展、头尾角度，每姿态是重新排布的蝙蝠设计

架构（替代 v302~v305 的 clean_layer 路线）：
  plate  = 源图 − 蝙蝠全部像素（紫区 inpaint / 大理石区旋转供体填补）
           环 / 盘缘 / 花瓣 / 浮雕字全保留 → 解决「浮在空紫上」
  pose   = plate + 身体(静) + 双翼(旋转+翼展缩放+腋膜补膜) + 头/尾(旋转) + 腿爪
"""
import math
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# ─── 配置 ────────────────────────────────────────────────
SRC_PATH = r"E:\Desktop\双接口\image-fission\ComfyUI\input\v300_text_free_source.png"
FONT_PATH = r"E:\Desktop\双接口\image-fission\ComfyUI\models\fonts\AbrilFatface-Regular.ttf"
OUT_DIR = r"E:\Desktop\双接口\image-fission\jobs\v306"
CHK = os.path.join(OUT_DIR, "_chk")
os.makedirs(CHK, exist_ok=True)

BG_RGB = np.array([181.0, 129.0, 171.0], dtype=np.float32)
RING_C = (775.5, 734.1)      # 圆环圆心 (x,y)
RING_R = 310.0
DISC_R = 326.0               # 大理石盘半径（同圆心）
MED_C = (772.8, 723.5)       # 章底紫圆圆心
MED_R = 224.0
CX = 776                     # 蝙蝠中轴
HY = 521                     # 头中心 y
NECK_Y = HY + 48
SHOULDER_Y = 850             # 翼 pivot y（cx∓120, 850）
HIP_Y = 862                  # 尾 pivot y

arr = np.array(Image.open(SRC_PATH).convert("RGB")).astype(np.float32)
H, W = arr.shape[:2]
R_, G_, B_ = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
maxc = np.maximum(np.maximum(R_, G_), B_)
ys_g, xs_g = np.mgrid[0:H, 0:W].astype(np.float32)
dist_ring = np.sqrt((xs_g - RING_C[0]) ** 2 + (ys_g - RING_C[1]) ** 2)
dist_med = np.sqrt((xs_g - MED_C[0]) ** 2 + (ys_g - MED_C[1]) ** 2)

def disc_mask(r, c=RING_C):
    m = np.zeros((H, W), np.uint8)
    cv2.circle(m, (int(c[0]), int(c[1])), int(r), 255, -1)
    return m

# ─── 1. v304 mask 链（已验证）────────────────────────────
badge_lim = disc_mask(RING_R + 30)
dark_strict = ((maxc < 75).astype(np.uint8) * 255)
dark_strict = cv2.bitwise_and(dark_strict, badge_lim)
dark_strict = cv2.morphologyEx(dark_strict, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
n, labels, stats, _ = cv2.connectedComponentsWithStats(dark_strict, connectivity=8)
keep_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
bat_core = (labels == keep_idx).astype(np.uint8) * 255
print(f"  bat_core(strict) area={int((bat_core>0).sum())}")

ext_edge = ((maxc >= 75) & (maxc < 150)).astype(np.uint8) * 255
ext_edge = cv2.bitwise_and(ext_edge, cv2.dilate(bat_core, np.ones((7, 7), np.uint8)))
n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(cv2.bitwise_or(bat_core, ext_edge), 8)
keep2 = 1 + int(np.argmax(stats2[1:, cv2.CC_STAT_AREA]))
bat_core = (labels2 == keep2).astype(np.uint8) * 255

solid = cv2.morphologyEx(bat_core, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
solid_m = solid.copy()
cnts, hier = cv2.findContours(solid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
if hier is not None:
    for i, h4 in enumerate(hier[0]):
        if h4[3] != -1:
            area = cv2.contourArea(cnts[i])
            if area <= 2500:
                cv2.drawContours(solid_m, cnts, i, 255, -1
                                 )
            else:
                _rect = cv2.minAreaRect(cnts[i])
                if min(_rect[1]) <= 22:
                    cv2.drawContours(solid_m, cnts, i, 255, -1)
halo_lim = cv2.dilate(bat_core, np.ones((9, 9), np.uint8))
solid_m = cv2.bitwise_and(solid_m, halo_lim)
print(f"  solid_m area={int((solid_m>0).sum())}")

# v306 新增：贴附蝙蝠的紫色部件（耳尖/翼指紫钩/尾缘高光/脚爪）
# 判据：紫色 & 距黑体≤7px & CC 面积≤3500（花瓣 17k 排除）
purple_like = ((maxc >= 85) & (maxc <= 180) & ((B_ - G_) > 32)).astype(np.uint8) * 255
purple_near = cv2.bitwise_and(purple_like, cv2.dilate(solid_m, np.ones((15, 15), np.uint8)))
pn, pl, ps, _ = cv2.connectedComponentsWithStats(purple_near, 8)
bat_extra = np.zeros((H, W), np.uint8)
claw_m = np.zeros((H, W), np.uint8)
claw_box = None
best_claw = None
for i in range(1, pn):
    x, y, w, h, a = ps[i]
    if a > 3500:
        continue
    m = (pl == i).astype(np.uint8) * 255
    bat_extra = cv2.bitwise_or(bat_extra, m)
    # 脚爪识别：小 CC（≤900px）且位于尾根左侧区域，取最大者（防 1px 杂点覆盖真爪）
    if a <= 900 and 690 <= x <= 760 and 740 <= y <= 810:
        if best_claw is None or a > best_claw[4]:
            best_claw = (x, y, w, h, a, i)
if best_claw is not None:
    x, y, w, h, a, idx = best_claw
    claw_m = (pl == idx).astype(np.uint8) * 255
    claw_box = (x, y, w, h)
    print(f"  claw cc bbox=({x},{y} {w}x{h}) area={a}")
bat_extra = cv2.bitwise_or(bat_extra, claw_m)
print(f"  bat_extra(purple parts) area={int((bat_extra>0).sum())}")

solid_all = cv2.bitwise_or(solid_m, bat_extra)

# ─── 2. 部位划分 ─────────────────────────────────────────
band_b = np.zeros((H, W), np.uint8)
band_b[:, CX - 48: CX + 48] = 255
feet_zone = np.zeros((H, W), np.uint8)
cv2.ellipse(feet_zone, (CX, 795), (58, 48), 0, 0, 360, 255, -1)
near_band = cv2.bitwise_or(band_b, feet_zone)

# 尾部区（下窄条 + 右侧尾缘高光带）
tail_zone = np.zeros((H, W), np.uint8)
tail_zone[HIP_Y - 10: 985, CX - 32: CX + 32] = 255
tail_zone[HIP_Y - 10: 985, CX + 32: CX + 68] = 255   # 尾缘紫色高光随尾走
# 头区（含耳回收 v305e）
head_zone = np.zeros((H, W), np.uint8)
cv2.ellipse(head_zone, (CX + 4, HY + 10), (38, 52), 0, 0, 360, 255, -1)
head_m = cv2.bitwise_and(head_zone, solid_all)
head_zone_big = np.zeros((H, W), np.uint8)
cv2.ellipse(head_zone_big, (CX + 4, 545), (42, 68), 0, 0, 360, 255, -1)
ear_cand = cv2.bitwise_and(head_zone_big, (maxc < 160).astype(np.uint8) * 255)
seed = cv2.bitwise_and(bat_core, head_zone_big)
cc_in = cv2.bitwise_or(ear_cand, seed)
n3, l3, _s3, _ = cv2.connectedComponentsWithStats(cc_in, 8)
main_lbl = np.bincount(l3[seed > 0]).argmax()
ear_ext = (l3 == main_lbl).astype(np.uint8) * 255
ear_ext = cv2.subtract(ear_ext, bat_core)
ear_ext = cv2.subtract(ear_ext, head_m)
ear_lim = cv2.distanceTransform(cv2.bitwise_not(seed), cv2.DIST_L2, 3)
ear_ext = cv2.bitwise_and(ear_ext, (ear_lim <= 14).astype(np.uint8) * 255)
head_m = cv2.bitwise_or(head_m, ear_ext)
print(f"  ear_ext px={int((ear_ext>0).sum())}  head_m px={int((head_m>0).sum())}")

body_static = cv2.bitwise_and(solid_all, near_band)
body_static = cv2.subtract(body_static, tail_zone)
body_static = cv2.subtract(body_static, claw_m)
# 尾缘高光上半段（贴体侧）归身体静区；y800 以下随尾旋转（fold 接缝连续）
# 只贴高光亮核（maxc>145）——整带贴回会带矩形直边台阶（up 实测梯形亮块）
rim_up = np.zeros((H, W), np.uint8)
rim_up[700:800, CX + 40: CX + 70] = 255
rim_dn = np.zeros((H, W), np.uint8)
rim_dn[800:HIP_Y - 10, CX + 32: CX + 70] = 255
_rim_up_core = cv2.bitwise_and(rim_up, (maxc > 145).astype(np.uint8) * 255)
body_static = cv2.bitwise_or(body_static, cv2.bitwise_and(bat_extra, _rim_up_core))

tail_zone[800:HIP_Y - 10, CX + 32: CX + 70] = 255   # 尾区上扩：携带 rump 高光下段
tail_m = cv2.bitwise_and(cv2.bitwise_or(tail_zone, rim_up), solid_all)
tail_m = cv2.subtract(tail_m, cv2.bitwise_and(tail_zone, body_static))
tail_m = cv2.bitwise_or(tail_m, cv2.bitwise_and(bat_extra, rim_dn))
tail_m[0:800, :] = 0

wing_m = solid_all.copy()
wing_m[body_static > 0] = 0
wing_m[tail_m > 0] = 0
wing_m[head_m > 0] = 0
wing_m[claw_m > 0] = 0
# 腿根 trim：原图大腿/腿根黑实体（躯干竖条以外 y745..875）不入任何部件——
# 新腿画到别处后旧腿根会悬空；bat_bin 已含它们 → plate 在其处填紫，剔除即露出
# 翼膜下缘根部剔除：躯干两侧 ±24..70、y768..852——翼外展后膜主体上移而
# pivot 附近根部内容几乎不动，留成"断梗"黑横杠（spread 实测）；排除 rim_dn
# 防伤尾缘高光。plate 已在其处填紫；腋部形态由 membrane_fill 渐变膜接管
leg_root = np.zeros((H, W), np.uint8)
leg_root[745:876, :] = 255
_lb = np.zeros((H, W), np.uint8)
cv2.rectangle(_lb, (CX - 47, 740), (CX + 47, 876), 255, -1)
leg_root = cv2.bitwise_and(leg_root, cv2.bitwise_not(_lb))
leg_root = cv2.bitwise_and(leg_root, solid_m)
_lr = np.zeros((H, W), np.uint8)
cv2.rectangle(_lr, (CX - 70, 768), (CX - 24, 852), 255, -1)
cv2.rectangle(_lr, (CX + 24, 768), (CX + 70, 852), 255, -1)
_lr = cv2.bitwise_and(_lr, solid_all)
leg_root = cv2.bitwise_or(leg_root, _lr)
leg_root = cv2.subtract(leg_root, rim_dn)
leg_root = cv2.subtract(leg_root, claw_m)
body_static = cv2.subtract(body_static, leg_root)
tail_m = cv2.subtract(tail_m, leg_root)
wing_m = cv2.subtract(wing_m, leg_root)
print(f"  leg_root trimmed px={int((leg_root>0).sum())}")
print(f"  parts: body={int((body_static>0).sum())} wing={int((wing_m>0).sum())} "
      f"head={int((head_m>0).sum())} tail={int((tail_m>0).sum())} claw={int((claw_m>0).sum())}")

# ─── 3. plate：源图 − 蝙蝠 → 修复 ────────────────────────
# rim light 描边归属：蝙蝠上缘浅色描边（翼指钩淡紫描边/翼顶亮线）是主体
# 风格一部分 → 按就近部件并入，随姿态一起移动；plate 填充其背后。
dist_to_bat0 = cv2.distanceTransform(cv2.bitwise_not(solid_all), cv2.DIST_L2, 3)
# 亮描边（翼指钩淡紫描边/翼顶亮线/尾缘 S 高光，含章底以外）
rim_bright = ((maxc > 150).astype(np.uint8) * 255)
rim_bright = cv2.bitwise_and(rim_bright, (dist_to_bat0 <= 16).astype(np.uint8) * 255)
rim_bright = cv2.bitwise_and(rim_bright, disc_mask(DISC_R - 2))
rim_bright = cv2.bitwise_and(rim_bright, (dist_med < 240).astype(np.uint8) * 255)  # 防误抓大理石盘缘亮带
# 粉色月牙晕（肩部 crescent，maxc 110-150，仅章底内——防止误抓大理石）
rim_pink = ((maxc > 110).astype(np.uint8) * 255)
rim_pink = cv2.bitwise_and(rim_pink, (G_ > 75).astype(np.uint8) * 255)
rim_pink = cv2.bitwise_and(rim_pink, (dist_to_bat0 <= 16).astype(np.uint8) * 255)
rim_pink = cv2.bitwise_and(rim_pink, (dist_med < MED_R).astype(np.uint8) * 255)
rim = cv2.bitwise_or(rim_bright, rim_pink)
part_lab = np.zeros((H, W), np.uint8)
part_lab[body_static > 0] = 1
part_lab[wing_m > 0] = 2
part_lab[head_m > 0] = 3
part_lab[tail_m > 0] = 4
inv_p = (part_lab == 0).astype(np.uint8)
_dl, _ll = cv2.distanceTransformWithLabels(inv_p, cv2.DIST_L2, 5,
                                           labelType=cv2.DIST_LABEL_PIXEL)
_p_map = np.zeros(int(_ll.max()) + 1, np.uint8)
_p_map[_ll[part_lab > 0]] = part_lab[part_lab > 0]
near_part = _p_map[_ll]
rim_cnt = {}
for pv, tgt in ((2, "wing"), (3, "head"), (4, "tail"), (1, "body")):
    m = ((rim > 0) & (near_part == pv)).astype(np.uint8) * 255
    rim_cnt[tgt] = int((m > 0).sum())
    if tgt == "wing":
        wing_m = cv2.bitwise_or(wing_m, m)
    elif tgt == "head":
        head_m = cv2.bitwise_or(head_m, m)
    elif tgt == "tail":
        tail_m = cv2.bitwise_or(tail_m, m)
    else:
        body_static = cv2.bitwise_or(body_static, m)
solid_all = cv2.bitwise_or(solid_all, rim)
print(f"  rim captured: {rim_cnt} total={int((rim>0).sum())}")

bat_all = cv2.bitwise_or(solid_all, head_m)

# 暗晕清除：蝙蝠周围 AA 过渡带（章底内 maxc<118；盘面区 maxc<95 近距）
dark2 = (((maxc < 118).astype(np.uint8) * 255)
         & cv2.dilate(bat_all, np.ones((19, 19), np.uint8)))
dark2_in = cv2.bitwise_and(dark2, (dist_med < MED_R).astype(np.uint8) * 255)
dark2_out = (((maxc < 95).astype(np.uint8) * 255)
             & (dist_to_bat0 <= 12).astype(np.uint8) * 255)
dark2_out = cv2.bitwise_and(dark2_out, disc_mask(DISC_R - 2))
dark2_out = cv2.subtract(dark2_out, dark2_in)
print(f"  dark halo px: in={int((dark2_in>0).sum())} out={int((dark2_out>0).sum())}")
bat_all = cv2.bitwise_or(bat_all, cv2.bitwise_or(dark2_in, dark2_out))
# dark3 泛捕获：盘内漏网暗 CC（翼端三角窗/头饰碎弧——halo 限制外的孤立暗块，
# plate 上留黑残块的根源）。蝙蝠黑体 B−G≈18，大理石紫裂纹 B−G≥32 → B−G<26 区分
dark3 = (((maxc < 100) & ((B_ - G_) < 26)).astype(np.uint8) * 255)
dark3 = cv2.bitwise_and(dark3, (dist_ring < 315).astype(np.uint8) * 255)
dark3 = cv2.subtract(dark3, bat_all)
_n3d, _l3d, _s3d, _ = cv2.connectedComponentsWithStats(dark3, 8)
_keep3 = np.zeros_like(dark3)
for _i in range(1, _n3d):
    if _s3d[_i, 4] >= 50:
        _keep3[_l3d == _i] = 255
dark3 = _keep3
bat_all = cv2.bitwise_or(bat_all, dark3)
# 腿根/翼膜根矩形内残余暗碎片强制清除：bat_bin 清掉横杠主体后，边缘碎片
# 成 CC<50 的小岛躲过 dark3 门槛（plate 上留 1/3 横杠残段，spread 实测）
_fq = np.zeros((H, W), np.uint8)
cv2.rectangle(_fq, (CX - 70, 768), (CX - 24, 852), 255, -1)
cv2.rectangle(_fq, (CX + 24, 768), (CX + 70, 852), 255, -1)
_fq = cv2.bitwise_and(_fq, ((maxc < 135) & ((B_ - G_) < 26)).astype(np.uint8) * 255)
bat_all = cv2.bitwise_or(bat_all, _fq)
print(f"  dark3 captured: {int((dark3>0).sum())} px, leg-root force-clear: {int((_fq>0).sum())} px")
bat_bin = (bat_all > 0).astype(np.uint8)

# 种子归类：蝙蝠环带上的紫 vs 非紫 → 最近种子传播
ring_zone = cv2.subtract(cv2.dilate(bat_bin * 255, np.ones((29, 29), np.uint8)), bat_bin * 255)
ring_zone = cv2.bitwise_and(ring_zone, disc_mask(DISC_R + 6))
seeds = np.zeros((H, W), np.uint8)
# 紫种子仅认章底/花瓣空间域内（dist_med<MED_R+8）——大理石暗裂纹纹理
# (maxc 85-150, B−G>32) 与花瓣色域重叠，不锁空间会把大理石判成紫区
seed_purple_ok = (dist_med < MED_R + 8).astype(np.uint8) * 255
seeds[(ring_zone > 0) & (purple_like > 0) & (seed_purple_ok > 0)] = 1
seeds[(ring_zone > 0) & (seeds == 0)] = 2
inv = (seeds == 0).astype(np.uint8)
_dist, lbl = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5,
                                             labelType=cv2.DIST_LABEL_PIXEL)
lab_ids = lbl[seeds > 0]
cls_ids = seeds[seeds > 0]
cls_of_label = np.zeros(int(lbl.max()) + 1, np.uint8)
cls_of_label[lab_ids] = cls_ids
nearest_cls = cls_of_label[lbl]

# 3.1 每个蝙蝠像素的修复方式：
#     章底圆内 → 纯紫填充（章底本就是平紫，杜绝大区域 inpaint 云团）
#     章底圆外 → 就近种子归类：紫(花瓣交界,小区域 inpaint) vs 大理石(纹理拼贴)
inside_med = dist_med < (MED_R - 2)
below_med = ys_g > (MED_C[1] + MED_R - 2)      # 章底以下（尾尖区）→ 大理石
fill_purple = (((nearest_cls == 1) | inside_med) & ~below_med) & (bat_bin > 0)
fill_marble = ((nearest_cls == 2) & ~inside_med) | (below_med & (bat_bin > 0))
fill_marble = fill_marble & (bat_bin > 0) & ~fill_purple
print(f"  plate fill: purple-assigned={int(fill_purple.sum())} marble-assigned={int(fill_marble.sum())}")

plate_f = arr.copy()
arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
rng = np.random.default_rng(7)

# 章底紫中值色（供填充与 membrane 兜底）
pm = purple_like.astype(bool) & (dist_med < MED_R - 6) & (bat_bin == 0)
MED_COL = np.median(arr[pm], axis=0) if pm.any() else np.array([93, 21, 125], np.float32)
print(f"  medallion purple median RGB={MED_COL.round(0)}")

# 翼采样净版源图：翼躯之间的紫盘竖条（v304 strip 教训）随翼旋转会拖出
# 垂直涂抹列——翼内容采样时把 strip 区改写为章底紫（与背后填充同色→隐形）
SRC_WING = arr.copy()
strip = np.zeros((H, W), np.uint8)
cv2.rectangle(strip, (CX - 92, 540), (CX - 30, 845), 255, -1)
cv2.rectangle(strip, (CX + 30, 540), (CX + 92, 845), 255, -1)
strip_mid = cv2.bitwise_and(cv2.bitwise_and(solid_all, strip),
                            ((maxc >= 75) & (maxc < 165)).astype(np.uint8) * 255)
SRC_WING[strip_mid > 0] = MED_COL
# strip 不做 mask 硬减（翼膜内缘会出笔直缺口边，fold 实测）——改在翼 soft 上
# 羽化减（soft 段 strip_soft），SRC_WING 该区回填 plate 像素兜底
print(f"  strip cleaned for wing sampling: {int((strip_mid>0).sum())} px")

# 3.2 紫区修复：tierA=章底内部平坦带 → 低频场外推；tierB=花瓣/边缘交界 → 小半径 inpaint
tierA = fill_purple & (dist_med < MED_R - 4) & (ys_g < 815)
tierB = fill_purple & ~tierA
# 章底低频场：非蝙蝠章底像素大核归一化卷积外推（自带二维光照渐变，
# 消除中值色+线性渐变模型造成的可见光晕色差）
bg_field = ((dist_med < MED_R) & (bat_bin == 0) & (ys_g < 815)).astype(np.float32)
numf = cv2.GaussianBlur(arr.astype(np.float32) * bg_field[..., None], (121, 121), 40)
denf = cv2.GaussianBlur(bg_field, (121, 121), 40)
lowfield = numf / np.clip(denf, 1e-2, None)[..., None]
plate_f[tierA] = lowfield[tierA]
base_u8 = np.clip(plate_f, 0, 255).astype(np.uint8)
maskB = tierB.astype(np.uint8) * 255
if maskB.any():
    inpB = cv2.inpaint(base_u8, maskB, 5, cv2.INPAINT_TELEA)
    plate_f[tierB] = inpB[tierB]
print(f"  purple fill: flat={int(tierA.sum())} inpaint={int(tierB.sum())}")

# 章底填充低频融合：大核模糊把周边原章底色调拖入填充区（消色调台阶）
low = cv2.GaussianBlur(np.clip(plate_f, 0, 255).astype(np.uint8), (41, 41), 8).astype(np.float32)
tierA_core = cv2.erode(tierA.astype(np.uint8) * 255, np.ones((7, 7), np.uint8)) > 0
plate_f[tierA_core] = np.clip(0.55 * plate_f[tierA_core] + 0.45 * low[tierA_core], 0, 255)

# 3.3 大理石区：干净大理石纹理块随机拼贴（quilt + 块界模糊去缝）
# 窗口 52 / 块 40（64/48 时干净窗口仅 9 个，拼贴重复感重）
need_m = fill_marble.copy()
cand = []
for by in range(420, 1020, 16):
    for bx in range(460, 1100, 16):
        dm_ = dist_ring[by:by + 52, bx:bx + 52]
        if dm_.shape != (52, 52) or (dm_ > DISC_R - 8).any():
            continue
        if bat_bin[by:by + 52, bx:bx + 52].any():
            continue
        if (G_[by:by + 52, bx:bx + 52] < 80).any():      # 章底深紫/环暗线
            continue
        if (maxc[by:by + 52, bx:bx + 52] > 212).any():   # 浮雕亮字
            continue
        cand.append((bx, by))
print(f"  marble donor patches: {len(cand)}")
if len(cand) < 8:
    for by in range(420, 1020, 10):
        for bx in range(460, 1100, 10):
            dm_ = dist_ring[by:by + 52, bx:bx + 52]
            if dm_.shape != (52, 52) or (dm_ > DISC_R - 8).any():
                continue
            if bat_bin[by:by + 52, bx:bx + 52].any():
                continue
            if (G_[by:by + 52, bx:bx + 52] < 70).any() or (maxc[by:by + 52, bx:bx + 52] > 218).any():
                continue
            cand.append((bx, by))
    cand = list(dict.fromkeys(cand))
    print(f"  relaxed marble donor patches: {len(cand)}")
assert cand, "no marble donor patches found"

stamped = plate_f.copy()
wsum = np.zeros((H, W), np.float32)
ys_n, xs_n = np.nonzero(need_m)
BS = 40
for by in range(int(ys_n.min()) // BS * BS, int(ys_n.max()) + 1, BS):
    for bx in range(int(xs_n.min()) // BS * BS, int(xs_n.max()) + 1, BS):
        blk = need_m[by:by + BS, bx:bx + BS]
        if not blk.any():
            continue
        pbx, pby = cand[int(rng.integers(len(cand)))]
        ox = int(rng.integers(0, 52 - BS + 1))
        oy = int(rng.integers(0, 52 - BS + 1))
        crop = arr_u8[pby + oy:pby + oy + BS, pbx + ox:pbx + ox + BS]
        region = stamped[by:by + BS, bx:bx + BS]
        region[blk] = crop[blk]
        wsum[by:by + BS, bx:bx + BS][blk] = 1.0
st_blur = cv2.GaussianBlur(stamped * wsum[..., None], (9, 9), 2.0)   # 归一化卷积分子
w_blur = cv2.GaussianBlur(wsum, (9, 9), 2.0)
quil = st_blur / np.clip(w_blur, 1e-2, None)[..., None]              # 有界，不爆白
alpha_q = np.clip(w_blur * 3.0, 0, 1).astype(np.float32)[..., None]
plate_f = plate_f * (1 - alpha_q) + quil * alpha_q
plate_f[need_m] = stamped[need_m]   # 核心区保留原始拼贴（块界已由混合带柔化）
print(f"  marble quilted px={int(need_m.sum())}")

# 填补区轻噪点 + 边界带柔化（隐藏接缝）
rng = np.random.default_rng(7)
noise = rng.normal(0, 1.3, (H, W, 1)).astype(np.float32)
allfill = (fill_purple | fill_marble)
band = cv2.subtract(cv2.dilate(allfill.astype(np.uint8) * 255, np.ones((7, 7), np.uint8)),
                    allfill.astype(np.uint8) * 255)
soft_plate = cv2.GaussianBlur(plate_f, (5, 5), 1.4)
plate_f[allfill] = np.clip(plate_f[allfill] + noise[allfill], 0, 255)
plate_f[band > 0] = soft_plate[band > 0]
# 大理石修复区旧化：低频云雾亮度扰动——修复区比周边 distressed 质感
# "太干净"会读作白斑（盘下缘浮雕字区三姿态实测）
_rng2 = np.random.default_rng(11)
_cloud = cv2.GaussianBlur(_rng2.normal(0, 1, (H, W)).astype(np.float32), (61, 61), 18)
_cloud = _cloud / max(float(np.abs(_cloud).max()), 1e-6) * 7.0
plate_f[need_m] = np.clip(plate_f[need_m] + _cloud[need_m][..., None], 0, 255)
plate = np.clip(plate_f, 0, 255).astype(np.uint8)
Image.fromarray(plate).save(os.path.join(CHK, "plate.png"))
Image.fromarray(plate[300:1150, 430:1130]).save(os.path.join(CHK, "plate_zoom.png"))
print("  plate saved")

# 翼采样净版升级：strip 区回填 plate 真实像素（含微渐变+噪点）——翼 warp 的
# backward mapping 命中 strip 时采到的是无缝章底，纯色 MED_COL 兜底作废
SRC_WING[strip_mid > 0] = plate[strip_mid > 0]

# 章底紫中值色（membrane 兜底色）
pm = purple_like.astype(bool) & (dist_med < MED_R - 6) & (bat_bin == 0)
MED_COL = np.median(plate[pm], axis=0) if pm.any() else np.array([93, 21, 125], np.float32)
print(f"  medallion purple median RGB={MED_COL.round(0)}")

# ─── 4. 通用放置 warp（内容一律取自源图 arr）─────────────
body_block = cv2.GaussianBlur(cv2.bitwise_and(
    (np.arange(W, dtype=np.int32)[None, :] >= CX - 22).astype(np.uint8) * 255 * 0
    + ((xs_g >= CX - 22) & (xs_g <= CX + 22) & (ys_g >= HY + 60) & (ys_g <= 875)).astype(np.uint8) * 255
    , bat_core), (5, 5), 1.5).astype(np.float32) / 255.0
body_keep = cv2.GaussianBlur(cv2.dilate(
    cv2.bitwise_and(((xs_g >= CX - 24) & (xs_g <= CX + 24)
                     & (ys_g >= HY + 58) & (ys_g <= 878)).astype(np.uint8) * 255,
                    bat_core), np.ones((5, 5), np.uint8)), (7, 7), 2).astype(np.float32) / 255.0

def _place(layer, soft, binm, theta, pivot, sx=1.0, shift=(0.0, 0.0),
           keep=None, protect_body=True, content=None):
    cos_t = np.float32(np.cos(theta)); sin_t = np.float32(np.sin(theta))
    dx = xs_g - pivot[0]; dy = ys_g - pivot[1]
    rx = pivot[0] + dx * cos_t + dy * sin_t
    ry = pivot[1] - dx * sin_t + dy * cos_t
    if sx != 1.0:
        rx = np.float32(pivot[0]) + (rx - np.float32(pivot[0])) / np.float32(sx)
    rx = rx - np.float32(shift[0]); ry = ry - np.float32(shift[1])
    rx = np.clip(rx, 0, W - 1).astype(np.float32)
    ry = np.clip(ry, 0, H - 1).astype(np.float32)
    rot_soft = cv2.remap(soft, rx, ry, cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    w_new = cv2.GaussianBlur(rot_soft, (3, 3), 1.0).astype(np.float32)
    if protect_body:
        w_new = w_new * (np.float32(1.0) - body_block)
    if keep is not None:
        w_new = w_new * (np.float32(1.0) - np.clip(keep, 0, 1))
    src_img = arr if content is None else content
    src_warp = cv2.remap(src_img, rx, ry, cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    out = layer * (np.float32(1.0) - w_new[..., None]) + src_warp * w_new[..., None]
    if protect_body:
        out = out * (np.float32(1.0) - body_keep[..., None]) + layer * body_keep[..., None]
    return out, w_new

def soft_of(mask, feather=2.5):
    d = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    return ((mask.astype(np.float32) / 255.0)
            * np.clip(d / np.float32(feather), 0, 1)).astype(np.float32)

# 身体静区绘制 mask（羽化 1.5px）
body_draw = cv2.GaussianBlur(body_static, (3, 3), 1.2).astype(np.float32) / 255.0
body_draw = np.clip(body_draw * 1.4, 0, 1).astype(np.float32)

# 翼 soft（v305：4px 窄羽化防烟熏雾）
dist_to_body = cv2.distanceTransform(cv2.bitwise_not(body_static), cv2.DIST_L2, 3)
wing_soft_full = ((wing_m.astype(np.float32) / 255.0)
                  * np.clip(dist_to_body / 4.0, 0, 1)).astype(np.float32)
# strip 缝在 soft 层羽化减除（翼躯间背景缝不随翼走；硬减会出笔直缺口边）
strip_soft = cv2.GaussianBlur((strip_mid > 0).astype(np.float32), (9, 9), 2.5)
wing_soft_full = np.clip(wing_soft_full - strip_soft, 0, 1).astype(np.float32)
xs_col = xs_g
left_soft = wing_soft_full * (xs_col < CX).astype(np.float32)
right_soft = wing_soft_full * (xs_col >= CX).astype(np.float32)
left_m = ((wing_m > 0) & (xs_col < CX)).astype(np.uint8) * 255
right_m = ((wing_m > 0) & (xs_col >= CX)).astype(np.uint8) * 255
head_soft = soft_of(head_m, 2.5)
tail_soft = soft_of(tail_m, 2.5)

shL = (CX - 120, SHOULDER_Y)
shR = (CX + 120, SHOULDER_Y)

# ─── 5. 腋膜补膜 ─────────────────────────────────────────
def membrane_fill(out_f, wing_keep, side, y0=620, y1=885, drawn=None):
    """翼与躯干之间被拉开的缝隙 → 按原图翼膜色渐变补膜。"""
    filled = np.zeros((H, W), np.float32)
    body_x = CX - 46 if side == "L" else CX + 46
    sgn = -1 if side == "L" else 1
    for y in range(y0, y1):
        if 786 <= y < 824:
            # 腋下禁区：翼外展后腋下应露出章底背景——此高度带补膜只会画成
            # 深色横杠（spread 实测；leg_root 剔除翼膜下缘根部后 gap 暴露）
            continue
        row = wing_keep[y]
        rng_ = slice(CX - 235, CX - 42) if side == "L" else slice(CX + 42, CX + 235)
        seg = row[rng_]
        idx = np.nonzero(seg > 0.35)[0]
        if len(idx) == 0:
            continue
        x_edge = (CX - 235 + idx.max()) if side == "L" else (CX + 42 + idx.min())
        gap = abs(body_x - x_edge)
        if gap < 10 or gap > 95:
            continue
        # 翼内容已越过体线（翼根搭在身上）→ 该行无需补膜，防止反向涂刷条纹
        if side == "L" and x_edge > body_x - 3:
            continue
        if side == "R" and x_edge < body_x + 3:
            continue
        # 翼侧颜色：向翼内逐级探测，取真实翼膜内容（避免采到软边/底色）
        c_wing = None
        for off in (10, 16, 24):
            xs_smp = (slice(x_edge + sgn * (off + 4), x_edge + sgn * off) if side == "L"
                      else slice(x_edge + sgn * off, x_edge + sgn * (off + 4)))
            if wing_keep[y, xs_smp].mean() > 0.5:
                c_wing = out_f[y, xs_smp].mean(axis=0)
                break
        if c_wing is None:
            c_wing = MED_COL
        # 体侧颜色
        xs_b = slice(body_x + 6, body_x + 10) if side == "L" else slice(body_x - 10, body_x - 6)
        c_body = out_f[y, xs_b].mean(axis=0)
        x0, x1 = (x_edge, body_x) if side == "L" else (body_x, x_edge)
        n_ = x1 - x0
        t = (np.arange(n_, dtype=np.float32) + 0.5) / max(n_, 1)
        cols = c_wing[None, :] * (1 - t[:, None]) + c_body[None, :] * t[:, None]
        a = np.clip(np.minimum(t, 1 - t) / 0.10, 0, 1).astype(np.float32)
        a = a * np.float32(np.clip(1.7 - gap / 70.0, 0.45, 1.0))  # 宽缝降alpha防假膜
        # 骨线提示（t≈0.35 一条暗线，仅够宽的膜才画，窄缝画了就是条纹噪声）
        if n_ >= 26:
            bone = np.exp(-((t - 0.35) ** 2) / (2 * 0.03 ** 2)).astype(np.float32) * 0.35
            cols = cols * (1 - bone[:, None] * 0.5)
        reg = slice(x0, x1)
        old_a = filled[y, reg]
        a = np.maximum(a, old_a)
        base = out_f[y, reg]
        out_f[y, reg] = base * (1 - a[:, None]) + cols * a[:, None]
        filled[y, reg] = a
    if drawn is not None:
        pass
    return filled

# ─── 6. 腿爪 ─────────────────────────────────────────────
if claw_box is not None:
    cbx, cby, cbw, cbh = claw_box
    pad = 4
    px0, py0 = max(0, cbx - pad), max(0, cby - pad)
    px1, py1 = min(W, cbx + cbw + pad), min(H, cby + cbh + pad)
    claw_patch = arr[py0:py1, px0:px1].copy()
    sub_maxc = maxc[py0:py1, px0:px1]
    sub_bg = (B_[py0:py1, px0:px1] - G_[py0:py1, px0:px1])
    a_raw = (((sub_maxc > 105) & (sub_bg > 28)).astype(np.uint8))
    # 只保留最大连通域（踢掉混进来的尾缘亮线/杂块），开运算去毛刺
    a_raw = cv2.morphologyEx(a_raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    na, la, sa, _ = cv2.connectedComponentsWithStats(a_raw, 8)
    if na > 1:
        ka = 1 + int(np.argmax(sa[1:, 4]))
        a_kept = (la == ka).astype(np.float32)
    else:
        a_kept = a_raw.astype(np.float32)
    a_kept = cv2.dilate(a_kept, np.ones((3, 3), np.uint8))
    claw_alpha = cv2.GaussianBlur(a_kept, (3, 3), 1.0)
    claw_alpha = np.clip(claw_alpha * 1.6, 0, 1)
    CLAW_C = (px0 + claw_patch.shape[1] / 2.0, py0 + claw_patch.shape[0] / 2.0)
    print(f"  claw patch {claw_patch.shape[:2]} center=({CLAW_C[0]:.0f},{CLAW_C[1]:.0f})")
else:
    claw_patch = None
    CLAW_C = (725, 779)

def paste_claw(out_f, target_c, mirror=False, cov=None):
    if claw_patch is None:
        return
    p = claw_patch[:, ::-1].copy() if mirror else claw_patch
    a = claw_alpha[:, ::-1].copy() if mirror else claw_alpha
    ph, pw = p.shape[:2]
    tx, ty = int(round(target_c[0] - pw / 2)), int(round(target_c[1] - ph / 2))
    x0, y0 = max(0, tx), max(0, ty)
    x1, y1 = min(W, tx + pw), min(H, ty + ph)
    if x1 <= x0 or y1 <= y0:
        return
    sub = out_f[y0:y1, x0:x1]
    sa = a[y0 - ty:y1 - ty, x0 - tx:x1 - tx][..., None]
    sp = p[y0 - ty:y1 - ty, x0 - tx:x1 - tx]
    out_f[y0:y1, x0:x1] = sub * (1 - sa) + sp * sa
    if cov is not None:
        cov[y0:y1, x0:x1] = np.maximum(cov[y0:y1, x0:x1], sa[..., 0])

def draw_limb(out_f, p0, p1, w0=13, w1=7, rim=True, cov=None):
    """体侧到爪的锥形腿段（体黑 + 外缘紫描边）。"""
    p0 = (float(p0[0]), float(p0[1])); p1 = (float(p1[0]), float(p1[1]))
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = max(math.hypot(dx, dy), 1.0)
    ux, uy = -dy / L, dx / L
    quad = np.array([[p0[0] + ux * w0 / 2, p0[1] + uy * w0 / 2],
                     [p1[0] + ux * w1 / 2, p1[1] + uy * w1 / 2],
                     [p1[0] - ux * w1 / 2, p1[1] - uy * w1 / 2],
                     [p0[0] - ux * w0 / 2, p0[1] - uy * w0 / 2]], np.float32)
    qi = quad.astype(np.int32)
    cv2.fillPoly(out_f, [qi], (20.0, 4.0, 22.0))
    r0 = r1 = None
    if rim:
        rim_col = np.array([147.0, 55.0, 150.0], np.float32)
        r0 = (p0[0] + ux * (w0 / 2 + 1.2), p0[1] + uy * (w0 / 2 + 1.2))
        r1 = (p1[0] + ux * (w1 / 2 + 1.0), p1[1] + uy * (w1 / 2 + 1.0))
        cv2.line(out_f, (int(r0[0]), int(r0[1])), (int(r1[0]), int(r1[1])),
                 rim_col.tolist(), 2, cv2.LINE_AA)
    if cov is not None:
        cv2.fillPoly(cov, [qi], 1.0)
        if r0 is not None:
            cv2.line(cov, (int(r0[0]) - 2, int(r0[1]) - 2),
                     (int(r1[0]) + 2, int(r1[1]) + 2), 1.0, 4, cv2.LINE_AA)

# ─── 7. 姿态合成 ─────────────────────────────────────────
POSES = {
    #  wingL, wingR, sxL, sxR, head_rot, head_dy, tail_rot, leg模式
    "up":     dict(wL=math.radians(18), wR=math.radians(-18), sxL=1.10, sxR=1.10,
                   hr=-8, hdy=-8, tr=-14, leg="dangle"),
    "spread": dict(wL=math.radians(-12), wR=math.radians(12), sxL=1.16, sxR=1.16,
                   hr=0, hdy=0, tr=0, leg="extend"),
    "fold":   dict(wL=math.radians(-33), wR=math.radians(33), sxL=0.92, sxR=0.92,
                   hr=8, hdy=8, tr=16, leg="tuck"),
}

def build_pose(name, p):
    out = plate.astype(np.float32).copy()
    # 身体静区
    out = out * (1 - body_draw[..., None]) + arr * body_draw[..., None]
    drawn = body_draw.copy()
    # 左翼
    out, keepL = _place(out, left_soft, left_m, p["wL"], shL, sx=p["sxL"], keep=drawn,
                        content=SRC_WING)
    drawn = np.maximum(drawn, cv2.GaussianBlur(keepL, (5, 5), 1.5))
    # 右翼
    out, keepR = _place(out, right_soft, right_m, p["wR"], shR, sx=p["sxR"], keep=drawn,
                        content=SRC_WING)
    drawn = np.maximum(drawn, cv2.GaussianBlur(keepR, (5, 5), 1.5))
    # 腋膜补膜
    memL = membrane_fill(out, keepL, "L")
    memR = membrane_fill(out, keepR, "R")
    drawn = np.maximum(drawn, np.maximum(memL, memR))
    # 头
    if abs(p["hr"]) > 0.001 or p["hdy"] != 0:
        out, keepH = _place(out, head_soft, head_m, math.radians(p["hr"]), (CX, NECK_Y),
                            shift=(0, p["hdy"]), keep=drawn, protect_body=False)
        drawn = np.maximum(drawn, cv2.GaussianBlur(keepH, (5, 5), 1.5))
    # 尾
    if abs(p["tr"]) > 0.001:
        out, keepT = _place(out, tail_soft, tail_m, math.radians(p["tr"]), (CX, HIP_Y),
                            keep=drawn, protect_body=False)
        drawn = np.maximum(drawn, cv2.GaussianBlur(keepT, (5, 5), 1.5))
    # 腿爪
    leg_cov = np.zeros((H, W), np.float32)
    lc = np.array(CLAW_C, np.float32)
    rc = np.array([2 * CX - CLAW_C[0], CLAW_C[1]], np.float32)
    if p["leg"] == "extend":
        lt = lc + np.array([-14, 52], np.float32)
        rt = rc + np.array([14, 52], np.float32)
        paste_claw(out, lt, mirror=False, cov=leg_cov)
        paste_claw(out, rt, mirror=True, cov=leg_cov)
        # 腿线真正斜向下（起点高、终点低）——早期终点与起点近同高，
        # 整条腿画成近水平黑横杠 + rim 描边（"浮空残块"的真身）
        draw_limb(out, (CX - 14, 788), (lt[0] + 4, lt[1] - 8), 13, 7, cov=leg_cov)
        draw_limb(out, (CX + 14, 788), (rt[0] - 4, rt[1] - 8), 13, 7, cov=leg_cov)
    elif p["leg"] == "dangle":
        lt = lc + np.array([-3, 26], np.float32)
        rt = rc + np.array([3, 26], np.float32)
        paste_claw(out, lt, mirror=False, cov=leg_cov)
        paste_claw(out, rt, mirror=True, cov=leg_cov)
        draw_limb(out, (CX - 16, 786), (lt[0] + 3, lt[1] - 8), 12, 8, cov=leg_cov)
        draw_limb(out, (CX + 16, 786), (rt[0] - 3, rt[1] - 8), 12, 8, cov=leg_cov)
    else:  # tuck
        paste_claw(out, lc + np.array([-2, -4], np.float32), mirror=False, cov=leg_cov)
    drawn = np.maximum(drawn, cv2.GaussianBlur(leg_cov, (5, 5), 1.5))
    return out, drawn

variants = {}
coverage = {}
for name, p in POSES.items():
    o, d = build_pose(name, p)
    variants[name] = o
    coverage[name] = d
    print(f"  composed {name}")

# ─── 8. 残留自检（对照 plate）────────────────────────────
disc_chk = dist_ring < (DISC_R - 2)
for name in POSES:
    body_u8 = np.clip(variants[name], 0, 255).astype(np.uint8)
    diff = np.abs(body_u8.astype(np.int16) - plate.astype(np.int16)).max(axis=2)
    cov = cv2.dilate((coverage[name] > 0.25).astype(np.uint8) * 255,
                     np.ones((7, 7), np.uint8)) > 0
    residue = (diff > 22) & ~cov & disc_chk
    swept = (bat_all > 0) & ~cov   # 旧蝙蝠位必须露出 plate
    swept_bad = (diff[swept] > 22).sum() if swept.any() else 0
    print(f"[check {name}] stray-diff px={int(residue.sum())} "
          f"swept-not-clean={int(swept_bad)}")
    vis = body_u8.copy()
    vis[residue] = (vis[residue] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
    vis[swept & (diff > 22)] = (vis[swept & (diff > 22)] * 0.4
                                + np.array([0, 220, 0]) * 0.6).astype(np.uint8)
    Image.fromarray(vis[300:1150, 430:1130]).save(os.path.join(CHK, f"check_{name}.png"))
    nb, lb, sb, _ = cv2.connectedComponentsWithStats(residue.astype(np.uint8), 8)
    order = np.argsort(-sb[1:, 4])[:6]
    for oi in order:
        i = 1 + int(oi)
        x, y, w, h, a = sb[i]
        if a < 30:
            continue
        print(f"   stray CC bbox=({x},{y} {w}x{h}) area={a}")

# ─── 9. 文字层（v304 定稿：顶弧 + EST/1862 + 大字 + 副字 + 三角）─
def render_text_layer(top_arc, brand, sub):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    drw = ImageDraw.Draw(layer)
    arc_r = 430.0
    ccx, ccy = 776, 746

    def _draw_arc_text(txt, size_pt):
        font = ImageFont.truetype(FONT_PATH, size_pt)
        bbox = drw.textbbox((0, 0), txt, font=font)
        text_w = bbox[2] - bbox[0]
        phi = min(math.pi * 0.30, text_w / (2.0 * arc_r))
        a_start = -math.pi / 2 - phi
        a_end = -math.pi / 2 + phi
        char_widths = [drw.textbbox((0, 0), c, font=font)[2] for c in txt]
        a = a_start
        for i, ch in enumerate(txt):
            w_c = char_widths[i]
            mid_a = a + (w_c / 2.0) / arc_r
            px = ccx + arc_r * math.cos(mid_a)
            py = ccy + arc_r * math.sin(mid_a)
            deg = math.degrees(mid_a) + 90
            ch_im = Image.new("RGBA", (w_c + 20, size_pt * 2), (0, 0, 0, 0))
            ImageDraw.Draw(ch_im).text((10, 0), ch, font=font, fill=(0, 0, 0, 255))
            ch_rot = ch_im.rotate(-deg, resample=Image.BICUBIC, expand=True)
            layer.paste(ch_rot, (int(px - ch_rot.width / 2), int(py - ch_rot.height / 2)), ch_rot)
            a += w_c / arc_r

    _draw_arc_text(top_arc, 78)
    est_font = ImageFont.truetype(FONT_PATH, 70)
    drw.text((432, 760), "EST.", font=est_font, fill=(0, 0, 0, 255))
    drw.text((1120, 760), "1862", font=est_font, fill=(0, 0, 0, 255))
    brand_font = ImageFont.truetype(FONT_PATH, 300)
    bb = drw.textbbox((0, 0), brand, font=brand_font)
    drw.text(((W - (bb[2] - bb[0])) // 2 - bb[0], 1060 - bb[1]), brand,
             font=brand_font, fill=(0, 0, 0, 255))
    sub_font = ImageFont.truetype(FONT_PATH, 150)
    sb = drw.textbbox((0, 0), sub, font=sub_font)
    drw.text(((W - (sb[2] - sb[0])) // 2 - sb[0], 1320 - sb[1]), sub,
             font=sub_font, fill=(0, 0, 0, 255))
    tri = [(W // 2 - 40, 1500 - 28), (W // 2 + 40, 1500 - 28), (W // 2, 1500 + 28)]
    drw.polygon(tri, fill=(0, 0, 0, 255))
    return layer

variants_text = {
    "up":     ("NOCTAVEN", "NOCTAVEN", "DISTILLERY"),
    "spread": ("DUSKBAT", "DUSKBAT", "RESERVE"),
    "fold":   ("MOONBAT", "MOONBAT", "NOCTURNE"),
}

final_imgs = {}
for name, base_arr in variants.items():
    base_u8 = np.clip(base_arr, 0, 255).astype(np.uint8)
    base_pil = Image.fromarray(base_u8).convert("RGBA")
    top, brand, sub = variants_text[name]
    composed = Image.alpha_composite(base_pil, render_text_layer(top, brand, sub)).convert("RGB")
    out_p = os.path.join(OUT_DIR, f"v306_{name}_final.png")
    composed.save(out_p)
    final_imgs[name] = out_p
    Image.fromarray(base_u8).save(os.path.join(OUT_DIR, f"v306_{name}_body.png"))
    print(f"  saved {out_p}")

# ─── 10. 徽章特写 + 三姿态对比 ───────────────────────────
BX0, BY0, BX1, BY1 = CX - 460, 280, CX + 460, 1120
ZOOM = 1.6
zw, zh = int((BX1 - BX0) * ZOOM), int((BY1 - BY0) * ZOOM)
for name in POSES:
    bimg = Image.open(os.path.join(OUT_DIR, f"v306_{name}_body.png"))
    bimg.crop((BX0, BY0, BX1, BY1)).resize((zw, zh), Image.LANCZOS).save(
        os.path.join(CHK, f"v306_{name}_badge_zoom.png"))
comp = Image.new("RGB", (zw * 3 + 40, zh), (30, 30, 30))
for i, name in enumerate(["up", "spread", "fold"]):
    z = Image.open(os.path.join(CHK, f"v306_{name}_badge_zoom.png"))
    comp.paste(z, (i * (zw + 20), 0))
comp.save(os.path.join(CHK, "v306_pose_compare.png"))
print("  saved badge zooms + pose compare")

grid = Image.new("RGB", (W * 2 + 30, H * 2 + 30), (200, 200, 200))
for name, (px_, py_) in {"up": (0, 0), "spread": (W + 30, 0),
                         "fold": (0, H + 30)}.items():
    grid.paste(Image.open(final_imgs[name]), (px_, py_))
panel4 = Image.new("RGB", (W, H), (50, 30, 60))
d4 = ImageDraw.Draw(panel4)
fn = ImageFont.truetype(FONT_PATH, 56)
d4.text((W // 2 - 330, H // 2 - 90), "v306 badge rebuild", font=fn, fill=(220, 200, 230))
fn2 = ImageFont.truetype(FONT_PATH, 38)
d4.text((W // 2 - 340, H // 2 + 10), "plate + regenerated bat per pose", font=fn2, fill=(180, 160, 190))
d4.text((W // 2 - 320, H // 2 + 70), "membrane fill / leg stretch / ring kept", font=fn2, fill=(220, 200, 230))
grid.paste(panel4, (W + 30, H + 30))
grid_path = os.path.join(OUT_DIR, "_grid_v306.png")
grid.save(grid_path)
print(f"\nGrid: {grid_path}")
print("All outputs ready.")
