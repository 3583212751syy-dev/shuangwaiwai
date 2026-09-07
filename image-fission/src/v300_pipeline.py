"""
v300 — TPS（薄板样条）控制点形变取代 falloff 位移场：
翼尖/翼峰/下缘控制点精确到位（falloff 会衰减位移、抹平翼峰），大位移无撕裂。
工具来源：github.com/cheind/py-thin-plate-spline（pip 已装 venv）。

根因（破 bug）：
  arc_text.draw_arc_text 内部把 img 转成 RGBA 再返回；
  v289/v291 的 burn_text 里 `draw = ImageDraw.Draw(img)` 在 ARC 之前绑定到 RGB img，
  ARC 返回后 img 已是 RGBA，但 draw 仍是旧 RGB 的 draw；
  后续 BIG/SUB/SIDE 的 draw.text 全部画到已被丢弃的 RGB buffer，主副字全丢。

修复：ARC 返回后重新 `draw = ImageDraw.Draw(img)`，并强制 img.convert("RGB") 避免模式混乱。

附加：把字再次过栅格化前用 try/except 包好，确保任何 PIL 异常都不会让主字丢失。
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path("E:/Desktop/双接口/image-fission")
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
JOB = PROJECT / "jobs" / "v300"
JOB.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT / "src"))
import arc_text

ORIG = "6978fabda2cc99629fa9e81f802762d3.jpg"
FONT = str(PROJECT / "ComfyUI" / "models" / "fonts" / "AbrilFatface-Regular.ttf")

TARGET_W, TARGET_H = 1024, 1280
INK = (26, 10, 31)
BG_PURPLE = (183, 127, 171)
RING_DEEP = (90, 20, 120)

cx, cy, r_badge = 776, 746, 300
ring_outer = 421

SUBJECT_VARIANTS = [
    ("up",     "NOCTAVEN", "DISTILLERY", "SHADOW OF THE WING",      300001),
    ("spread", "DUSKBAT",  "RESERVE",    "WINGS OF TWILIGHT",       300101),
    ("fold",   "MOONBAT",  "NOCTURNE",   "GUARDIAN OF THE DARK",    300201),
]


def _ring_sector_mask(h, w, c_x, c_y, r_in, r_out, a0, a1):
    ys, xs = np.ogrid[:h, :w]
    d2 = (xs - c_x) ** 2 + (ys - c_y) ** 2
    in_ring = (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
    sweep = (a1 - a0) % 360
    if sweep == 0 and (a1 - a0) != 0:  # 0,360 -> 360° 而非 0°
        sweep = 360
    if sweep >= 359.99:
        return in_ring  # 完整圆盘
    ang = (np.degrees(np.arctan2(ys - c_y, xs - c_x)) - a0) % 360
    in_angle = ang <= sweep
    return in_ring & in_angle


def _detect_bat_mask(arr):
    """蝙蝠 mask：cx,cy 周围 r_badge+30 + ring 区域（覆盖翼尖完整保留）。"""
    h, w = arr.shape[:2]
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    maxc = np.maximum(np.maximum(R, G), B)
    stdc = np.std(arr.astype(np.float32), axis=2)
    dark = (maxc < 80) & (stdc < 25)
    mask = dark.astype(np.uint8) * 255
    # 关键：限制圆形必须覆盖到 ring_outer 才能让翼尖被纳入 mask
    badge = np.zeros((h, w), np.uint8)
    cv2.circle(badge, (cx, cy), ring_outer + 30, 255, -1)
    mask = cv2.bitwise_and(mask, badge)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.dilate(mask, np.ones((8, 8), np.uint8), iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        sizes = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + int(np.argmax(sizes))
        mask = (labels == keep).astype(np.uint8) * 255
    return mask


def _only_keep_dark_text_in_arc_skip(arr, mask_region):
    """把原图黑色像素位置限制在 mask_region 之外保留非黑（环带本身），字像素填紫底"""
    arr_copy = arr.copy()
    arr_copy[mask_region] = BG_PURPLE  # 整段环带 → 紫色底
    return arr_copy


def make_clean_source():
    """v291 改进版：拱门环带保留原图（非黑像素），字像素填紫底。"""
    orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
    arr = np.array(orig).astype(np.float32)
    h, w = arr.shape[:2]

    corners = np.stack([arr[5, 5], arr[5, w-5], arr[h-5, 5], arr[h-5, w-5]], axis=0)
    bg_color = np.median(corners, axis=0).astype(np.float32)
    print(f"[bg] {tuple(bg_color.astype(int))}")

    clean = np.full_like(arr, bg_color)

    # 0) 先在原图上找蝙蝠精确 mask（用于环带清除时只保留蝙蝠像素）
    bat_mask_pre = _detect_bat_mask(arr)
    print(f"[protect] bat_mask_pre sum={int(bat_mask_pre.sum()/255)}px")

    # 1) 拱门环带：只保留蝙蝠剪影像素，其余全部填成紫底
    arc_band = _ring_sector_mask(h, w, cx, cy, ring_outer - 95, ring_outer + 65, 185, 355)
    inner = _ring_sector_mask(h, w, cx, cy, 0, ring_outer - 30, 0, 360)
    arc_band_keep = arc_band & ~inner
    # 环带内只有蝙蝠像素保留原图，其他（字/描边/装饰线）全清
    in_band_not_bat = arc_band_keep & (bat_mask_pre == 0)
    clean[in_band_not_bat] = bg_color
    print(f"[arc] arc_band={int(arc_band.sum())} inner={int(inner.sum())} arc_band_keep={int(arc_band_keep.sum())} bat_mask={int((bat_mask_pre>0).sum())} inter={int((arc_band_keep & (bat_mask_pre>0)).sum())} cleared={int(in_band_not_bat.sum())}")
    # debug vis
    debug_img = np.zeros((h, w, 3), dtype=np.uint8)
    debug_img[arc_band_keep] = (255, 0, 0)  # 红
    debug_img[bat_mask_pre > 127] = (0, 255, 0)  # 绿
    debug_img[arc_band_keep & (bat_mask_pre > 127)] = (255, 255, 0)  # 黄=交集
    Image.fromarray(debug_img).save(r"E:/Desktop/双接口/image-fission/jobs/v300/_debug_mask.png")

    # 2) 中心徽章保留 (不消黑字，里面只有蝙蝠剪影)
    inner_badge = _ring_sector_mask(h, w, cx, cy, 0, ring_outer - 95, 0, 360)
    clean[inner_badge] = arr[inner_badge]
    print(f"[inner] 中心徽章 {int(inner_badge.sum())}px (跳过黑字清除，保护蝙蝠)")

    # 3) 蝙蝠剪影保留（从原图 arr 提取，避免 clean 上被改动）
    bat_pixels = arr.copy()
    bat_pixels[bat_mask_pre == 0] = 0
    clean[bat_mask_pre > 127] = bat_pixels[bat_mask_pre > 127]
    print(f"[bat] {int((bat_mask_pre>127).sum())}px")

    # 4) 底部矩形清黑字（maxc<70 纯黑字；sum<200 会误杀深紫圆暗部——圆盘下半被抹的元凶）
    rects = [
        (1080, 1200, 50,   1500),
        (1200, 1400, 200,  1350),
        (700,  920,  240,  700),
        (700,  920,  870,  1320),
    ]
    for y0, y1, x0, x1 in rects:
        region = np.zeros((h, w), dtype=bool)
        region[y0:y1, x0:x1] = True
        black_in_rect = (arr.max(axis=2) < 70) & region
        clean[black_in_rect] = bg_color
        print(f"[rect] {y0}-{y1},{x0}-{x1} 黑字清除 {int(black_in_rect.sum())}px")

    # 5) 盘外 y>940 非蝙蝠 → 填外圈紫底；盘内字残留只在固定小矩形内清（maxc<90 含抗锯齿）
    #    严禁盘内大面积整块清——会抹掉深紫圆下半，下翼膜弧口失去对比（"翅膀被剪"的真相）
    ys_grid, xs_grid = np.mgrid[0:h, 0:w]
    lower_zone = ys_grid > 940
    not_bat = bat_mask_pre == 0
    inner_badge_bool = inner_badge > 0 if not isinstance(inner_badge, bool) else inner_badge
    outside_disk = ~inner_badge_bool
    clear_outside = lower_zone & not_bat & outside_disk
    clean[clear_outside] = bg_color
    text_rect = np.zeros((h, w), dtype=bool)
    text_rect[935:1085, 560:990] = True
    # 字身（maxc<90）+ BACARDÍ 浅色描边（min>195）都清；深紫圆（min 25-60/max 80-140）保留
    clear_inside = text_rect & not_bat & inner_badge_bool & ((arr.max(axis=2) < 90) | (arr.min(axis=2) > 195))
    clean[clear_inside] = bg_color  # 占位，稍后 inpaint
    print(f"[lower] 盘外清 {int(clear_outside.sum())}px 盘内字残留待修复 {int(clear_inside.sum())}px")

    # 6) 圆盘内非蝙蝠纯黑杂点（maxc<70）→ 收集为待修复（min<80 会误判深紫圆暗部）
    stray = ((arr.max(axis=2) < 70) & inner_badge_bool & not_bat).astype(np.uint8)
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(stray, connectivity=8)
    disk_fix = clear_inside.copy()
    removed = 0
    for i in range(1, n_lab):
        if stats[i, cv2.CC_STAT_AREA] < 600:
            disk_fix |= (labels == i)
            removed += 1
    print(f"[stray] 收集小杂点 {removed} 个")

    # 7) 盘内待修复区域用邻域修复（小面积 NS inpaint 安全；v290 失败是大面积弧带的教训）
    if disk_fix.any():
        inp_src = np.clip(clean, 0, 255).astype(np.uint8)
        inp_mask = disk_fix.astype(np.uint8) * 255
        fixed = cv2.inpaint(inp_src, inp_mask, 5, cv2.INPAINT_NS)
        clean = fixed.astype(np.float32)
        print(f"[fix] 盘内邻域修复 {int(disk_fix.sum())}px")

    return Image.fromarray(np.clip(clean, 0, 255).astype(np.uint8))


def _warp_bat_by_wings(bat_pixels, mask, left_delta, right_delta, shoulder_l, shoulder_r, falloff=110):
    h, w = mask.shape
    ys, xs = np.mgrid[:h, :w].astype(np.float32)
    dl = np.sqrt((xs - shoulder_l[0]) ** 2 + (ys - shoulder_l[1]) ** 2)
    dr = np.sqrt((xs - shoulder_r[0]) ** 2 + (ys - shoulder_r[1]) ** 2)
    wl = np.clip((dl - 25) / falloff, 0, 1) * (xs < cx).astype(np.float32)
    wr = np.clip((dr - 25) / falloff, 0, 1) * (xs > cx).astype(np.float32)
    dx_map = wl * left_delta[0] + wr * right_delta[0]
    dy_map = wl * left_delta[1] + wr * right_delta[1]
    x_map = xs - dx_map
    y_map = ys - dy_map
    warped_mask = cv2.remap(mask, x_map.astype(np.float32), y_map.astype(np.float32),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_pixels = cv2.remap(bat_pixels, x_map.astype(np.float32), y_map.astype(np.float32),
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    badge = np.zeros((h, w), np.uint8)
    cv2.circle(badge, (cx, cy), r_badge, 255, -1)
    warped_mask = cv2.bitwise_and(warped_mask, badge)
    warped_pixels = warped_pixels * (badge[:, :, None] > 0)
    _, warped_mask = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)
    warped_mask = cv2.GaussianBlur(warped_mask, (3, 3), 0)
    # 底部安全线：蝙蝠不得压到主字区（主字顶部约 h*0.49），线设在 h*0.47
    safe_line = int(h * 0.47)
    warped_mask[safe_line:, :] = 0
    alpha = warped_mask.astype(np.float32) / 255.0
    return warped_pixels, alpha


def _find_wing_landmarks(mask):
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None
    upper = ys < cy
    if upper.sum() < 20:
        upper = ys < cy + 80
    left_idx = xs[upper].argmin()
    right_idx = xs[upper].argmax()
    left_tip = (int(xs[upper][left_idx]), int(ys[upper][left_idx]))
    right_tip = (int(xs[upper][right_idx]), int(ys[upper][right_idx]))
    shoulder_y = int((left_tip[1] + cy) * 0.55)
    row = mask[shoulder_y, :]
    row_idx = np.where(row > 127)[0]
    if len(row_idx) >= 2:
        left_shoulder = (int(row_idx[row_idx < cx].max()) if np.any(row_idx < cx) else cx - 80, shoulder_y)
        right_shoulder = (int(row_idx[row_idx > cx].min()) if np.any(row_idx > cx) else cx + 80, shoulder_y)
    else:
        left_shoulder = (cx - 90, shoulder_y)
        right_shoulder = (cx + 90, shoulder_y)
    return {"left_tip": left_tip, "right_tip": right_tip,
            "left_shoulder": left_shoulder, "right_shoulder": right_shoulder}


def _extract_bat_keypoints(bat_m, lm):
    """从蝙蝠 mask 自动提取 TPS 控制点：翼尖/上缘尖峰/下缘/尾/头/肩。"""
    ys, xs = np.where(bat_m > 127)
    h_img, w_img = bat_m.shape
    pts = {}

    # 每侧翅膀：翼展范围（该侧 x 极值），上缘尖峰 = 该侧 y 最小像素
    for side, sel in (("L", xs < cx), ("R", xs > cx)):
        sxs, sys = xs[sel], ys[sel]
        top_i = int(sys.argmin())
        pts[f"{side}_peak"] = (float(sxs[top_i]), float(sys[top_i]))
        # 下缘：该侧 y 最大但排除尾巴（|x-cx|<90 视为身体/尾）
        wing_only = np.abs(sxs - cx) >= 90
        if wing_only.sum() > 20:
            bxs, bys = sxs[wing_only], sys[wing_only]
            bot_i = int(bys.argmax())
            pts[f"{side}_bottom"] = (float(bxs[bot_i]), float(bys[bot_i]))
        else:
            pts[f"{side}_bottom"] = (float(sxs[int(len(sxs)*0.8)]), float(sys.max()))
    pts["L_tip"] = (float(lm["left_tip"][0]), float(lm["left_tip"][1]))
    pts["R_tip"] = (float(lm["right_tip"][0]), float(lm["right_tip"][1]))
    # 头顶 / 尾尖 / 双肩
    head_band = np.abs(xs - cx) < 100
    head_i = int(ys[head_band].argmin())
    pts["head"] = (float(xs[head_band][head_i]), float(ys[head_band][head_i]))
    tail_i = int(ys.argmax())
    pts["tail"] = (float(xs[tail_i]), float(ys[tail_i]))
    pts["L_sh"] = (float(lm["left_shoulder"][0]), float(lm["left_shoulder"][1]))
    pts["R_sh"] = (float(lm["right_shoulder"][0]), float(lm["right_shoulder"][1]))
    return pts


def _rot_warp(layer_in, wing_soft, wing_bin, body_block, body_keep, bg_layer, theta, rot_c, h, w, keep_new=None):
    """刚体旋转形变（骨骼式）：纯翅膀半翼绕 rot_c 整体旋转，三层混合：
      out = src_warp*w_new + bg_layer*w_old + layer_in*(1-w_new-w_old)
      wing_soft = 源空间贴回权重（翼根距身体 0~22px 渐变 0→1）。
        ★ 翼根渐变必须编码在源空间、随内容一起旋转（v300 十三轮定稿）。
          输出空间的邻域渐变（prox 系）会罩住新翅膀贴入区（恰在身体邻域），
          把新位置 w_new 清零 → 新膜消失只剩翼骨 = 纱状膜（十一~十二轮事故）。
      w_old（旧位置）= bg_layer 平滑背景。
        ★ 旧位置严禁用 R^{-1} 采样 reveal（v300 七轮教训）：盘内翅膀紧邻浅紫锯齿
          底座纹理，旋转采样把锯齿"拖拽复制"到旧位置 = 枝状破碎物。
      keep_new = 前序 pass 的 w_new 区（软权重）。
        ★ 十六轮教训（up 姿态专属事故）：高扬时翼内段越过中线，撞进对侧翼的
          旧位置区——pass2 的 w_old 若不扣除该区，会把 pass1 刚贴上的新翼
          用 bg_layer 整块盖掉（左右翼连环互擦）。keep_new 区强制 rest=1
          → 原样保留 layer_in（= pass1 成果）。
      body_block/body_keep 只保护确定部件（脸/耳；躯干翼可自然遮挡——
      翅膀在身体前面是自然遮挡顺序，十七轮起躯干/尾条不再进 body_block，
      只留头椭圆，防翼内段越线时被躯干条挖出断口）。"""
    # float32 固定：np.cos(python float) 返回 float64 会污染网格，OpenCV 5 remap 拒收
    cos_t = np.float32(np.cos(theta))
    sin_t = np.float32(np.sin(theta))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xs - rot_c[0]
    dy = ys - rot_c[1]
    # backward 采样：输出点 p 处取源点 R^{-1}(p)
    rx = rot_c[0] + dx * cos_t + dy * sin_t
    ry = rot_c[1] - dx * sin_t + dy * cos_t
    # 新位置 = 源权重随内容旋转（软权重 + 窄羽化贴边）
    rot_soft = cv2.remap(wing_soft, rx, ry, cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    w_new = cv2.GaussianBlur(rot_soft, (5, 5), 1.5).astype(np.float32)
    w_new = w_new * (np.float32(1.0) - body_block)
    # 旧位置 = 原翅膀减去新位置重叠与前序成果（重叠区分别由 w_new/layer_in 全权负责）
    w_old = cv2.GaussianBlur(wing_bin, (3, 3), 0.8).astype(np.float32) / 255.0
    w_old = np.clip(w_old - w_new, 0, np.float32(1.0))
    if keep_new is not None:
        w_old = np.clip(w_old - keep_new, 0, np.float32(1.0))
    src_warp = cv2.remap(layer_in, rx, ry, cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    rest = np.float32(1.0) - w_new - w_old
    if keep_new is not None:
        # keep 区直接采用 layer_in（pass1 输出原样），压掉 w_new 计算残差
        rest = np.maximum(rest, keep_new)
        w_new = w_new * (np.float32(1.0) - keep_new)
        w_old = w_old * (np.float32(1.0) - keep_new)
    out = (src_warp * w_new[..., None] + bg_layer * w_old[..., None]
           + layer_in * rest[..., None])
    # 输出级身体兜底：确定部件强制原像素（任何上游 mask 缺陷都不该能擦动头/尾）
    out = out * (np.float32(1.0) - body_keep)[..., None] + layer_in * body_keep[..., None]
    return out, w_new


def make_variants(src_img):
    """v300 骨骼式刚体旋转：三姿态 = 不同的旋转中心+角度组合。
    刚体旋转保证翅膀内部零失真；边界窄羽化过渡。"""
    arr = np.array(src_img).astype(np.float32)
    h, w = arr.shape[:2]

    # 1) 主体层 = 内盘圆 r=360，边缘轻羽化
    layer_mask = np.zeros((h, w), np.uint8)
    cv2.circle(layer_mask, (cx, cy), 360, 255, -1)
    layer_alpha = cv2.GaussianBlur(layer_mask, (5, 5), 1.2).astype(np.float32) / 255.0
    layer_pixels = arr.copy()

    # 2) 蝙蝠暗色检测（仅用于找 landmark，不做抠图）
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    maxc = np.maximum(np.maximum(R, G), B)
    dark = (maxc < 95).astype(np.uint8) * 255
    badge_lim = np.zeros((h, w), np.uint8)
    cv2.circle(badge_lim, (cx, cy), ring_outer + 30, 255, -1)
    dark = cv2.bitwise_and(dark, badge_lim)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if n > 1:
        sizes = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + int(np.argmax(sizes))
        bat_m = (labels == keep).astype(np.uint8) * 255
    else:
        bat_m = dark
    lm = _find_wing_landmarks(bat_m)
    if lm is None:
        raise RuntimeError("未找到蝙蝠 landmark")
    shoulder_y = lm["left_shoulder"][1]
    lm["left_shoulder"] = (cx - 120, shoulder_y)
    lm["right_shoulder"] = (cx + 120, shoulder_y)

    kp = _extract_bat_keypoints(bat_m, lm)
    print(f"[v300] keypoints: {kp}")

    # 实心化（v300 十轮定稿）：CLOSE 21 桥接开口的浅紫骨纹条（宽 10~20px）+
    # 只填真洞。骨纹不是封闭洞（与外部背景连通），RETR_CCOMP 直接填不到——
    # 不桥接的话深色膜转走了、骨纹留在原地半透明混 bg_layer = "纱状膜"。
    # CLOSE 核不能大：翅膀弧口宽 40px+，CLOSE 21 不会误填弧口（大核外轮廓
    # 填充把弧口全填掉 = 六轮事故，废弃路线）。
    solid = cv2.morphologyEx(bat_m, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    solid_m = solid.copy()
    cnts, hier = cv2.findContours(solid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is not None:
        for i, h4 in enumerate(hier[0]):
            if h4[3] != -1:  # 有父轮廓 = 是内部洞
                cv2.drawContours(solid_m, cnts, i, 255, -1)

    # —— 身体/翅膀分离 + 源空间翼根渐变（v300 十三轮定稿）——
    # body_core = 中线 ±48 内的暗色区（躯干两侧缘附近），作剪切线：
    # wing_m 只含 body_core 之外的翅膀；32° 剪切藏在身体边缘褶皱暗色区。
    band_b = np.zeros((h, w), np.uint8)
    band_b[:, cx - 48:cx + 48] = 255
    body_core = cv2.bitwise_and(bat_m, band_b)
    wing_m = solid_m.copy()
    wing_m[body_core > 0] = 0
    # 源空间翼根渐变：距身体 0~22px 贴回权重 0→1，随翅膀内容一起旋转。
    # ★ 翼根渐变必须编码在源空间（十三轮定稿）。输出空间的邻域渐变（prox 系）
    #   会罩住新翅膀贴入区（恰在身体邻域），把新位置 w_new 清零 → 新膜消失
    #   只剩翼骨 = 纱状膜（十一~十二轮事故）。
    dist = cv2.distanceTransform(cv2.bitwise_not(body_core), cv2.DIST_L2, 3)
    wing_soft = (wing_m.astype(np.float32) / 255.0) * np.clip(dist / 22.0, 0, 1).astype(np.float32)

    # 确定不可被翅膀遮挡的部件：只有脸/耳（蝙蝠身份核心）。
    # ★ 头部椭圆严禁开大（十四轮教训）：(42,46) 把头部两侧膜面划进保护，
    #   新翅膀内段扫过该区贴回全被清零 = 纱状区。只罩脸，耳旁膜允许按
    #   "翅膀在身体前"的遮挡逻辑自然覆盖。
    # ★ 十七轮定稿：躯干/尾条撤出 protect——高扬姿态翼内段自然越过躯干
    #   （翅膀在身体前面），躯干条会把越线翼段挖出断口（up 事故主因之二）。
    #   身体像素安全由 w_old 只在翼 mask 内 reveal 保证（身体从不在翼区）。
    hx, hy = kp["head"]
    protect = np.zeros((h, w), np.uint8)
    cv2.ellipse(protect, (int(hx) + 6, int(hy) + 6), (24, 28), 0, 0, 360, 255, -1)
    protect = cv2.bitwise_and(protect, bat_m)
    body_block = cv2.GaussianBlur(protect, (5, 5), 1.5).astype(np.float32) / 255.0
    body_keep = cv2.GaussianBlur(
        cv2.dilate(protect, np.ones((5, 5), np.uint8)), (7, 7), 2
    ).astype(np.float32) / 255.0

    # ★ 膜面修复层已废弃（v300 十五轮回退）：CLOSE 桥接的浅紫区（骨纹开放条+
    #   膜外缝）是一大片连通网络，TELEA 修复色=深膜与浅紫背景的混合=纱色，越修越纱。
    #   外缝浅紫与骨纹浅紫颜色本就接近，随膜旋转后出现在新膜前缘 = 膜面纹理的一部分，
    #   直接用原像素采样（wing_src 方案废弃，src_warp 采样 arr/wing_src 均验证）。

    # bg_layer：翅膀挖除后的平滑背景层（旧位置 reveal 用）。
    # 挖洞 = 蝙蝠膨胀 4px 但避开身体（身体保持原样）。
    # ★ v300 十六轮教训：1/4 尺度 TELEA 填"整只蝙蝠"大洞会把紫冠深紫/盘缘高光/
    #   底座锯齿往洞里拖成龟裂白纹（翼后面是紫冠图案而非平滑渐变，TELEA 不识别
    #   结构）。升到 1/2 尺度 + inpaint 半径 9，拖纹距离减半，斑驳显著收敛；
    #   旧位置 reveal 的 w_old 本身只是"新翼未覆盖的月牙差集"，非全翼暴露。
    dig = cv2.dilate(bat_m, np.ones((9, 9), np.uint8), iterations=1)
    dig[cv2.dilate(body_core, np.ones((3, 3), np.uint8)) > 0] = 0
    scale = 2
    small = cv2.resize(arr, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    dig_s = cv2.resize(dig, (w // scale, h // scale), interpolation=cv2.INTER_NEAREST)
    small[dig_s > 0] = 0
    # TELEA 仅支持 8-bit 3 通道，float32 三通道拒收
    small_u8 = np.clip(small, 0, 255).astype(np.uint8)
    small_u8 = cv2.inpaint(small_u8, dig_s, 9, cv2.INPAINT_TELEA)
    bg_up = cv2.resize(small_u8, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    # 上采样后再做一次 5x5 轻高斯，压掉块状插值纹理
    bg_up = cv2.GaussianBlur(bg_up, (5, 5), 1.2)
    bg_layer = arr.copy()
    bg_layer[dig > 0] = bg_up[dig > 0]

    # 半翼拆分（纯翅膀）：中线分割
    xs_col = np.tile(np.arange(w, dtype=np.int32), (h, 1))
    left_m = wing_m.copy()
    left_m[xs_col >= cx] = 0
    right_m = wing_m.copy()
    right_m[xs_col < cx] = 0
    left_soft = wing_soft.copy()
    left_soft[xs_col >= cx] = 0
    right_soft = wing_soft.copy()
    right_soft[xs_col < cx] = 0

    # 旋转中心 = 翼膜竖向中点（肩点下方 60px），不是肩点本身。
    # v300 八轮教训：绕肩点旋转时，肩点上方的翼尖上扬、肩点下方的下翼角反向
    # 下甩 30px+，视觉=身体两侧长出枝状物。中心下移后下翼角贴近转轴几乎不动，
    # 翼尖照常大幅旋转，全翼一致动作（已验算：up32° 翼尖位移 114px/下翼角 17px）。
    shL = (cx - 120, shoulder_y + 60)
    shR = (cx + 120, shoulder_y + 60)

    def two_wing(thetaL, thetaR):
        """左翼绕左轴旋转 thetaL，右翼绕右轴旋转 thetaR（串联：pass2 以 pass1
        输出为底层；★ pass2 携带 keep_new=pass1 的 w_new 区，防止右翼旧位置
        的 bg_layer reveal 把左翼新贴成果盖掉——高扬姿态翼内段越中线必撞）。"""
        wpL, wnL = _rot_warp(arr, left_soft, left_m, body_block, body_keep, bg_layer, thetaL, shL, h, w)
        wp, _ = _rot_warp(wpL, right_soft, right_m, body_block, body_keep, bg_layer, thetaR, shR, h, w,
                          keep_new=cv2.GaussianBlur(wnL, (7, 7), 2))
        return wp, layer_alpha

    # theta>0 = 左翼上扬 / 右翼配 -theta 上扬（镜像对：L+θ 与 R-θ 同为扬）。
    # v300 十七轮定稿三档姿态：
    #   up     +18° 双翼轻扬 V 形（翼尖位移 ~63px）。★ 32° 翼峰距头仅 32px 怼脸
    #          （十七轮验算），高扬角上限 ~20°，这是本图几何的硬约束。
    #   spread -12° 微垂平展（翼尖下移 ~30px）
    #   fold   -33° 大幅垂翼（翼尖下移 ~96px；-45 会把翼峰甩出盘 30px 硬切，
    #          -33 翼峰距盘心 364 在羽化容差内，切翅红线不可碰）
    variants = {
        "up":     two_wing(math.radians(18), math.radians(-18)),   # 双翼轻扬 V 形
        "spread": two_wing(math.radians(-12), math.radians(12)),   # 微垂平展
        "fold":   two_wing(math.radians(-33), math.radians(33)),   # 大幅垂翼
    }
    return variants


def _calibrate_font(text, font_path, start_size, max_w):
    lo, hi = 8, start_size
    best = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(font_path, mid)
        if font.getlength(text) <= max_w:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def burn_text(img, arc, big, sub, color=INK):
    """v300 关键修复：每步后重新绑定 draw + 强制 RGB。"""
    img = img.convert("RGB")
    w, h = img.width, img.height

    # ARC：draw_arc_text 内部把图转 RGBA，返回 RGBA 图
    fs_arc = int(w * 0.045)
    radius = int(w * 0.46)
    cy_arc = int(h * 0.13) + radius
    while fs_arc > 18:
        arc_len = arc_text.fit_arc_text_width(arc, FONT, fs_arc, radius, char_spacing_px=3)
        total_deg = math.degrees(arc_len / radius)
        if total_deg <= 170:
            break
        fs_arc = int(fs_arc * 0.92)
    start = 270 - total_deg / 2
    end = 270 + total_deg / 2
    img = arc_text.draw_arc_text(img, arc, FONT, fs_arc, color,
                                 center=(w // 2, cy_arc), radius=radius,
                                 start_angle_deg=start, end_angle_deg=end,
                                 char_spacing_px=3, flip_180=False)
    # 关键 ①：arc_text 返回 RGBA，立刻转 RGB 并重绑 draw
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    print(f"[burn] arc ok; img mode {img.mode}")

    fs_big = _calibrate_font(big, FONT, int(h * 0.145), max_w=int(w * 0.82))
    font = ImageFont.truetype(FONT, fs_big)
    print(f"[burn] BIG fs={fs_big} len={font.getlength(big):.1f}")
    draw.text((w // 2, int(h * 0.555)), big, font=font, fill=color, anchor="mm")

    fs_sub = _calibrate_font(sub, FONT, int(h * 0.075), max_w=int(w * 0.60))
    font = ImageFont.truetype(FONT, fs_sub)
    print(f"[burn] SUB fs={fs_sub} len={font.getlength(sub):.1f}")
    draw.text((w // 2, int(h * 0.655)), sub, font=font, fill=color, anchor="mm")

    f_side = ImageFont.truetype(FONT, int(h * 0.028))
    draw.text((int(w * 0.305), int(h * 0.415)), "EST.", font=f_side, fill=color, anchor="mm")
    draw.text((int(w * 0.695), int(h * 0.415)), "1862", font=f_side, fill=color, anchor="mm")

    # 底三角 ▼（深紫菱形/箭头）
    tri_cx = w // 2
    tri_cy = int(h * 0.83)
    tri_w = int(w * 0.038)
    tri_h = int(h * 0.025)
    draw.polygon([
        (tri_cx, tri_cy + tri_h),
        (tri_cx - tri_w, tri_cy - tri_h // 3),
        (tri_cx + tri_w, tri_cy - tri_h // 3),
    ], fill=color)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src_path = COMFY_INPUT / "v300_text_free_source.png"

    if not src_path.exists() or args.force:
        src = make_clean_source()
        src.save(src_path, quality=98)
        orig = Image.open(COMFY_INPUT / ORIG).convert("RGB")
        check = Image.new("RGB", (1552 * 2 + 24, 2000), "white")
        check.paste(orig, (0, 0))
        check.paste(src, (1552 + 24, 0))
        check.save(JOB / "_source_check.png", quality=95)
        print(f"[v300] clean source -> {src_path}")
    else:
        print(f"[v300] reuse clean source")

    text_free = Image.open(src_path).convert("RGB")
    bg_arr = np.array(text_free).astype(np.float32)
    variants = make_variants(text_free)

    files = []
    for tag, big, sub, arc, _ in SUBJECT_VARIANTS:
        warped_pixels, alpha = variants[tag]
        composed = bg_arr * (1 - alpha[:, :, None]) + warped_pixels * alpha[:, :, None]
        composed = Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8))
        composed.save(JOB / f"v300_{tag}_composed.png", quality=95)

        check = Image.new("RGB", (1552 * 2 + 24, 2000), "white")
        check.paste(text_free, (0, 0))
        check.paste(composed, (1552 + 24, 0))
        check.save(JOB / f"_compose_check_{tag}.png", quality=95)

        final = burn_text(composed, arc, big, sub)
        final_path = JOB / f"v300_{tag}_final.png"
        final.save(final_path, quality=95)
        files.append(final_path)
        print(f"[v300] {tag} final -> {final_path.name}")

    orig_small = Image.open(COMFY_INPUT / ORIG).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS)
    grid = Image.new("RGB", (TARGET_W * 2 + 24, TARGET_H * 2 + 24), "white")
    grid.paste(orig_small, (0, 0))
    grid.paste(Image.open(files[0]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, 0))
    grid.paste(Image.open(files[1]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (0, TARGET_H + 24))
    grid.paste(Image.open(files[2]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS), (TARGET_W + 24, TARGET_H + 24))
    gp = JOB / "_grid_v300.png"
    grid.save(gp, quality=92)
    print(f"[v300] grid -> {gp}")


if __name__ == "__main__":
    main()
