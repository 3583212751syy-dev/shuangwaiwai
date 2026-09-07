# -*- coding: utf-8 -*-
"""v300 诊断：dump 蝙蝠 mask 链 / bg_layer / up 变体的 w_new·w_old·src_warp，
另跑 theta=0 对照组，定位斑驳与水平硬边的来源层。"""
import sys, math
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, "src")
import v300_pipeline as V

SRC = r"ComfyUI\input\v300_text_free_source.png"

orig = Image.open(SRC).convert("RGB")
arr = np.array(orig).astype(np.float32)
h, w = arr.shape[:2]

# ---- 复刻 make_variants 的 mask 链（直接借用其内部逻辑,精简 dump）----
R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
maxc = np.maximum(np.maximum(R, G), B)
dark = (maxc < 95).astype(np.uint8) * 255
badge_lim = np.zeros((h, w), np.uint8)
cv2.circle(badge_lim, (V.cx, V.cy), V.ring_outer + 30, 255, -1)
dark = cv2.bitwise_and(dark, badge_lim)
dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
sizes = stats[1:, cv2.CC_STAT_AREA]
keep = 1 + int(np.argmax(sizes))
bat_m = (labels == keep).astype(np.uint8) * 255
print(f"[dbg] bat_m={int((bat_m>0).sum())}  cy={V.cy} cx={V.cx}")

solid = cv2.morphologyEx(bat_m, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
solid_m = solid.copy()
cnts, hier = cv2.findContours(solid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
if hier is not None:
    for i, h4 in enumerate(hier[0]):
        if h4[3] != -1:
            cv2.drawContours(solid_m, cnts, i, 255, -1)

band_b = np.zeros((h, w), np.uint8)
band_b[:, V.cx - 48:V.cx + 48] = 255
body_core = cv2.bitwise_and(bat_m, band_b)
wing_m = solid_m.copy()
wing_m[body_core > 0] = 0
dist = cv2.distanceTransform(cv2.bitwise_not(body_core), cv2.DIST_L2, 3)
wing_soft = (wing_m.astype(np.float32) / 255.0) * np.clip(dist / 22.0, 0, 1).astype(np.float32)

hx_v = None  # 头部
ys, xs = np.nonzero(bat_m)
hx = float(xs[np.abs(xs - V.cx) < 100][np.abs(ys[np.abs(xs - V.cx) < 100] - ys[np.abs(xs - V.cx) < 100].min()).argmin()])
hy = float(ys.min())
protect = np.zeros((h, w), np.uint8)
cv2.ellipse(protect, (int(hx) + 6, int(hy) + 6), (24, 28), 0, 0, 360, 255, -1)
protect[875:985, V.cx - 26:V.cx + 26] = 255
protect[560:875, V.cx - 22:V.cx + 22] = 255
protect = cv2.bitwise_and(protect, bat_m)
body_block = cv2.GaussianBlur(protect, (5, 5), 1.5).astype(np.float32) / 255.0
body_keep = cv2.GaussianBlur(cv2.dilate(protect, np.ones((5, 5), np.uint8)), (7, 7), 2).astype(np.float32) / 255.0

dig = cv2.dilate(bat_m, np.ones((9, 9), np.uint8), iterations=1)
dig[cv2.dilate(body_core, np.ones((3, 3), np.uint8)) > 0] = 0
small = cv2.resize(arr, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
dig_s = cv2.resize(dig, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
small[dig_s > 0] = 0
small_u8 = np.clip(small, 0, 255).astype(np.uint8)
small_u8 = cv2.inpaint(small_u8, dig_s, 15, cv2.INPAINT_TELEA)
bg_up = cv2.resize(small_u8, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
bg_layer = arr.copy()
bg_layer[dig > 0] = bg_up[dig > 0]

xs_col = np.tile(np.arange(w, dtype=np.int32), (h, 1))
left_m = wing_m.copy();  left_m[xs_col >= V.cx] = 0
right_m = wing_m.copy(); right_m[xs_col < V.cx] = 0
left_soft = wing_soft.copy();  left_soft[xs_col >= V.cx] = 0
right_soft = wing_soft.copy(); right_soft[xs_col < V.cx] = 0

shoulder_y = hy + 261  # 近似肩高(日志 L_sh y=790, hy≈529)
shL = (V.cx - 120, shoulder_y + 60)
shR = (V.cx + 120, shoulder_y + 60)

def vis_mask(m, name):
    if m.dtype != np.uint8:
        m8 = np.clip(m * 255, 0, 255).astype(np.uint8)
    else:
        m8 = m
    Image.fromarray(m8).save(f"jobs/v300/_chk/dbg_{name}.png")

vis_mask(bat_m, "bat_m"); vis_mask(solid_m, "solid_m"); vis_mask(wing_m, "wing_m")
vis_mask(body_core, "body_core"); vis_mask(protect, "protect")
vis_mask(bg_layer / 255.0, "bg_layer")

def dbg_rot(theta, tag):
    out = V._rot_warp(arr, left_soft, left_m, body_block, body_keep, bg_layer, theta, shL, h, w)
    cos_t = np.float32(np.cos(theta)); sin_t = np.float32(np.sin(theta))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - shL[0]; dy = yy - shL[1]
    rx = shL[0] + dx * cos_t + dy * sin_t
    ry = shL[1] - dx * sin_t + dy * cos_t
    rot_soft = cv2.remap(left_soft, rx, ry, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    w_new = cv2.GaussianBlur(rot_soft, (5, 5), 1.5).astype(np.float32) * (np.float32(1.0) - body_block)
    w_old = np.clip(cv2.GaussianBlur(left_m, (3, 3), 0.8).astype(np.float32) / 255.0 - w_new, 0, 1)
    src_warp = cv2.remap(arr, rx, ry, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(f"jobs/v300/_chk/dbg_pass1_{tag}.png")
    vis_mask(w_new, f"wnew_{tag}"); vis_mask(w_old, f"wold_{tag}")
    vis_mask(src_warp / 255.0, f"srcwarp_{tag}")
    # 翼尖轨迹验证
    tipx, tipy = 524.0, 692.0
    dx_t, dy_t = tipx - shL[0], tipy - shL[1]
    nrx = shL[0] + dx_t * cos_t + dy_t * sin_t
    nry = shL[1] - dx_t * sin_t + dy_t * cos_t
    print(f"[dbg {tag}] L_tip ({tipx:.0f},{tipy:.0f}) -> ({nrx:.0f},{nry:.0f})")

dbg_rot(math.radians(32), "up32")
dbg_rot(0.0, "zero")
print("[dbg] done -> jobs/v300/_chk/")
