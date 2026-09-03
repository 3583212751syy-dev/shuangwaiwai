"""
v34-pose: composite front print onto a v34 pure-back base (pose-matched, beige bg).
Unlike v33 (which assumed a CLEAN white-shirt / gray-shorts base like v23 s2024),
this one:
  1. PRE-CLEANS the base: inpaints dark pixels (lum<60) within the figure's
     garment region to remove the IP-Adapter-generated ink splatter and the
     solid black waistband artifact.
  2. Uses GrabCut-figure-based garment masks (not gray-range) so it works on
     the v34 white-T + white-shorts base.

Result: pose-matched (hands in pockets), pixel-perfect front print, matching
warm-beige background, 1350x1800 portrait.
"""
import cv2, numpy as np, sys, os, time

FRONT = cv2.imread('ComfyUI/input/front_model.jpg')
TARGET_W, TARGET_H = 1350, 1800
BG = (180, 170, 158)  # warm beige (close to front bg 181,164,142)

# FS source quads in FRONT image coords (from v33; full front image bounds)
FS_shirt = np.array([[150, 180], [1300, 180], [1300, 1480], [150, 1480]], np.float32)
FS_pants = np.array([[280, 1180], [1100, 1180], [1100, 1600], [280, 1600]], np.float32)

back_path = sys.argv[1]
out_prefix = sys.argv[2] if len(sys.argv) > 2 else 'v34_pose'
t0 = time.time()

# ---- 1. Load + canvas-fit the back base ----
back_raw = cv2.imread(back_path)
Hb, Wb = back_raw.shape[:2]
scale = TARGET_H / Hb
new_w = int(Wb * scale); new_h = TARGET_H
br = cv2.resize(back_raw, (new_w, new_h), cv2.INTER_LANCZOS4)
canvas = np.full((TARGET_H, TARGET_W, 3), BG, np.uint8)
x_off = (TARGET_W - new_w) // 2
canvas[:, x_off:x_off + new_w] = br
back = canvas

# ---- 2. GrabCut figure ----
mask_gc = np.zeros(back.shape[:2], np.uint8)
bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
cv2.grabCut(back, mask_gc, (150, 50, 1050, 1700), bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
figure = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

g = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)

# ---- 3. y_split detection (gray drop in central columns) ----
cx0, cx1 = x_off + new_w // 4, x_off + 3 * new_w // 4
central_means = []
for y in range(0, TARGET_H, 5):
    row_central = g[y, cx0:cx1]
    fig_in_row = (figure[y, cx0:cx1] > 0).sum()
    if fig_in_row > 20:
        central_means.append((y, row_central.mean()))
y_split = None
max_drop = 0
for i in range(1, len(central_means)):
    y, m = central_means[i]
    py, pm = central_means[i - 1]
    drop = pm - m
    if drop > max_drop and y > 600:
        max_drop = drop
        y_split = (py + y) // 2
if y_split is None or max_drop < 5:
    y_split = TARGET_H * 2 // 3
print(f'[{time.time()-t0:.1f}s] y_split={y_split}')

# ---- 4. PRE-CLEAN: inpaint dark pixels in garment region ----
shirt_y_range = (50, y_split)
shorts_y_range = (y_split, TARGET_H - 50)
cleaned = back.copy()
for (y0, y1) in [shirt_y_range, shorts_y_range]:
    region_mask = np.zeros(back.shape[:2], np.uint8)
    region_mask[y0:y1, :] = 255
    region_mask = cv2.bitwise_and(region_mask, region_mask, mask=figure)
    dark = ((g < 60) & (region_mask > 0)).astype(np.uint8) * 255
    if dark.sum() == 0:
        continue
    dark = cv2.dilate(dark, np.ones((5, 5), np.uint8), iterations=1)
    cleaned = cv2.inpaint(cleaned, dark, 9, cv2.INPAINT_TELEA)
g_clean = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
print(f'[{time.time()-t0:.1f}s] cleaned dark pixels in garment region')

# ---- 5. Shirt mask: figure ∩ y<y_split → close → largest → convex hull ----
shirt_m = np.zeros(back.shape[:2], np.uint8)
shirt_m[shirt_y_range[0]:shirt_y_range[1], :] = 255
shirt_m = cv2.bitwise_and(shirt_m, shirt_m, mask=figure)
shirt_m = cv2.morphologyEx(shirt_m, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8))
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(shirt_m, connectivity=8)
if num_labels <= 1:
    print('NO SHIRT'); sys.exit(1)
areas = stats[1:, cv2.CC_STAT_AREA]
largest_label = 1 + int(np.argmax(areas))
shirt_main = np.where(labels == largest_label, 255, 0).astype(np.uint8)
contours, _ = cv2.findContours(shirt_main, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
biggest = max(contours, key=cv2.contourArea)
hull = cv2.convexHull(biggest)
hull_mask = np.zeros((TARGET_H, TARGET_W), np.uint8)
cv2.fillConvexPoly(hull_mask, hull, 255)
hull_mask = cv2.erode(hull_mask, np.ones((10, 10), np.uint8), iterations=1)
hull_mask[shirt_y_range[1]:, :] = 0
ys_s, xs_s = np.where(hull_mask > 0)
sh_x0 = int(xs_s.min()); sh_x1 = int(xs_s.max())
sh_y0 = int(ys_s.min()); sh_y1 = int(ys_s.max())
print(f'[{time.time()-t0:.1f}s] shirt hull x[{sh_x0}..{sh_x1}] y[{sh_y0}..{sh_y1}]')

# ---- 6. Shorts mask: figure ∩ y>y_split → close → largest → convex hull ----
shorts_m = np.zeros(back.shape[:2], np.uint8)
shorts_m[shorts_y_range[0]:shorts_y_range[1], :] = 255
shorts_m = cv2.bitwise_and(shorts_m, shorts_m, mask=figure)
shorts_m = cv2.morphologyEx(shorts_m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
num_p, labels_p, stats_p, _ = cv2.connectedComponentsWithStats(shorts_m, connectivity=8)
if num_p > 1:
    areas_p = stats_p[1:, cv2.CC_STAT_AREA]
    largest_p = 1 + int(np.argmax(areas_p))
    shorts_main = np.where(labels_p == largest_p, 255, 0).astype(np.uint8)
else:
    shorts_main = shorts_m
contours_p, _ = cv2.findContours(shorts_main, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
if not contours_p:
    print('NO SHORTS'); sys.exit(1)
hull_p = cv2.convexHull(max(contours_p, key=cv2.contourArea))
hull_p_mask = np.zeros((TARGET_H, TARGET_W), np.uint8)
cv2.fillConvexPoly(hull_p_mask, hull_p, 255)
hull_p_mask = cv2.erode(hull_p_mask, np.ones((10, 10), np.uint8), iterations=1)
ys_p, xs_p = np.where(hull_p_mask > 0)
if len(xs_p) == 0:
    print('NO SHORTS HULL'); sys.exit(1)
pa_x0 = int(xs_p.min()); pa_x1 = int(xs_p.max())
pa_y0 = int(ys_p.min()); pa_y1 = int(ys_p.max())
print(f'[{time.time()-t0:.1f}s] shorts hull x[{pa_x0}..{pa_x1}] y[{pa_y0}..{pa_y1}]')

# ---- 7. Warp front onto cleaned base ----
DS_shirt = np.array([[sh_x0, sh_y0], [sh_x1, sh_y0], [sh_x1, sh_y1], [sh_x0, sh_y1]], np.float32)
DS_pants = np.array([[pa_x0, pa_y0], [pa_x1, pa_y0], [pa_x1, pa_y1], [pa_x0, pa_y1]], np.float32)
M_S = cv2.getPerspectiveTransform(FS_shirt, DS_shirt)
M_P = cv2.getPerspectiveTransform(FS_pants, DS_pants)
sw = cv2.warpPerspective(FRONT, M_S, (TARGET_W, TARGET_H), borderMode=cv2.BORDER_CONSTANT, borderValue=BG)
pw = cv2.warpPerspective(FRONT, M_P, (TARGET_W, TARGET_H), borderMode=cv2.BORDER_CONSTANT, borderValue=BG)

# Quad masks to prevent leakage
quad_S = np.zeros((TARGET_H, TARGET_W), np.uint8)
cv2.fillPoly(quad_S, [DS_shirt.astype(np.int32)], 255)
sw = cv2.bitwise_and(sw, sw, mask=quad_S)
quad_P = np.zeros((TARGET_H, TARGET_W), np.uint8)
cv2.fillPoly(quad_P, [DS_pants.astype(np.int32)], 255)
pw = cv2.bitwise_and(pw, pw, mask=quad_P)

# ---- 8. Composite onto CLEANED base ----
# Shirt: full alpha blend (replace back shirt with warped front)
soft_mask = cv2.GaussianBlur(hull_mask, (31, 31), 4)
sa = soft_mask.astype(np.float32) / 255.0
sa3 = np.dstack([sa] * 3)

# Shorts: only ink pixels (preserves back's clean shorts shading)
pg = cv2.cvtColor(pw, cv2.COLOR_BGR2GRAY)
ink = (pg < 130).astype(np.uint8) * 255
ink = cv2.bitwise_and(ink, ink, mask=quad_P)
ink = cv2.erode(ink, np.ones((3, 3), np.uint8), iterations=1)
ink = cv2.GaussianBlur(ink, (21, 21), 4)
ia = ink.astype(np.float32) / 255.0
ia3 = np.dstack([ia] * 3)

out = cleaned.astype(np.float32)
out = out * (1 - sa3) + sw.astype(np.float32) * sa3
ma = ia3 > 0.05
out = np.where(ma, out * (1 - ia3) + pw.astype(np.float32) * ia3, out)
out = np.clip(out, 0, 255).astype(np.uint8)

# ---- 9. Verification ----
g_out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
ink_in_shirt = (g_out[sh_y0:sh_y1, sh_x0:sh_x1] < 60).sum() / max(1, (sh_y1 - sh_y0) * (sh_x1 - sh_x0)) * 100
ink_in_shorts = (g_out[pa_y0:pa_y1, pa_x0:pa_x1] < 60).sum() / max(1, (pa_y1 - pa_y0) * (pa_x1 - pa_x0)) * 100
print(f'[{time.time()-t0:.1f}s] ink%% shirt={ink_in_shirt:.1f}%% shorts={ink_in_shorts:.1f}%%')

# ---- 10. Save ----
tag = os.path.basename(back_path).replace('.png', '')
on = f'results/back_{out_prefix}_{tag}.png'
cv2.imwrite(on, out)
gap = 20
cmp = np.full((TARGET_H, TARGET_W * 2 + gap, 3), [220, 215, 205], np.uint8)
cmp[:, :TARGET_W] = FRONT
cmp[:, TARGET_W + gap:] = out
cn = on.replace('back_', 'cmp_')
cv2.imwrite(cn, cmp)
print(f'[{time.time()-t0:.1f}s] DONE -> {on} | {cn}')
