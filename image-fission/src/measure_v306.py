# -*- coding: utf-8 -*-
"""v306 measure: ring circle / medallion circle / disc edge / claws / colors."""
import numpy as np
import cv2
from PIL import Image

SRC = r"E:\Desktop\双接口\image-fission\ComfyUI\input\v300_text_free_source.png"
arr = np.array(Image.open(SRC).convert("RGB")).astype(np.float32)
H, W = arr.shape[:2]
R_, G_, B_ = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
maxc = np.maximum(np.maximum(R_, G_), B_)

cx0, cy0 = 776.0, 745.0
ys, xs = np.mgrid[0:H, 0:W]
dist = np.sqrt((xs - cx0) ** 2 + (ys - cy0) ** 2)

# ---------- 1) ring circle fit ----------
dark = maxc < 75
ring_cand = dark & (dist > 300) & (dist < 400)
pts = np.column_stack(np.nonzero(ring_cand))
px = pts[:, 1].astype(np.float64)
py = pts[:, 0].astype(np.float64)

def fit_circle(px, py):
    A = np.column_stack([px, py, np.ones_like(px)])
    b = px ** 2 + py ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = np.sqrt(sol[2] + cx ** 2 + cy ** 2)
    return cx, cy, r

for _ in range(8):
    rcx, rcy, rr = fit_circle(px, py)
    d = np.sqrt((px - rcx) ** 2 + (py - rcy) ** 2)
    keep = np.abs(d - rr) < 5
    px, py = px[keep], py[keep]
print("RING fit center=(%.1f,%.1f) R=%.1f npts=%d" % (rcx, rcy, rr, len(px)))
ang = np.degrees(np.arctan2(py - rcy, px - rcx))
hist, _ = np.histogram(ang, bins=36, range=(-180, 180))
print("ring angle hist(10deg, -180..180):", hist.tolist())

# ---------- 2) medallion circle (boundary fit) ----------
purple = (maxc >= 95) & (maxc <= 170) & ((B_ - G_) > 45)
purple = purple & (dist < 340)
# boundary: purple pixel whose outward radial +10px neighbor is marble (light, maxc>175)
out_x = (xs + 10 * (xs - rcx) / np.maximum(dist, 1)).astype(int)
out_y = (ys + 10 * (ys - rcy) / np.maximum(dist, 1)).astype(int)
ok = (out_x >= 0) & (out_x < W) & (out_y >= 0) & (out_y < H)
out_maxc = np.zeros((H, W), np.float32)
out_maxc[ok] = maxc[out_y[ok], out_x[ok]]
bd = purple & (out_maxc > 175) & (dist > 120)
pts2 = np.column_stack(np.nonzero(bd))
qx = pts2[:, 1].astype(np.float64)
qy = pts2[:, 0].astype(np.float64)
for _ in range(8):
    mcx, mcy, mr = fit_circle(qx, qy)
    d = np.sqrt((qx - mcx) ** 2 + (qy - mcy) ** 2)
    keep = np.abs(d - mr) < 6
    qx, qy = qx[keep], qy[keep]
print("MEDALLION fit center=(%.1f,%.1f) R=%.1f npts=%d" % (mcx, mcy, mr, len(qx)))

# ---------- 3) disc edge per angle ----------
def is_bg(y, x):
    p = arr[int(y), int(x)]
    return abs(p[0] - 171) < 22 and abs(p[1] - 129) < 22 and abs(p[2] - 181) < 22

edge_pts = []
for adeg in range(-180, 180, 3):
    a = np.radians(adeg)
    hit = None
    for rr_ in np.arange(430, 240, -2.0):
        x = int(round(cx0 + rr_ * np.cos(a)))
        y = int(round(cy0 + rr_ * np.sin(a)))
        if 0 <= x < W and 0 <= y < H and not is_bg(y, x):
            hit = (rr_, adeg, maxc[y, x])
            break
    if hit:
        edge_pts.append(hit)
ers = np.array([e[0] for e in edge_pts])
print("DISC edge radius: median=%.0f p10=%.0f p90=%.0f (n=%d)" % (
    np.median(ers), np.percentile(ers, 10), np.percentile(ers, 90), len(ers)))
low = [(e[1], round(e[0]), round(e[2])) for e in edge_pts if e[0] < np.percentile(ers, 10) + 5]
print("shallow edge angles (angle, r, maxc):", low[:20])

# ---------- 4) bat approx & wing-beyond-medallion ----------
bat_approx = (dark & (dist < 330)).astype(np.uint8) * 255
bat_approx = cv2.morphologyEx(bat_approx, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(bat_approx, 8)
k = 1 + int(np.argmax(st[1:, 4]))
bat_m = (lab == k).astype(np.uint8) * 255
bx, by, bw, bh, ba = st[k]
print("BAT bbox x%d..%d y%d..%d area=%d" % (bx, bx + bw, by, by + bh, ba))
med_c = np.zeros((H, W), np.uint8)
cv2.circle(med_c, (int(mcx), int(mcy)), int(mr + 8), 255, -1)
out_med = (bat_m > 0) & (med_c == 0)
print("bat px outside medallion+8: %d (%.1f%%)" % (int(out_med.sum()), 100.0 * out_med.sum() / max(1, (bat_m > 0).sum())))
ys_o, xs_o = np.nonzero(out_med)
if len(ys_o):
    print("  outside bbox x%d..%d y%d..%d" % (xs_o.min(), xs_o.max(), ys_o.min(), ys_o.max()))

# ---------- 5) claws near tail base ----------
claw_zone = np.zeros((H, W), np.uint8)
claw_zone[700:840, 660:900] = 255
claw = (maxc >= 85) & (maxc <= 175) & ((B_ - G_) > 35)
claw = cv2.bitwise_and(claw.astype(np.uint8) * 255, claw_zone)
n3, l3, s3, _ = cv2.connectedComponentsWithStats(claw, 8)
for i in range(1, n3):
    x, y, w, h, a = s3[i]
    if a >= 60:
        print("CLAW cc bbox=(%d,%d %dx%d) area=%d meanRGB=%s" % (
            x, y, w, h, a, arr[l3 == i].mean(axis=0).round(0)))

# ---------- 6) petals (purple lobes below body) ----------
pet_zone = np.zeros((H, W), np.uint8)
pet_zone[820:1100, 500:1060] = 255
pet = cv2.bitwise_and(purple.astype(np.uint8) * 255, pet_zone)
n4, l4, s4, _ = cv2.connectedComponentsWithStats(pet, 8)
big = [i for i in range(1, n4) if s4[i][4] > 3000]
print("PETALS: %d big lobes" % len(big))
for i in big:
    x, y, w, h, a = s4[i]
    print("  petal bbox=(%d,%d %dx%d) area=%d" % (x, y, w, h, a))

# ---------- 7) colors ----------
def patch_stat(name, x, y, r=8):
    p = arr[y - r:y + r, x - r:x + r].reshape(-1, 3)
    m = p.mean(axis=0)
    s = p.std(axis=0).mean()
    print("%-18s RGB mean=(%5.0f,%5.0f,%5.0f) std=%.1f" % (name, m[0], m[1], m[2], s))

patch_stat("marble_clean_L", 500, 800)
patch_stat("marble_clean_R", 1050, 800)
patch_stat("marble_top", 776, 480)
patch_stat("medallion_purple", 700, 700)
patch_stat("petal_purple", 640, 900)
patch_stat("bg_outside", 300, 300)
# ring thickness: sample radial profile at -45deg
a = np.radians(-45)
prof = []
for rr_ in np.arange(rr - 14, rr + 14, 1.0):
    x = int(round(rcx + rr_ * np.cos(a)))
    y = int(round(rcy + rr_ * np.sin(a)))
    prof.append(int(maxc[y, x]))
print("ring radial maxc profile at -45deg (r-14..r+14):", prof)
print("done")
