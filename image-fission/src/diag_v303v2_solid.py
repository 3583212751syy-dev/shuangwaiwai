"""diag_solid: 检查 dome/花瓣是否在 bat_m/solid_m 内 + 各部位颜色值"""
import numpy as np, cv2
from PIL import Image

SRC_PATH = r"E:\Desktop\双接口\image-fission\ComfyUI\input\v300_text_free_source.png"
arr = np.array(Image.open(SRC_PATH).convert("RGB")).astype(np.float32)
H, W = arr.shape[:2]
cx, cy = 776, 746
R, G, Bc = arr[:,:,0], arr[:,:,1], arr[:,:,2]
maxc = np.maximum(np.maximum(R, G), Bc)
dark = (maxc < 150).astype(np.uint8) * 255
badge_lim = np.zeros((H, W), np.uint8)
cv2.circle(badge_lim, (cx, cy), 390, 255, -1)
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

# 采样点：dome(冠顶弧)、花瓣、翅膀膜、头、躯干、尾
samples = {
    "dome_top(776,560)":   (776, 560),
    "dome_L(650,650)":     (650, 650),
    "dome_R(900,650)":     (900, 650),
    "petal_L(640,920)":    (640, 920),
    "petal_R(912,920)":    (912, 920),
    "petal_L2(680,960)":   (680, 960),
    "wing_mem_L(600,700)": (600, 700),
    "head(780,520)":       (780, 520),
    "torso(776,800)":      (776, 800),
    "tail(776,950)":       (776, 950),
    "bg_inner(500,1100)":  (500, 1100),
}
print(f"{'point':<22}{'RGB':<18}{'maxc':<7}{'bat_m':<7}{'solid_m':<8}")
for name, (x, y) in samples.items():
    rgb = arr[y, x].astype(int)
    m = maxc[y, x]
    bm = int(bat_m[y, x] > 0); sm = int(solid_m[y, x] > 0)
    print(f"{name:<22}{str(rgb.tolist()):<18}{m:<7.0f}{bm:<7}{sm:<8}")

# 可视化：solid_m 红罩 + bat_m 绿罩
vis = arr.astype(np.uint8).copy()
vis[solid_m > 0] = (0.5 * vis[solid_m > 0] + np.array([128, 0, 0]) * 0.5).astype(np.uint8)
vis[bat_m > 0] = (0.6 * vis[bat_m > 0] + np.array([0, 128, 0]) * 0.4).astype(np.uint8)
badge = vis[cy-450:cy+450, cx-450:cx+450]
Image.fromarray(badge).save(r"E:\Desktop\双接口\image-fission\jobs\v303v2\_chk\diag_solid_vis.png")

# 翼 mask 可视化（v303v2 逻辑：wing_m = solid_m - body_core）
band_b = np.zeros((H, W), np.uint8)
band_b[:, cx-48:cx+48] = 255
body_core = cv2.bitwise_and(bat_m, band_b)
wing_m = solid_m.copy()
wing_m[body_core > 0] = 0
vis2 = arr.astype(np.uint8).copy()
vis2[wing_m > 0] = (0.5 * vis2[wing_m > 0] + np.array([0, 0, 160]) * 0.5).astype(np.uint8)
badge2 = vis2[cy-450:cy+450, cx-450:cx+450]
Image.fromarray(badge2).save(r"E:\Desktop\双接口\image-fission\jobs\v303v2\_chk\diag_wing_vis.png")
print("\nSaved diag_solid_vis.png / diag_wing_vis.png")
