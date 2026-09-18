import cv2
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
import os

# Paths
jimeng_path = "E:/Desktop/茶叶/成品_样稿_confirm/_jimeng_front.png"
label_path = "E:/Desktop/茶叶/成品_样稿_confirm/_label.png"
out_dir = "E:/Desktop/茶叶/成品_第十一轮_正背组合重制/_assets"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "product_front_label_fixed.png")

# Load with PIL
front_pil = Image.open(jimeng_path).convert("RGB")
label_pil = Image.open(label_path).convert("RGBA")

front_rgb = cv2.cvtColor(np.array(front_pil), cv2.COLOR_RGB2BGR)
label_rgba = cv2.cvtColor(np.array(label_pil), cv2.COLOR_RGBA2BGRA)

print(f"Front size: {front_rgb.shape}, Label size: {label_rgba.shape}")

label_alpha = label_rgba[:, :, 3]
label_rgb = cv2.cvtColor(label_rgba, cv2.COLOR_BGRA2BGR)

# ---- Compute alpha mask from ORIGINAL front image using grabCut ----
# This cleanly separates the pouch from white background and bottom shadow
h, w = front_rgb.shape[:2]
rect = (int(w * 0.18), int(h * 0.03), int(w * 0.66), int(h * 0.92))
mask = np.zeros(front_rgb.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
cv2.grabCut(front_rgb, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
fg = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8) * 255

# Keep largest component (the pouch)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
if num_labels > 1:
    sizes = stats[1:, cv2.CC_STAT_AREA]
    largest = np.argmax(sizes) + 1
    fg = (labels == largest).astype(np.uint8) * 255

# Fill holes to include the light label area fully
fg = ndimage.binary_fill_holes(fg // 255).astype(np.uint8) * 255

# Smooth edges
fg = cv2.GaussianBlur(fg, (5, 5), 0)

# Smooth mask edges
fg_smooth = cv2.GaussianBlur(fg, (5,5), 0)
mask = fg_smooth.astype(float) / 255.0
print(f"Mask coverage: {mask.mean():.2%}")

# ---- ORB feature matching to align label onto front ----
orb = cv2.ORB_create(5000)
kp1, des1 = orb.detectAndCompute(label_rgb, None)
kp2, des2 = orb.detectAndCompute(front_rgb, None)

if des1 is None or des2 is None:
    raise ValueError("Feature detection failed")

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Keep top matches
num_good = max(20, int(len(matches) * 0.15))
num_good = min(num_good, len(matches))
good = matches[:num_good]
print(f"Matches: {len(matches)}, using top {len(good)}")

src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
if H is None:
    raise ValueError("Homography estimation failed")

print(f"Homography found, inliers: {inlier_mask.sum()}/{len(inlier_mask)}")

# Warp label and alpha to front dimensions
h, w = front_rgb.shape[:2]
warped_label = cv2.warpPerspective(label_rgb, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
warped_alpha = cv2.warpPerspective(label_alpha, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

# Normalize alpha
warped_alpha = warped_alpha.astype(float) / 255.0

# Blend: only replace where warped_alpha > 0
result = front_rgb.copy().astype(float)
for c in range(3):
    result[:, :, c] = warped_alpha * warped_label[:, :, c] + (1 - warped_alpha) * result[:, :, c]
result = np.clip(result, 0, 255).astype(np.uint8)

# Convert to RGBA and apply pre-computed mask
result_rgba = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
result_rgba[:, :, 3] = (mask * 255).astype(np.uint8)

# Save with PIL
result_pil = Image.fromarray(cv2.cvtColor(result_rgba, cv2.COLOR_BGRA2RGBA))
result_pil.save(out_path)

# Debug: blended without alpha
debug_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
debug_pil.save(os.path.join(out_dir, "_debug_front_label_blended.jpg"), quality=95)

# Debug: mask
Image.fromarray(fg).save(os.path.join(out_dir, "_debug_front_mask.jpg"))

print(f"Saved: {out_path}")
