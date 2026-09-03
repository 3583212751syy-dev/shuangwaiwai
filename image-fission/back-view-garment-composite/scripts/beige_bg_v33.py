"""
beige_bg_v33.py — Post-process: replace the v33 back result's gray studio bg
with the front's warm beige (181,164,142), so the back view matches the front's bg.

Steps:
  1. GrabCut the figure (rect, 5 iter) → fg mask
  2. Morphology clean (close 5x5 iter2, open 5x5 iter1)
  3. Erode figure slightly (3px) to keep beige inside the safe border
  4. Replace all non-figure pixels with beige (BGR 142,164,181)
  5. Soft 7px blur on the alpha for a seamless edge
  6. Save to results/back_v33_v23s2024_beige.png + a final comparison
"""
import cv2, numpy as np, os
ROOT = r'D:\.workbuddy\2026-08-16-00-13-40\image-fission'
SRC   = os.path.join(ROOT, 'results', 'back_v33_v23s2024_s2024_00001_.png')
CMP   = os.path.join(ROOT, 'results', 'cmp_v33_v23s2024_s2024_00001_.png')
FRONT = os.path.join(ROOT, 'ComfyUI', 'input', 'front_model.jpg')
OUT   = os.path.join(ROOT, 'results', 'back_v33_v23s2024_beige.png')
CMP_F = os.path.join(ROOT, 'results', 'cmp_v33_v23s2024_beige.png')
# Front-sampled beige (BGR)
BEIGE = np.array([142, 164, 181], np.float32)

img = cv2.imread(SRC)
H, W = img.shape[:2]
print('src', SRC, img.shape)

# GrabCut
mask = np.zeros((H, W), np.uint8)
bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
rx0, ry0 = int(W * 0.10), int(H * 0.02)
rw, rh   = int(W * 0.80), int(H * 0.96)
cv2.grabCut(img, mask, (rx0, ry0, rw, rh), bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
k = 5
fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=2)
fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  np.ones((k, k), np.uint8), iterations=1)
# Erode 3px (keep beige cleanly behind figure edge)
fg = cv2.erode(fg, np.ones((3, 3), np.uint8), iterations=1)
# Soft alpha
alpha = cv2.GaussianBlur(fg, (7, 7), 0).astype(np.float32) / 255.0
a3 = np.stack([alpha] * 3, -1)

out = (img.astype(np.float32) * a3 + BEIGE * (1 - a3)).astype(np.uint8)
cv2.imwrite(OUT, out)
print('saved', OUT)

# Final comparison (front | back)
front = cv2.imread(FRONT)
cmp = np.hstack([front, out])
cv2.imwrite(CMP_F, cmp)
print('saved', CMP_F, cmp.shape)
