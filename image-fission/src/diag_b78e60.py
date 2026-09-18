import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from PIL import Image
import fission_v322_per_element as F

SRC = "E:/Desktop/图裂变测试图/b78e60de8dfdf44acda99395326a7298.jpg"
img = Image.open(SRC).convert("RGB")
arr = np.array(img)
H, W = arr.shape[:2]
gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

# 1) 重复 pipeline 的 Otsu+Canny 前景（与 _detect_small_elements 一致）
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, fg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
edges = cv2.Canny(blur, 30, 100)
fg = cv2.bitwise_or(fg, edges)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=1)
fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)

# 2) 排除文字带（用 analyze 得到的 bands）
prof = F.analyze_image(SRC)
tb_mask = F._build_text_band_mask(prof.text_bands, W, H)
fg_no_text = cv2.bitwise_and(fg, cv2.bitwise_not(tb_mask))

print(f"img H={H} W={W}  text_bands={[(b.y0,b.y1,b.x0,b.x1) for b in prof.text_bands]}")
n, labels, stats, cents = cv2.connectedComponentsWithStats(fg_no_text, connectivity=8)
print(f"连通域总数(排除文字带后): {n-1}")
print(f"{'#':>3} {'y0':>4}-{'y1':<4} {'x0':>4}-{'x1':<4} {'area':>7} {'aH/W':>5} {'rel%':>5} {'cy%':>5} {'tb':>2}  ->class")
for lab in range(1, n):
    x,y,w,h,area = stats[lab]
    cx,cy = cents[lab]
    if area < 80 or area > H*W*0.30: continue
    aH = h/max(w,1); aW = w/max(h,1)
    rel = area/(W*H); cyp = cy/H
    tb_ = (x<=5 or y<=5 or x+w>=W-5 or y+h>=H-5)
    cls = F._classify_small_element(x,y,w,h,area,aH,tb_,cx,cy,W,H)
    print(f"{lab:>3} {y:>4}-{y+h:<4} {x:>4}-{x+w:<4} {area:>7} {aH:>5.2f} {rel*100:>5.1f} {cyp*100:>5.1f} {str(tb_)[0]:>2}  ->{cls}")

# 3) 亮度 ASCII 图（粗看布局）
print("\n--- 亮度 ASCII (暗=#, 亮=. ; 文字带行标 T) ---")
cols, rows = 80, 40
small = cv2.resize(gray, (cols, rows))
chars = " .:-=+*#%@"
lines=[]
for ry in range(rows):
    line=""
    for rx in range(cols):
        v = small[ry,rx]
        line += chars[min(9, v*10//256)]
    lines.append(line)
# 标注文字带
for b in prof.text_bands:
    ry0 = b.y0*rows//H; ry1 = b.y1*rows//H
    for ry in range(max(0,ry0), min(rows,ry1+1)):
        if ry < len(lines):
            lines[ry] = lines[ry][:cols] + "  <-TEXTBAND"
print("\n".join(lines))
