"""v259 分析3: 用真实 bat 掩码排除 bat, 再找近墨文字块 (BACARDÍ/WHEART/Est1862/三角) 真实位置。"""
import numpy as np, cv2
import importlib.util
from PIL import Image
spec = importlib.util.spec_from_file_location('v253', 'src/v253_bat_logo_inpaint.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

rgb = np.array(Image.open(m.COMFY_INPUT / m.REF_IMG).convert('RGB'))
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
h, w = bgr.shape[:2]
yy, xx = np.mgrid[:h, :w]
dist = np.sqrt((xx - m.RING['cx'])**2 + (yy - m.RING['cy'])**2)

# 真实 bat 掩码: bbox 内 gray<50 (近似, 与 extract_bat_strict 同思路)
bx, by, bw, bh = m.BAT_BBOX
bat = np.zeros_like(gray, bool)
bat[by:by+bh, bx:bx+bw] = gray[by:by+bh, bx:bx+bw] < 50
ring = (dist > (m.RING['inner_r']-10)) & (dist < (m.RING['outer_r']+10))

dark = gray < 70
cand = dark & ~bat & ~ring

num, labels, stats, centroids = cv2.connectedComponentsWithStats((cand.astype(np.uint8))*255, 8)
print(f"候选连通域(排除bat/ring, gray<70)数={num}")
rows = []
for i in range(1, num):
    x, y, cw, ch, area = stats[i]
    if area < 300:
        continue
    cyy, cxx = centroids[i]
    # 该块平均灰度
    mg = gray[labels == i].mean()
    rows.append((area, int(cxx), int(cyy), int(mg), x, y, cw, ch))
rows.sort(reverse=True)
print(f"{'area':>7} {'cx':>5} {'cy':>5} {'meanGray':>8}  bbox(x,y,w,h)")
for area, cx, cy, mg, x, y, cw, ch in rows[:30]:
    print(f"{area:>7} {cx:>5} {cy:>5} {mg:>8.1f}  ({x},{y},{cw},{ch})")
