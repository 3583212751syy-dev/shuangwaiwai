"""v259 连通域分析: 精确找原图各文字块包围盒 (排除 bat 与外环)。"""
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

# 暗像素候选 (文字), 排除 bat(gray<50) 与外环带(dist 397~431)
dark = gray < 120
bat = gray < 50
ring = (dist > (m.RING['inner_r']-10)) & (dist < (m.RING['outer_r']+10))
cand = dark & ~bat & ~ring
# 只保留徽章下方 + 内圈顶部的文字区, 去掉背景杂点
cand &= ~((dist > m.RING['outer_r']+10) & (yy < 700))  # 徽章外上方空白

# 连通域
num, labels, stats, centroids = cv2.connectedComponentsWithStats((cand.astype(np.uint8))*255, 8)
print(f"连通域数(含背景)={num}")
rows = []
for i in range(1, num):
    x, y, bw, bh, area = stats[i]
    cx, cy = centroids[i]
    if area < 200:   # 跳过小噪点
        continue
    rows.append((area, int(cx), int(cy), x, y, bw, bh))
rows.sort(reverse=True)
print(f"\n{'area':>7} {'cx':>5} {'cy':>5}  bbox(x,y,w,h)")
for area, cx, cy, x, y, bw, bh in rows[:25]:
    print(f"{area:>7} {cx:>5} {cy:>5}  ({x},{y},{bw},{bh})")
