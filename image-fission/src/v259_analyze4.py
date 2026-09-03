"""v259 分析4: 用真实 bat 剪影掩码(extract_bat_strict 思路)排除 bat, 找真正的文字块。"""
import numpy as np, cv2, importlib.util, sys
from pathlib import Path
from PIL import Image
spec = importlib.util.spec_from_file_location('v253', 'src/v253_bat_logo_inpaint.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sys.path.insert(0, 'src')
import v259_clean_fission as v259

rgb = np.array(Image.open(m.COMFY_INPUT / m.REF_IMG).convert('RGB'))
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
h, w = bgr.shape[:2]
yy, xx = np.mgrid[:h, :w]
dist = np.sqrt((xx - m.RING['cx'])**2 + (yy - m.RING['cy'])**2)

bat = v259.extract_bat_strict(bgr)        # 真实 bat 剪影 (含翼)
ring = (dist > (m.RING['inner_r']-10)) & (dist < (m.RING['outer_r']+10))
dark = gray < 80
cand = dark & ~bat & ~ring

num, labels, stats, centroids = cv2.connectedComponentsWithStats((cand.astype(np.uint8))*255, 8)
print(f"排除 bat剪影+ring 后, 近墨文字连通域数={num}")
rows = []
for i in range(1, num):
    x, y, cw, ch, area = stats[i]
    if area < 400:
        continue
    cyy, cxx = centroids[i]
    rows.append((area, int(cxx), int(cyy), x, y, cw, ch))
rows.sort(reverse=True)
print(f"{'area':>7} {'cx':>5} {'cy':>5}  bbox(x,y,w,h)")
for area, cx, cy, x, y, cw, ch in rows[:22]:
    print(f"{area:>7} {cx:>5} {cy:>5}  ({x},{y},{cw},{ch})")
