"""Round 18 rebuild v4: make ENTIRE bag interior kraft brown (uniform).
Only the dark fold at bottom and the hanger cut at top stay original.
Real label pasted at exact label-center.
"""
import numpy as np
from PIL import Image

JIMENG = r"E:/Desktop/茶叶/成品_样稿_confirm/_jimeng_front.png"
LABEL  = r"E:/Desktop/茶叶/成品_样稿_confirm/_label.png"
OUT    = r"E:/Desktop/茶叶/成品_第十八轮_品牌叙事D/_assets"
import os; os.makedirs(OUT, exist_ok=True)

bag = Image.open(JIMENG).convert("RGBA")
arr = np.array(bag); H,W = arr.shape[:2]

r,g,bv = arr[:,:,0].astype(int), arr[:,:,1].astype(int), arr[:,:,2].astype(int)
alpha = arr[:,:,3]
# BAG = warm-brown + dark area (the bag occupies central vertical area)
bag_mask = ((r < 218) & (g < 213) & (bv < 198)) | ((r < 150) & (g < 110) & (bv < 80))
bag_mask = bag_mask & (alpha > 30)
# expand to fill any holes inside the bag
import scipy.ndimage as ndi
bag_mask = ndi.binary_fill_holes(bag_mask)
# clean up small specks
bag_mask = ndi.binary_closing(bag_mask, iterations=8)
bag_mask = ndi.binary_fill_holes(bag_mask)
ys, xs = np.where(bag_mask)
bx0,by0 = int(xs.min()), int(ys.min())
bx1,by1 = int(xs.max()), int(ys.max())
bw, bh = bx1-bx0, by1-by0
print(f"FILLED bag bbox: {bx0},{by0}  ->  {bx1},{by1}  ({bw}x{bh})")

# target kraft color — explicit dark kraft for visual contrast against yellow bg
# (avoid sampling median since bag interior is biased light — use a known kraft tone)
KRAFT_BASE = (155, 105, 65)
KRAFT_DARKER = (118, 75, 42)
bag_brown = (*KRAFT_BASE, 255)
print("bag_brown:", bag_brown[:3], " (forced kraft)")

# fill ENTIRE bag interior with bag_brown (erase AI highlights + label)
filled = arr.copy()
inset = 4   # keep 4px of original bag edge for natural silhouette
x0, y0 = bx0+inset, by0+inset
x1, y1 = bx1-inset, by1-inset
for y in range(y0, y1):
    for x in range(x0, x1):
        if arr[y,x,3] > 10:
            filled[y,x] = bag_brown
# add vertical gradient for natural kraft + horizontal "wrinkle" hint
for y in range(y0, y1):
    sh = (y - y0) / max(1, y1 - y0)
    # bottom is slightly darker (deeper kraft)
    adjust = int(20 * sh)
    for x in range(x0, x1):
        if filled[y,x,3] > 10:
            np_arr = np.array(filled[y,x,0:3])
            filled[y,x,0:3] = np.maximum(0, np_arr - adjust)
# add subtle vertical fold-shadows on left and right sides (slight darker edges)
left_dark = np.array(KRAFT_DARKER)
for x in range(x0, x0+8):
    for y in range(y0, y1):
        if filled[y,x,3] > 10:
            filled[y,x,0:3] = np.maximum(0, np.array(filled[y,x,0:3]) - 14)
right_dark = np.array(KRAFT_DARKER)
for x in range(x1-8, x1):
    for y in range(y0, y1):
        if filled[y,x,3] > 10:
            filled[y,x,0:3] = np.maximum(0, np.array(filled[y,x,0:3]) - 14)

bag2 = Image.fromarray(filled, "RGBA")

# === paste real label, contained, centered at detected AI cream label box ===
# find cream label box (within bag bbox only)
gray = 0.299*r + 0.587*g + 0.114*bv
cream_mask = (gray > 215) & (np.maximum(np.maximum(r,g),bv) - np.minimum(np.minimum(r,g),bv) < 25)
cream_inside = np.zeros_like(cream_mask)
cream_inside[by0+int(bh*0.20):by0+int(bh*0.65), bx0+int(bw*0.18):bx0+int(bw*0.82)] = True
cream_mask = cream_mask & cream_inside
ys2, xs2 = np.where(cream_mask)
if len(xs2) > 100:
    fx,fy = int(xs2.min()), int(ys2.min())
    fw,fh = int(xs2.max()-xs2.min()), int(ys2.max()-ys2.min())
else:
    fx,fy = bx0+int(bw*0.25), by0+int(bh*0.25)
    fw,fh = int(bw*0.50), int(bh*0.45)
print(f"label box: {fx},{fy}  {fw}x{fh}")

# pasted real label COVER-MODE (slightly larger to fully hide AI label residue)
lbl = Image.open(LABEL).convert("RGBA")
la = np.array(lbl)
lys, lxs = np.where(la[:,:,3]>10)
lx0,ly0 = int(lxs.min()), int(lys.min())
lw2, lh2 = int(lxs.max()-lxs.min()), int(lys.max()-lys.min())
lbl_c = lbl.crop((lx0,ly0,lx0+lw2,ly0+lh2))
ratio_l = lw2/lh2
ratio_b = fw/fh
# cover-mode: make label slightly larger than AI label area (1.10x)
if ratio_l > ratio_b:
    tw = int(fw * 1.05)
    th = max(1, int(tw/ratio_l))
else:
    th = int(fh * 1.05)
    tw = max(1, int(th*ratio_l))
print(f"label cover: {tw}x{th} (ratio {ratio_l:.3f})")
lbl_s = lbl_c.resize((tw,th), Image.LANCZOS)
px = fx + (fw - tw)//2
py = fy + (fh - th)//2
bag2.alpha_composite(lbl_s, (px,py))

# make outside-bag transparent
final = np.array(bag2)
mask_in = np.zeros((H,W), dtype=bool)
mask_in[by0:by0+bh, bx0:bx0+bw] = True
mask_in = ndi.binary_erosion(mask_in, iterations=2)
final_alpha = np.where(mask_in, final[:,:,3], np.uint8(0))
bag3 = Image.fromarray(np.dstack([final[:,:,:3], final_alpha[:,:,None]]), "RGBA")

out_path = os.path.join(OUT, "product_front_real.png")
bag3.save(out_path, "PNG")
print("SAVED", out_path, bag3.size)

ys3,xs3 = np.where(np.array(bag3)[:,:,3]>30)
print("BAG bbox:", (int(xs3.min()),int(ys3.min()),int(xs3.max()),int(ys3.max())), "w", int(xs3.max()-xs3.min()), "h", int(ys3.max()-ys3.min()))
