"""Build front|v34-pose comparison and front|old-v33|new-v34-pose 3-panel."""
from PIL import Image, ImageDraw, ImageFont

ROOT = r'D:\.workbuddy\2026-08-16-00-13-40\image-fission'
front = Image.open(f'{ROOT}/ComfyUI/input/front_model.jpg').convert('RGB')
new_v34 = Image.open(f'{ROOT}/results/back_v34_pose_back_view_v34_s_00002_.png').convert('RGB')
old_v33 = Image.open(f'{ROOT}/results/back_v33_v23s2024_beige.png').convert('RGB')

TGT_W, TGT_H = 1350, 1800
BG = (181, 164, 142)

def fit(img):
    iw, ih = img.size
    s = min(TGT_W/iw, TGT_H/ih)
    nw, nh = int(iw*s), int(ih*s)
    img2 = img.resize((nw, nh), Image.LANCZOS)
    bg = Image.new('RGB', (TGT_W, TGT_H), BG)
    bg.paste(img2, ((TGT_W-nw)//2, (TGT_H-nh)//2))
    return bg

front_f = fit(front)
new_f = new_v34.resize((TGT_W, TGT_H), Image.LANCZOS)
old_f = old_v33.resize((TGT_W, TGT_H), Image.LANCZOS)

# 2-panel: front | new v34-pose
gap = 30; label_h = 80
total_w = TGT_W * 2 + gap
total_h = TGT_H + label_h
canvas2 = Image.new('RGB', (total_w, total_h), (28, 28, 32))
canvas2.paste(front_f, (0, label_h))
canvas2.paste(new_f, (TGT_W + gap, label_h))
try:
    font = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc', 44)
except Exception:
    font = ImageFont.load_default()
draw = ImageDraw.Draw(canvas2)
for i, (txt, col) in enumerate([('FRONT (input)', (255,255,255)),
                                 ('v34 POSE-MATCHED (NEW DELIVERABLE)', (120,255,120))]):
    x = i * (TGT_W + gap) + 24
    draw.text((x, 20), txt, font=font, fill=col)
view2 = canvas2.resize((canvas2.size[0]//2, canvas2.size[1]//2), Image.LANCZOS)
out2 = f'{ROOT}/results/2panel_front_v34pose.png'
view2.save(out2, optimize=True)
print('saved', out2, view2.size)

# 3-panel: front | old v33 | new v34-pose
total_w3 = TGT_W * 3 + gap * 2
canvas3 = Image.new('RGB', (total_w3, total_h), (28, 28, 32))
canvas3.paste(front_f, (0, label_h))
canvas3.paste(old_f, (TGT_W + gap, label_h))
canvas3.paste(new_f, (2*(TGT_W + gap), label_h))
draw3 = ImageDraw.Draw(canvas3)
for i, (txt, col) in enumerate([('FRONT', (255,255,255)),
                                 ('OLD: v33 + gray-base + beige-bg', (255,200,120)),
                                 ('NEW: v34-pose (hands-in-pockets base)', (120,255,120))]):
    x = i * (TGT_W + gap) + 24
    draw3.text((x, 20), txt, font=font, fill=col)
view3 = canvas3.resize((canvas3.size[0]//2, canvas3.size[1]//2), Image.LANCZOS)
out3 = f'{ROOT}/results/3panel_old_vs_new.png'
view3.save(out3, optimize=True)
print('saved', out3, view3.size)
