"""Build a 3-panel comparison: front | cv2 deliverable | CatVTON git-model."""
from PIL import Image, ImageDraw, ImageFont

ROOT = r'D:\.workbuddy\2026-08-16-00-13-40\image-fission'
front = Image.open(f'{ROOT}/ComfyUI/input/front_model.jpg').convert('RGB')
cv2_result = Image.open(f'{ROOT}/results/back_v33_v23s2024_beige.png').convert('RGB')
catvton = Image.open(f'{ROOT}/catvton_back_1350x1800_beige.png').convert('RGB')

# Normalize all to 1350x1800 (portrait) by letterboxing to beige (181,164,142).
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
cv2_f   = fit(cv2_result)
cv2_f   = cv2_f.resize((TGT_W, TGT_H), Image.LANCZOS)
catvton_f = catvton.resize((TGT_W, TGT_H), Image.LANCZOS)

panel_w = TGT_W
panel_h = TGT_H
gap = 30
label_h = 80
total_w = panel_w * 3 + gap * 2
total_h = panel_h + label_h

canvas = Image.new('RGB', (total_w, total_h), (28, 28, 32))
canvas.paste(front_f, (0, label_h))
canvas.paste(cv2_f, (panel_w + gap, label_h))
canvas.paste(catvton_f, (2*(panel_w + gap), label_h))

# Headers
try:
    font = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc', 48)
except Exception:
    font = ImageFont.load_default()
draw = ImageDraw.Draw(canvas)
labels = [('FRONT (input)', (255,255,255)),
          ('cv2 pixel-perfect (DELIVERABLE)', (120,255,120)),
          ('CatVTON git-model (FAILED)', (255,120,120))]
for i, (txt, col) in enumerate(labels):
    x = i * (panel_w + gap) + 24
    draw.text((x, 18), txt, font=font, fill=col)

# Resize for easier viewing (half)
view = canvas.resize((canvas.size[0]//2, canvas.size[1]//2), Image.LANCZOS)
out = f'{ROOT}/results/3panel_front_cv2_catvton.png'
view.save(out, optimize=True)
print('saved', out, view.size)
