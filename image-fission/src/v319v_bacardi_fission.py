"""
v319v_bacardi: Purple bat badge fission (final)
Source: BACARDI purple bat badge (1552x2000)
Strategy:
  - Use connected-component filtered text mask (purple R<130 G<30 B 90-160 + black R<30)
  - Inpaint with NS (Navier-Stokes) preserves background better than TELEA on flat purple
  - Render new text at original position/size with Black Ops One (matching style)
  - Keep bat body untouched (text mask excludes it)
"""
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

JOB = r"E:\Desktop\双接口\image-fission\jobs\v319v_bat_style"
SRC = os.path.join(JOB, "source_bacardi.jpg")
OUT = os.path.join(JOB, "fission_v319v.jpg")
OUT_CLEAN = os.path.join(JOB, "fission_v319v_cleaned.png")
OUT_CMP = os.path.join(JOB, "fission_v319v_compare.jpg")

FONT_BLACK = r"E:\Desktop\双接口\image-fission\fonts\BlackOpsOne-Regular.ttf"
FONT_NARROW_BOLD = r"C:\Windows\Fonts\ARIALNB.TTF"

def cv2_read(path):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def cv2_write(path, img):
    ext = os.path.splitext(path)[1]
    cv2.imencode(ext, img)[1].tofile(path)


def get_text_mask(arr_bgr):
    """Detect text pixels by color: pure purple (R<130 G<30 B 90-160) + black (R<30).
    Filter by CC shape: keep only text-like components (area 20-8000, aspect 0.1-10)."""
    rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    text_mask = ((r < 130) & (g < 30) & (b > 90) & (b < 160)) | ((r < 30) & (g < 30) & (b < 30))
    text_mask |= ((r < 110) & (g < 35) & (b > 100) & (b < 160))
    mask_u = text_mask.astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u, connectivity=8)
    keep = np.zeros_like(text_mask, dtype=bool)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 20 or area > 8000: continue
        aspect = w / max(h, 1)
        if aspect > 10 or aspect < 0.05: continue
        keep[labels == i] = True
    # Dilate slightly to capture anti-alias fringe
    keep_u = (keep.astype(np.uint8)) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    keep_u = cv2.dilate(keep_u, k, iterations=1)
    return keep_u


def inpaint_ns(arr_bgr, mask, radius=3):
    return cv2.inpaint(arr_bgr, mask, radius, cv2.INPAINT_NS)


def draw_arc_text(canvas_rgb, text, font_path, fsize, cx, cy, arc_radius,
                  sweep_deg=80, color=(40, 25, 60), arc_up=True):
    """Draw text along an arc. arc_up=True means text curves upward (cap points up)."""
    font = ImageFont.truetype(font_path, fsize)
    # Measure all char widths
    char_data = []
    for ch in text:
        if ch == " ":
            char_data.append(("sp", fsize // 2))
            continue
        bbox = font.getbbox(ch)
        cw = bbox[2] - bbox[0]
        ch_h = bbox[3] - bbox[1]
        layer = Image.new("RGBA", (cw + 10, ch_h + 10), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.text((5 - bbox[0], 5 - bbox[1]), ch, font=font, fill=color)
        layer = layer.crop(layer.getbbox())
        char_data.append(("ch", layer))
    # Total width
    total_w = sum(c[1].size[0] for c in char_data if c[0] == "ch") + \
              sum(c[1] for c in char_data if c[0] == "sp") + \
              6 * len(char_data)  # inter-char spacing
    # Walk along arc
    arc_len = np.deg2rad(sweep_deg) * arc_radius
    scale = arc_len / max(total_w, 1)
    if arc_up:
        start_deg = -90 - sweep_deg / 2
    else:
        start_deg = -90 + sweep_deg / 2
    cur_angle = start_deg
    for kind, val in char_data:
        if kind == "ch":
            cw, ch_h = val.size
            half_angle = (cw * scale / 2) / arc_radius
            mid_angle = cur_angle + half_angle
            rad = np.deg2rad(mid_angle)
            px = cx + arc_radius * np.cos(rad)
            py = cy + arc_radius * np.sin(rad)
            rot_angle_deg = -mid_angle - 90 if arc_up else -mid_angle + 90
            rot = val.rotate(rot_angle_deg, resample=Image.BICUBIC, expand=True)
            rw, rh = rot.size
            canvas_rgb.paste(rot, (int(px - rw/2), int(py - rh/2)), rot)
            cur_angle += 2 * half_angle + (6 * scale) / arc_radius
        else:
            cur_angle += val * scale / arc_radius


def draw_text_centered(canvas_rgb, text, font_path, fsize, cx, cy, color=(40, 25, 60),
                       scale_x=1.0, max_w=None):
    """Draw text centered at (cx, cy). Optional scale_x for horizontal stretch."""
    font = ImageFont.truetype(font_path, fsize)
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad = 10
    layer = Image.new("RGBA", (text_w + 2 * pad, text_h + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=color)
    layer = layer.crop(layer.getbbox())
    lw, lh = layer.size
    if scale_x != 1.0 or (max_w and lw > max_w):
        if max_w and lw > max_w:
            scale_x = max_w / lw
        new_w = int(round(lw * scale_x))
        layer = layer.resize((new_w, lh), Image.LANCZOS)
        lw = new_w
    paste_x = cx - lw // 2
    paste_y = cy - lh // 2
    canvas_rgb.paste(layer, (paste_x, paste_y), layer)
    return lw, lh, paste_x, paste_y


def main():
    arr = cv2_read(SRC)
    H, W = arr.shape[:2]
    print(f"Source: {W}x{H}")

    # Step 1: text mask + inpaint
    mask = get_text_mask(arr)
    n_text = (mask > 0).sum()
    print(f"Text pixels detected: {n_text}")
    cleaned = inpaint_ns(arr, mask, radius=3)
    cv2_write(OUT_CLEAN, cleaned)
    print(f"Saved cleaned: {OUT_CLEAN}")

    # Step 2: render new text on cleaned canvas
    canvas = Image.open(OUT_CLEAN).convert("RGB")
    # Color: same dark purple as original text
    color = (60, 30, 90)

    # Band 1: top arc "LA CASA DEL VAMPIRO" at y_center ~ 380, arc_radius ~ 530, cx=776
    # Original arc text is at y=320-580, mid_y ~ 450. But "Est." is at y=600-700.
    # Actually looking at mask image: arc top is y~320, arc bottom y~510.
    # Let me look at original arc shape: text starts at y=320 and curves down to y=510
    # This is an UPPER-curved arc (opens downward), so cap points UP = arc_up=True
    # cx of arc: average of x extent (centroid of text)
    # Approx arc center: x=776 (center), radius ~600, arc covers from y=320 to y=510
    # That means arc_radius = (510 - 320) / 2 * something... let me just use direct coords

    # Top arc
    draw_arc_text(canvas, "LA CASA DEL VAMPIRO", FONT_BLACK, 60,
                  cx=W//2, cy=200, arc_radius=580, sweep_deg=85,
                  color=color, arc_up=True)

    # "Est." left side at y~640, cx~440
    draw_text_centered(canvas, "EST.", FONT_NARROW_BOLD, 60, cx=480, cy=640, color=color)

    # "1862" right side at y~640, cx~1100
    draw_text_centered(canvas, "1862", FONT_NARROW_BOLD, 60, cx=1070, cy=640, color=color)

    # Main brand "NOCTAVEN" at y~870 (replacing BACARDI), cx=776
    # Original BACARDI is at y=823-906 (h=84), main visual center ~870
    draw_text_centered(canvas, "NOCTAVEN", FONT_BLACK, 170, cx=W//2, cy=870, color=color)

    # Sub brand "DUSKBAT" at y~1075 (replacing MCKEART), cx=776
    draw_text_centered(canvas, "DUSKBAT", FONT_BLACK, 100, cx=W//2, cy=1075, color=color)

    canvas.save(OUT, quality=92)
    print(f"Saved: {OUT}")

    # Side-by-side compare
    src_pil = Image.open(SRC).convert("RGB")
    panel_w = 600
    panel_h = int(panel_w * H / W)
    big = Image.new("RGB", (panel_w * 2 + 20, panel_h + 40), "white")
    d = ImageDraw.Draw(big)
    for i, (label, img) in enumerate([("ORIGINAL (BACARDI purple bat)", src_pil),
                                       ("FISSION v319v (NOCTAVEN bat)", canvas)]):
        crop = img.resize((panel_w, panel_h))
        big.paste(crop, (i * (panel_w + 20), 40))
        d.text((i * (panel_w + 20) + 10, 10), label, fill="red")
    big.save(OUT_CMP, quality=85)
    print(f"Saved: {OUT_CMP}")


if __name__ == "__main__":
    main()
