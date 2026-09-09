"""v320: generate position masks (RGBA white text on transparent) for AnyText2 text-editing.

For each image we render the NEW text at the ORIGINAL text position/size so AnyText2
generates new content in the same location and (matching) style as the reference.

Mask format: RGBA, text pixels (255,255,255,255), background (0,0,0,0).
ComfyUI LoadImage MASK = alpha channel = white text on black  -> correct for AnyText2.
"""
import os
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = r"E:\Desktop\双接口\image-fission\fonts"
# Serif for BACARDÍ (Didone-like), military for eagle, heavy for denim
PLAYFAIR = os.path.join(FONT_DIR, "PlayfairDisplay-Bold.ttf")
BLACKOPS = os.path.join(FONT_DIR, "BlackOpsOne-Regular.ttf")
ARCHIVO = os.path.join(FONT_DIR, "ArchivoBlack-Regular.ttf")
ARIAL = r"C:\Windows\Fonts\arial.ttf"

OUT = r"E:\Desktop\双接口\image-fission\ComfyUI\input"
os.makedirs(OUT, exist_ok=True)


def make_mask(W, H, items, out_name):
    """items: list of (text, font_path, target_w, cx, cap_top_y)
    Renders white text on transparent RGBA at given position/size."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for text, font_path, target_w, cx, cap_top_y in items:
        # find fsize so rendered text width ~ target_w
        fsize = 40
        font = ImageFont.truetype(font_path, fsize)
        nat_w = d.textlength(text, font=font)
        fsize = int(fsize * target_w / max(nat_w, 1))
        font = ImageFont.truetype(font_path, fsize)
        nat_w = d.textlength(text, font=font)
        cap_h = font.getbbox("A")[3] - font.getbbox("A")[1]
        # center horizontally, cap top at cap_top_y
        x = int(cx - nat_w / 2)
        # draw with anchor 'la' (left-ascender); cap top = y + bbox_A[1]
        bbox_A = font.getbbox("A")
        y = int(cap_top_y - bbox_A[1])
        d.text((x, y), text, font=font, fill=(255, 255, 255, 255))
        print(f"  '{text}': fsize={fsize} w={nat_w:.0f}/{target_w} capH={cap_h} x={x} y={y}")
    img.save(os.path.join(OUT, out_name))
    print(f"  saved {out_name} ({W}x{H})")


# ---- Image 1: BACARDÍ 1552x2000 ----
# main BACARDI: y=820-920 (capH~98), centered x=776, width~758
# sub  MCKEART: y=1003-1145 (capH~78), centered x=776, width~927
make_mask(1552, 2000, [
    ("NOCTAVEN", PLAYFAIR, 720, 776, 822),
    ("DARK RESERVE", PLAYFAIR, 880, 776, 1005),
], "mask_bacardi.png")

# ---- Image 2: EAGLE 964x1280 ----
# banner JACKE DIANNIES: y=780-880 (capH~78), centered x=482, width~560
# bottom TALIBAN LONDON DIT: y=970-1080 (capH~64), centered x=482, width~780
make_mask(964, 1280, [
    ("IRON EAGLES", BLACKOPS, 540, 482, 782),
    ("BORN TO RIDE", BLACKOPS, 760, 482, 972),
], "mask_eagle.png")

# ---- Image 3: DENIM 736x1308 ----
# UPGY: y=80-350 (capH~200), centered x=368, width~560
make_mask(736, 1308, [
    ("LUCKY", ARCHIVO, 560, 368, 85),
], "mask_denim.png")

print("ALL MASKS DONE")
