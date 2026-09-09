"""v319t_fonttest: render 'ARMED' with multiple candidate fonts at matching capH,
side-by-side with the original band crop, to pick the closest match.
"""
import os, sys
from PIL import Image, ImageDraw, ImageFont

ROOT = r"E:\Desktop\双接口\image-fission"
JOB = os.path.join(ROOT, "jobs", "v319t")
os.makedirs(JOB, exist_ok=True)

# Candidate fonts (Windows paths)
CANDIDATES = [
    ("arial_black", r"C:\Windows\Fonts\ariblk.ttf"),
    ("impact", r"C:\Windows\Fonts\impact.ttf"),
    ("stencil", r"C:\Windows\Fonts\stencil.ttf"),
    ("bahnschrift_cond", r"C:\Windows\Fonts\Bahnschrift.ttf"),  # may be variable
    ("arial_narrow_bold", r"C:\Windows\Fonts\ARIALNB.TTF"),
    ("franklin_gothic_bold", r"C:\Windows\Fonts\FRAMD.TTF"),
    ("calibri_bold", r"C:\Windows\Fonts\calibrib.ttf"),
    ("verdana_bold", r"C:\Windows\Fonts\verdanab.ttf"),
    ("georgia_bold", r"C:\Windows\Fonts\georgiab.ttf"),
    ("segoeui_black", r"C:\Windows\Fonts\seguisb.ttf"),
    ("consolas_bold", r"C:\Windows\Fonts\consolab.ttf"),
    ("trebuchet_bold", r"C:\Windows\Fonts\trebucbd.ttf"),
    ("tahoma_bold", r"C:\Windows\Fonts\tahomabd.ttf"),
    ("constantia_bold", r"C:\Windows\Fonts\constanb.ttf"),
    ("cambria_bold", r"C:\Windows\Fonts\cambriab.ttf"),
    ("candara_bold", r"C:\Windows\Fonts\candarab.ttf"),
    ("corbel_bold", r"C:\Windows\Fonts\corbelb.ttf"),
]

# Test word: "ARMED" — match original capH = 99 in a 138 font size target
# Render at fsize=138 with default 'la' anchor so caps fit in 99px band
TEST_WORD = "ARMED"
TARGET_CAP_H = 99  # original ARMED capH
TARGET_FSIZE = 138

# Original ARMED width for reference
ORIG_W = 1322  # original ARMED total width in source

results = []
for name, path in CANDIDATES:
    if not os.path.exists(path):
        continue
    try:
        font = ImageFont.truetype(path, TARGET_FSIZE)
        # Get rendered size
        bbox = font.getbbox(TEST_WORD)  # (l, t, r, b)
        nat_w = bbox[2] - bbox[0]
        # Measure cap height (cap H) by rendering 'A' alone
        cap_bbox = font.getbbox("A")
        cap_h = cap_bbox[3] - cap_bbox[1]
        ascent, descent = font.getmetrics()
        results.append((name, path, nat_w, cap_h, ascent, descent, bbox))
    except Exception as e:
        results.append((name, path, -1, -1, -1, -1, str(e)))

# Print table
print(f"{'font':<22} {'nat_w':>7} {'cap_h':>7} {'ascent':>7} {'descent':>8} {'width_ratio_vs_orig':>22}")
print("-" * 80)
for name, path, nat_w, cap_h, ascent, descent, bbox in results:
    if nat_w > 0:
        ratio = nat_w / ORIG_W
        print(f"{name:<22} {nat_w:>7} {cap_h:>7} {ascent:>7} {descent:>8} {ratio:>22.3f}")
    else:
        print(f"{name:<22} (error: {bbox})")

# Render each into a strip image for visual comparison
strip_h = 140
strip_w = 1400  # wide enough
n_found = sum(1 for r in results if r[2] > 0)
strip = Image.new("RGB", (strip_w, strip_h * n_found), "white")
draw = ImageDraw.Draw(strip)
y = 0
for name, path, nat_w, cap_h, ascent, descent, bbox in results:
    if nat_w < 0:
        continue
    try:
        font = ImageFont.truetype(path, TARGET_FSIZE)
        # draw the word left-aligned
        draw.text((20, y + 20), TEST_WORD, font=font, fill="black")
        # label
        lbl_font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 16)
        draw.text((20, y + 5), f"{name} (w={nat_w} capH={cap_h})", font=lbl_font, fill="red")
    except Exception:
        pass
    y += strip_h

strip_path = os.path.join(JOB, "font_strip_ARMED.jpg")
strip.save(strip_path, quality=92)
print(f"\nStrip saved: {strip_path}")
print(f"Target: cap_h~{TARGET_CAP_H}, nat_w_to_fill={ORIG_W}")
