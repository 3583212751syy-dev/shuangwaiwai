"""debug burn_text to find why MOONBAT main word doesn't show"""
import sys
sys.path.insert(0, r"E:/Desktop/双接口/image-fission/src")
import traceback
import math

import arc_text
from PIL import Image, ImageDraw, ImageFont

FONT = r"E:/Desktop/双接口/image-fission/ComfyUI/models/fonts/AbrilFatface-Regular.ttf"
INK = (26, 10, 31)


def burn_text_debug(img, arc, big, sub, color=INK):
    print(f"\n[burn] arc={arc!r} big={big!r} sub={sub!r}")
    img = img.convert("RGB")
    w, h = img.width, img.height
    print(f"[burn] img size {w}x{h}")
    draw = ImageDraw.Draw(img)

    # ARC
    fs_arc = int(w * 0.045)
    radius = int(w * 0.46)
    cy_arc = int(h * 0.13) + radius
    print(f"[burn] arc initial: fs={fs_arc} rad={radius} cy={cy_arc}")
    while fs_arc > 18:
        arc_len = arc_text.fit_arc_text_width(arc, FONT, fs_arc, radius, char_spacing_px=3)
        total_deg = math.degrees(arc_len / radius)
        print(f"[burn] arc try fs={fs_arc} total_deg={total_deg:.2f}")
        if total_deg <= 170:
            break
        fs_arc = int(fs_arc * 0.92)
    start = 270 - total_deg / 2
    end = 270 + total_deg / 2
    print(f"[burn] arc final fs={fs_arc} start={start} end={end}")
    img = arc_text.draw_arc_text(img, arc, FONT, fs_arc, color,
                                 center=(w // 2, cy_arc), radius=radius,
                                 start_angle_deg=start, end_angle_deg=end,
                                 char_spacing_px=3, flip_180=False)
    print(f"[burn] arc drawn OK")

    # BIG
    fs_big_target = int(h * 0.145)
    max_w_big = int(w * 0.82)
    lo, hi = 8, fs_big_target
    best = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(FONT, mid)
        if font.getlength(big) <= max_w_big:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    fs_big = best
    font = ImageFont.truetype(FONT, fs_big)
    print(f"[burn] BIG: target={fs_big_target} max_w={max_w_big} final fs={fs_big} len={font.getlength(big):.1f}")
    draw.text((w // 2, int(h * 0.555)), big, font=font, fill=color, anchor="mm")
    print(f"[burn] BIG drawn OK")

    # SUB
    fs_sub_target = int(h * 0.075)
    max_w_sub = int(w * 0.60)
    lo, hi = 8, fs_sub_target
    best = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(FONT, mid)
        if font.getlength(sub) <= max_w_sub:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    fs_sub = best
    font = ImageFont.truetype(FONT, fs_sub)
    print(f"[burn] SUB: target={fs_sub_target} max_w={max_w_sub} final fs={fs_sub} len={font.getlength(sub):.1f}")
    draw.text((w // 2, int(h * 0.655)), sub, font=font, fill=color, anchor="mm")
    print(f"[burn] SUB drawn OK")

    # SIDE
    f_side = ImageFont.truetype(FONT, int(h * 0.028))
    draw.text((int(w * 0.305), int(h * 0.415)), "EST.", font=f_side, fill=color, anchor="mm")
    draw.text((int(w * 0.695), int(h * 0.415)), "1862", font=f_side, fill=color, anchor="mm")
    print(f"[burn] SIDE drawn OK")

    return img


img = Image.new("RGB", (1552, 2000), (183, 127, 171))
try:
    result = burn_text_debug(img, "GUARDIAN OF THE DARK", "MOONBAT", "NOCTURNE")
    result.save(r"E:/Desktop/双接口/image-fission/jobs/v291/_debug_fold.png")
    print(f"[done] saved _debug_fold.png")
except Exception as e:
    traceback.print_exc()
    print(f"[done] FAILED: {e}")
