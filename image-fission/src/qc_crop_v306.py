# -*- coding: utf-8 -*-
"""v306 detail crops for visual QC."""
import sys
from PIL import Image

JOBS = r"E:\Desktop\双接口\image-fission\jobs\v306"

def crop(src, box, zoom, out):
    im = Image.open(src).crop(box)
    im = im.resize((int(im.width * zoom), int(im.height * zoom)), Image.LANCZOS)
    im.save(out)
    print("saved", out, im.size)

# spread 右翼根（条纹伪影区）
crop(JOBS + r"\v306_spread_body.png", (800, 560, 990, 800), 3.0,
     JOBS + r"\_chk\qc_spread_rwingroot.png")
# spread 腿爪区
crop(JOBS + r"\v306_spread_body.png", (680, 770, 880, 910), 3.5,
     JOBS + r"\_chk\qc_spread_legs.png")
# up 右翼根 + 头
crop(JOBS + r"\v306_up_body.png", (790, 480, 1000, 760), 2.6,
     JOBS + r"\_chk\qc_up_rhead.png")
# fold 左翼根
crop(JOBS + r"\v306_fold_body.png", (550, 560, 760, 800), 2.8,
     JOBS + r"\_chk\qc_fold_lwing.png")
