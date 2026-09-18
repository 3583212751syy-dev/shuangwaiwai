#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单株棕榈大尺寸诊断"""
import sys
from pathlib import Path

from PIL import Image

ROOT = Path("E:/Desktop/双接口/image-fission")
sys.path.insert(0, str(ROOT))
from styles.palm_draw import make_palm              # noqa: E402

OUT = ROOT / "jobs" / "v364_study"
H = 500
cfgs = [dict(seed=201, kind='solid', crown_scale=1.0),
        dict(seed=207, kind='solid', crown_scale=1.05),
        dict(seed=202, kind='scribble', crown_scale=1.0),
        dict(seed=208, kind='scribble', crown_scale=0.95)]
ims = [make_palm(H, **c).convert('RGB') for c in cfgs]
gap = 20
W = sum(i.width for i in ims) + gap * (len(ims) + 1)
sh = Image.new('RGB', (W, H + 2 * gap), (240, 240, 240))
x = gap
for i in ims:
    sh.paste(i, (x, gap))
    x += i.width + gap
sh = sh.resize((int(sh.width * 1.5), int(sh.height * 1.5)), Image.LANCZOS)
sh.save(OUT / "palm_big.jpg", quality=95)
print(sh.size)
