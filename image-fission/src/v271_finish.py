"""手动完成 v271 的 composite + burn_text."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import importlib.util

PROJECT = Path("E:/Desktop/双接口/image-fission")
spec = importlib.util.spec_from_file_location("v271", str(PROJECT / "src" / "v271_pipeline.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

style_ref = Image.open(mod.COMFY_INPUT / "v271_style_ref.png").convert("RGB")
bat_mask_im = Image.open(mod.COMFY_INPUT / "v271_bat_mask.png").convert("L")
bat_mask = (np.array(bat_mask_im) > 127)

variants = [
    ("up", "NIGHTBAT", "SHADOW OF THE WING", "ECHO HUNT"),
    ("spread", "DUSKBAT", "WINGS OF TWILIGHT", "SILENT FLIGHT"),
    ("fold", "MOONBAT", "GUARDIAN OF THE DARK", "SONAR OATH"),
]

finals = []
for tag, big, arc, sub in variants:
    raw = Image.open(mod.COMFY_OUTPUT / f"v271_{tag}_00001_.png").convert("RGB")
    comp = mod.composite_bat(style_ref, raw, bat_mask)
    final = mod.burn_text(comp, big, arc, sub)
    out = mod.JOB / f"v271_{tag}_final.png"
    final.save(out, quality=95)
    finals.append(out)
    print(f"saved {out}")

if finals:
    imgs = [Image.open(p).convert("RGB") for p in finals]
    w, h = imgs[0].size
    gap = 14
    grid = Image.new("RGB", (w * len(imgs) + gap * (len(imgs) + 1), h), "white")
    for i, im in enumerate(imgs):
        grid.paste(im, (gap + i * (w + gap), 0))
    grid.save(mod.JOB / "_grid_v271.png", quality=92)
    print(f"saved {mod.JOB / '_grid_v271.png'}")
