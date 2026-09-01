"""Reliable OCR re-check for v195 outputs: load via PIL (cv2.imread is broken on these JPGs),
pass the numpy array to easyocr to bypass cv2's file-reader bug."""
import numpy as np
from PIL import Image
from pathlib import Path
import easyocr

PROJECT = Path(__file__).resolve().parents[1]
JOB = PROJECT / "jobs" / "smoke_v195"
IDS = ["camo_classic","floral_bw","denim_patch","palm_camo","skull_snake_rose","eagle_skull_metal"]

reader = easyocr.Reader(['en'], gpu=False, verbose=False)
for cid in IDS:
    p = JOB / f"v195_{cid}.jpg"
    if not p.exists():
        print(f"[skip] {cid}"); continue
    arr = np.array(Image.open(p).convert("RGB"))[:, :, ::-1].copy()  # RGB->BGR array
    res = reader.readtext(arr, detail=1)
    if res:
        items = ", ".join(f'"{t}"({c:.2f})' for _, t, c in res)
        print(f"{cid:18s} ⚠ 残留 {len(res)} 处: {items}")
    else:
        print(f"{cid:18s} ✅ 无可读字符残留")
