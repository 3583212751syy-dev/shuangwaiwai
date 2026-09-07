"""Sanity: import v292's _detect_bat_mask, run on real image, check sum"""
import sys
sys.path.insert(0, r"E:/Desktop/双接口/image-fission/src")
import importlib
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# Force fresh import
if "v292_pipeline" in sys.modules:
    del sys.modules["v292_pipeline"]
if "v291_pipeline" in sys.modules:
    del sys.modules["v291_pipeline"]

import v292_pipeline

# Override constants if necessary
v292_pipeline.ORIG = "6978fabda2cc99629fa9e81f802762d3.jpg"

img = Image.open(Path(r"E:/Desktop/双接口/image-fission/ComfyUI/input") / v292_pipeline.ORIG).convert("RGB")
arr = np.array(img).astype(np.float32)
bat_mask = v292_pipeline._detect_bat_mask(arr)
print(f"bat_mask sum: {int(bat_mask.sum()/255)}px")
print(f"bat_mask shape: {bat_mask.shape}")
print(f"badge r: {v292_pipeline.ring_outer + 30} (= ring_outer+30)")

# Visualize
viz = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
viz[bat_mask > 127] = (0, 255, 0)
Image.fromarray(viz).save(r"E:/Desktop/双接口/image-fission/jobs/v292/_debug_bat_viz.png")
print("saved _debug_bat_viz.png")
