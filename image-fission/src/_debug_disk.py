"""对比原图 vs clean source 的内盘区域，定位翅膀被切的步骤。"""
from PIL import Image
import numpy as np

ORIG = "E:/Desktop/双接口/image-fission/ComfyUI/input/6978fabda2cc99629fa9e81f802762d3.jpg"
CLEAN = "E:/Desktop/双接口/image-fission/ComfyUI/input/v299_text_free_source.png"

cx, cy, R = 776, 746, 380

orig = Image.open(ORIG).convert("RGB")
clean = Image.open(CLEAN).convert("RGB")

box = (cx - R, cy - R, cx + R, cy + R)
o_crop = orig.crop(box)
c_crop = clean.crop(box)

W, H = o_crop.width, o_crop.height
canvas = Image.new("RGB", (W * 2 + 20, H), "white")
canvas.paste(o_crop, (0, 0))
canvas.paste(c_crop, (W + 20, 0))
canvas.save("E:/Desktop/双接口/image-fission/jobs/v299/_debug_disk_compare.png")
print("saved _debug_disk_compare.png", canvas.size)
