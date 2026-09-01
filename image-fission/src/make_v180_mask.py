"""为 v179b 生成顶部 UPGY 字母区 inpaint mask（白色=要重绘，黑色=保留）。

v179b 实际像素：768x1152（ComfyUI SDXL 默认 ~768x1152）。
UPGY 字母区大致在 y=0.00-0.20 高度区间 + x=0.18-0.82 宽度区间。
羽化 mask 边缘 30px 让 ControlNet inpaint 接缝更自然。"""
from PIL import Image, ImageFilter
from pathlib import Path

ROOT = Path("E:/Desktop/双接口/image-fission")
src = Image.open(ROOT / "outputs/v179b/v179b_denim_3.jpg").convert("RGB")
W, H = src.size
print(f"v179b size = {W}x{H}")

mask = Image.new("L", (W, H), 0)  # 全黑
# 顶部 UPGY 字母矩形区：x=18%..82%, y=0%..20%
x0 = int(W * 0.18); x1 = int(W * 0.82)
y0 = int(H * 0.00); y1 = int(H * 0.20)
# 白色矩形（要重绘的区）
from PIL import ImageDraw
ImageDraw.Draw(mask).rectangle([x0, y0, x1, y1], fill=255)
# 羽化边缘 30px
mask = mask.filter(ImageFilter.GaussianBlur(30))
out = ROOT / "outputs/v179b/mask_top_letters.png"
mask.save(out)
print(f"mask saved: {out}  size={W}x{H}  inpaint region=[{x0},{y0}]..[{x1},{y1}]")