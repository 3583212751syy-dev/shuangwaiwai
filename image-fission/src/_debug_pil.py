"""Quick debug: PIL draw.text 在 comfUI 环境是否正常？"""
import sys
sys.path.insert(0, r"E:/Desktop/双接口/image-fission/src")
from PIL import Image, ImageDraw, ImageFont

FONT = r"E:/Desktop/双接口/image-fission/ComfyUI/models/fonts/AbrilFatface-Regular.ttf"

img = Image.new("RGB", (1552, 2000), (183, 127, 171))
draw = ImageDraw.Draw(img)
font = ImageFont.truetype(FONT, 280)
print(f"[debug] font size 280 getlength NOCTAVEN = {font.getlength('NOCTAVEN'):.1f}")
draw.text((776, 1110), "NOCTAVEN", font=font, fill=(26, 10, 31), anchor="mm")
img.save(r"E:/Desktop/双接口/image-fission/jobs/v291/_debug_draw_text.png")
print(f"[debug] saved")

# Test if AbrilFatface is corrupt:
import os
size = os.path.getsize(FONT)
print(f"[debug] FONT size = {size} bytes")

font2 = ImageFont.truetype(FONT, 100)
print(f"[debug] font2 size 100 getlength M = {font2.getlength('M'):.1f}")
