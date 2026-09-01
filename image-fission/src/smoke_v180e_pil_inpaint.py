"""v180e 纯 PIL/OpenCV inpaint 后期（放弃 ComfyUI ControlNet inpaint 路线）：

4 次 ComfyUI inpaint 翻车根因：
  · destitech controlnet-inpaint-dreamer-sdxl 是 outpainting 模型,
    对「局部小区域去字母」任务不适用
  · denoise=1.0 时 SetLatentNoiseMask 被忽略,SDXL 重绘整图(蝴蝶变金黄)
  · denoise=0.85 时 mask 部分生效但 D 字母顽固保留(原图 Canny 边缘锁定太死)
  · 调 mask 范围/prompt/grow_mask_by 都改不掉 D 字母

v180e 改用 OpenCV Telea inpaint 算法（社区标准后期 inpaint）：
  1. 输入 v179b 输出（牛仔布料蝴蝶到位）
  2. mask 覆盖顶部 UPGY 字母区 (x=0..1, y=0..0.25)
  3. cv2.inpaint (Telea 算法) 智能填充 mask 区,根据周围像素自然过渡
  4. 输出 v180e_denim_3.jpg

无需 SDXL 调用，~5s 完成，无 AI 字母顽固问题。
"""
import sys, shutil
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs" / "v180e"
OUT.mkdir(parents=True, exist_ok=True)
FINAL = HERE.parent / "outputs" / "final"
GALLERY = HERE.parent / "web_gallery" / "img"

SRC_IMG = HERE.parent / "outputs" / "v179b" / "v179b_denim_3.jpg"
assert SRC_IMG.exists()

src = Image.open(SRC_IMG).convert("RGB")
W, H = src.size
print(f"[v180e] 输入 v179b = {W}x{H}")

# 生成 mask：覆盖顶部 UPGY 字母区（x=0..1, y=0..0.25 全宽覆盖）
mask = Image.new("L", (W, H), 0)
x0 = 0; x1 = W
y0 = 0; y1 = int(H * 0.25)
ImageDraw.Draw(mask).rectangle([x0, y0, x1, y1], fill=255)
mask = mask.filter(ImageFilter.GaussianBlur(20))  # 轻度羽化让 inpaint 边缘自然
mask.save(OUT / "mask_used.png")
print(f"[v180e] mask 覆盖顶部 25% 高度 + 全宽 (x=0..1, y=0..0.25)")

# 用 OpenCV Telea inpaint 算法填充 mask 区
img_np = np.array(src)
mask_np = np.array(mask)

if HAS_CV2:
    print(f"[v180e] 用 OpenCV cv2.inpaint (Telea 算法, inpaintRadius=5)")
    inpainted = cv2.inpaint(img_np, mask_np, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    out_img = Image.fromarray(inpainted)
else:
    print(f"[v180e] OpenCV 不可用，回退到 PIL 填白 + 模糊")
    # PIL fallback：把 mask 区填白 + 边缘模糊过渡
    arr = np.array(src).copy()
    mask_arr = np.array(mask)[:, :, None] / 255.0  # 0..1
    white = np.ones_like(arr) * 255
    out_arr = (arr * (1 - mask_arr) + white * mask_arr).astype(np.uint8)
    out_img = Image.fromarray(out_arr)
    # 边缘 blur
    out_img = out_img.filter(ImageFilter.GaussianBlur(8))

# USM 锐化
out_img = out_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

out = OUT / "v180e_denim_3.jpg"
out_img.save(out, quality=95)
print(f"[v180e] ✓ {out} {out.stat().st_size/1024/1024:.1f}MB")

# 复制到 final + gallery
shutil.copyfile(out, FINAL / "denim_3.jpg")
shutil.copyfile(out, GALLERY / "fiss_denim_3.jpg")
print(f"[v180e] ✓ denim_3 → final + gallery")

# 自检
from selfcheck_metrics import color_intersect, ssim, frag_ratio
orig = GALLERY / "orig_denim_3.jpg"
ci = color_intersect(str(orig), str(out))
sm = ssim(str(orig), str(out))
fr = frag_ratio(str(orig), str(out))
print(f"[v180e] denim_3 metrics: color∩={ci:.3f}  ssim={sm:.3f}  fragR={fr:.3f}")

print(f"\n[v180e] DONE  顶部 OpenCV inpaint 智能填充（无 AI 字母顽固问题）")