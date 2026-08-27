"""v127 终版核验：裁剪文字带单独 OCR（去掉背景噪声），证明 PIL 烧的字就是目标词。
PIL 用已知字符串绘制，无 AI 歧义；装饰字体 OCR 难，但裁掉背景后应能读出目标词。
"""
import easyocr
from PIL import Image
import numpy as np
from pathlib import Path

OUT = Path(r"C:\Users\lenovo\WorkBuddy\2026-08-24-16-39-13\outputs\v127")
REPORT = OUT / "ocr_crop_report.txt"

# (文件前缀, 期望词, y_ratio起始, 高度倍率)
ITEMS = [
    ("eagle_2", "DOMINION", 0.13, 0.22),
    ("denim_3", "UPCY", 0.10, 0.20),
    ("skull_5", "VENOM", 0.13, 0.20),
    ("metal_6", "MRCHGSR", 0.06, 0.18),
]

def main():
    reader = easyocr.Reader(["en"], gpu=True)
    lines = []
    for pre, exp, yr, hm in ITEMS:
        im = Image.open(OUT / f"{pre}_final.jpg").convert("RGB")
        w, h = im.size
        y0 = int(h * yr)
        y1 = int(h * (yr + hm))
        crop = im.crop((0, y0, w, y1))
        # 放大 2x 提升 OCR
        crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
        arr = np.array(crop)
        res = reader.readtext(arr, detail=0, paragraph=False, contrast_ths=0.1)
        txt = " ".join(res).strip()
        ok = exp.replace(" ", "").upper() in txt.replace(" ", "").upper()
        lines.append(f"{pre}_final.jpg 期望={exp!r} 裁剪OCR={txt!r} 匹配={ok}\n")
        print(lines[-1], flush=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写 {REPORT}")

if __name__ == "__main__":
    main()
