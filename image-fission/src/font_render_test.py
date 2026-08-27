"""白底渲染核验：用相同字体把目标词画在纯白背景上 OCR，证明字形即目标词（与背景无关）。
"""
import easyocr
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path

FONTS = {
    "metal_mania": r"E:/Desktop/双接口/image-fission/fonts/MetalMania-Regular.ttf",
    "pirata_one":  r"E:/Desktop/双接口/image-fission/fonts/PirataOne-Regular.ttf",
    "rye":         r"E:/Desktop/双接口/image-fission/fonts/Rye-Regular.ttf",
}
WORDS = [
    ("eagle_2", "DOMINION", "pirata_one", (20, 12, 8)),
    ("denim_3", "UPCY", "rye", (50, 60, 90)),
    ("skull_5", "VENOM", "pirata_one", (215, 30, 40)),
    ("metal_6", "MRCHGSR", "metal_mania", (10, 10, 10)),
]

def main():
    reader = easyocr.Reader(["en"], gpu=True)
    for pre, word, font, fill in WORDS:
        fs = 200
        font_obj = ImageFont.truetype(FONTS[font], fs)
        img = Image.new("RGB", (1400, 400), (255, 255, 255))
        d = ImageDraw.Draw(img)
        w = d.textlength(word, font=font_obj)
        d.text(((1400 - w) / 2, 100), word, font=font_obj, fill=fill)
        arr = np.array(img)
        res = reader.readtext(arr, detail=0, paragraph=False, contrast_ths=0.1)
        txt = " ".join(res).strip()
        ok = word.upper() in txt.upper()
        print(f"{pre}: 目标={word!r} 白底OCR={txt!r} 匹配={ok}", flush=True)

if __name__ == "__main__":
    main()
