"""OCR 自检 v127 终版：验证 PIL 烧字后文字是否清晰可读、拼写正确。
文件为 PNG 内容(.jpg 扩展名) → 用 PIL 读 numpy 送 easyocr。
"""
import easyocr
from PIL import Image
import numpy as np
from pathlib import Path

OUT = Path(r"C:\Users\lenovo\WorkBuddy\2026-08-24-16-39-13\outputs\v127")
REPORT = OUT / "ocr_report.txt"

EXPECT = {
    "eagle_2": "DOMINION",
    "denim_3": "UPCY",
    "skull_5": "VENOM",
    "metal_6": "MRCHGSR",
}

def main():
    reader = easyocr.Reader(["en"], gpu=True)
    imgs = sorted(OUT.glob("*_final.jpg"))
    lines = []
    for im in imgs:
        exp = next((v for k, v in EXPECT.items() if im.name.startswith(k)), None)
        tag = "无字" if exp is None else "有字"
        try:
            arr = np.array(Image.open(im).convert("RGB"))
            res = reader.readtext(arr, detail=0, paragraph=False, contrast_ths=0.3)
            txt = " ".join(res).strip()
            if exp:
                ok = exp.replace(" ", "").upper() in txt.replace(" ", "").upper()
                lines.append(f"{im.name} [{tag}]\n   OCR={txt!r}\n   期望={exp!r}  匹配={ok}\n")
            else:
                lines.append(f"{im.name} [{tag}]\n   OCR={txt!r}\n   期望=无字  (应为空或极少量)\n")
        except Exception as e:
            lines.append(f"{im.name}\n   OCR异常={e!r}\n")
        print(lines[-1], flush=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写 {REPORT}")

if __name__ == "__main__":
    main()
