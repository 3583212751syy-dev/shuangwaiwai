"""OCR 自检 v126：用 PIL 读图(文件实为 PNG 内容+.jpg 扩展名) → numpy 数组送 easyocr。
不使用 allowlist，诚实读取 AI 实际画出的字符。
"""
import easyocr
from PIL import Image
import numpy as np
from pathlib import Path

JOB = Path(r"E:\Desktop\双接口\image-fission\jobs\smoke_v126_1787808430")
OUT = Path(r"E:\Desktop\双接口\image-fission\jobs\smoke_v126_1787808430\ocr_report.txt")

EXPECT = {
    "eagle_2": "DOMINION",
    "denim_3": "UPCY",
    "skull_5": "VENOM",
    "metal_6": "MRCHGSR",
}

def main():
    reader = easyocr.Reader(["en"], gpu=True)
    imgs = sorted([p for p in JOB.glob("*.jpg") if p.name != ".presented_marker"])
    lines = []
    for im in imgs:
        exp = next((v for k, v in EXPECT.items() if k in im.name), "")
        try:
            arr = np.array(Image.open(im).convert("RGB"))
            res = reader.readtext(arr, detail=0, paragraph=False, contrast_ths=0.3)
            txt = " ".join(res).strip()
            ok = exp and exp.replace(" ", "").upper() in txt.replace(" ", "").upper()
            lines.append(f"{im.name}\n   OCR={txt!r}\n   期望={exp!r}  匹配={ok}\n")
        except Exception as e:
            lines.append(f"{im.name}\n   OCR异常={e!r}\n")
        print(lines[-1], flush=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写 {OUT}")

if __name__ == "__main__":
    main()
