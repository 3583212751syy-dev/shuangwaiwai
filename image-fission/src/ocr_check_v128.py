"""OCR verification of v128 final outputs (确认 PIL 烧字正确)."""
import os
import numpy as np
from PIL import Image
import easyocr

OUT_FINAL = r"C:\Users\lenovo\WorkBuddy\2026-08-24-16-39-13\outputs\v128"

EXPECTED = {
    "eagle_2":  ("DOMINION", "哥特"),
    "denim_3":  ("UPCY",     "牛仔"),
    "skull_5":  ("VENOM",    "哥特"),
    "metal_6":  ("MRCHGSR",  "金属"),
}

def main():
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    print("OCR 文字验证（v128 PIL 烧字 + 渲染清晰度）")
    print("=" * 80)
    for key, (expected, font) in EXPECTED.items():
        path = os.path.join(OUT_FINAL, f"{key}_final.jpg")
        if not os.path.isfile(path):
            print(f"  [{key}] 缺失: {path}")
            continue
        im = Image.open(path)
        # OCR full + 顶部+底部带状 (text 区域)
        results = reader.readtext(np.array(im), detail=1, paragraph=False)
        # 过滤掉非字母字符 + short noise
        words = []
        for r in results:
            t = r[1].strip()
            if not t or not any(c.isalpha() for c in t): continue
            words.append(t)
        ocr_text = " ".join(words)
        print(f"\n【{key}】 期望: {expected!r}  ({font})")
        print(f"   OCR 全图读出: {ocr_text or '(空)'}")
        # 验证: 字母是否齐全
        exp_set = set(expected.upper())
        ocr_set = set(ocr_text.upper().replace(" ", ""))
        missing = exp_set - ocr_set
        extra = ocr_set - exp_set - {" "}
        verdict = "✅" if not missing else "❌ 缺字: " + ",".join(missing)
        print(f"   结论: {verdict}")

if __name__ == "__main__":
    import sys
    sys.exit(main())
