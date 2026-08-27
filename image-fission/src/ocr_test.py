import easyocr, cv2, json
from pathlib import Path
from PIL import Image
import numpy as np

reader = easyocr.Reader(['en'], gpu=True)
src = Path(r"E:/Desktop/图裂变测试图")
out = {}
for p in sorted(src.glob("*.jpg")):
    img = np.array(Image.open(str(p)).convert("RGB"))
    h, w = img.shape[:2]
    res = reader.readtext(img, detail=1, paragraph=False)
    boxes = []
    for (bbox, text, conf) in res:
        xs = [pt[0] for pt in bbox]; ys = [pt[1] for pt in bbox]
        boxes.append({"text": text, "conf": round(float(conf), 2),
                      "x1": int(min(xs)), "y1": int(min(ys)),
                      "x2": int(max(xs)), "y2": int(max(ys))})
    out[p.name] = {"w": w, "h": h, "boxes": boxes}
    print(f"\n=== {p.name} ({w}x{h}) ===")
    for b in boxes:
        print(f"  {b['text']!r} conf={b['conf']} box=({b['x1']},{b['y1']})-({b['x2']},{b['y2']})")
json.dump(out, open(r"E:/Desktop/双接口/image-fission/src/ocr_result.json", "w"), ensure_ascii=False, indent=2)
print("\nSAVED ocr_result.json")
