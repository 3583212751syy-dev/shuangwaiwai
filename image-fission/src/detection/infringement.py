"""
侵权元素检测（独立 Python 预步骤，不依赖 ComfyUI 运行）。
- logo / brand / trademark / watermark：YOLO-World（ultralytics 开放词汇）
- 真人肖像：InsightFace FaceAnalysis
输出一张灰度蒙版 PNG（白=需屏蔽/剥离区域），并外扩避免残留。
后续由 ComfyUI 工作流加载该蒙版，在生成时擦除/重绘这些区域，
保证产出图不再带明显侵权元素。
"""
import os
import cv2
import numpy as np

# 延迟导入重依赖，避免 import 即报错
_YOLO = None
_FACE = None


def _load_yolo():
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        # ultralytics 首次会自动下载 yolov8x-world.pt（走代理）
        _YOLO = YOLO("yolov8x-world.pt")
        _YOLO.set_classes(["logo", "brand", "trademark", "watermark"])
    return _YOLO


def _load_face():
    global _FACE
    if _FACE is None:
        from insightface.app import FaceAnalysis
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "ComfyUI", "models", "insightface")
        _FACE = FaceAnalysis(name="antelopev2", root=root)
        _FACE.prepare(ctx_id=0, det_size=(640, 640))
    return _FACE


def _dilate(mask, px):
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    return cv2.dilate(mask, k, iterations=1)


def detect(image_path: str, out_mask_path: str,
           conf: float = 0.25, dilation_frac: float = 0.05,
           detect_logo: bool = True, detect_face: bool = True) -> dict:
    """
    返回：{
      "mask": out_mask_path,
      "regions": [{"type","score","bbox":[x1,y1,x2,y2]}, ...],
      "has_infringement": bool
    }
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    regions = []

    if detect_logo:
        try:
            yolo = _load_yolo()
            res = yolo.predict(image_path, conf=conf, verbose=False)[0]
            for b in res.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                regions.append({"type": "logo/brand", "score": float(b.conf[0]),
                                "bbox": [x1, y1, x2, y2]})
        except Exception as e:
            print(f"[warn] YOLO-World 检测失败（跳过 logo）: {e}")

    if detect_face:
        try:
            face = _load_face()
            faces = face.get(img)
            for f in faces:
                b = f.bbox.astype(int)
                x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                regions.append({"type": "face", "score": float(f.det_score),
                                "bbox": [int(x1), int(y1), int(x2), int(y2)]})
        except Exception as e:
            print(f"[warn] InsightFace 检测失败（跳过 face）: {e}")

    dil = max(1, int(dilation_frac * min(h, w)))
    mask = _dilate(mask, dil)
    os.makedirs(os.path.dirname(out_mask_path) or ".", exist_ok=True)
    cv2.imwrite(out_mask_path, mask)
    return {
        "mask": out_mask_path,
        "regions": regions,
        "has_infringement": len(regions) > 0,
    }


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    out = sys.argv[2] if len(sys.argv) > 2 else "infringement_mask.png"
    r = detect(p, out)
    print(r)
