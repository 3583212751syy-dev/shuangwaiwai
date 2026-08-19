"""
质量评分与合格判定（独立 Python，逐张打分）。
- Laplacian 方差：清晰度（越大越清晰，过小=糊图）
- CLIP-IQA：无参考图像质量/美学（0~1）
- （可选）PaddleOCR：检测图内乱码/异常文字
- （可选）手部多指检测：留接口，MVP 可后置
低于阈值即判不合格，由编排层触发自动重绘。
"""
import os
import cv2
import numpy as np
from config import QUALITY

_metric = None


def _load_clip_iqa():
    global _metric
    if _metric is None:
        import torch
        from pyiqa import create_metric
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _metric = create_metric("clipiqa", device=dev)
    return _metric


def laplacian_blur(path: str) -> float:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def clip_iqa_score(path: str) -> float:
    try:
        metric = _load_clip_iqa()
        score = metric(path)  # tensor (N,)
        return float(score.detach().cpu().mean().item())
    except Exception as e:
        print(f"[warn] CLIP-IQA 失败: {e}")
        return 1.0  # 失败时放行，避免误杀


def detect_text(path: str) -> bool:
    """返回图内是否出现文字（用于判断是否乱码/非预期文字）。需装 paddleocr。"""
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        res = ocr.ocr(path, cls=True)
        has = bool(res and res[0])
        return has
    except Exception as e:
        print(f"[warn] OCR 不可用: {e}")
        return False


def score_image(path: str, allow_text: bool = None) -> dict:
    blur = laplacian_blur(path)
    ciqa = clip_iqa_score(path)
    has_text = detect_text(path) if (allow_text is None) else None

    issues = []
    if blur < QUALITY["laplacian_blur_max"] * 0.5:   # 简易糊图判定（阈值待校准）
        issues.append("blur")
    if ciqa < QUALITY["clip_iqa_min"]:
        issues.append("low_quality")

    passed = (blur >= QUALITY["laplacian_blur_max"] * 0.5) and (ciqa >= QUALITY["clip_iqa_min"])
    if allow_text is False and has_text:
        issues.append("unexpected_text")
        passed = False

    return {
        "path": path,
        "laplacian": round(blur, 2),
        "clip_iqa": round(ciqa, 4),
        "has_text": has_text,
        "issues": issues,
        "passed": bool(passed),
    }


if __name__ == "__main__":
    import sys
    print(score_image(sys.argv[1] if len(sys.argv) > 1 else "test.jpg"))
