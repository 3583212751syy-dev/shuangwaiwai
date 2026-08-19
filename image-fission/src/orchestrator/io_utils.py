"""文件暂存与落盘工具。"""
import os
import shutil
from config import COMFYUI_DIR, JOBS_DIR, OUTPUT_FORMAT, OUTPUT_QUALITY

INPUT_DIR = os.path.join(COMFYUI_DIR, "input")
OUTPUT_ROOT = os.path.join(COMFYUI_DIR, "output")


def stage_input(original_path: str, job_id: str) -> str:
    """把原图拷到 ComfyUI/input，返回工作流引用的文件名（含相对路径）。"""
    os.makedirs(INPUT_DIR, exist_ok=True)
    ext = os.path.splitext(original_path)[1] or ".png"
    fname = f"{job_id}_orig{ext}"
    shutil.copyfile(original_path, os.path.join(INPUT_DIR, fname))
    return fname


def save_result(bytes_data: bytes, job_id: str, batch: int, idx: int) -> str:
    """把生成的图字节存为 JPG 到 jobs/<job_id>/。返回路径。"""
    from PIL import Image
    import io
    d = os.path.join(JOBS_DIR, job_id)
    os.makedirs(d, exist_ok=True)
    im = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    out = os.path.join(d, f"b{batch:03d}_{idx}.{OUTPUT_FORMAT}")
    im.save(out, OUTPUT_FORMAT.upper(), quality=OUTPUT_QUALITY)
    return out
