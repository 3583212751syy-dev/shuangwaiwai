"""styles/base.py — 所有风格模块共享的工具函数。

这里只放"与具体风格无关"的底层能力：图像读写、ComfyUI 调用封装、LAB 颜色锁、
LaMa 抹字封装、文字带检测封装、字体度量。风格专属逻辑一律放在各自模块，不进 base。
"""
from __future__ import annotations
import os, sys, subprocess, json, shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 项目根（styles/ 的上两级：image-fission/）
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
VENV_PY = ROOT / "venv" / "Scripts" / "python.exe"
FISSION_CLI = SRC / "fission.py"
COMFYUI_URL = "http://127.0.0.1:8188"

# 字体路径（Windows 无 Bodoni/Didot，用 OFL 替代）
FONTS = {
    "blackopsone": ROOT / "fonts" / "BlackOpsOne-Regular.ttf",
    "playfair": ROOT / "fonts" / "PlayfairDisplay-Bold.ttf",
}

# 把 src 加入 sys.path 以便复用已有工具
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def run_fission_cli(args: list[str], cwd: str | None = None) -> int:
    """调用现有 fission.py CLI（本地 ComfyUI 出图）。返回 exit code。"""
    cmd = [str(VENV_PY), str(FISSION_CLI), *args]
    print(f"[base] run_fission_cli: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=cwd)


def comfyui_ready(timeout_s: int = 5) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=timeout_s) as r:
            return r.status == 200
    except Exception:
        return False


def lab_color_lock(img: Image.Image, ref: Image.Image, alpha: float = 0.85) -> Image.Image:
    """Reinhard LAB 颜色锁：把 img 的色族迁移到 ref（保留 ref 色族 + 部分 img 纹理）。

    alpha=1.0 完全锁成 ref 色；alpha<1 允许轻微色族内偏移（让裂变差异肉眼可见）。
    """
    a = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    b = np.asarray(ref.convert("RGB"), dtype=np.float64) / 255.0
    # 近似 LAB（用简单的均值/标准差迁移，避免额外依赖）
    def stats(x):
        return x.mean(axis=(0, 1)), x.std(axis=(0, 1)) + 1e-6
    ma, sa = stats(a)
    mb, sb = stats(b)
    out = (a - ma) * (sb / sa) + mb
    out = np.clip(out, 0, 1)
    blended = alpha * out + (1 - alpha) * a
    return Image.fromarray((blended * 255).astype(np.uint8), "RGB")


def font_capH_ratio(fp: Path, probe_size: int = 200) -> float:
    """测量字体 capH/fontSize 比例（v324 校准法）。"""
    f = ImageFont.truetype(str(fp), probe_size)
    bb = f.getbbox("A")
    probe_capH = bb[3] - bb[1]
    return probe_capH / probe_size


def target_font_size(capH_target: int, fp: Path, canvas_scale: int = 4) -> int:
    """给定目标 capH（原图像素），返回 4x canvas 下应设的字体字号。"""
    ratio = font_capH_ratio(fp)
    return int(capH_target / ratio * canvas_scale * 0.95)


def detect_text_lines(image_path: str):
    """复用 src/detect_text_lines.py 的程序化文字带检测。

    返回 list[{'bbox':(x0,y0,x1,y1), 'cx','cy','capH'...}]。风格模块如需原位改写文字调用它。
    """
    try:
        import detect_text_lines as dtl
        return dtl.detect_text_lines(image_path)
    except Exception as e:
        print(f"[base] detect_text_lines failed: {e}")
        return []


def lama_inpaint(image_path: str, mask_path: str, out_path: str):
    """复用 src/v268_lama_clean.lama_inpaint 做本地 Big-LaMa 抹字（dilate 8）。"""
    try:
        import v268_lama_clean as lc
        lc.lama_inpaint(image_path, mask_path, out_path)
        return out_path
    except Exception as e:
        print(f"[base] lama_inpaint failed: {e}")
        return None


def render_text_band(img: Image.Image, word: str, bbox: tuple, font_key: str,
                     color=(0, 0, 0)) -> Image.Image:
    """按原文字带 bbox 的同位置/同高度渲染新词（整词居中 + 字距自适应）。

    仅作风格模块调用的标准接口，保证"同排版/同字号/只改内容"。
    """
    x0, y0, x1, y1 = bbox
    band_w = x1 - x0
    capH_target = int((y1 - y0) * 0.78)  # 经验：文字 capH 约占带高 0.78
    fp = FONTS.get(font_key, FONTS["blackopsone"])
    # 4x canvas 超采样后再缩回
    scale = 4
    font_size = target_font_size(capH_target, fp, scale)
    font = ImageFont.truetype(str(fp), font_size)
    canvas = Image.new("RGBA", (img.width * scale, img.height * scale), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    nat_w = d.textlength(word, font=font)
    gap = (band_w * scale - nat_w) / max(1, len(word) - 1) if len(word) > 1 else 0
    d.text((x0 * scale, y0 * scale), word, font=font, fill=color + (255,), spacing=max(0, int(gap)))
    canvas = canvas.resize((img.width, img.height), Image.LANCZOS)
    out = img.convert("RGBA")
    out.alpha_composite(canvas)
    return out.convert("RGB")


def save_variant(img: Image.Image, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    img.save(p, "JPEG", quality=92)
    print(f"[base] saved variant -> {p}")
    return p
