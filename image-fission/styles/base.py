"""styles/base.py — 所有风格模块共享的工具函数。

这里只放"与具体风格无关"的底层能力：图像读写、ComfyUI 调用封装、LAB 颜色锁、
LaMa 抹字封装、按原排版重绘文字（replace_text_plan）、字体度量。
风格专属逻辑一律放在各自模块，不进 base。
"""
from __future__ import annotations
import os, sys, subprocess, json, shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from skimage.color import rgb2lab, lab2rgb

# 项目根（styles/ 的上两级：image-fission/）
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
VENV_PY = ROOT / "venv" / "Scripts" / "python.exe"
FISSION_CLI = SRC / "fission.py"
COMFYUI_URL = "http://127.0.0.1:8188"

# 字体路径（Windows 无 Bodoni/Didot，用 OFL 替代；Black 版 Playfair 损坏，用 Bold）
FONTS = {
    "blackopsone": ROOT / "fonts" / "BlackOpsOne-Regular.ttf",   # military / stencil display
    "playfair":    ROOT / "fonts" / "PlayfairDisplay-Bold.ttf",  # Didone serif (BACARDÍ 类)
    "metal":       ROOT / "fonts" / "MetalMania-Regular.ttf",    # spiky gothic metal (ARCHOR 类)
    "denim":       ROOT / "fonts" / "LeagueSpartan-Black.ttf",   # bold blocky (UPCY 类)
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
    """真正 Reinhard LAB 颜色锁：把 img 的色族迁移到 ref（保留 ref 色族 + 部分 img 纹理）。

    alpha=1.0 完全锁成 ref 色；alpha<1 允许轻微色族内偏移（让裂变差异肉眼可见）。
    使用 skimage rgb2lab/lab2rgb 在感知均匀空间做均值/方差匹配，避免 RGB 逐通道匹配导致的色相漂移。
    """
    a_rgb = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    b_rgb = np.asarray(ref.convert("RGB"), dtype=np.float64) / 255.0

    def stats(x):
        return x.mean(axis=(0, 1)), x.std(axis=(0, 1)) + 1e-6

    a = rgb2lab(a_rgb)
    b = rgb2lab(b_rgb)
    ma, sa = stats(a)
    mb, sb = stats(b)
    matched = (a - ma) * (sb / sa) + mb
    matched = np.clip(matched, [0, -128, -128], [100, 127, 127])
    blended = alpha * matched + (1 - alpha) * a
    blended = np.clip(blended, [0, -128, -128], [100, 127, 127])
    rgb = (lab2rgb(blended) * 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


# ---------------------------------------------------------------------------
# 文字原位替换：LaMa 抹旧字 + PIL 按原字体/字号/位置重画新词
# ---------------------------------------------------------------------------

def _lama_module():
    """惰性加载 v268_lama_clean（含本地 Big-LaMa）。"""
    import v268_lama_clean as lc
    return lc


def _capH_ratio(fp: Path, probe_size: int = 200) -> float:
    f = ImageFont.truetype(str(fp), probe_size)
    bb = f.getbbox("A")
    return (bb[3] - bb[1]) / probe_size


def _fit_font_size(word: str, fp: Path, target_capH: int, band_w: int, scale: int = 4) -> int:
    """返回在 4x 超采样画布上应使用的字体字号（显示 capH≈target_capH 且宽度不溢出）。"""
    ratio = _capH_ratio(fp)
    S = int(target_capH * scale / ratio * 0.95)
    font = ImageFont.truetype(str(fp), S)
    tw = font.getlength(word)
    disp_w = tw / scale
    if disp_w > band_w * 0.96:
        S = int(S * (band_w * 0.96 * scale) / tw)
    return max(12, S)


def _render_word(img: Image.Image, word: str, bbox, font_key: str, color) -> Image.Image:
    """在 img 的 bbox 内按原排版（同位置/同高度）渲染新词（4x 超采样后缩回，整词居中）。

    对超大图（如 pinterest6 3543×4961）做超采样上限保护，避免一次性构造过大画布吃掉内存。
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    fp = FONTS.get(font_key, FONTS["blackopsone"])
    target_capH = int(bh * 0.78)
    scale = 4
    # 超采样保护：宽度超采样后超过 ~14000px 时降档，控制瞬时内存
    if img.width * scale > 14000:
        scale = max(2, 14000 // img.width)
    S = _fit_font_size(word, fp, target_capH, bw, scale)
    font = ImageFont.truetype(str(fp), S)
    W, H = img.width, img.height
    canvas = Image.new("RGBA", (W * scale, H * scale), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    tw = d.textlength(word, font=font)
    cx = (x1 + x2) / 2 * scale
    cy = (y1 + y2) / 2 * scale
    capH = S * _capH_ratio(fp)
    start_x = cx - tw / 2
    top_y = cy - capH / 2
    d.text((start_x, top_y), word, font=font, fill=tuple(color) + (255,))
    canvas = canvas.resize((W, H), Image.LANCZOS)
    out = img.convert("RGBA")
    out.alpha_composite(canvas)
    return out.convert("RGB")


def replace_text_plan(img: Image.Image, plan: list[dict], dilate: int = 12) -> Image.Image:
    """按 text_plan 逐条原位改写文字：LaMa 抹旧字 → PIL 重画新词。

    plan 每条: {"bbox":[x1,y1,x2,y2], "word":str, "font":"blackopsone|playfair|metal|denim",
                "color":[r,g,b], "dilate":int(可选)}。坐标为原图像素坐标。
    禁止矩形色块遮盖——LaMa 只抹文字区、保周围纹理，PIL 直接贴新字。

    内存安全：对超大图（如 pinterest6 3543×4961）只裁剪文字 bbox 周边区域做 LaMa，
    不在整张大图上跑 inpaint，避免内存峰值把进程打挂。
    """
    if not plan:
        return img
    lc = _lama_module()
    out = img.convert("RGB")
    W, H = out.size
    for item in plan:
        bbox = item.get("bbox")
        word = (item.get("word") or "").strip()
        if not bbox or not word:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        item_dilate = int(item.get("dilate", dilate))
        # 裁剪框：以文字 bbox 为中心，上下各留 margin（取文字高度的 ~0.8 或至少 60px）
        bh = y2 - y1
        margin = max(int(bh * 0.8), 60)
        cx1 = max(0, x1 - margin)
        cy1 = max(0, y1 - margin)
        cx2 = min(W, x2 + margin)
        cy2 = min(H, y2 + margin)
        out_crop = out.crop((cx1, cy1, cx2, cy2))
        # 文字 bbox 相对裁剪框的坐标
        bx1, by1, bx2, by2 = x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1
        # 1) 构建文字区 mask（白=去除），膨胀覆盖笔画边缘
        mask = Image.new("L", out_crop.size, 0)
        mdraw = ImageDraw.Draw(mask)
        mdraw.rectangle([bx1, by1, bx2, by2], fill=255)
        mask = mask.filter(ImageFilter.MaxFilter(item_dilate * 2 + 1))
        # 2) LaMa 只抹裁剪框内文字区（生成匹配背景的纹理）
        try:
            cleaned_crop = lc.lama_inpaint(out_crop, mask, removal_strength=235, edge_smoothness=6)
        except Exception as e:
            print(f"[base] lama_inpaint failed for {bbox}: {e}")
            cleaned_crop = out_crop
        # 3) 按原排版重画新词（坐标相对裁剪框）
        cleaned_crop = _render_word(cleaned_crop, word, [bx1, by1, bx2, by2],
                                    item.get("font", "blackopsone"),
                                    item.get("color", [0, 0, 0]))
        # 4) 贴回原图（裁剪框周边留足 buffer，LaMa 边缘过渡无缝）
        out.paste(cleaned_crop, (cx1, cy1))
    return out


def get_image_cfg(cfg: dict, image_path: str) -> dict:
    """从 set.json 找到当前图对应的完整配置条目（按 path/filename 匹配）。"""
    target = str(image_path)
    for img in cfg.get("images", []):
        if img.get("path") == target or img.get("filename") in target:
            return img
    return {}


def get_text_plan_for(cfg: dict, image_path: str) -> list[dict]:
    """从 set.json 找到当前图对应的 text_plan（按 path/filename 匹配）。"""
    return get_image_cfg(cfg, image_path).get("text_plan") or []


def save_variant(img: Image.Image, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    img.save(p, "JPEG", quality=92)
    print(f"[base] saved variant -> {p}")
    return p
