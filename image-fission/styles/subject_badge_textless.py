"""styles/subject_badge_textless.py — 有主体 + 本应带文字但裂变后文字缺失（鹰骷髅金属）。

用户硬规则（2026-09-12）：
- 主体裂变风格即用户所需（Pinterest(6) 已被明确点赞）→ **保留该主体裂变效果，不可改**
- 只补上文字（金属尖刺字体按原排版/字号替换内容词）

重要工程约束（回归实测）：Pinterest(6) 这张 3543×4961 超长竖图在 mode3(img2img) 路径下
会**可复现地 ComfyUI 执行挂死**（running=1 / models_loaded=[] / 无输出），已重启多次复现。
因此该风格**必须用 mode1**（txt2img + IPAdapter 双锁）来得到用户点赞的主体裂变效果。
"""
from __future__ import annotations
from pathlib import Path
from . import base

STYLE_KEY = "subject_badge_textless"
DESCRIPTION = "有主体(文字缺失)：保留点赞的主体裂变效果，补文字"
COVERS = ["pinterest6"]


def default_params() -> dict:
    return {
        "mode": "mode1",                 # 关键：Pinterest(6) mode3 会挂死，必须用 mode1
        "canny_res": 512,
        "canny_strength": 0.75,
        "tile_strength": 0.45,
        "ipa_weight": 0.55,
        "ipadapter_noise": 0.10,
        "denoise1": 0.52,
        "denoise2": 0.15,
        "steps": 28,
        "cfg": 5.0,
        "color_strength": 0.6,
        "composition_strength": 0.55,
        "lab_alpha": 0.85,
        "count": 1,
        "seed": 8888,
    }


def classify(image_path: str) -> float:
    return 0.5


def fission(image_path: str, out_dir: Path, cfg: dict, seed: int | None = None) -> list[Path]:
    """跑该风格：mode1 出主体裂变（=用户点赞效果）+ 颜色锁 + 补文字。"""
    p = default_params()
    p.update({k: v for k, v in (cfg.get("comfyui_params", {}).get("subject_badge", {}) or {}).items()})
    if seed is not None:
        p["seed"] = seed
    prompts = cfg.get("prompts") or ["black metal poster, eagle on horned skull, lightning, spiked gothic typography, no real band name"]
    prompt = prompts[0]

    out_dir = Path(out_dir)
    args = [
        "--input", image_path,
        "--mode", p["mode"],
        "--prompts", prompt,
        "--count", str(p["count"]),
        "--out", str(out_dir),
        "--seed", str(p["seed"]),
        "--steps", str(p["steps"]),
        "--cfg", str(p["cfg"]),
        "--color-strength", str(p["color_strength"]),
        "--composition-strength", str(p["composition_strength"]),
        "--ipadapter-noise", str(p["ipadapter_noise"]),
    ]
    rc = base.run_fission_cli(args)

    out_paths = sorted(out_dir.glob("*.jpg")) if out_dir.exists() else []
    if out_paths:
        ref = base.Image.open(image_path).convert("RGB")
        for op in out_paths:
            gen = base.Image.open(op).convert("RGB")
            locked = base.lab_color_lock(gen, ref, alpha=p["lab_alpha"])
            locked = _add_text(locked, cfg)
            base.save_variant(locked, out_dir, op.name)
    print(f"[subject_badge_textless] done rc={rc} variants={len(out_paths)}")
    return out_paths


def _add_text(img: Image.Image, cfg: dict) -> Image.Image:
    """补文字：金属尖刺字体按原排版/字号替换内容（ARCHOR→候选）。当前占位。"""
    tr = cfg.get("text_replacements") or {}
    if not tr:
        return img
    print("[subject_badge_textless] text add: not yet implemented (placeholder)")
    return img


def selfcheck_notes() -> str:
    return (
        "自检要点：① 主体裂变效果与用户点赞样例一致（不可改）；② 文字已补上且按原排版/字号/金属尖刺字体；"
        "③ 配色锁原色族。注意 mode3 挂死，本风格强制 mode1。"
    )
