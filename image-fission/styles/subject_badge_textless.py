"""styles/subject_badge_textless.py — 有主体 + 本应带文字但裂变后文字缺失（鹰骷髅金属）。

用户硬规则（2026-09-12）：
- 主体裂变风格即用户所需（Pinterest(6) 已被明确点赞）→ **保留该主体裂变效果，不可改**
- 只补上文字（金属尖刺字体按原排版/字号替换内容词）

重要工程约束（回归实测）：Pinterest(6) 这张 3543×4961 超长竖图在 mode3(img2img) 路径下
会**可复现地 ComfyUI 执行挂死**（running=1 / models_loaded=[] / 无输出），已重启多次复现。
因此该风格**必须用 mode1**（txt2img + IPAdapter 双锁）来得到用户点赞的主体裂变效果。
"""
from __future__ import annotations
import shutil
from pathlib import Path
from . import base

STYLE_KEY = "subject_badge_textless"
DESCRIPTION = "有主体(文字缺失)：保留点赞的主体裂变效果，补文字"
COVERS = ["pinterest6"]


def default_params() -> dict:
    return {
        "mode": "mode1",                 # 关键：Pinterest(6) mode3 会挂死，必须用 mode1
        "color_strength": 0.60,
        "composition_strength": 0.55,
        "ipadapter_noise": 0.10,
        "controlnet_strength": 0.50,    # Canny 中强：锁主体轮廓，保留用户点赞的裂变形态
        "controlnet_end": 0.90,
        "lab_alpha": 0.85,
        "steps": 28,
        "cfg": 5.0,
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
    prompts = base.get_image_cfg(cfg, image_path).get("prompts") or ["black metal poster, eagle on horned skull, lightning, spiked gothic typography, no real band name"]
    prompt = prompts[0]

    out_dir = Path(out_dir)
    im = base.Image.open(image_path).convert("RGB")
    W, H = im.size
    w4 = max(64, ((W // 4) // 8) * 8)
    h4 = max(64, ((H // 4) // 8) * 8)

    args = [
        "--input", image_path,
        "--mode", p["mode"],
        "--prompts", prompt,
        "--count", str(p["count"]),
        "--out", str(out_dir),
        "--seed", str(p["seed"]),
        "--steps", str(p["steps"]),
        "--cfg", str(p["cfg"]),
        "--width", str(w4),
        "--height", str(h4),
        "--color-strength", str(p["color_strength"]),
        "--composition-strength", str(p["composition_strength"]),
        "--ipadapter-noise", str(p["ipadapter_noise"]),
        "--controlnet-strength", str(p["controlnet_strength"]),
        "--controlnet-end", str(p["controlnet_end"]),
    ]
    rc = base.run_fission_cli(args)

    out_paths = sorted({p for p in out_dir.rglob("*.jpg") if p.is_file()}) if out_dir.exists() else []
    if out_paths:
        ref = base.Image.open(image_path).convert("RGB")
        for op in out_paths:
            gen = base.Image.open(op).convert("RGB").resize((W, H), base.Image.LANCZOS)
            locked = base.lab_color_lock(gen, ref, alpha=p["lab_alpha"])
            locked = _add_text(locked, image_path, cfg)
            base.save_variant(locked, out_dir, op.name)
    print(f"[subject_badge_textless] done rc={rc} variants={len(out_paths)}")
    return out_paths


def _add_text(img: "base.Image.Image", image_path: str, cfg: dict) -> "base.Image.Image":
    """补文字：按 set.json text_plan（金属尖刺字体按原排版/字号替换内容）。"""
    plan = base.get_text_plan_for(cfg, image_path)
    if not plan:
        return img
    return base.replace_text_plan(img, plan, dilate=10)


def selfcheck_notes() -> str:
    return (
        "自检要点：① 主体裂变效果与用户点赞样例一致（不可改）；② 文字已补上且按原排版/字号/金属尖刺字体；"
        "③ 配色锁原色族。注意 mode3 挂死，本风格强制 mode1。"
    )
