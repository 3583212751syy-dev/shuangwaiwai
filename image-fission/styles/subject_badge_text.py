"""styles/subject_badge_text.py — 有主体 + 文字（蝙蝠徽章 / 牛仔蝴蝶）。

用户硬规则（2026-09-12）：
- 主体物种不变（蝙蝠还是蝙蝠、蝴蝶还是蝴蝶）
- 改角度 / 姿态 / 细节裂变（异内容同构）
- 文字按原图字体 / 字号 / 排版位置**只改内容单词**（原位改写，禁色块遮盖）
- 侵权品牌字（BACARDÍ/MCKHEART 等）必须原位改写成占位词（NOCTAVEN/DUSKBAT/MOONBAT 等）

本模块独立持有该风格的 ComfyUI 参数（主体保身份、适度裂变）与文字替换策略。
字体：Didone serif → PlayfairDisplay-Bold；military display → BlackOpsOne。
"""
from __future__ import annotations
from pathlib import Path
from . import base

STYLE_KEY = "subject_badge_text"
DESCRIPTION = "有主体+文字：主体物种不变改角度细节、文字按原排版替换"
COVERS = ["6978", "pinterest3"]


def default_params() -> dict:
    return {
        "mode": "mode3",                 # img2img + 双锁：保主体身份 + 结构
        "redraw_amount": 0.50,          # 比 v324 baseline 0.45 略高让角度/细节真正裂变
        "canny_res": 768,
        "canny_strength": 0.55,
        "tile_strength": 0.40,
        "ipa_weight": 0.55,
        "ipadapter_noise": 0.08,
        "denoise1": 0.50,               # 抬高：让主体角度/姿态裂变但不换物种
        "denoise2": 0.15,
        "steps": 28,
        "cfg": 5.0,
        "color_strength": 0.6,
        "composition_strength": 0.55,
        "lab_alpha": 0.85,              # 强锁原色族
        "count": 1,
        "seed": 8888,
    }


def classify(image_path: str) -> float:
    return 0.5


def fission(image_path: str, out_dir: Path, cfg: dict, seed: int | None = None) -> list[Path]:
    """跑主体类元素裂变：保物种改角度细节 + 颜色锁 + 文字原位替换。"""
    p = default_params()
    p.update({k: v for k, v in (cfg.get("comfyui_params", {}).get("subject_badge", {}) or {}).items()})
    if seed is not None:
        p["seed"] = seed
    prompts = cfg.get("prompts") or ["vintage emblem, same subject redesigned at new angle, no real brand"]
    prompt = prompts[0]

    out_dir = Path(out_dir)
    args = [
        "--input", image_path,
        "--mode", p["mode"],
        "--redraw-amount", str(p["redraw_amount"]),
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
            locked = _replace_text(locked, cfg)
            base.save_variant(locked, out_dir, op.name)
    print(f"[subject_badge_text] done rc={rc} variants={len(out_paths)}")
    return out_paths


def _replace_text(img: Image.Image, cfg: dict) -> Image.Image:
    """文字原位替换：检测文字带 → LaMa 抹旧字 → 按原排版/字号/字体重画新词。

    字体：Didone serif（BACARDÍ 类）→ Playfair；military（ARMED FORCES 类）→ BlackOpsOne。
    当前占位，后续本风格单独升级实现。
    """
    tr = cfg.get("text_replacements") or {}
    if not tr:
        return img
    print("[subject_badge_text] text replacement: not yet implemented (placeholder)")
    return img


def selfcheck_notes() -> str:
    return (
        "自检要点：① 主体物种不变（蝙蝠/蝴蝶），角度/姿态/细节已裂变；② 文字按原字体/字号/排版只改内容词；"
        "③ 侵权品牌字已原位改写（禁色块遮盖）；④ 配色锁原色族（LAB 0.85）。"
    )
