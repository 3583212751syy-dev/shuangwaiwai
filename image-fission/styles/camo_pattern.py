"""styles/camo_pattern.py — 迷彩类（军标狗牌迷彩 / 迷彩棕榈树）。

用户硬规则（2026-09-12）：
- 保留原色族（颜色不变）
- 把迷彩"湖泊"色块改变形状 / 角度 / 大小 / 选取范围（不同角度长度大小的裂变）
- 小元素（dog tag / 棕榈树）改角度与内容，但物种不变（狗牌还是狗牌、棕榈还是棕榈）
- 文字按原图字体 / 排版 / 字号只改内容单词（原位改写，禁色块遮盖）

本模块独立持有该风格的 ComfyUI 参数（camo blob 形变强、色锁强）与文字替换策略。
"""
from __future__ import annotations
from pathlib import Path
from . import base

STYLE_KEY = "camo_pattern"
DESCRIPTION = "迷彩类：改湖泊色块形变、小元素物种不变、文字按原排版替换"
# 该风格覆盖的回归图 id（用于路由回查）
COVERS = ["b78e60", "pinterest4"]


def default_params() -> dict:
    return {
        "mode": "mode1",                 # 双锁：保整体迷彩构图 + 风格锁
        "canny_res": 512,                # 中分 Canny 锁大块边界
        "canny_strength": 0.78,
        "tile_strength": 0.50,
        "ipa_weight": 0.45,
        "ipadapter_noise": 0.10,
        "denoise1": 0.50,                # 略高让色块形状/角度/大小裂变
        "denoise2": 0.15,
        "steps": 28,
        "cfg": 5.0,
        "color_strength": 0.62,
        "composition_strength": 0.58,
        "lab_alpha": 0.80,               # 强锁原色族（迷彩颜色必须保留）
        "count": 1,
        "seed": 8888,
    }


def classify(image_path: str) -> float:
    return 0.5  # 由 set.json 显式指派优先


def fission(image_path: str, out_dir: Path, cfg: dict, seed: int | None = None) -> list[Path]:
    """跑迷彩类元素裂变：湖泊色块形变 + 小元素角度/内容裂变 + 颜色锁 + 文字原位替换。"""
    p = default_params()
    p.update({k: v for k, v in (cfg.get("comfyui_params", {}).get("camo_pattern", {}) or {}).items()})
    if seed is not None:
        p["seed"] = seed
    prompts = cfg.get("prompts") or ["tropical camouflage pattern with palm silhouettes, lake-like blobs reshaped, original palette preserved, no text"]
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
            # 文字原位替换（若有）：按 set.json text_replacements 逐条 LaMa 抹 + PIL 重画
            locked = _replace_text(locked, cfg)
            base.save_variant(locked, out_dir, op.name)
    print(f"[camo_pattern] done rc={rc} variants={len(out_paths)}")
    return out_paths


def _replace_text(img: Image.Image, cfg: dict) -> Image.Image:
    """迷彩类文字替换：检测文字带 → LaMa 抹旧字 → 按原排版/字号/字体重画新词。

    字体选择：military display（dog tag 类）→ BlackOpsOne；其余按检测。目前用 BlackOpsOne 兜底。
    """
    tr = cfg.get("text_replacements") or {}
    if not tr:
        return img
    # TODO(per-style): 接入 detect_text_lines + lama_inpaint + render_text_band
    # 当前仅占位，避免阻塞出图；后续本风格单独升级时实现
    print("[camo_pattern] text replacement: not yet implemented (placeholder)")
    return img


def selfcheck_notes() -> str:
    return (
        "自检要点：① 迷彩'湖泊'色块已改变形状/角度/大小/选取范围；② 原色族保留（LAB 0.80）；"
        "③ 小元素物种不变（棕榈仍是棕榈、狗牌仍是狗牌）仅角度/内容变；④ 文字按原字体/排版/字号只改内容词。"
    )
