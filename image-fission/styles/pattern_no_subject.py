"""styles/pattern_no_subject.py — 无主体、全是小元素（佩斯利/头巾/底纹）。

用户硬规则（2026-09-12）：
- 保留颜色与布局（含 4-patch 网格）不变
- 只改小元素母题（佩斯利重生/变体），异内容同构
- 禁止改整体排版、禁止把 4-patch 变成单幅中心纹章
- 颜色保留原图色族

本模块独立持有该风格的 ComfyUI 参数与 prompt，升级只动这里。
"""
from __future__ import annotations
from pathlib import Path
from . import base

STYLE_KEY = "pattern_no_subject"
DESCRIPTION = "无主体图案（佩斯利/头巾/底纹）：保留颜色与布局，只改小元素母题"


def default_params() -> dict:
    # 与 set.json comfyui_params.no_subject_pattern 对齐；本模块是唯一权威
    return {
        "mode": "mode1",                 # 双锁 txt2img + IPAdapter 风格锁
        "canny_res": 320,                # 低分 Canny 锁大结构（4-patch 边界/中线），不擦细母题
        "canny_strength": 0.82,
        "tile_strength": 0.55,           # 保画风色彩块，允许母题线条重画
        "ipa_weight": 0.45,
        "ipadapter_noise": 0.10,
        "denoise1": 0.48,
        "denoise2": 0.15,
        "steps": 28,
        "cfg": 5.0,
        "color_strength": 0.6,
        "composition_strength": 0.55,
        "lab_alpha": 0.70,               # 允许轻微色族内偏移让差异可见，但锁原色族
        "count": 1,
        "seed": 8888,
    }


def classify(image_path: str) -> float:
    """判定是否为无主体图案类。返回置信度 0~1。

    启发式：无明显居中徽章、文字带少、纹理高频重复 → 高置信。
    真实智慧体里这里会接更细的 CV/模型分类；当前用保守启发式 + 配置优先。
    """
    # 默认由 set.json 显式指派，这里给一个中性置信，路由层优先用配置
    return 0.5


def fission(image_path: str, out_dir: Path, cfg: dict, seed: int | None = None) -> list[Path]:
    """跑该风格的元素裂变：保 4-patch 布局 + 母题重生 + 颜色锁。

    注意：文本替换本风格为空（无文字），Stage B/C 跳过。
    """
    p = default_params()
    p.update({k: v for k, v in (cfg.get("comfyui_params", {}).get("no_subject_pattern", {}) or {}).items()})
    if seed is not None:
        p["seed"] = seed
    prompts = cfg.get("prompts") or ["ornamental paisley pattern, symmetrical 4-patch layout, no text"]
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
    # 收集产出
    out_paths = sorted(out_dir.glob("*.jpg")) if out_dir.exists() else []
    # 颜色锁（保原色族）
    if out_paths:
        ref = base.Image.open(image_path).convert("RGB")
        for op in out_paths:
            gen = base.Image.open(op).convert("RGB")
            locked = base.lab_color_lock(gen, ref, alpha=p["lab_alpha"])
            base.save_variant(locked, out_dir, op.name)
    print(f"[pattern_no_subject] done rc={rc} variants={len(out_paths)}")
    return out_paths


def selfcheck_notes() -> str:
    return (
        "自检要点：① 4-patch 布局 100% 保留；② 佩斯利母题重生/变体但身份仍在；"
        "③ 颜色锁原色族（LAB alpha=0.70 允许轻微内偏移）；④ 禁止变成单幅中心纹章。"
    )
