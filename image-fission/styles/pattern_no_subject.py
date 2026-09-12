"""styles/pattern_no_subject.py — 无主体、全是小元素（佩斯利/头巾/底纹）。

用户硬规则（2026-09-12）：
- 保留颜色与布局（含 4-patch 网格）不变
- 只改小元素母题（佩斯利重生/变体），异内容同构
- 禁止改整体排版、禁止把 4-patch 变成单幅中心纹章
- 颜色保留原图色族

实现（v325 升级）：
- 用 mode3 img2img：以原图当底图，denoise 抬到 0.55 → 保住 4-patch 网格与整体布局，
  只重绘佩斯利母题（异内容同构）。比旧 mode1 自由生成安全（不会再变成单幅纹章）。
- Canny 中强(0.5) 锁 4-patch 网格线，避免母题重绘时网格漂移。
- 低构图锁(0.35) 给母题变化留空间；颜色锁 0.55 + LAB 0.82 锁原色族。
- 无文字，跳过文本替换。
"""
from __future__ import annotations
import shutil
from pathlib import Path
from . import base

STYLE_KEY = "pattern_no_subject"
DESCRIPTION = "无主体图案（佩斯利/头巾/底纹）：保留颜色与布局，只改小元素母题"


def default_params() -> dict:
    return {
        "mode": "mode3",                 # img2img 保 4-patch 布局，只重绘母题
        "redraw_amount": 0.55,          # 抬到 0.55：母题重生但网格/布局保留
        "color_strength": 0.55,         # 锁原色族
        "composition_strength": 0.35,   # 低：给母题变化留空间
        "ipadapter_noise": 0.12,
        "controlnet_strength": 0.50,    # Canny 中强锁 4-patch 网格线
        "controlnet_end": 0.90,
        "lab_alpha": 0.82,
        "steps": 28,
        "cfg": 5.0,
        "count": 1,
        "seed": 8888,
    }


def classify(image_path: str) -> float:
    return 0.5  # 由 set.json 显式指派优先


def fission(image_path: str, out_dir: Path, cfg: dict, seed: int | None = None) -> list[Path]:
    p = default_params()
    p.update({k: v for k, v in (cfg.get("comfyui_params", {}).get("no_subject_pattern", {}) or {}).items()})
    if seed is not None:
        p["seed"] = seed
    prompts = base.get_image_cfg(cfg, image_path).get("prompts") or ["ornamental paisley pattern, symmetrical 4-patch layout, intricate damask motifs, terracotta and cream, decorative borders, no text"]
    prompt = prompts[0]

    out_dir = Path(out_dir)
    im = base.Image.open(image_path).convert("RGB")
    W, H = im.size

    # mode3 以原图为底图，输出 = 原图尺寸，无需传宽高；Canny 直接锁原图边缘
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
        "--controlnet-strength", str(p["controlnet_strength"]),
        "--controlnet-end", str(p["controlnet_end"]),
    ]
    rc = base.run_fission_cli(args)

    out_paths = sorted({p for p in out_dir.rglob("*.jpg") if p.is_file()}) if out_dir.exists() else []
    if out_paths:
        ref = base.Image.open(image_path).convert("RGB")
        for op in out_paths:
            gen = base.Image.open(op).convert("RGB").resize((W, H), base.Image.LANCZOS)
            gen = base.lab_color_lock(gen, ref, alpha=p["lab_alpha"])
            # 无文字，跳过文本替换
            base.save_variant(gen, out_dir, op.name)
    print(f"[pattern_no_subject] done rc={rc} variants={len(out_paths)}")
    return out_paths


def selfcheck_notes() -> str:
    return (
        "自检要点：① 4-patch 布局 100% 保留（mode3 img2img + Canny 锁网格）；"
        "② 佩斯利母题重生/变体但身份仍在；③ 颜色锁原色族（LAB 0.82）；"
        "④ 禁止变成单幅中心纹章。"
    )
