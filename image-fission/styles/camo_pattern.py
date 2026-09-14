"""styles/camo_pattern.py — 迷彩类（军标狗牌迷彩 / 迷彩棕榈树）。

用户硬规则（2026-09-12）：
- 保留原色族（颜色不变）
- 把迷彩"湖泊"色块改变形状 / 角度 / 大小 / 选取范围（不同角度长度大小的裂变）
- 小元素（dog tag / 棕榈树）改角度与内容，但物种不变（狗牌还是狗牌、棕榈还是棕榈）
- 文字按原图字体 / 排版 / 字号只改内容单词（原位改写，禁色块遮盖）

本模块独立持有该风格的 ComfyUI 参数与文字替换策略。升级只改本模块，不污染其他风格。
"""
from __future__ import annotations
import shutil
from pathlib import Path
from . import base

STYLE_KEY = "camo_pattern"
DESCRIPTION = "迷彩类：改湖泊色块形变、小元素物种不变、文字按原排版替换"
COVERS = ["b78e60", "pinterest4"]


def default_params() -> dict:
    return {
        "mode": "mode1",                 # 双锁：保整体迷彩构图 + 风格锁
        "color_strength": 0.72,         # 强锁原色族（迷彩颜色必须保留）
        "composition_strength": 0.58,
        "ipadapter_noise": 0.10,
        "controlnet_strength": 0.72,    # Canny 偏强：锁小元素(棕榈/狗牌)轮廓，放湖泊色块形变
        "controlnet_end": 0.90,
        "lab_alpha": 0.80,              # 强锁原色族（迷彩颜色必须保留）
        "blob_morph_strength": 0.030,   # 新增：湖泊色块保色几何形变强度（像素级搬运，原色不变）
        "steps": 28,
        "cfg": 5.0,
        "count": 1,
        "seed": 8888,
    }


def classify(image_path: str) -> float:
    return 0.5  # 由 set.json 显式指派优先


def fission(image_path: str, out_dir: Path, cfg: dict, seed: int | None = None) -> list[Path]:
    """跑迷彩类元素裂变：湖泊色块形变 + 小元素角度/内容裂变 + 颜色锁 + 文字原位替换。"""
    p = default_params()
    img_cfg = base.get_image_cfg(cfg, image_path)
    p.update({k: v for k, v in (img_cfg.get("comfyui_params", {}).get("camo_pattern", {}) or {}).items()})
    if seed is not None:
        p["seed"] = seed
    prompts = base.get_image_cfg(cfg, image_path).get("prompts") or ["urban gray camouflage poster, military dog tag pendant, black ops stencil typography, monochrome tactical pattern, no text"]
    prompt = prompts[0]

    out_dir = Path(out_dir)
    im = base.Image.open(image_path).convert("RGB")
    W, H = im.size
    # mode1/3 用原图 gen_scale 宽高保比例，避免拉伸；默认 1/4，狗牌图可调高以保小元素物种
    scale = float(base.get_image_cfg(cfg, image_path).get("gen_scale", 0.25))
    w = max(64, int(W * scale))
    h = max(64, int(H * scale))
    w4 = max(64, (w // 8) * 8)
    h4 = max(64, (h // 8) * 8)

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
    ]
    if p["mode"] == "mode3":
        # img2img 重绘幅度；迷彩类用 0.35-0.45 让色块可见变化，又保狗牌/链不消失
        args += ["--redraw-amount", str(p.get("redraw_amount", 0.38))]
    if p.get("controlnet_strength", 0) > 0:
        args += [
            "--controlnet-strength", str(p["controlnet_strength"]),
            "--controlnet-end", str(p["controlnet_end"]),
        ]
    rc = base.run_fission_cli(args)

    out_paths = sorted({p for p in out_dir.rglob("*.jpg") if p.is_file()}) if out_dir.exists() else []
    # 文字 bbox 排除区：保色形变不移动文字所在区域，确保 LaMa + 重绘坐标对齐
    text_bboxes = [tuple(it["bbox"]) for it in (base.get_text_plan_for(cfg, image_path) or []) if it.get("bbox")]
    if out_paths:
        ref = base.Image.open(image_path).convert("RGB")
        for op in out_paths:
            # 4x 超分图 -> resize 回原图尺寸，使 text_plan 的 bbox 坐标对齐
            gen = base.Image.open(op).convert("RGB").resize((W, H), base.Image.LANCZOS)
            locked = base.lab_color_lock(gen, ref, alpha=p["lab_alpha"])
            # 新增：迷彩「湖泊」色块保色几何形变（不同角度/长度/大小，原色不变）
            locked = base.camo_blob_morph(locked, seed=p["seed"],
                                          strength=p["blob_morph_strength"],
                                          exclude_bboxes=text_bboxes or None)
            locked = _replace_text(locked, image_path, cfg)
            base.save_variant(locked, out_dir, op.name)
    print(f"[camo_pattern] done rc={rc} variants={len(out_paths)}")
    return out_paths


def _replace_text(img: "base.Image.Image", image_path: str, cfg: dict) -> "base.Image.Image":
    """文字原位替换：按 set.json text_plan 逐条 LaMa 抹旧字 + PIL 重画新词。"""
    plan = base.get_text_plan_for(cfg, image_path)
    if not plan:
        return img
    return base.replace_text_plan(img, plan, dilate=10)


def selfcheck_notes() -> str:
    return (
        "自检要点：① 迷彩'湖泊'色块经保色几何形变（camo_blob_morph，不同角度/长度/大小/选取范围），原色不变；"
        "② 原色族保留（LAB 0.80）；③ 小元素物种不变（棕榈仍是棕榈、狗牌仍是狗牌）仅角度/内容变；"
        "④ 文字按原字体/排版/字号只改内容词，无矩形色块遮字（文字区已排除形变保对齐）。"
    )
