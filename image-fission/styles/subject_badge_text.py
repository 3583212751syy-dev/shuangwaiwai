"""styles/subject_badge_text.py — 有主体 + 文字（蝙蝠徽章 / 牛仔蝴蝶）。

用户硬规则（2026-09-12）：
- 主体物种不变（蝙蝠还是蝙蝠、蝴蝶还是蝴蝶）
- 改角度 / 姿态 / 细节裂变（异内容同构）
- 文字按原图字体 / 字号 / 排版位置**只改内容单词**（原位改写，禁色块遮盖）
- 侵权品牌字（BACARDÍ/MCKHEART 等）必须原位改写成占位词（NOCTAVEN/DUSKBAT/MOONBAT 等）

本模块独立持有该风格的 ComfyUI 参数（主体保身份、适度裂变）与文字替换策略。
字体：Didone serif → PlayfairDisplay-Bold；military display → BlackOpsOne。

重要工程约束（回归实测）：6978 / pinterest3 在 mode3(img2img)+Canny 路径下会**可复现 ComfyUI 挂死**
（与 pinterest6 此前 mode3 挂死同因）。故本风格**强制 mode1**（txt2img + IPAdapter 双锁 + Canny 锁轮廓），
尺寸取原图 1/4 向下取 8 的倍数，与 subject_badge_textless 一致。
"""
from __future__ import annotations
import shutil
from pathlib import Path
from . import base

STYLE_KEY = "subject_badge_text"
DESCRIPTION = "有主体+文字：主体物种不变改角度细节、文字按原排版替换"
COVERS = ["6978", "pinterest3"]


def default_params() -> dict:
    return {
        "mode": "mode1",                 # 关键：mode3+Canny 在该风格会挂死，强制 mode1
        "color_strength": 0.60,
        "composition_strength": 0.55,
        "ipadapter_noise": 0.10,
        "controlnet_strength": 0.50,    # Canny 中强：锁主体轮廓，放角度/姿态变化
        "controlnet_end": 0.90,
        "lab_alpha": 0.85,              # 强锁原色族
        "steps": 28,
        "cfg": 5.0,
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
    prompts = base.get_image_cfg(cfg, image_path).get("prompts") or ["vintage emblem, same subject redesigned at new angle, no real brand"]
    prompt = prompts[0]

    out_dir = Path(out_dir)
    im = base.Image.open(image_path).convert("RGB")
    W, H = im.size
    # 每图可在 set.json 指定 gen_scale（默认 0.25=原图 1/4），用于控制 mode1 生成分辨率
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
        "--controlnet-strength", str(p["controlnet_strength"]),
        "--controlnet-end", str(p["controlnet_end"]),
    ]
    rc = base.run_fission_cli(args)

    out_paths = sorted({p for p in out_dir.rglob("*.jpg") if p.is_file()}) if out_dir.exists() else []
    if out_paths:
        ref = base.Image.open(image_path).convert("RGB")
        for op in out_paths:
            # mode1 输出=4x 超采样，resize 回精确原图尺寸以对齐 text_plan bbox
            gen = base.Image.open(op).convert("RGB").resize((W, H), base.Image.LANCZOS)
            locked = base.lab_color_lock(gen, ref, alpha=p["lab_alpha"])
            locked = _replace_text(locked, image_path, cfg)
            base.save_variant(locked, out_dir, op.name)
    print(f"[subject_badge_text] done rc={rc} variants={len(out_paths)}")
    return out_paths


def _replace_text(img: "base.Image.Image", image_path: str, cfg: dict) -> "base.Image.Image":
    """文字原位替换：按 set.json text_plan 逐条 LaMa 抹旧字 + PIL 重画新词。"""
    plan = base.get_text_plan_for(cfg, image_path)
    if not plan:
        return img
    return base.replace_text_plan(img, plan, dilate=10)


def selfcheck_notes() -> str:
    return (
        "自检要点：① 主体物种不变（蝙蝠/蝴蝶），角度/姿态/细节已裂变；② 文字按原字体/字号/排版只改内容词；"
        "③ 侵权品牌字已原位改写（禁色块遮盖）；④ 配色锁原色族（LAB 0.85）。注意 mode3 挂死，本风格强制 mode1。"
    )
