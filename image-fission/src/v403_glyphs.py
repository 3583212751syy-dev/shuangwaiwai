# -*- coding: utf-8 -*-
"""v403_glyphs.py —— p6 标题「单词裂变」字标生成（词库）

用户要求：「字体裂变，我要的是把文本单词都裂变」「文本裂变按照之前把单词都裂变成
别的意思的那一个代码」。

本脚本 = 把 p6 的标题做**词库**：每个候选词是**另一个意思**的英文自造词（黑金属命名风），
用与既有 n1~n6 完全相同的生成器产出字标（SDXL 1.0 + Harrlogos_XL_v2 LoRA），
再交给 make_v329.py 的 P6_WORD_SRC 逐个贴到主体定稿上 → 得到可直接对比的多版文本裂变。

生成器出处：jobs/_probe/_t_logo_mid.py（n1~n6 就是它生成的）。
产出：jobs/_probe/elemgen_mid/w_<WORD>.png

用法：
    python src/v403_glyphs.py              # 只生成缺的字标
    python src/v403_glyphs.py --force      # 全部重生成
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "."), sys.path.insert(0, "src")
from PIL import Image                                   # noqa: E402
from engine.comfy_client import ComfyClient             # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
OUTD = PROJ / "jobs" / "_probe" / "elemgen_mid"
OUTD.mkdir(parents=True, exist_ok=True)

CKPT = "sd_xl_base_1.0.safetensors"
LORA = "Harrlogos_XL_v2.safetensors"
SEED = 770315

NEG = ("bird, animal, skull, illustration, drawing, photo, 3d, color, gradient, frame, border, "
       "two words, repeated text, duplicated letters, stacked text, extra letters, watermark, "
       "cluttered, blurry, thin hairlines, small, faint")

# ---------------------------------------------------------------- 词库
# 已有字标（n1~n6 生成于 09-15）→ 直接复用，不重复渲染。
REUSE = {
    "RAVEN": "n1_0.png",
    "MOURNGRAVE": "n4_0.png",
    "VORGRAVEN": "n5_0.png",
    "SKARVALD": "n6_0.png",
}

# 本轮新裂变的词：**每个词是另一个意思**（不再是 RAVEN 的变体拼写）
NEW_WORDS = [
    ("IRONVEIL",   1344, 768),   # 铁幕
    ("GRAVETIDE",  1344, 768),   # 墓潮
    ("STORMHELM",  1344, 768),   # 风暴舵
    ("ASHREAVER",  1344, 768),   # 灰烬掠夺者
    ("DUSKBANE",   1344, 768),   # 暮祸
    ("THORNMOURN", 1344, 768),   # 荆棘哀悼
]


def wf(pos, neg, w, h, seed, steps=30, cfg=6.5):
    g = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "LoraLoader", "inputs": {
            "lora_name": LORA, "strength_model": 1.0, "strength_clip": 1.0,
            "model": ["1", 0], "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["2", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["2", 1]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "dpmpp_2m",
            "scheduler": "karras", "denoise": 1.0, "model": ["2", 0],
            "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0]}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "wbank"}},
    }
    return g


def pos_for(word: str) -> str:
    return (f"black metal band logo lettering spelling {word}, spiky thorny letters, "
            "white letters on pure black background, text only")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="已存在也重生成")
    args = ap.parse_args()

    # 1) 复用既有字标 → 复制成 w_<WORD>.png
    for word, src in REUSE.items():
        dst = OUTD / f"w_{word}.png"
        s = OUTD / src
        if s.exists() and (args.force or not dst.exists()):
            shutil.copy2(s, dst)
            print(f"[reuse] {word:12s} <- {src}")

    # 2) 生成新词
    todo = [(w, cw, ch) for (w, cw, ch) in NEW_WORDS
            if args.force or not (OUTD / f"w_{w}.png").exists()]
    if not todo:
        print("[done] 字标已齐，无需生成")
        return

    cl = ComfyClient()
    for word, cw, ch in todo:
        t = time.time()
        try:
            res = cl.run(wf(pos_for(word), NEG, cw, ch, SEED), timeout=1200)
        except Exception as e:                                  # noqa: BLE001
            print(f"[FAIL] {word}: {e}", flush=True)
            continue
        saved = 0
        for _node, blobs in res.items():
            for i, b in enumerate(blobs):
                Image.open(io.BytesIO(b)).convert("RGB").save(OUTD / f"w_{word}.png")
                saved += 1
        print(f"[ok] {word:12s} {cw}x{ch} {time.time() - t:.0f}s ({saved})", flush=True)
    print("[done]")


if __name__ == "__main__":
    main()
