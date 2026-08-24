"""
v6 方向验证：每个原图只跑第一个主题，共 6 张。
快速确认：去实体化 / 图案质量 / 主题差异。
"""
import os
import sys
import time
import shutil
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo_batch_6x4 import (
    COMFYUI_INPUT, JOBS_BASE, ORIGINALS, ORIGINALS_CONFIG,
    STYLE_PREFIX, STYLE_SUFFIX, SIM, IPA_WEIGHT_TYPE, IPA_NOISE, IPA_END,
    COMPOSITION_STRENGTH, CONTROLNET, USE_CONTROLNET, CN_LOW, CN_HIGH,
    TEXTILE_LORA, TEXTILE_LORA_STRENGTH,
)
from engine.comfy_client import ComfyClient
from pipelines.build import build_mode1


def main():
    out_dir = os.path.join(JOBS_BASE, f"smoke_v6_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")

    client = ComfyClient()
    results = []

    for orig_seed, orig_label in ORIGINALS:
        seed_src = os.path.join(COMFYUI_INPUT, f"{orig_seed}.jpg")
        if not os.path.exists(seed_src):
            print(f"[WARN] 跳过: {seed_src}")
            continue
        seed_name = f"smoke_v6_{orig_label}_{int(time.time()*1000)}.jpg"
        shutil.copy(seed_src, os.path.join(COMFYUI_INPUT, seed_name))

        cfg = ORIGINALS_CONFIG[orig_label]
        sub_label, sub_prompt = cfg["subjects"][0]
        style_words = cfg.get("style_words", "")
        cn_strength = cfg.get("cn_strength", 0.45)
        prompt = ", ".join(filter(None, [STYLE_PREFIX, style_words, sub_prompt, STYLE_SUFFIX]))
        seed = random.randint(1, 999999999)

        params = {
            "style_prompt": prompt,
            "similarity": SIM,
            "ipadapter_weight_type": IPA_WEIGHT_TYPE,
            "ipadapter_noise": IPA_NOISE,
            "ipadapter_end": IPA_END,
            "composition_strength": COMPOSITION_STRENGTH,
            "controlnet_name": CONTROLNET if USE_CONTROLNET else "",
            "controlnet_strength": cn_strength if USE_CONTROLNET else 0.0,
            "controlnet_low_threshold": CN_LOW,
            "controlnet_high_threshold": CN_HIGH,
            "usdu_model": "4x_NMKD-Siax_200k.pth",
            "negative_prompt": (
                "photography, product photo, 3d render, realistic texture, fabric folds, "
                "wrinkles, shadows, depth of field, blurry, deformed, low quality, "
                "text, watermark, signature, cropped, out of frame"
            ),
            "width": 1024, "height": 1024, "batch_per_run": 1,
            "steps": 35, "cfg": 6.0,
            "seed": seed,
            "lora_name": TEXTILE_LORA,
            "lora_strength": TEXTILE_LORA_STRENGTH,
        }

        g = build_mode1(seed_name, params, f"smoke_v6_{orig_label}_{sub_label}")
        t0 = time.time()
        try:
            res = client.run(g, timeout=360)
            data = next(iter(res.values()))[0]
            out_path = os.path.join(out_dir, f"{orig_label}_{sub_label}.jpg")
            with open(out_path, "wb") as f:
                f.write(data)
            dt = time.time() - t0
            print(f"[OK] {orig_label} × {sub_label}  seed={seed}  {dt:.0f}s -> {out_path}")
            results.append(out_path)
        except Exception as e:
            print(f"[FAIL] {orig_label} × {sub_label}: {repr(e)}")

    print(f"\n[DONE] {len(results)}/6")
    if results:
        from make_gallery import build as build_lean
        build_lean(out_dir, os.path.join(out_dir, "gallery_lean.html"))
        print(f"[gallery] {out_dir}/gallery_lean.html")


if __name__ == "__main__":
    main()
