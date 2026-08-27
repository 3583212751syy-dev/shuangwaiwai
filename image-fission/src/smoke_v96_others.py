r"""v9.6 剩余 5 张裂变：高清参考 + Proteus + v8.6 去毛躁参数。

illust_1 已在 v9.5 完成（jobs/smoke_v95_illust_*/illust_1_peacock_floral.jpg，4 维自检 8.5+）。
本脚本跑剩余 5 张：eagle_2 / denim_3 / camo_4 / skull_5 / metal_6。
每张自动从 db.ORIGINALS_CONFIG[label]["subjects"] 取第一个 subject。
"""
import os
import sys
import time
import random
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import demo_batch_6x4 as db
from engine.comfy_client import ComfyClient
from pipelines.build import build_mode1
from demo_batch_6x4 import (
    STYLE_PREFIX, STYLE_SUFFIX, IPA_WEIGHT_TYPE, IPA_NOISE, IPA_END,
    CN_LOW, CN_HIGH, CONTROLNET, USE_CONTROLNET, USDU_MODEL,
    HIRES_SCALE, TEXTILE_LORA, TEXTILE_LORA_STRENGTH,
)

# v9.0 切底模 + v8.6 去毛躁
db._cfg.SDXL_CHECKPOINT = "ProteusV0.4.safetensors"
db.HIRES_DENOISE = 0.20
db.HIRES_STEPS = 40

SKIP = {"illust_1"}  # 已用 v9.5 完成


def resolve_ref_path(orig_seed):
    hd = os.path.join(db.COMFYUI_INPUT, f"{orig_seed}_hd.png")
    if os.path.exists(hd):
        print(f"  ref=HD {os.path.basename(hd)} ({os.path.getsize(hd)/1024/1024:.1f}MB)")
        return hd, ".png"
    jpg = os.path.join(db.COMFYUI_INPUT, f"{orig_seed}.jpg")
    print(f"  ref=fallback {os.path.basename(jpg)} ({os.path.getsize(jpg)/1024:.0f}KB)")
    return jpg, ".jpg"


def main():
    out_dir = os.path.join(db.JOBS_BASE, f"smoke_v96_others_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")
    print(f"[v9.6 cfg] checkpoint={db._cfg.SDXL_CHECKPOINT}")
    print(f"[v9.6 cfg] hires_denoise={db.HIRES_DENOISE}  hires_steps={db.HIRES_STEPS}")
    print(f"[v9.6 cfg] VECTOR=0.55  cfg=7.5  TEXTILE={TEXTILE_LORA}@{TEXTILE_LORA_STRENGTH}")

    client = ComfyClient()
    results = []

    for orig_seed, orig_label in db.ORIGINALS:
        if orig_label in SKIP:
            continue
        cfg = db.ORIGINALS_CONFIG.get(orig_label)
        if not cfg:
            print(f"[WARN] no config for {orig_label}, skip")
            continue
        # 取第一个 subject（subjects 是 list，dict() 转成 {name: prompt}）
        sub_name, sub_prompt = next(iter(dict(cfg["subjects"]).items()))
        print(f"\n=== {orig_label}  sub={sub_name} ===")

        style_words = cfg.get("style_words", "")
        sim = cfg.get("sim", 0.85)
        comp = cfg.get("comp", 0.72)
        cn_v = cfg.get("cn", 0.62)
        extra_lora = cfg.get("extra_lora")
        extra_lora_strength = 0.55  # v8.6 改进

        seed_src, ext = resolve_ref_path(orig_seed)
        if not os.path.exists(seed_src):
            print(f"  [WARN] skip {orig_seed}: no ref")
            continue
        seed_name = f"smoke_{orig_label}_{int(time.time()*1000)}{ext}"
        shutil.copy(seed_src, os.path.join(db.COMFYUI_INPUT, seed_name))

        prompt = ", ".join(filter(None, [STYLE_PREFIX, style_words, sub_prompt, STYLE_SUFFIX]))
        seed = random.randint(1, 999999999)

        params = {
            "style_prompt": prompt,
            "similarity": sim,
            "ipadapter_weight_type": IPA_WEIGHT_TYPE,
            "ipadapter_noise": IPA_NOISE,
            "ipadapter_end": IPA_END,
            "composition_strength": comp,
            "controlnet_name": CONTROLNET if USE_CONTROLNET else "",
            "controlnet_strength": cn_v if USE_CONTROLNET else 0.0,
            "controlnet_low_threshold": CN_LOW,
            "controlnet_high_threshold": CN_HIGH,
            "usdu_model": USDU_MODEL,
            "hires_scale": HIRES_SCALE,
            "hires_denoise": db.HIRES_DENOISE,
            "hires_steps": db.HIRES_STEPS,
            "negative_prompt": (
                "embedding:NegativeXL, embedding:unaestheticXL, "
                "photography, product photo, 3d render, realistic texture, fabric folds, "
                "wrinkles, shadows, depth of field, blurry, deformed, low quality, "
                "garbled text, gibberish text, pseudo-script, fake characters, "
                "runic nonsense, occult sigils, runes, talisman symbols, "
                "malformed letters, repeating letters, double letters, misspelled words, "
                "illegible text, scribbles resembling text, watermark, copyright logo, "
                "cropped, out of frame, mockup, garment, "
                "solid color text block, plain banner, plain sans-serif text, Helvetica, "
                "Arial, modern UI font, UI overlay sticker, mismatched typography, "
                "text style different from illustration style, standalone text banner, "
                "noise, grain, film grain, sensor noise, compression artifacts, "
                "speckles, dust spots, harsh jagged edges, fuzzy halftone, "
                "rough sketch, pencil smudge, dirty paper texture, "
                "uneven ink, broken outlines, scratchy strokes"
            ),
            "width": 1024, "height": 1024, "batch_per_run": 1,
            "steps": 45, "cfg": 7.5,
            "seed": seed,
            "lora_name": TEXTILE_LORA,
            "lora_strength": TEXTILE_LORA_STRENGTH,
            "lora_name_2": extra_lora,
            "lora_strength_2": extra_lora_strength,
        }

        g = build_mode1(seed_name, params, f"smoke_{orig_label}_{sub_name}")
        t0 = time.time()
        try:
            res = client.run(g, timeout=600)
            data = next(iter(res.values()))[0]
            out_path = os.path.join(out_dir, f"{orig_label}_{sub_name}.jpg")
            with open(out_path, "wb") as f:
                f.write(data)
            dt = time.time() - t0
            size_mb = os.path.getsize(out_path) / 1024 / 1024
            print(f"  [OK] {dt:.0f}s  {size_mb:.1f}MB")
            results.append((orig_label, sub_name, out_path))
        except Exception as e:
            print(f"  [FAIL] {repr(e)}")

    print(f"\n[DONE] {len(results)}/{len(db.ORIGINALS)-len(SKIP)}")
    for r in results:
        print(f"  - {r[0]}_{r[1]}.jpg")
    return out_dir


if __name__ == "__main__":
    main()
