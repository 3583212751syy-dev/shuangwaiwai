r"""v9.5 illust_1 单图验证：高清化参考图 + Proteus 底模 + v8.6 hires 改进。

根因（用户 2026-08-25 15:14）：
  "可能我给的参考图是糊的，将其高清化去毛边去噪点再裂变做高质量图片"
  - 参考图 pinterest_illust_1.jpg 仅 558x960 / 90KB（JPEG 压缩重）
  - IPAdapter 把它从 558x960 上采到 1024 编码时，引入大量灰阶/抗锯齿/压缩块
  - 4x USDU 把这些"软边"放大成"画面毛躁"——这是比"底模"更上游的根因

v9.5 改动（在 v9.0 基础上叠加）：
  - 优先读 pinterest_illust_1_hd.png（由 src/enhance_references.py 生成，4096² PNG）
  - 保留 v9.0 的 Proteus 切底模
  - 保留 v8.6 的 hires_denoise 0.20 / hires_steps 40 / cfg 7.5 / VECTOR 0.55

执行：
  1. 先跑 src/enhance_references.py（生成 pinterest_illust_1_hd.png 到 ComfyUI/input）
  2. 再跑本脚本
"""
import os
import sys
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import demo_batch_6x4 as db

# v9.0 改动：切底模
db._cfg.SDXL_CHECKPOINT = "ProteusV0.4.safetensors"

# v8.6 改进：治毛躁（保留）
db.HIRES_DENOISE = 0.20
db.HIRES_STEPS = 40

from engine.comfy_client import ComfyClient

ONLY_ORIG = "illust_1"
SUBJECT_PICK = "peacock_floral"


def resolve_ref_path(orig_seed: str) -> tuple[str, str]:
    """优先读 _hd.png 高清版，fallback 原 .jpg。返回 (路径, 后缀)。"""
    hd = os.path.join(db.COMFYUI_INPUT, f"{orig_seed}_hd.png")
    if os.path.exists(hd):
        print(f"[v9.5] 使用高清参考 {os.path.basename(hd)} "
              f"({os.path.getsize(hd)/1024/1024:.1f}MB)")
        return hd, ".png"
    jpg = os.path.join(db.COMFYUI_INPUT, f"{orig_seed}.jpg")
    print(f"[v9.5] fallback 原图 {os.path.basename(jpg)} "
          f"({os.path.getsize(jpg)/1024:.0f}KB)")
    return jpg, ".jpg"


def main():
    out_dir = os.path.join(db.JOBS_BASE, f"smoke_v95_illust_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")
    print(f"[v9.5 cfg] checkpoint={db._cfg.SDXL_CHECKPOINT}")
    print(f"[v9.5 cfg] hires_denoise={db.HIRES_DENOISE}  hires_steps={db.HIRES_STEPS}")

    client = ComfyClient()
    results = []

    for orig_seed, orig_label in db.ORIGINALS:
        if orig_label != ONLY_ORIG:
            continue
        cfg = db.ORIGINALS_CONFIG.get(orig_label)
        sub_prompt = dict(cfg["subjects"])[SUBJECT_PICK]

        cfg_v95 = dict(cfg)
        cfg_v95["extra_lora_strength"] = 0.55
        cfg_v95["sim"] = cfg.get("sim", 0.85)
        cfg_v95["comp"] = cfg.get("comp", 0.72)
        cfg_v95["cn"] = cfg.get("cn", 0.62)

        seed_src, ext = resolve_ref_path(orig_seed)
        if not os.path.exists(seed_src):
            print(f"[WARN] skip {orig_seed}: neither _hd.png nor .jpg found")
            continue
        seed_name = f"smoke_{orig_label}_{int(time.time()*1000)}{ext}"
        shutil.copy(seed_src, os.path.join(db.COMFYUI_INPUT, seed_name))

        from demo_batch_6x4 import (
            STYLE_PREFIX, STYLE_SUFFIX, IPA_WEIGHT_TYPE, IPA_NOISE, IPA_END,
            CN_LOW, CN_HIGH, CONTROLNET, USE_CONTROLNET, USDU_MODEL,
            HIRES_SCALE, HIRES_DENOISE, HIRES_STEPS, TEXTILE_LORA,
            TEXTILE_LORA_STRENGTH,
        )
        import random
        from pipelines.build import build_mode1

        style_words = cfg_v95.get("style_words", "")
        sim = cfg_v95.get("sim", 0.70)
        comp = cfg_v95.get("comp", 0.55)
        cn_v = cfg_v95.get("cn", 0.40)
        extra_lora = cfg_v95.get("extra_lora")
        extra_lora_strength = cfg_v95.get("extra_lora_strength", 0.40)

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
            "hires_denoise": HIRES_DENOISE,
            "hires_steps": HIRES_STEPS,
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

        g = build_mode1(seed_name, params, f"smoke_{orig_label}_{SUBJECT_PICK}")
        t0 = time.time()
        try:
            res = client.run(g, timeout=600)
            data = next(iter(res.values()))[0]
            out_path = os.path.join(out_dir, f"{orig_label}_{SUBJECT_PICK}.jpg")
            with open(out_path, "wb") as f:
                f.write(data)
            dt = time.time() - t0
            size_mb = os.path.getsize(out_path) / 1024 / 1024
            print(f"[OK] {orig_label} x {SUBJECT_PICK}  seed={seed}  ckpt={db._cfg.SDXL_CHECKPOINT}  "
                  f"VECTOR={extra_lora_strength}  hires={HIRES_DENOISE}/{HIRES_STEPS}  "
                  f"ref=HD  {dt:.0f}s  {size_mb:.1f}MB")
            results.append((orig_label, SUBJECT_PICK, out_path))
        except Exception as e:
            print(f"[FAIL] {orig_label} x {SUBJECT_PICK}: {repr(e)}")

    print(f"\n[DONE] {len(results)}/1")
    return out_dir


if __name__ == "__main__":
    main()
