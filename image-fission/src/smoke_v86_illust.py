"""v8.6 illust_1 单图验证：治"画面毛躁"。

毛躁根因（看 v8.5 illust_1 实际产物）：
  1. DD-vector 0.35 太弱 —— 治不了 Juggernaut 写实质感残留的颗粒噪点
  2. Juggernaut Ragnarok 是写实底模，对矢量平面控制弱
  3. USDU 4x-UltraSharp 把潜空间残留噪点放大成"右侧黑底灰色碎渣"

v8.6 改动（保守，不破坏 v8.5 已成功的整体构图）：
  - hires_denoise 0.28 → 0.20   更轻细化 = 保留细节 + 更清噪
  - hires_steps    30   → 40    更彻底清理
  - VECTOR 0.35   → 0.55       适度强化矢量（不冲到 0.85 避免盖过 ornamental 花卉）
  - cfg 7.0       → 7.5        更严守 prompt，治伪元素/碎线条

不动：底模（等 Proteus 切）/ sim 0.85 / cn 0.62 / ControlNet（构图锁已对）
       / TEXTILE / TSHIRTS LoRA（v8.5 illust_1 没用第三个 LoRA）

治本留给 v9.0（Proteus 底模到位后做完整切底模）。
"""
import os
import sys
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import demo_batch_6x4 as db

# v8.6 改进：治毛躁。模块级覆盖 v8.5 默认值（只影响本次 run_one 调用）
db.HIRES_DENOISE = 0.20   # 0.28 → 0.20（更轻细化 = 保留细节 + 更清噪）
db.HIRES_STEPS = 40        # 30 → 40（更彻底清理）

from engine.comfy_client import ComfyClient

ONLY_ORIG = "illust_1"
SUBJECT_PICK = "peacock_floral"


def main():
    out_dir = os.path.join(db.JOBS_BASE, f"smoke_v86_illust_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")
    print(f"[v8.6 cfg] hires_denoise={db.HIRES_DENOISE}  hires_steps={db.HIRES_STEPS}")
    print(f"[v8.6 cfg] USDU={db.USDU_MODEL}  checkpoint={db._cfg.SDXL_CHECKPOINT}")

    client = ComfyClient()
    results = []

    for orig_seed, orig_label in db.ORIGINALS:
        if orig_label != ONLY_ORIG:
            continue
        cfg = db.ORIGINALS_CONFIG.get(orig_label)
        sub_prompt = dict(cfg["subjects"])[SUBJECT_PICK]

        # v8.6 在 cfg 层做只读副本，避免污染 v8.5 全局
        cfg_v86 = dict(cfg)
        cfg_v86["extra_lora_strength"] = 0.55  # 0.35 → 0.55 适度强化矢量
        cfg_v86["sim"] = cfg.get("sim", 0.70)  # 保留 v8.5 锁主体值
        cfg_v86["comp"] = cfg.get("comp", 0.55)
        cfg_v86["cn"] = cfg.get("cn", 0.40)

        seed_src = os.path.join(db.COMFYUI_INPUT, f"{orig_seed}.jpg")
        if not os.path.exists(seed_src):
            print(f"[WARN] skip {orig_seed}")
            continue
        seed_name = f"smoke_{orig_label}_{int(time.time()*1000)}.jpg"
        shutil.copy(seed_src, os.path.join(db.COMFYUI_INPUT, seed_name))

        # cfg 8.5 → 8.6 通过覆盖 cfg dict 注入到 run_one 里的 params["cfg"]
        # run_one 不读 cfg["cfg"]，而是用 params["cfg"]。这里 monkey patch:
        _orig_run_one = db.run_one
        def run_one_with_cfg(client, orig_label, sub_label, sub_prompt, seed_name, cfg_in,
                             out_dir, idx, cfg_override=7.5):
            # 把 cfg 注入：临时把全局 cfg 改写到下一行 build_mode1 用的 params["cfg"]
            # run_one 不接受 cfg override 参数，所以简单办法：直接在 params 里改 cfg
            # —— 但 run_one 是闭包函数，我们在外层 monkey patch 通过改 cfg 的方式不可行。
            # 改方案：直接在 cfg 里加一个 cfg_scale key，让 run_one 误读……不行。
            # 最简办法：直接复制 run_one 的核心调用并改 cfg。
            from demo_batch_6x4 import (
                STYLE_PREFIX, STYLE_SUFFIX, IPA_WEIGHT_TYPE, IPA_NOISE, IPA_END,
                CN_LOW, CN_HIGH, CONTROLNET, USE_CONTROLNET, USDU_MODEL,
                HIRES_SCALE, HIRES_DENOISE, HIRES_STEPS, TEXTILE_LORA,
                TEXTILE_LORA_STRENGTH,
            )
            import random
            style_words = cfg_in.get("style_words", "")
            sim = cfg_in.get("sim", 0.70)
            comp = cfg_in.get("comp", 0.55)
            cn_v = cfg_in.get("cn", 0.40)
            extra_lora = cfg_in.get("extra_lora")
            extra_lora_strength = cfg_in.get("extra_lora_strength", 0.40)

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
                "steps": 45, "cfg": cfg_override,  # v8.6: 7.0 → 7.5
                "seed": seed,
                "lora_name": TEXTILE_LORA,
                "lora_strength": TEXTILE_LORA_STRENGTH,
                "lora_name_2": extra_lora,
                "lora_strength_2": extra_lora_strength,
            }
            from pipelines.build import build_mode1
            g = build_mode1(seed_name, params, f"smoke_{orig_label}_{sub_label}")
            t0 = time.time()
            try:
                res = client.run(g, timeout=600)
                data = next(iter(res.values()))[0]
                out_path = os.path.join(out_dir, f"{orig_label}_{sub_label}.jpg")
                with open(out_path, "wb") as f:
                    f.write(data)
                dt = time.time() - t0
                print(f"[OK] {orig_label} × {sub_label}  seed={seed}  cfg={cfg_override}  "
                      f"VECTOR={extra_lora_strength}  hires={HIRES_DENOISE}/{HIRES_STEPS}  {dt:.0f}s")
                return (orig_label, sub_label, out_path)
            except Exception as e:
                print(f"[FAIL] {orig_label} × {sub_label}: {repr(e)}")
                return None

        r = run_one_with_cfg(client, orig_label, SUBJECT_PICK, sub_prompt,
                             seed_name, cfg_v86, out_dir, 0, cfg_override=7.5)
        if r:
            results.append(r)

    print(f"\n[DONE] {len(results)}/1")
    return out_dir


if __name__ == "__main__":
    main()