r"""v9.0 illust_1 单图验证：切 Proteus v0.4 底模，治 Juggernaut 写实质感残留。

根因（用户 14:58 反馈 "画面毛躁"）：
  - Juggernaut Ragnarok 是写实底模，训练集多为摄影/写实人像，
    对矢量平面控制弱，残留"颗粒噪点"被 USDU 4x 放大成可见碎裂感。
  - v8.6 已通过强化细化 KSampler + VECTOR LoRA 把噪点降到 8.5/10，
    但主体复现度（孔雀剪影）仍弱——Juggernaut 不擅长画孔雀。

v9.0 改动：
  - 切底模 juggernautXL_ragnarokBy → ProteusV0.4
    Proteus 是 typography 优化底模（dataautogpt3/ProteusV0.4），
    对细节和轮廓更干净，预期降低 AI 风残留 + 改善主体辨识度。
  - 保留 v8.6 所有改进：hires_denoise 0.20 / hires_steps 40 / cfg 7.5
  - VECTOR LoRA 保留 0.55（v8.6 已验证）

执行：
  - Proteus 落盘路径：C:\Users\lenovo\WorkBuddy\2026-08-24-16-39-13\models_extra\Proteus_v0.4.safetensors
  - 拷贝到：E:\Desktop\双接口\image-fission\ComfyUI\models\checkpoints\ProteusV0.4.safetensors
  - 然后跑本脚本
"""
import os
import shutil
import sys
import time

# ---- 1. 拷贝 Proteus 到 ComfyUI checkpoints ----
SRC = r"C:\Users\lenovo\WorkBuddy\2026-08-24-16-39-13\models_extra\Proteus_v0.4.safetensors"
DST_DIR = r"E:\Desktop\双接口\image-fission\ComfyUI\models\checkpoints"
DST = os.path.join(DST_DIR, "ProteusV0.4.safetensors")

def ensure_proteus():
    """确保 Proteus 在 ComfyUI checkpoints 目录。如已在，直接跳过；不在则拷贝。"""
    if os.path.exists(DST):
        size = os.path.getsize(DST)
        if size > 6 * 1024**3:  # >6GB 视为完整
            print(f"[proteus] 已存在 {DST} ({size/1024**3:.2f}GB)")
            return True
        else:
            print(f"[proteus] {DST} 存在但只有 {size/1024**3:.2f}GB，疑似不完整")
            return False
    if not os.path.exists(SRC):
        print(f"[proteus] 源文件不存在: {SRC}")
        print("        请等 task 1IlHRK 下载完成")
        return False
    size = os.path.getsize(SRC)
    print(f"[proteus] 源文件 {SRC} ({size/1024**3:.2f}GB)")
    if size < 6 * 1024**3:
        print(f"[proteus] 源文件不完整（{size/1024**3:.2f}GB < 6GB），跳过")
        return False
    print(f"[proteus] 拷贝到 ComfyUI checkpoints ...")
    shutil.copy2(SRC, DST)
    print(f"[proteus] 完成 {DST}")
    return True


# ---- 2. 跑 v9.0 illust_1 单图 ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import demo_batch_6x4 as db

# v9.0 关键改动：切底模
db._cfg.SDXL_CHECKPOINT = "ProteusV0.4.safetensors"

# v9.0 保留 v8.6 改进
db.HIRES_DENOISE = 0.20
db.HIRES_STEPS = 40

from engine.comfy_client import ComfyClient

ONLY_ORIG = "illust_1"
SUBJECT_PICK = "peacock_floral"


def main():
    if not ensure_proteus():
        print("[abort] Proteus 不可用")
        return None

    out_dir = os.path.join(db.JOBS_BASE, f"smoke_v90_illust_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")
    print(f"[v9.0 cfg] checkpoint={db._cfg.SDXL_CHECKPOINT}")
    print(f"[v9.0 cfg] hires_denoise={db.HIRES_DENOISE}  hires_steps={db.HIRES_STEPS}")

    client = ComfyClient()
    results = []

    for orig_seed, orig_label in db.ORIGINALS:
        if orig_label != ONLY_ORIG:
            continue
        cfg = db.ORIGINALS_CONFIG.get(orig_label)
        sub_prompt = dict(cfg["subjects"])[SUBJECT_PICK]

        cfg_v90 = dict(cfg)
        cfg_v90["extra_lora_strength"] = 0.55  # 保留 v8.6
        cfg_v90["sim"] = cfg.get("sim", 0.85)
        cfg_v90["comp"] = cfg.get("comp", 0.72)
        cfg_v90["cn"] = cfg.get("cn", 0.62)

        seed_src = os.path.join(db.COMFYUI_INPUT, f"{orig_seed}.jpg")
        if not os.path.exists(seed_src):
            print(f"[WARN] skip {orig_seed}")
            continue
        seed_name = f"smoke_{orig_label}_{int(time.time()*1000)}.jpg"
        shutil.copy(seed_src, os.path.join(db.COMFYUI_INPUT, seed_name))

        # 复用 v8.6 的内联 run_one_with_cfg 模式
        from demo_batch_6x4 import (
            STYLE_PREFIX, STYLE_SUFFIX, IPA_WEIGHT_TYPE, IPA_NOISE, IPA_END,
            CN_LOW, CN_HIGH, CONTROLNET, USE_CONTROLNET, USDU_MODEL,
            HIRES_SCALE, HIRES_DENOISE, HIRES_STEPS, TEXTILE_LORA,
            TEXTILE_LORA_STRENGTH,
        )
        import random
        from pipelines.build import build_mode1

        style_words = cfg_v90.get("style_words", "")
        sim = cfg_v90.get("sim", 0.70)
        comp = cfg_v90.get("comp", 0.55)
        cn_v = cfg_v90.get("cn", 0.40)
        extra_lora = cfg_v90.get("extra_lora")
        extra_lora_strength = cfg_v90.get("extra_lora_strength", 0.40)

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
            print(f"[OK] {orig_label} × {SUBJECT_PICK}  seed={seed}  ckpt={db._cfg.SDXL_CHECKPOINT}  "
                  f"VECTOR={extra_lora_strength}  hires={HIRES_DENOISE}/{HIRES_STEPS}  {dt:.0f}s")
            results.append((orig_label, sub_label, out_path))
        except Exception as e:
            print(f"[FAIL] {orig_label} × {sub_label}: {repr(e)}")

    print(f"\n[DONE] {len(results)}/1")
    return out_dir


if __name__ == "__main__":
    main()