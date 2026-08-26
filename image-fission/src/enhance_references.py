r"""参考图预处理：高清化 + 去噪点 + 去毛边。

用户判断（2026-08-25 15:14）：给的参考图本身就是糊的（小尺寸 + JPEG 压缩块），
IPAdapter 把它锁进风格向量，再被 4x USDU 放大成"画面毛躁"的真正上游。
本脚本在裂变前先对参考图做"真实超分 + 潜空间轻去噪"，输出 4096² PNG 作为高质量参考。

工作流（10 节点，1 张参考约 20-30 秒）：
  LoadImage(orig.jpg)
    -> ImageScale(1024, lanczos)  -- 统一画布到 1024²
    -> VAEEncode(Proteus)         -- 1024² 像素 -> 128² latent
    -> KSampler(denoise 0.10, steps 15, cfg 6)  -- 潜空间轻去噪（去 JPEG 压缩块 / 颗粒）
    -> VAEDecode                  -- latent -> 1024² 像素
    -> ImageUpscaleWithModel(4x-UltraSharp)  -- 真实超分（恢复边缘细节 + 锐化去毛边）
    -> SaveImage(prefix=enhance_ref)          -- 落到 ComfyUI/output/enhance_ref_*.png

为何这样：
  - 4x-UltraSharp 是学得的高清模型，恢复 1024→4096 时会"创造"细节（vs Lanczos 只能插值模糊），
    这是治"毛边 / 模糊"的关键。
  - KSampler(denoise 0.10) 在潜空间清理 JPEG 压缩块，权重极低不会重画原图（denoise>0.20 才明显重画）。
  - 用 Proteus v0.4 底模（与裂变保持一致），CLIP/VAE 与后续风格编码同源。

执行：
  - 必须先跑 src/smoke_v90_illust.py 之前的 ensure_proteus() 步骤把模型拷到 ComfyUI/checkpoints
  - 本脚本会自行 ensure_proteus()
"""
import os
import sys
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as _cfg
import demo_batch_6x4 as db
from engine.comfy_client import ComfyClient

# 切到 Proteus v0.4 底模（与裂变管线一致）
_cfg.SDXL_CHECKPOINT = "ProteusV0.4.safetensors"

PROTEUS_SRC = r"C:\Users\lenovo\WorkBuddy\2026-08-24-16-39-13\models_extra\Proteus_v0.4.safetensors"
PROTEUS_DST = os.path.join(
    r"E:\Desktop\双接口\image-fission\ComfyUI\models\checkpoints",
    "ProteusV0.4.safetensors",
)


def ensure_proteus():
    if os.path.exists(PROTEUS_DST) and os.path.getsize(PROTEUS_DST) > 6 * 1024**3:
        size_gb = os.path.getsize(PROTEUS_DST) / 1024**3
        print(f"[proteus] 已就位 {PROTEUS_DST} ({size_gb:.2f}GB)")
        return True
    if not os.path.exists(PROTEUS_SRC):
        print(f"[proteus] 源文件不存在: {PROTEUS_SRC}")
        return False
    print(f"[proteus] 拷贝到 ComfyUI checkpoints ...")
    shutil.copy2(PROTEUS_SRC, PROTEUS_DST)
    size_gb = os.path.getsize(PROTEUS_DST) / 1024**3
    print(f"[proteus] 完成 {size_gb:.2f}GB")
    return True


def build_enhance_ref_workflow(input_filename: str, params: dict) -> dict:
    """高清化 + 去噪 + 锐化（去毛边）参考图工作流。"""
    g = {}
    # 1. Checkpoint (Proteus)
    g["1"] = {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": _cfg.SDXL_CHECKPOINT}}
    # 2. LoadImage
    g["2"] = {"class_type": "LoadImage",
              "inputs": {"image": input_filename, "upload": "image"}}
    # 3. ImageScale -> 1024² lanczos
    g["3"] = {"class_type": "ImageScale",
              "inputs": {"image": ["2", 0],
                         "width": 1024, "height": 1024,
                         "crop": "disabled",
                         "upscale_method": "lanczos"}}
    # 4/5. CLIP (pos=让模型"高保真复原"，neg=拒绝噪点/模糊/压缩块)
    pos = ("high quality illustration, sharp fine linework, clean smooth shapes, "
           "vector ink, high resolution, intricate detail")
    neg = ("blurry, noisy, grain, film grain, sensor noise, compression artifacts, "
           "jpeg artifacts, speckles, dust spots, halftone, rough paper texture, "
           "dirty, smudge, distorted, deformed, low quality, "
           "watermark, signature, text, letters")
    g["4"] = {"class_type": "CLIPTextEncode",
              "inputs": {"text": pos, "clip": ["1", 1]}}
    g["5"] = {"class_type": "CLIPTextEncode",
              "inputs": {"text": neg, "clip": ["1", 1]}}
    # 6. VAEEncode
    g["6"] = {"class_type": "VAEEncode",
              "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    # 7. KSampler 轻去噪
    g["7"] = {"class_type": "KSampler",
              "inputs": {
                  "seed": params.get("seed", 12345),
                  "steps": params.get("steps", 15),
                  "cfg": params.get("cfg", 6),
                  "sampler_name": "dpmpp_2m",
                  "scheduler": "karras",
                  "denoise": params.get("denoise", 0.10),
                  "model": ["1", 0],
                  "positive": ["4", 0],
                  "negative": ["5", 0],
                  "latent_image": ["6", 0],
              }}
    # 8. VAEDecode
    g["8"] = {"class_type": "VAEDecode",
              "inputs": {"samples": ["7", 0], "vae": ["1", 2]}}
    # 9. UpscaleModelLoader (4x-UltraSharp)
    g["9"] = {"class_type": "UpscaleModelLoader",
              "inputs": {"model_name": params.get("upscale_model", "4x-UltraSharp.pth")}}
    # 10. ImageUpscaleWithModel (真实 4x 超分，1024 -> 4096)
    g["10"] = {"class_type": "ImageUpscaleWithModel",
               "inputs": {"upscale_model": ["9", 0], "image": ["8", 0]}}
    # 11. SaveImage
    g["11"] = {"class_type": "SaveImage",
               "inputs": {"images": ["10", 0], "filename_prefix": "enhance_ref"}}
    return g


def main():
    if not ensure_proteus():
        print("[abort] Proteus 不可用")
        return None

    out_dir = os.path.join(db.JOBS_BASE, f"enhance_refs_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")
    print(f"[enhance_ref cfg] checkpoint={_cfg.SDXL_CHECKPOINT}  upscale=4x-UltraSharp")
    print(f"                   denoise=0.10  steps=15  cfg=6")

    # 全部 6 张参考图一次性高清化（去噪 + 去毛边 + 真实超分）
    targets = [
        ("pinterest_illust_1", "illust_1"),
        ("pinterest_eagle_2",  "eagle_2"),
        ("pinterest_denim_3_flat", "denim_3"),
        ("pinterest_camo_4",   "camo_4"),
        ("pinterest_skull_5",  "skull_5"),
        ("pinterest_metal_6",  "metal_6"),
    ]

    params = {
        "seed": 12345,
        "steps": 15,
        "cfg": 6,
        "denoise": 0.10,  # 极轻：只清 JPEG 压缩块 / 微噪点，不重画
        "upscale_model": "4x-UltraSharp.pth",
    }

    client = ComfyClient()
    results = []

    for orig_seed, orig_label in targets:
        src = os.path.join(db.COMFYUI_INPUT, f"{orig_seed}.jpg")
        if not os.path.exists(src):
            print(f"[skip] {orig_seed}.jpg missing")
            continue
        src_size = os.path.getsize(src)
        print(f"[enhance] {orig_label}  source={src_size/1024:.0f}KB")

        g = build_enhance_ref_workflow(f"{orig_seed}.jpg", params)
        t0 = time.time()
        try:
            res = client.run(g, timeout=300)
            data = next(iter(res.values()))[0]
            # Save to both ComfyUI/input (for fission to pick up) and jobs (for record)
            input_dst = os.path.join(db.COMFYUI_INPUT, f"{orig_seed}_hd.png")
            jobs_dst = os.path.join(out_dir, f"{orig_label}_hd.png")
            with open(input_dst, "wb") as f:
                f.write(data)
            shutil.copy2(input_dst, jobs_dst)
            dt = time.time() - t0
            size_mb = os.path.getsize(input_dst) / 1024 / 1024
            print(f"[OK] {orig_label} -> {input_dst}  {size_mb:.1f}MB  {dt:.0f}s")
            results.append((orig_label, input_dst, size_mb))
        except Exception as e:
            print(f"[FAIL] {orig_label}: {repr(e)}")

    print(f"\n[DONE] {len(results)}/{len(targets)}")
    for label, path, size_mb in results:
        print(f"  - {label}: {path}  ({size_mb:.1f}MB)")
    return out_dir


if __name__ == "__main__":
    main()
