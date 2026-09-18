"""_p6_vec.py — p6 向量风格 mode1 试验（第 18 轮）。

问题量化（1:1 目检 + Laplacian 方差）：v386 的 p6 主体源是 Juggernaut(照片模型) 出的
**油画感软画面**，且 base 880×1240 经 4x NMKD 超分到 3543 → 鹰区 Laplacian 方差仅
461（原图 2715，差 5.9 倍）= 用户所说"错乱/糊"。

本试验两处改动：
  ① **换向量风**：ProteusV0.4(插画) / Juggernaut + `DD-vector-v2` LoRA(平涂矢量)
     + prompt 明写 flat vector / bold outlines / screen print + NEG 压 painterly/photo；
  ② **抬高 base 到 1216×1704**（≈2.07MP）→ 4x 得 4864×6816 > 原图 3543 → 末端
     由"放大"变"缩小"，硬边得以保留。
"""
import os
import sys
import time
import shutil

ROOT = r"E:\Desktop\双接口\image-fission"
sys.path.insert(0, os.path.join(ROOT, "src"))
from engine.comfy_client import ComfyClient           # noqa: E402
from pipelines.build import build_mode1               # noqa: E402
import config as _cfg                                 # noqa: E402

SRC = r"E:\Desktop\图裂变测试图\Pinterest (6).jpg"
INP = os.path.join(ROOT, "ComfyUI", "input")
os.makedirs(INP, exist_ok=True)
NAME = "p6vec_seed.jpg"
shutil.copy(SRC, os.path.join(INP, NAME))

PROMPT = ("flat vector illustration of a fierce bald eagle perched on a horned human skull, "
          "bold clean black outlines, flat cel shading, screen print, high contrast, "
          "white brown black palette, vintage heavy metal band poster art, no text")
NEG = ("painterly, airbrush, soft shading, blur, blurry, gradient, photo, photorealistic, "
       "3d render, low quality, deformed, watermark, text, letters")

VARIANTS = [
    ("V1_proteus_ddv_base1216", "ProteusV0.4.safetensors",
     "DD-vector-v2.safetensors", 0.70, 1216, 1704),
    ("V2_jugg_ddv_base1216", "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
     "DD-vector-v2.safetensors", 0.90, 1216, 1704),
]

OUT = os.path.join(ROOT, "jobs", "v388_p6vec")
os.makedirs(OUT, exist_ok=True)
client = ComfyClient()
orig = _cfg.SDXL_CHECKPOINT
try:
    for tag, ckpt, lora, lw, w, h in VARIANTS:
        _cfg.SDXL_CHECKPOINT = ckpt
        p = dict(style_prompt=PROMPT, negative_prompt=NEG, seed=4242,
                 width=w, height=h, batch_per_run=1, steps=30, cfg=5.0,
                 color_strength=0.55, composition_strength=0.45,
                 ipadapter_noise=0.12, ipadapter_end=0.85,
                 lora_name=lora, lora_strength=lw,
                 controlnet_name="controlnet-canny-sdxl-1.0.fp16.safetensors",
                 controlnet_strength=0.45, controlnet_end=0.55)
        g = build_mode1(NAME, p, f"p6vec_{tag}")
        t0 = time.time()
        try:
            res = client.run(g, timeout=1800)
            _, imgs = next(iter(res.items()))
            dst = os.path.join(OUT, f"{tag}.jpg")
            with open(dst, "wb") as f:
                f.write(imgs[0])
            print(f"[OK] {tag} {time.time()-t0:.0f}s -> {dst} "
                  f"({os.path.getsize(dst)} bytes)")
        except Exception as e:
            print(f"[FAIL] {tag}: {repr(e)}")
finally:
    _cfg.SDXL_CHECKPOINT = orig
print("[done]")
