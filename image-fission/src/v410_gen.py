# -*- coding: utf-8 -*-
"""v410_gen.py — p6 主体移植：① 文生图重画「鹰踩角骷髅」 ② 按原位置贴回（第 29 轮）

为什么再换一次机制
------------------
第 29 轮先试了"掩膜内 denoise=1.0 自由重画"（`v409_p6free.py`）。结果：
姿态确实换了（剪影 IoU 0.67–0.71、逐行宽度相关 0.94–0.97，对比 v408 的 0.87/0.998），
但**质量塌**：区域提示词在 denoise=1.0 下互相打架 → 鹰头糊、胸前一团泥、
前额长黄色斑块、羽层碎。原因不是参数没调好，而是**"在既有图里挖洞重生"这条路
天然要同时满足"位置锁死 + 8 个区域互不干扰 + 解剖合理"三个互相拉扯的目标**。

改成两步分离（各自做自己最擅长的事）：
  ① **生成**：完全不看原图，纯 txt2img 在 SDXL 原生分辨率画一张干净的
     「展翼老鹰踩在带角骷髅头上，哥特丝网印」——SDXL 对这类题材的生成质量极高；
  ② **移植**：把新图的**主体（自有剪影）**按原主体 bbox 缩放对齐后贴回原图，
     掩膜外（黑底 / 白色射线 / 标题带）原图零改动。

于是：变化幅度 = 两张独立生成的差异（必然大）；质量 = 干净 txt2img（必然高）；
位置/配色/版式 = 由对齐 + LAB Reinhard 保证（🔴4）。

用法
----
  python src/v410_gen.py --poses T1,T2,T3,T4 --seeds 111,222,333 --out-dir jobs/v410_t2i
  python src/v410_transplant.py --pick <新图> --tag v410a          # 贴回
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import urllib.request as ur
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from v342_rebirth import COMFY, CUI_IN, _fetch, comfy_submit      # noqa: E402

OUT = ROOT / "jobs" / "v410_t2i"
OUT.mkdir(parents=True, exist_ok=True)

CKPT = "juggernautXL_ragnarokBy.safetensors"
LORA_DETAIL = "add-detail-xl.safetensors"

# 主体 bbox 宽高比 ≈ 3130 : 3388 = 0.924 → 取 SDXL 友好的 1024×1152（0.889）与
# 1152×1280（0.900），两者都在原生训练尺度内、不触发拼接崩坏。
W, H = 1024, 1152
STEPS, CFG = 30, 7.0

# 公用的"风格 + 配色 + 解剖"锚（锁定 🔴4 色族 + 🔴35 解剖红线）
STYLE = (
    "gothic tattoo illustration, bold screen print t-shirt graphic, flat vector illustration, "
    "heavy black linework, crisp hard clean edges, solid flat colors, NO gradients, NO soft shading, "
    "pure black background, "
    "a white and silver human skull, dark chocolate brown eagle wing feathers, "
    "tan ribbed horns, bright golden-yellow beak and talons, fierce yellow eyes, "
    "white lightning bolts behind the skull, "
    "high contrast, centered symmetrical composition, "
    "no text, no letters, no words, no banner, no signature, no frame, no border"
)

NEG = (
    "photo, realistic, 3d render, airbrush, painterly, soft gradients, smooth shading, "
    "gray haze, grey haze, fog, mist, smoke, glow, halo, gradient background, "
    "gray background, dirty background, noise, speckle, "
    "yellow patches on the skull, yellow stains on the bone, yellow nose, "
    "beak on the skull, second beak, melted beak, deformed beak, missing eye, "
    "extra heads, extra skulls, extra birds, extra wings, extra horns, extra legs, "
    "deformed, anatomically incorrect, broken bones, missing claws, floating yellow blobs, "
    "fused elements, melted edges, crowded center, cluttered middle, "
    "text, letters, words, watermark, signature, numbers, frame, border, "
    "blurry, low quality, out of focus, jpeg artifacts, "
    "pale beak, white beak, desaturated, washed out, pastel"
)

# 强调"骷髅必须完整可见"的骨架句（T1~T4 常把骷髅整只画丢）
_SK = ("a LARGE WHITE HORNED SKULL clearly visible and fully readable in the LOWER HALF of the image, "
       "two big tan ribbed horns curving out to the sides, dark empty eye sockets and nasal cavity, "
       "the eagle's golden talons gripping the TOP of that skull, ")

POSES = {
    # T1 双翼高展成 V、正面对镜头 —— 消灭"横向大展翼"这个最抓眼的同构特征
    "T1": ("a bald eagle with BOTH WINGS SPREAD WIDE AND SWEPT UPWARD IN A TALL V shape, "
           "wings dominating the upper half of the image, wingtips pointing up and outward, "
           "the eagle's head FACING THE CAMERA head-on with a fierce open golden beak, "
           "both wings fully feathered in dark chocolate brown with long separated primary feathers, "
           "the eagle perched with golden talons gripped onto the top of a large horned demon skull, "
           "front view of the skull, " + STYLE),
    # T2 鹰侧脸尖叫 + 骷髅正面
    "T2": ("a bald eagle in a fierce SCREAMING SIDE PROFILE, head turned to one side with a wide open "
           "golden hooked beak, single fierce yellow eye, wings raised high and swept back behind the body, "
           "long primary feathers trailing, the eagle perched with golden talons gripped onto a large "
           "horned demon skull below it, front view of the skull, " + STYLE),
    # T3 骷髅 3/4 + 下颌张开；鹰在上双臂展翼
    "T3": ("a bald eagle with wide spread wings perched on top of a huge horned demon skull that is "
           "turned at a THREE-QUARTER angle, the skull's jaw WIDE OPEN screaming with a full row of teeth, "
           "deep cracks and fractures across the white bone, the eagle facing the camera with a fierce eye, "
           "golden talons gripping the skull crown, " + STYLE),
    # T4 鹰低伏收翅
    "T4": ("a bald eagle CROUCHED LOW with its wings HALF-FOLDED back along its flanks like a bird that "
           "just landed, compact powerful low silhouette, head LOWERED and thrust forward with a fierce "
           "open golden beak pointing down, both eyes visible, golden talons gripping the crown of a "
           "large horned demon skull that dominates the lower half, " + STYLE),
    # ── T5~T8：把「骷髅必须完整可见地在鹰下方」写死（T1~T4 常把骷髅画丢）──
    "T5": ("a bald eagle FACING THE CAMERA with both wings spread wide and swept upward in a tall V, "
           "fierce yellow eyes and hooked golden beak, the eagle standing ON TOP OF and perched upon "
           + _SK + "white lightning bolts radiating behind the skull, " + STYLE),
    "T6": ("a bald eagle in a fierce SCREAMING SIDE PROFILE with a wide open golden hooked beak, wings "
           "raised high behind the body, the eagle perched on top of "
           + _SK + "white lightning bolts behind, " + STYLE),
    "T7": ("a bald eagle with wings raised in a V, perched with golden talons on top of "
           + _SK + "the skull turned at a THREE-QUARTER angle with its jaw WIDE OPEN screaming, "
           "deep cracks across the bone, white lightning bolts behind, " + STYLE),
    "T8": ("a bald eagle with wide spread wings perched on top of "
           + _SK + "the skull tilted and cracked with a wide open jaw, "
           "the eagle's head held high, both wings bright with long separated feathers, "
           "white lightning bolts behind the skull, " + STYLE),
}


def build(seed: int, prompt: str, w=W, h=H, lora_w=0.5, steps=STEPS, cfg=CFG) -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "7": {"class_type": "LoraLoader",
              "inputs": {"model": ["1", 0], "clip": ["1", 1], "lora_name": LORA_DETAIL,
                         "strength_model": lora_w, "strength_clip": lora_w}},
        "pg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": prompt}},
        "ng": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "10": {"class_type": "KSampler",
               "inputs": {"model": ["7", 0], "positive": ["pg", 0], "negative": ["ng", 0],
                          "latent_image": ["5", 0], "seed": seed, "steps": steps, "cfg": cfg,
                          "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "13": {"class_type": "SaveImage",
               "inputs": {"images": ["12", 0], "filename_prefix": f"v410_{seed}"}},
    }


def gen(seed: int, pose: str, out_dir=None, **kw) -> Path:
    out_dir = Path(out_dir) if out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"t2i_{pose}_s{seed}.jpg"
    if dst.exists():
        print(f"[v410] 复用 {dst.name}")
        return dst
    t0 = time.time()
    outs = comfy_submit(build(seed, POSES[pose], **kw))
    im = None
    for _n, o in outs.items():
        if "images" in o:
            im = _fetch(o); break
    if im is None:
        raise RuntimeError("no comfy output")
    im.convert("RGB").save(dst, quality=95)
    print(f"[v410] {dst.name} {im.size} {time.time() - t0:.0f}s", flush=True)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", default="T1,T2,T3,T4")
    ap.add_argument("--seeds", default="111,222,333")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--lora", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--cfg", type=float, default=CFG)
    a = ap.parse_args()
    for p in [x.strip() for x in a.poses.split(",") if x.strip()]:
        if p not in POSES:
            raise SystemExit(f"未知 pose {p}")
        for s in [int(x) for x in a.seeds.split(",") if x.strip()]:
            gen(s, p, out_dir=a.out_dir, lora_w=a.lora, steps=a.steps, cfg=a.cfg)


if __name__ == "__main__":
    main()
