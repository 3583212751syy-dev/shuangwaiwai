# -*- coding: utf-8 -*-
"""v405_bank_style.py —— p6 词库**全部 10 个词**统一按原图 MRCHOSR 的词体风格重做

用户第 26 轮口谕：「字体按照2 —— 一般根据原图字体风格设计，做相同风格的字体设计。」
第 25 轮的 10 个词风格各飞（RAVEN 黑字浅底 / IRONVEIL 细针 / n4 高大方块…），
排版高度 1787~1930px 远超原图标题带 926px → 压住老鹰。

第 26 轮为 VORGRAVEN 摸出的定式（见 v404_vorg_style.py 第 4/5 轮）：
  **MetalMania 排版骨架（锁拼写） + canny 弱引导（cn .50 / end .50） + Harrlogos 高权重（1.30）长尖刺**
本脚本把这套定式**批量套到全部 10 个词**，每个词出 2 个种子，按指标挑最优。

产出：jobs/v405_bank/w_<WORD>.png + S_词库风格重做_对比.jpg
用法：python src/v405_bank_style.py            # 只做缺的
      python src/v405_bank_style.py --force    # 全部重做
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from engine.comfy_client import ComfyClient, COMFYUI_URL         # noqa: E402
import requests                                                 # noqa: E402
from scipy import ndimage as ndi                                # noqa: E402

OUT = ROOT / "jobs" / "v405_bank"
OUT.mkdir(parents=True, exist_ok=True)
GLY = ROOT / "jobs" / "_probe" / "elemgen_mid"
ORIG = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")
FONT = ROOT / "fonts" / "MetalMania-Regular.ttf"

W, H = 1536, 512
CN, CN_END, LORA, CFG = 0.50, 0.50, 1.30, 7.5
SEEDS = [888, 3571]

BANK = ["RAVEN", "MOURNGRAVE", "SKARVALD", "IRONVEIL", "GRAVETIDE",
        "STORMHELM", "ASHREAVER", "DUSKBANE", "THORNMOURN"]

CKPT = "sd_xl_base_1.0.safetensors"
LORA_NAME = "Harrlogos_XL_v2.safetensors"
CN_NAME = "controlnet-canny-sdxl-1.0.fp16.safetensors"

STYLE = ("bold fat wide letterforms, thick heavy chunky letters, "
         "letters stretched wide across the frame forming a very wide flat horizontal logo, "
         "long curved thorny blades sweeping outward from the letters, "
         "hooked curved wing spikes clustered at the left and right ends, "
         "short spikes above and below, solid white letters with thin black slits, "
         "symmetrical death metal band logo")

NEG = ("tall letters, vertical layout, narrow tall composition, one huge central spike going up, "
       "thin needle spikes, spindly hairline spikes, stretched vertical letters, "
       "plain flat font, uniform plain letters, sans-serif, boring plain typeface, no spikes, "
       "clean smooth letterforms, gray background, grey background, gradient background, "
       "misspelled letters, wrong spelling, garbled text, scrambled letters, two words, split word, "
       "sparse, thin hairlines, small letters, tiny text, faint, "
       "bird, animal, skull, illustration, drawing, figure, character, photo, 3d render, color, "
       "frame, border, repeated text, duplicated letters, stacked text, "
       "extra letters, watermark, cluttered, blurry")


def skeleton(word: str) -> Path:
    p = OUT / f"_skel_{word}.png"
    if p.exists():
        return p
    lo, hi, best = 8.0, float(H * 4), None
    for _ in range(40):
        mid = (lo + hi) / 2.0
        bb = ImageFont.truetype(str(FONT), max(8, int(mid))).getbbox(word)
        if (bb[2] - bb[0]) <= W * 0.94:
            best = (max(8, int(mid)), bb)
            lo = mid
        else:
            hi = mid
    size, bb = best
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    d.text(((W - (bb[2] - bb[0])) / 2 - bb[0], (H - (bb[3] - bb[1])) / 2 - bb[1]),
           word, font=ImageFont.truetype(str(FONT), size), fill=255)
    img.convert("RGB").save(p)
    return p


def wf(word: str, seed: int, cn_img: str) -> dict:
    pos = (f"black metal band logo lettering spelling {word}, {STYLE}, "
           "white on pure black background, text only")
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "LoraLoader", "inputs": {
            "lora_name": LORA_NAME, "strength_model": LORA, "strength_clip": LORA,
            "model": ["1", 0], "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["2", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["2", 1]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 30, "cfg": CFG, "sampler_name": "dpmpp_2m",
            "scheduler": "karras", "denoise": 1.0, "model": ["2", 0],
            "positive": ["13", 0], "negative": ["13", 1], "latent_image": ["5", 0]}},
        "10": {"class_type": "LoadImage", "inputs": {"image": cn_img}},
        "11": {"class_type": "Canny", "inputs": {
            "image": ["10", 0], "low_threshold": 0.30, "high_threshold": 0.70}},
        "12": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_NAME}},
        "13": {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": ["3", 0], "negative": ["4", 0], "control_net": ["12", 0],
            "image": ["11", 0], "strength": CN, "start_percent": 0.0, "end_percent": CN_END}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "wbank5"}},
    }


def ink_metrics(path: Path) -> dict:
    g = np.asarray(Image.open(path).convert("L"), np.float32) / 255.0
    bw = max(2, int(round(min(g.shape) * 0.04)))
    bd = np.concatenate([g[:bw].ravel(), g[-bw:].ravel(), g[:, :bw].ravel(), g[:, -bw:].ravel()])
    if float(np.median(bd)) > 0.5:
        g = 1.0 - g
    ink = g > 0.45
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return {"fill": 0, "asp": 0, "score": -9}
    sub = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    rows = sub.sum(1)
    dense = np.where(rows > rows.max() * 0.25)[0]
    dh = int(dense.max() - dense.min() + 1) if len(dense) else sub.shape[0]
    A = int(sub.sum())
    per = int((sub & ~ndi.binary_erosion(sub)).sum())
    fill = A / sub.size
    asp = sub.shape[1] / max(dh, 1)
    # 目标：fill≈0.40（原图 0.403）、asp≈4.5（原图 4.49）→ 分越小越好
    return {"fill": fill, "asp": asp,
            "score": abs(fill - 0.40) * 3.0 + abs(min(asp, 7.0) - 4.5) * 0.10}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cl = ComfyClient()
    up: dict = {}
    for word in BANK:
        dst = GLY / f"w_{word}.png"
        if dst.exists() and not a.force:
            print(f"[skip] {word}")
            continue
        if word not in up:
            with open(skeleton(word), "rb") as f:
                r = requests.post(COMFYUI_URL + "/upload/image",
                                  files={"image": (f"v405_skel_{word}.png", f, "image/png")},
                                  timeout=120)
            r.raise_for_status()
            up[word] = r.json().get("name", f"v405_skel_{word}.png")
        best, best_p = None, None
        for sd in SEEDS:
            t = time.time()
            try:
                res = cl.run(wf(word, sd, up[word]), timeout=1800)
            except Exception as e:                              # noqa: BLE001
                print(f"[FAIL] {word} s{sd}: {e}", flush=True)
                continue
            for _n, blobs in res.items():
                for b in blobs:
                    p = OUT / f"{word}_s{sd}.png"
                    Image.open(io.BytesIO(b)).convert("RGB").save(p)
            cand = OUT / f"{word}_s{sd}.png"
            m = ink_metrics(cand)
            print(f"  {word:12s} s{sd:<6d} {time.time() - t:5.0f}s  "
                  f"fill={m['fill']:.3f} asp={m['asp']:.2f} score={m['score']:.3f}", flush=True)
            if best is None or m["score"] < best["score"]:
                best, best_p = m, cand
        if best_p:
            shutil.copy2(best_p, dst)
            print(f"[pick] {word:12s} <- {best_p.name}  fill={best['fill']:.3f} asp={best['asp']:.2f}")

    # 对比图
    tiles = [("ORIG MRCHOSR（风格基准）", Image.open(ORIG).convert("RGB").crop((0, 130, 3543, 1100)))]
    for word in ["VORGRAVEN"] + BANK:
        p = GLY / f"w_{word}.png"
        if p.exists():
            tiles.append((word, Image.open(p).convert("RGB")))
    tw, gap, cap, cols = 900, 8, 24, 3
    s = [(l, im.resize((tw, max(1, int(im.height * tw / im.width))), Image.LANCZOS)) for l, im in tiles]
    th = max(x[1].height for x in s)
    rows = (len(s) + cols - 1) // cols
    sh = Image.new("RGB", (tw * cols + gap * (cols + 1), (th + cap) * rows + gap), (22, 22, 22))
    d = ImageDraw.Draw(sh)
    for i, (l, im) in enumerate(s):
        r, c = divmod(i, cols)
        x = gap + c * (tw + gap)
        y = gap + r * (th + cap)
        sh.paste(im, (x, y + cap))
        d.text((x + 4, y + 5), l, fill=(255, 214, 110))
    p = OUT / "S_词库风格重做_对比.jpg"
    sh.save(p, quality=92)
    print("sheet ->", p, sh.size)


if __name__ == "__main__":
    main()
