# -*- coding: utf-8 -*-
"""v407_p6_title.py —— p6 新文本**按原图排版位置 + 大概形状构图**生成（第 27 轮）

用户第 27 轮口谕（原话）：
  「按照原图文本排版位置，大概形状构图去生成新的文本样子，
    不允许换位置生成，也不允许用色块背景遮盖原图内容，
    主体老鹰跟骷髅头的裂变需要跟原图拉开，棕榈树可以先保留只做老鹰这一张，
    严格记住我的要求不要让我一直讲」

量测到的问题（本脚本要修的）
----------------------------
原图 MRCHOSR 标题（左区 x<1150，无鹰头干扰）：
    密集字身带  y[359,1286]  高 **927**     墨迹 x[12,3524]（满幅 3513）
v405 的 VORGRAVEN（同一量法）：
    密集字身带  y[192,730]   高 **538**     且 y>800 后几乎无墨（无下垂长须）
=> 新字**只有原图的 58% 高、且整体偏上、缺少下垂尖刺** → 用户看到的就是
   "一排整齐小字浮在上方"，而不是原图那种**顶天立地、扎进鹰头的巨大尖刺字标**。

做法
----
1. 骨架 = MetalMania 排 `{WORD}`（锁拼写，🔴43）**再纵向拉伸 VSTRETCH 倍**，
   使 `dense_h / ink_w` 从 0.170 提到原图的 **0.264**（VSTRETCH≈1.55）；
   画布同步加高到 `W x H2`，否则拉伸后的骨架会被裁掉。
2. 可选**斜切 SHEAR**（原图 MRCHOSR 的字是右倾的）。
3. canny 弱引导 `cn .50 / end .50` + Harrlogos `1.30`（🔴43 定式）。**关键改动**：
   旧 NEG 里写着 `tall letters / vertical layout / stretched vertical letters`
   —— 正是它把字压成"矮扁横条"。本轮把这些从 NEG 移除，POS 改成
   高壮字身（`tall heavy chunky letters`），让模型长成原图那种高度。
4. 出多 seed 候选，按「密集字身带高 ≈ 927（换算到宽 3513 后）」+ 拼写可读挑最优。

产出：jobs/v407_title/w_<WORD>.png + jobs/v407_title/S_候选对比.jpg
用法：python src/v407_p6_title.py --words VORGRAVEN
      python src/v407_p6_title.py --words VORGRAVEN --vstretch 1.55 --shear 0
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

OUT = ROOT / "jobs" / "v407_title"
OUT.mkdir(parents=True, exist_ok=True)
ORIG = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")
FONT = ROOT / "fonts" / "MetalMania-Regular.ttf"

W = 1536
# ⚠️ 画布高度**必须让拉伸后的骨架填满**，否则模型会把空白区填满荆棘纹理
# → 合成后标题周围糊一层灰雾/乱刺（第 27 轮首试 H2=768 实测就是这个毛病）。
# 骨架纵拉 1.55 后 ≈1438x470（aspect 3.06）→ 配 1536x512 刚好填满（留 ~21px 边）。
H2 = 560
CN, CN_END, LORA, CFG = 0.50, 0.50, 1.30, 7.5

# 目标（原图 MRCHOSR 实测，全分辨率）
TARGET_DENSE = 927.0        # 密集字身带高
TARGET_W = 3513.0           # 墨迹满幅宽
TARGET_RATIO = TARGET_DENSE / TARGET_W      # 0.2639
# 骨架墨迹高宽比目标：由 VORGRAVEN 定稿实测反推（骨架 1438x515 → 成品密集带 968 ≈ 原图 976）
TARGET_SKEL_RATIO = 0.358
# 原图密集字身带顶 / 底（全分辨率）→ 排版落位目标
TARGET_DENSE_TOP = 359.0
TARGET_DENSE_BOT = 1286.0

CKPT = "sd_xl_base_1.0.safetensors"
LORA_NAME = "Harrlogos_XL_v2.safetensors"
CN_NAME = "controlnet-canny-sdxl-1.0.fp16.safetensors"

STYLE = ("very thick heavy strokes, massive fat solid letterforms filling the space, "
         "tall heavy chunky letters, thick bold letterforms, "
         "letters standing tall and proud filling the frame vertically, "
         "long curved thorny blades sweeping outward from the letters, "
         "hooked curved wing spikes clustered at the left and right ends, "
         "jagged needles hanging down from the letter bottoms, "
         "solid white letters with thin black slits, "
         "wide death metal band logo")
NEG = ("thin outline letters, wire thin strokes, hollow letters, thin spindly needles, "
       "plain flat font, uniform plain letters, sans-serif, boring plain typeface, no spikes, "
       "clean smooth letterforms, gray background, grey background, gradient background, "
       "mottled background, textured background, thorny vine background, bramble background, "
       "leaves behind letters, branches behind letters, clutter behind letters, "
       "vignette, dark vignette, smoke, haze, fog, noise texture, scratched background, "
       "misspelled letters, wrong spelling, garbled text, scrambled letters, two words, split word, "
       "sparse, thin hairlines, small letters, tiny text, faint, "
       "bird, animal, skull, illustration, drawing, figure, character, photo, 3d render, color, "
       "frame, border, repeated text, duplicated letters, stacked text, "
       "extra letters, watermark, cluttered, blurry")


def skeleton(word: str, vstretch: float, shear_deg: float) -> Path:
    """MetalMania 排字 → **自适应纵拉**（把字标墨迹高宽比归一到 TARGET_SKEL_RATIO）
    → 可选斜切 → 高画布居中。

    ⚠️ 为什么必须"自适应"而不是固定倍数（第 27 轮实测）：
    MetalMania 各字母宽度差异极大（I/L 很窄、M/O 很宽）→ 同样"排满 94% 宽"，
    含窄字母的词（IRONVEIL 有两个 I）要用**更大的字号** → 排出来更高。
    实测固定 1.70 倍下：IRONVEIL 骨架 1440x685（**超出 560 画布被裁**）、
    RAVEN 成品 placed_dh=1290、THORNMOURN 只 832 —— 同一批词高矮乱飞。
    归一化后所有词骨架比例一致 → 成品密集字身带都落在原图的 976 附近。
    """
    p = OUT / f"_skel_{word}_v{vstretch:.2f}_s{shear_deg:.0f}_h{H2}_n2.png"
    if p.exists():
        return p
    W0 = W
    # ① 先按 W0 宽排满 94%
    lo, hi, best = 8.0, float(W0 * 2), None
    for _ in range(40):
        mid = (lo + hi) / 2.0
        bb = ImageFont.truetype(str(FONT), max(8, int(mid))).getbbox(word)
        if (bb[2] - bb[0]) <= W0 * 0.94:
            best = (max(8, int(mid)), bb)
            lo = mid
        else:
            hi = mid
    size, bb = best
    base = Image.new("L", (W0, int(size * 2.2)), 0)
    d = ImageDraw.Draw(base)
    d.text(((W0 - (bb[2] - bb[0])) / 2 - bb[0], (base.height - (bb[3] - bb[1])) / 2 - bb[1]),
           word, font=ImageFont.truetype(str(FONT), size), fill=255)
    # ② 裁到墨迹 bbox
    a = np.asarray(base) > 40
    ys, xs = np.nonzero(a)
    base = base.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    bw_, bh_ = base.size
    # ③ **自适应纵向定标**：把墨迹高宽比归一到 TARGET_SKEL_RATIO（vstretch 作微调系数）
    target = TARGET_SKEL_RATIO * (vstretch / 1.70)
    new_w = int(round(min(W0 * 0.94, H2 * 0.95 / target)))
    new_h = max(8, int(round(new_w * target)))
    base = base.resize((new_w, new_h), Image.LANCZOS)
    # ④ 斜切（正=右倾，和原图字势一致）
    if abs(shear_deg) > 0.01:
        sh = np.tan(np.radians(shear_deg))
        pad = int(abs(sh) * new_h) + 8
        nb = new_w + pad * 2
        base = base.transform((nb, new_h), Image.AFFINE,
                              (1.0, -sh, pad + (sh * new_h if sh < 0 else 0.0), 0.0, 1.0, 0.0),
                              resample=Image.BICUBIC)
    # ⑤ 缩回画布内、居中
    lim_w, lim_h = int(W0 * 0.96), int(H2 * 0.96)
    k = min(1.0, lim_w / base.width, lim_h / base.height)
    if k < 1.0:
        base = base.resize((max(8, int(base.width * k)), max(8, int(base.height * k))), Image.LANCZOS)
    cv = Image.new("L", (W0, H2), 0)
    cv.paste(base, ((W0 - base.width) // 2, (H2 - base.height) // 2))
    cv.convert("RGB").save(p)
    print(f"[skel] {word} size={size} ink={bw_}x{bh_} -> {base.size} on {cv.size} "
          f"(target ratio {target:.3f})")
    return p


def wf(word: str, seed: int, cn_img: str, lora: float = LORA) -> dict:
    pos = (f"black metal band logo lettering spelling {word}, {STYLE}, "
           "white on pure black background, text only")
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "LoraLoader", "inputs": {
            "lora_name": LORA_NAME, "strength_model": lora, "strength_clip": lora,
            "model": ["1", 0], "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["2", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["2", 1]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": W, "height": H2, "batch_size": 1}},
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
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "v407title"}},
    }


def ink_metrics(path: Path) -> dict:
    g = np.asarray(Image.open(path).convert("L"), np.float32) / 255.0
    bw = max(2, int(round(min(g.shape) * 0.04)))
    bd = np.concatenate([g[:bw].ravel(), g[-bw:].ravel(),
                         g[:, :bw].ravel(), g[:, -bw:].ravel()])
    if float(np.median(bd)) > 0.5:
        g = 1.0 - g
    ink = g > 0.45
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return {"ok": False}
    iw = int(xs.max() - xs.min() + 1)
    ih = int(ys.max() - ys.min() + 1)
    sub = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    rows = sub.sum(1)
    dens = np.where(rows > rows.max() * 0.25)[0]
    dh = int(dens.max() - dens.min() + 1) if len(dens) else ih
    d_off = int(dens.min()) if len(dens) else 0
    sc = TARGET_W / iw
    placed_dh = dh * sc
    placed_ih = ih * sc
    placed_dtop = d_off * sc
    # **背景洁净度**（第 27 轮新增，必查）：字标"白字纯黑底"才算合格。
    # 实测 RAVEN / IRONVEIL 两个种子吐的是"灰底荆棘纹理"（白墨像素 0%）——
    # 单看 placed_dh 很准（975/1001）却完全不可用（合成后字母发灰/消失）。
    # 判据：中灰像素占比 mid% 低 + 白墨占比 ink% 足够。
    mid_pct = float(((g > 0.28) & (g < 0.72)).mean()) * 100.0
    ink_pct = float((g > 0.75).mean()) * 100.0
    clean = (mid_pct < 12.0) and (ink_pct > 10.0)
    score = abs(placed_dh - TARGET_DENSE) / 100.0 + (0.0 if clean else 3.0) + mid_pct / 40.0
    return dict(ok=True, iw=iw, ih=ih, dh=dh, ratio=dh / iw,
                placed_dh=placed_dh, placed_ih=placed_ih, placed_dtop=placed_dtop,
                mid_pct=mid_pct, ink_pct=ink_pct, clean=clean,
                fill=float(sub.sum()) / sub.size, score=score)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", default="VORGRAVEN")
    ap.add_argument("--vstretch", type=float, default=1.55)
    ap.add_argument("--shear", type=float, default=0.0)
    ap.add_argument("--lora", type=float, default=LORA)
    ap.add_argument("--tag", default="", help="输出文件名后缀，便于多配置并存")
    ap.add_argument("--seeds", default="888,3571,4242")
    a = ap.parse_args()

    words = [w.strip().upper() for w in a.words.split(",") if w.strip()]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    cl = ComfyClient()
    up: dict = {}
    picks = []
    for word in words:
        sp = skeleton(word, a.vstretch, a.shear)
        if word not in up:
            with open(sp, "rb") as f:
                r = requests.post(COMFYUI_URL + "/upload/image",
                                  files={"image": (f"v407_skel_{word}.png", f, "image/png")},
                                  timeout=120)
            r.raise_for_status()
            up[word] = r.json().get("name", f"v407_skel_{word}.png")
        best, best_p, best_m = None, None, None
        for sd in seeds:
            t = time.time()
            try:
                res = cl.run(wf(word, sd, up[word], a.lora), timeout=1800)
            except Exception as e:                              # noqa: BLE001
                print(f"[FAIL] {word} s{sd}: {e}", flush=True)
                continue
            for _n, blobs in res.items():
                for b in blobs:
                    Image.open(io.BytesIO(b)).convert("RGB").save(OUT / f"{word}{a.tag}_s{sd}.png")
            cand = OUT / f"{word}{a.tag}_s{sd}.png"
            m = ink_metrics(cand)
            if not m.get("ok"):
                continue
            print(f"  {word:12s} s{sd:<6d} {time.time() - t:5.0f}s  "
                  f"dh={m['placed_dh']:.0f} mid={m['mid_pct']:.1f}% ink={m['ink_pct']:.1f}% "
                  f"{'OK ' if m['clean'] else 'DIRTY'} score={m['score']:.3f}", flush=True)
            if best is None or m["score"] < best["score"]:
                best, best_p, best_m = m, cand, m
        if best_p:
            dst = OUT / f"w_{word}.png"
            shutil.copy2(best_p, dst)
            picks.append((word, dst, best_m))
            print(f"[pick] {word} <- {best_p.name}  placed_dh={best_m['placed_dh']:.0f} "
                  f"(目标 {TARGET_DENSE:.0f})  ratio={best_m['ratio']:.4f}")

    # 候选拼版（含骨架）
    tiles = [("ORIG MRCHOSR 标题带", Image.open(ORIG).convert("RGB").crop((0, 0, 3543, 1560)))]
    for word in words:
        sp = OUT / f"_skel_{word}_v{a.vstretch:.2f}_s{a.shear:.0f}_h{H2}_n2.png"
        if sp.exists():
            tiles.append((f"骨架 {word} v{a.vstretch} sh{a.shear}", Image.open(sp).convert("RGB")))
        for sd in seeds:
            p = OUT / f"{word}{a.tag}_s{sd}.png"
            if p.exists():
                tiles.append((f"{word}{a.tag} s{sd}", Image.open(p).convert("RGB")))
    tw, gap, cap, cols = 1000, 8, 22, 2
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
        d.text((x + 4, y + 4), l, fill=(255, 214, 110))
    p = OUT / "S_候选对比.jpg"
    sh.save(p, quality=92)
    print("sheet ->", p, sh.size)


if __name__ == "__main__":
    main()
