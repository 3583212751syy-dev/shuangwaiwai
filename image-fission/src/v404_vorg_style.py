# -*- coding: utf-8 -*-
"""v404_vorg_style.py —— p6 标题「VORGRAVEN」按**原图 MRCHOSR 的字体风格**重做

用户第 26 轮口谕
----------------
「棕榈2和3，字体按照2 —— 一般根据原图字体风格设计，做相同风格的字体设计。」

含义：p6 词库选定 **#2 = VORGRAVEN**；但它的字标不能是现在这种"细高垂直针"
（n5 那版 war-metal 风），要**照原图 MRCHOSR 的风格重做**。

原图 MRCHOSR 风格实测（标题带 y154~1080）
----------------------------------------
  · 宽高比 ≈ 3.8（横向铺满、字身带矮）
  · 字身**粗壮宽体**，笔画实心白，内部有细黑缝
  · **长弯钩刀刃**从字母左右向外甩（两侧各一簇"翼刺"）
  · 上下有短尖刺，但**不是**贯穿全幅的高针
对比 n5(VORGRAVEN)：字身带宽高比 2.2、笔画细、中央一根通天高针 → 完全另一流派。

做法
----
1. txt2img（SDXL + Harrlogos_XL_v2）用**风格锁死**的 prompt 重出 VORGRAVEN，
   画布取**宽幅**（1536x672 / 1600x640），逼模型横向铺字而不是竖着长针；
   负提示压 "thin needle / very tall vertical letters / narrow"。
2. 另出**canny 引导**一档：把原图标题带当结构参考（低强度 0.10~0.14），
   只借它的"宽体+弯钩"气质，不锁字母轮廓。
3. 每张算风格指标（字身带宽高比 asp / 填充率 fill / 笔画宽 stroke），
   与 ORIG 对比，拼成对比图供肉眼挑。

用法：
    python src/v404_vorg_style.py            # 出全部候选
    python src/v404_vorg_style.py --sheet    # 只重建对比图
产出：jobs/v404_vorg/cand_*.png + S_风格对比.jpg
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from engine.comfy_client import ComfyClient, COMFYUI_URL          # noqa: E402
import requests                                                   # noqa: E402
from scipy import ndimage as ndi                                  # noqa: E402

FONT = ROOT / "fonts" / "MetalMania-Regular.ttf"

OUT = ROOT / "jobs" / "v404_vorg"
OUT.mkdir(parents=True, exist_ok=True)

CKPT = "sd_xl_base_1.0.safetensors"
LORA = "Harrlogos_XL_v2.safetensors"
CN = "controlnet-canny-sdxl-1.0.fp16.safetensors"
ORIG = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")

WORD = "VORGRAVEN"

# ── 原图风格参考（标题带）─────────────────────────────────────────────────
REF_BOX = (0, 130, 3543, 1100)          # 原图字母带（含两侧翼刺）

# ── 风格锁：把 MRCHOSR 的特征写成 prompt ──────────────────────────────────
# ⚠️ 关键：**不要靠画布比例去逼"宽"**（宽画布必拼错，见 CANDS_R1 结论）。
#    改为在 1024² 里用 prompt 要求"横向铺开的扁宽构图" —— 模型会把字标画成
#    宽墨迹块 + 上下留黑边，字标 ink bbox 自然变宽，且拼写不崩。
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

# ── 候选：(tag, 画布W, 画布H, seed, cn_strength)  cn=None 走纯 txt2img ─────
# 第 1 轮（宽画布 1536x640 / 1600x640 / 1344x768）：**风格对了但拼写全崩**
#   （9 字母长词在 ≥2:1 画布上必被拼错 —— 🔴14 已记过这条）。结论：宽画布不能用。
CANDS_R1 = [
    ("A_w1536_s770",  1536, 640, 770315, None),
    ("B_w1536_s888",  1536, 640,   888, None),
    ("C_w1600_s4242", 1600, 640,  4242, None),
    ("D_w1344_s770",  1344, 768, 770315, None),
    ("E_cn10_s770",   1536, 640, 770315, 0.10),
    ("F_cn14_s888",   1536, 640,   888, 0.14),
    ("G_cn10_s2024",  1600, 640,  2024, 0.10),
    ("H_w1536_s6161", 1536, 640,  6161, None),
]

# 第 2 轮：**保住拼写**优先 —— 回到 1024²（n1~n6 拼写正常的画布）+ 少量 1344x768；
# 风格靠 prompt 逼（宽体 / 弯钩 / 两侧翼刺），不靠画布横向拉伸；cfg 提到 7.0 帮拼写。
CANDS_R2 = [
    ("R2a_sq_s770",   1024, 1024, 770315, None),
    ("R2b_sq_s888",   1024, 1024,    888, None),
    ("R2c_sq_s4242",  1024, 1024,   4242, None),
    ("R2d_sq_s2024",  1024, 1024,   2024, None),
    ("R2e_sq_s6161",  1024, 1024,   6161, None),
    ("R2f_sq_s1349",  1024, 1024,   1349, None),
    ("R2g_w_s770",    1344,  768, 770315, None),
    ("R2h_w_s888",    1344,  768,    888, None),
]

# 第 6 轮：第 5 轮最优参数（cn .50 / end .55 / lora 1.25~1.3）+ **两端大弯钩刀刃**强调。
#   第 5 轮已长出尖刺（R5b/R5g），但缺原图那种"两端甩出去的大弯钩"。本轮靠 prompt 补。
CANDS_R6 = [
    ("R6a_l130_e55_s888",  1792, 640,   888, 0.50, "font", 0.55, 1.30),
    ("R6b_l130_e55_s3571", 1792, 640,  3571, 0.50, "font", 0.55, 1.30),
    ("R6c_l135_e50_s4242", 1792, 640,  4242, 0.50, "font", 0.50, 1.35),
    ("R6d_l135_e50_s770",  1792, 640, 770315, 0.50, "font", 0.50, 1.35),
    ("R6e_l130_e55_s6161", 1536, 512,  6161, 0.50, "font", 0.55, 1.30),
    ("R6f_l135_e50_s2024", 2048, 704,  2024, 0.50, "font", 0.50, 1.35),
]

CANDS = CANDS_R1

# 第 3 轮：**加黑边把原图字带塞进画布中央**当 canny 结构参考。
#   思路：宽画布拼写崩 / 方画布构图不宽 —— 那就用"方/近方画布 + 居中一条扁宽参考"
#   逼模型在画布中间画一条横向扁宽字标（字够大能拼对，构图又是宽的）。
#   结果：R3e asp=4.74 / R3h asp=3.86 **构图对上了，但拼写全掉字母** ——
#   说明 LoRA 在"大体量尖刺"下守不住 9 字母。
CANDS_R3 = [
    ("R3a_sq26_c10_s770",  1344, 1024,  770315, 0.10, 0.26),
    ("R3b_sq26_c10_s888",  1344, 1024,     888, 0.10, 0.26),
    ("R3c_sq34_c12_s4242", 1344, 1024,    4242, 0.12, 0.34),
    ("R3d_sq34_c12_s2024", 1344, 1024,    2024, 0.12, 0.34),
    ("R3e_w26_c10_s6161",  1536,  768,    6161, 0.10, 0.26),
    ("R3f_w26_c12_s1349",  1536,  768,    1349, 0.12, 0.26),
    ("R3g_w34_c10_s3571",  1536,  768,    3571, 0.10, 0.34),
    ("R3h_w30_c14_s9111",  1536,  768,    9111, 0.14, 0.30),
]

# 第 4 轮：**用金属字体排版骨架锁拼写**（PIL + MetalMania 渲染 VORGRAVEN 当 canny 骨架）。
#   结果：**拼写 100% 对**（骨架生效），但 cn 0.6~0.75 / end 0.85 把风格压平了 ——
#   输出 ≈ MetalMania 原样，尖刺/弯钩刀刃没长出来，且底色变灰、有两版极性翻成黑字。
CANDS_R4 = [
    ("R4a_c60_s770",  1536, 512, 770315, 0.60, "font"),
    ("R4b_c60_s888",  1536, 512,    888, 0.60, "font"),
    ("R4c_c70_s4242", 1536, 512,   4242, 0.70, "font"),
    ("R4d_c70_s2024", 1536, 512,   2024, 0.70, "font"),
    ("R4e_c60_s6161", 1792, 640,   6161, 0.60, "font"),
    ("R4f_c70_s1349", 1792, 640,   1349, 0.70, "font"),
    ("R4g_c55_s3571", 1536, 512,   3571, 0.55, "font"),
    ("R4h_c75_s9111", 1536, 512,   9111, 0.75, "font"),
]

# 第 5 轮：**骨架守拼写 + 提前松手让 LoRA 长刺**。
#   ① cn 强度降到 0.42~0.55（骨架只当"位置提示"，不当"描红"）
#   ② cn_end 提前到 0.45~0.60（后半程自由发挥 → 长尖刺/弯钩）
#   ③ LoRA 权重提到 1.25~1.45（压过骨架的"平"）
#   字段：(tag, W, H, seed, cn, ref, cn_end, lora)
CANDS_R5 = [
    ("R5a_c50_e50_l13_s770",  1536, 512, 770315, 0.50, "font", 0.50, 1.30),
    ("R5b_c50_e50_l13_s888",  1536, 512,    888, 0.50, "font", 0.50, 1.30),
    ("R5c_c55_e60_l14_s4242", 1536, 512,   4242, 0.55, "font", 0.60, 1.40),
    ("R5d_c55_e60_l14_s2024", 1536, 512,   2024, 0.55, "font", 0.60, 1.40),
    ("R5e_c42_e45_l13_s6161", 1792, 640,   6161, 0.42, "font", 0.45, 1.30),
    ("R5f_c45_e50_l14_s1349", 1792, 640,   1349, 0.45, "font", 0.50, 1.40),
    ("R5g_c50_e55_l125_s3571", 1536, 512,  3571, 0.50, "font", 0.55, 1.25),
    ("R5h_c55_e55_l145_s9111", 1536, 512,  9111, 0.55, "font", 0.55, 1.45),
]


def pos(extra: str = "") -> str:
    tail = f", {extra}" if extra else ""
    return (f"black metal band logo lettering spelling {WORD}, {STYLE}{tail}, "
            "white on pure black background, text only")


def _skeleton_png(w: int, h: int) -> Path:
    """MetalMania 把 VORGRAVEN 排成一行 → 白字黑底骨架（给 canny 当**拼写骨架**）。

    拼写交给确定性排版，AI 只负责在骨架上长尖刺 —— 这是第 4 轮的核心思路。
    """
    p = OUT / f"_skel_{WORD}_{w}x{h}.png"
    if p.exists():
        return p
    from PIL import ImageFont
    lo, hi, best = 8.0, float(h * 4), None
    for _ in range(40):
        mid = (lo + hi) / 2.0
        f = ImageFont.truetype(str(FONT), max(8, int(mid)))
        bb = f.getbbox(WORD)
        if (bb[2] - bb[0]) <= w * 0.94:
            best = (max(8, int(mid)), bb)
            lo = mid
        else:
            hi = mid
    size, bb = best
    f = ImageFont.truetype(str(FONT), size)
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    d.text(((w - tw) / 2 - bb[0], (h - th) / 2 - bb[1]), WORD, font=f, fill=255)
    img.convert("RGB").save(p)
    print(f"[skel] {WORD} font={size}px  文字块 {tw}x{th} -> {p.name}")
    return p


def _ref_png(band_frac: float | None = None, canvas: tuple[int, int] | None = None) -> Path:
    """原图字母带 → 白字黑底 PNG（给 canny 当结构参考）。

    band_frac/canvas 给定时：把字带**等比缩放后居中贴进 canvas**，上下/左右留黑边
    —— 这样 canny 传达的是「画布中间一条横向扁宽字标」的构图，而不是字母轮廓。
    """
    tag = "plain" if band_frac is None else f"lb{int(band_frac * 100)}_{canvas[0]}x{canvas[1]}"
    p = OUT / f"_ref_title_{tag}.png"
    if p.exists():
        return p
    im = Image.open(ORIG).convert("L").crop(REF_BOX)
    ink = (np.asarray(im, np.float32) > 110)
    if band_frac is None:
        out = np.where(ink, 255, 0).astype(np.uint8)
    else:
        W, H = canvas
        bh = max(8, int(round(H * band_frac)))
        bw = max(8, int(round(bh * ink.shape[1] / ink.shape[0])))
        if bw > W:                                    # 太宽就按宽度回缩
            bw = W
            bh = max(8, int(round(bw * ink.shape[0] / ink.shape[1])))
        band = Image.fromarray(np.where(ink, 255, 0).astype(np.uint8), "L").resize(
            (bw, bh), Image.LANCZOS)
        out = np.zeros((H, W), np.uint8)
        y0, x0 = (H - bh) // 2, (W - bw) // 2
        out[y0:y0 + bh, x0:x0 + bw] = np.maximum(out[y0:y0 + bh, x0:x0 + bw], np.asarray(band))
    Image.fromarray(out, "L").convert("RGB").save(p)
    return p


def _upload(path: Path, name: str) -> str:
    with open(path, "rb") as f:
        r = requests.post(COMFYUI_URL + "/upload/image",
                          files={"image": (name, f, "image/png")}, timeout=120)
    r.raise_for_status()
    return r.json().get("name", name)


def wf(w: int, h: int, seed: int, cn_strength: float | None, cn_img: str | None,
       cfg: float = 6.5, cn_end: float = 0.50, lora: float = 1.0, extra: str = "") -> dict:
    g = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "LoraLoader", "inputs": {
            "lora_name": LORA, "strength_model": lora, "strength_clip": lora,
            "model": ["1", 0], "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": pos(extra), "clip": ["2", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["2", 1]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 30, "cfg": cfg, "sampler_name": "dpmpp_2m",
            "scheduler": "karras", "denoise": 1.0,
            "model": ["2", 0], "positive": ["3", 0], "negative": ["4", 0],
            "latent_image": ["5", 0]}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "vorg"}},
    }
    if cn_strength is not None and cn_img:
        g["10"] = {"class_type": "LoadImage", "inputs": {"image": cn_img}}
        g["11"] = {"class_type": "Canny", "inputs": {
            "image": ["10", 0], "low_threshold": 0.30, "high_threshold": 0.70}}
        g["12"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN}}
        # 只借"宽体+弯钩"的气质：强度低 + end<1 提前松手 → 不锁字母轮廓
        g["13"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": ["3", 0], "negative": ["4", 0], "control_net": ["12", 0],
            "image": ["11", 0],             "strength": cn_strength,
            "start_percent": 0.0, "end_percent": cn_end}}
        g["6"]["inputs"]["positive"] = ["13", 0]
        g["6"]["inputs"]["negative"] = ["13", 1]
    return g


# ---------------------------------------------------------------- 风格指标
def metrics_from_ink(ink: np.ndarray) -> dict:
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return {}
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = ink[y0:y1, x0:x1]
    H, W = sub.shape
    rows = sub.sum(1)
    dense = np.where(rows > rows.max() * 0.25)[0]          # 字身带（去尖刺）
    dh = int(dense.max() - dense.min() + 1) if len(dense) else H
    A = int(sub.sum())
    per = int((sub & ~ndi.binary_erosion(sub)).sum())
    return {"W": W, "H": H, "dense_h": dh, "asp": W / max(dh, 1),
            "fill": A / (W * H), "stroke": 2.0 * A / max(per, 1),
            "ink_ratio": float(ink.sum()) / ink.size}


def orig_metrics() -> dict:
    im = Image.open(ORIG).convert("RGB").crop(REF_BOX)
    a = np.asarray(im, np.float32)
    mx, mn = a.max(2), a.min(2)
    sat = (mx - mn) / np.maximum(mx, 1) * 255
    return metrics_from_ink((a.mean(2) > 150) & (sat < 62))


def glyph_metrics(p: Path) -> dict:
    g = np.asarray(Image.open(p).convert("L"), np.float32) / 255.0
    bw = max(2, int(round(min(g.shape) * 0.04)))
    border = np.concatenate([g[:bw].ravel(), g[-bw:].ravel(),
                             g[:, :bw].ravel(), g[:, -bw:].ravel()])
    if float(np.median(border)) > 0.5:
        g = 1.0 - g
    return metrics_from_ink(g > 0.45)


def build_sheet() -> Path:
    o = Image.open(ORIG).convert("RGB").crop(REF_BOX)
    cands = sorted(OUT.glob("cand_*.png"))
    tw = 900
    tiles = [("ORIG 原标题 MRCHOSR（风格基准）", o)]
    for p in cands:
        tiles.append((p.stem.replace("cand_", ""), Image.open(p).convert("RGB")))
    scaled = []
    for lab, im in tiles:
        scaled.append((lab, im.resize((tw, max(1, int(im.height * tw / im.width))), Image.LANCZOS)))
    gap, cap = 10, 26
    cols = 3
    th = max(s[1].height for s in scaled)
    rows_n = (len(scaled) + cols - 1) // cols
    sheet = Image.new("RGB", (tw * cols + gap * (cols + 1),
                              (th + cap) * rows_n + gap), (22, 22, 22))
    d = ImageDraw.Draw(sheet)
    for i, (lab, im) in enumerate(scaled):
        r, c = divmod(i, cols)
        x = gap + c * (tw + gap)
        y = gap + r * (th + cap)
        sheet.paste(im, (x, y + cap))
        d.text((x + 4, y + 6), lab, fill=(255, 214, 110))
    p = OUT / "S_风格对比.jpg"
    sheet.save(p, quality=92)
    print("sheet ->", p, sheet.size)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", action="store_true", help="只重建对比图")
    ap.add_argument("--only", default="", help="只跑这些 tag（逗号分隔）")
    ap.add_argument("--set", default="r2", choices=["r1", "r2", "r3", "r4", "r5", "r6"],
                    help="用哪一轮候选表")
    ap.add_argument("--cfg", type=float, default=0.0, help="覆盖 cfg（0=用默认）")
    ap.add_argument("--extra", default="", help="追加到正向 prompt 的风格强调")
    args = ap.parse_args()

    cands = {"r1": CANDS_R1, "r2": CANDS_R2, "r3": CANDS_R3,
             "r4": CANDS_R4, "r5": CANDS_R5, "r6": CANDS_R6}[args.set]
    cfg = args.cfg if args.cfg > 0 else (6.5 if args.set == "r1" else 7.5)

    m0 = orig_metrics()
    print(f"ORIG  字身带宽高比 asp={m0['asp']:.2f}  fill={m0['fill']:.3f}  "
          f"stroke={m0['stroke']:.1f}  dense_h={m0['dense_h']}  W={m0['W']}")

    if not args.sheet:
        only = [t.strip() for t in args.only.split(",") if t.strip()]
        up: dict = {}
        cl = ComfyClient()
        for c in cands:
            tag, w, h, seed, cns = c[:5]
            ref = c[5] if len(c) > 5 else None          # None | float(band) | "font"
            cn_end = c[6] if len(c) > 6 else (0.85 if ref == "font" else 0.55)
            lora = c[7] if len(c) > 7 else 1.0
            if only and tag not in only:
                continue
            dst = OUT / f"cand_{tag}.png"
            if dst.exists():
                print(f"[skip] {tag}")
                continue
            cn_img = None
            if cns is not None:
                key = (ref, w, h)
                if key not in up:
                    src = _skeleton_png(w, h) if ref == "font" else _ref_png(ref, (w, h))
                    up[key] = _upload(src, f"v404_ref_{tag}.png")
                cn_img = up[key]
            t = time.time()
            try:
                res = cl.run(wf(w, h, seed, cns, cn_img, cfg, cn_end, lora, args.extra), timeout=1800)
            except Exception as e:                              # noqa: BLE001
                print(f"[FAIL] {tag}: {e}", flush=True)
                continue
            for _n, blobs in res.items():
                for b in blobs:
                    Image.open(io.BytesIO(b)).convert("RGB").save(dst)
            mm = glyph_metrics(dst)
            print(f"[ok] {tag:22s} {w}x{h} s{seed} cn={cns}/{cn_end} lora={lora} ref={ref} "
                  f"cfg={cfg}  {time.time() - t:.0f}s  asp={mm.get('asp', 0):.2f} "
                  f"fill={mm.get('fill', 0):.3f} stroke={mm.get('stroke', 0):.1f}", flush=True)

    print()
    for p in sorted(OUT.glob("cand_*.png")):
        mm = glyph_metrics(p)
        print(f"{p.stem:22s} asp={mm.get('asp',0):.2f} fill={mm.get('fill',0):.3f} "
              f"stroke={mm.get('stroke',0):.1f} dense_h={mm.get('dense_h',0)} W={mm.get('W',0)}")
    build_sheet()


if __name__ == "__main__":
    main()
