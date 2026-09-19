# -*- coding: utf-8 -*-
"""v411_skel.py — p6 主体「新骨架引导重画」（第 29 轮，定稿路线）

为什么是它
----------
本轮把三条路都试穿了：
  · 第 28 轮 `v408`（区域提示 + Tile/Canny 自引导）→ 剪影逐行宽度相关 **0.998**，
    等于姿态没动（用户"没拉开"的量化证据）。
  · 第 29 轮 `v409`（掩膜内 denoise=1.0 自由重画）→ wcorr 0.94~0.97，
    姿态真变了但**质量塌**（8 条区域提示词在 denoise=1.0 下互相打架：鹰头糊、
    胸前一团泥、前额黄斑）。
  · 第 29 轮 `v410`（文生图重画 + 按位贴回）→ wcorr 0.55~0.70（变化最大），
    但**拼贴感**：新图自带一套射线、原图残射线对不上，接缝可见。

结论：**"改多少"要靠结构，不该靠提示词；"画多好"要靠重绘，不该靠贴图。**
于是本脚本把两者拆开：
  ① 用几何手段**造一张新骨架**（只看结构，不看像素）：
     鹰的双翼绕肩部旋转（上举 或 下掠）、骷髅绕自身质心倾斜 —— 姿态是**被指定**的，
     不是"求"模型做出来的；
  ② 掏空成纯结构图（只留剪影 + 边缘），当 **canny 引导**喂 SDXL；
     再配 v409 那套逐元素区域提示词（照骨架重画羽层/颅骨裂纹/角的节数）；
  ③ denoise 0.85 + hires 自引导细节遍 → 像素全部重生，成品是**新画的一张**，
     不是原图的旋转副本（骨架只提供"姿势"，不提供"内容"）。

⛔️ 三条红线不动：不叠原图像素（🔴34）、不色块遮盖（🔴6）、成品分层序与配色照原图（🔴4）。

用法
  python src/v411_skel.py --pose V1_wingV --wing 26 --skull 14 --seeds 888
  python src/v411_skel.py --pose V3_lowbrace --wing -22 --skull 10 --seeds 888   # 负角=下掠
产物：jobs/v411_skel/p6_<tag>.jpg（含原标题带的整图，喂 P6_REBIRTH）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from v342_rebirth import (CUI_IN, _disk, _fetch, _lum, comfy_submit,          # noqa: E402
                          mask_p6, _match_region)
from v377_p6rb import snap_p6                                                  # noqa: E402
from v409_p6free import (CKPT, CN_CANNY, CN_TILE, LORA_DETAIL, GLOBAL_POS,     # noqa: E402
                         GLOBAL_NEG, VARIANTS, _add_regions, _bg_protect)

SRC = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")
OUT = ROOT / "jobs" / "v411_skel"
OUT.mkdir(parents=True, exist_ok=True)
TITLE_LINE = 1500

PASS1_MP = 1.10
TARGET = 2048
DENOISE = 0.85
CN_S, CN_E = 0.42, 0.58
HIRES_DN = 0.42
HC_S, HC_E = 0.15, 0.50
HT_S, HT_E = 0.45, 0.65
REGION_SCALE = 0.80
MASK_GROW = 26


# ─────────────────────────────────────────────── 骨架
def body_mask(orig: Image.Image) -> np.ndarray:
    """主体实体（含深棕翼、含骷髅），但**排除细长闪电**。"""
    lum = _lum(np.asarray(orig, np.float32))
    m = ndi.binary_closing(lum > 22.0, structure=_disk(5))
    m = ndi.binary_opening(m, structure=_disk(6))
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    if n:
        sz = np.bincount(lab.ravel())
        keep = np.zeros(n + 1, bool); keep[1:] = sz[1:] >= 15000
        m = keep[lab]
    m[:TITLE_LINE, :] = False                        # 标题带不动
    return m


def make_skeleton(orig: Image.Image, subj: np.ndarray, wing_deg: float,
                  skull_deg: float, body_frac: float = 0.22) -> np.ndarray:
    """只旋**两侧翼板**，头与躯干保持不动（枢轴放在翼根）。

    ⚠️ 第 29 轮踩到的坑：按 x 中线硬切会把鹰头切成两半各转各的 → 头中间留白块。
    正解：留一条以体心为中线的"核心带"（宽度 = 主体宽的 `body_frac`）**不参与旋转**，
    只把核心带外侧的翼板绕**翼根**旋转 —— 翼板与躯干因此始终相连，不会撕开。
    正角 = 逆时针（PIL 约定）：左翼用 -deg、右翼用 +deg ⇒ 双翼同时上举；反向则下掠。
    `skull_deg` 保留给骷髅整体倾斜（默认 0，交给区域提示词处理更安全）。
    """
    H, W = subj.shape
    yy, xx = np.mgrid[0:H, 0:W]
    ys, xs = np.where(subj)
    y_split = int(np.quantile(ys, 0.42))              # 鹰/骷髅分界（实测 ≈0.39）
    upper = subj & (yy < y_split)
    uxs = np.where(upper.any(axis=0))[0]
    cx = float(np.median(uxs)) if len(uxs) else W / 2.0
    hw = max(60.0, body_frac * (xs.max() - xs.min()) / 2.0)
    uys = np.where(upper.any(axis=1))[0]
    y_sh = float(np.median(uys)) if len(uys) else y_split / 2.0
    core = upper & (np.abs(xx - cx) <= hw)
    left = upper & (xx < cx - hw)
    right = upper & (xx > cx + hw)

    base = np.zeros((H, W, 3), np.uint8)
    base[core] = np.asarray(orig)[core]
    base[subj & (yy >= y_split)] = np.asarray(orig)[subj & (yy >= y_split)]

    def rot(sel, deg, center):
        a = np.zeros((H, W, 3), np.uint8)
        a[sel] = np.asarray(orig)[sel]
        im = Image.fromarray(a).rotate(deg, center=center, resample=Image.BICUBIC, fillcolor=(0, 0, 0))
        return np.asarray(im)

    if left.any():
        base = np.maximum(base, rot(left, -wing_deg, (cx - hw, y_sh)))
    if right.any():
        base = np.maximum(base, rot(right, wing_deg, (cx + hw, y_sh)))
    if skull_deg and (subj & (yy >= y_split)).any():
        sk = subj & (yy >= y_split)
        sy, sx = np.where(sk)
        base = np.maximum(base, rot(sk, skull_deg, (float(sx.mean()), float(sy.mean()))))
    canvas = np.ascontiguousarray(base).copy()
    canvas[:TITLE_LINE] = 0
    return canvas


def skel_mask(canvas: np.ndarray) -> np.ndarray:
    lum = _lum(canvas.astype(np.float32))
    m = lum > 18
    m = ndi.binary_closing(m, structure=_disk(9))
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    if n:
        sz = np.bincount(lab.ravel())
        keep = np.zeros(n + 1, bool); keep[1:] = sz[1:] >= 8000
        m = keep[lab]
    return ndi.binary_dilation(m, structure=_disk(MASK_GROW))


# ─────────────────────────────────────────────── ComfyUI
def build_wf(crop: Image.Image, guide: Image.Image, mask_png: str, seed: int, pose: str, *,
             pass1_mp=PASS1_MP, steps=26, canny_s=CN_S, canny_e=CN_E, region_scale=REGION_SCALE,
             target=TARGET, hires_dn=HIRES_DN, steps2=18,
             hc_s=HC_S, hc_e=HC_E, ht_s=HT_S, ht_e=HT_E, lora_w=0.5, **ignored) -> dict:
    ts = str(int(time.time() * 1000))
    n1, n2 = f"v411_src_{ts}.png", f"v411_gd_{ts}.png"
    crop.save(str(CUI_IN / n1))
    guide.save(str(CUI_IN / n2))
    cw, ch = crop.size
    sc = min(1.0, target / float(max(cw, ch)))
    tw = max(8, (int(round(cw * sc)) // 8) * 8)
    th = max(8, (int(round(ch * sc)) // 8) * 8)

    g: dict = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "LoadImage", "inputs": {"image": n1}},
        "2g": {"class_type": "LoadImage", "inputs": {"image": n2}},
        "2m": {"class_type": "LoadImage", "inputs": {"image": mask_png}},
        "2mm": {"class_type": "ImageToMask", "inputs": {"image": ["2m", 0], "channel": "red"}},
        "3": {"class_type": "ImageScaleToTotalPixels",
              "inputs": {"image": ["2", 0], "upscale_method": "lanczos",
                         "megapixels": pass1_mp, "resolution_steps": 64}},
        "3g": {"class_type": "ImageScaleToTotalPixels",
               "inputs": {"image": ["2g", 0], "upscale_method": "lanczos",
                          "megapixels": pass1_mp, "resolution_steps": 64}},
        "4": {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}},
        "4m": {"class_type": "SetLatentNoiseMask",
               "inputs": {"samples": ["4", 0], "mask": ["2mm", 0]}},
        "31": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "32": {"class_type": "ImageScale", "inputs": {"image": ["31", 0], "upscale_method": "lanczos",
                                                      "width": tw, "height": th, "crop": "disabled"}},
        "7": {"class_type": "LoraLoader",
              "inputs": {"model": ["1", 0], "clip": ["1", 1], "lora_name": LORA_DETAIL,
                         "strength_model": lora_w, "strength_clip": lora_w}},
    }
    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": GLOBAL_POS}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": GLOBAL_NEG}}
    # 骨架 canny（**来源是新骨架，不是原图**）
    g["20"] = {"class_type": "CannyEdgePreprocessor",
               "inputs": {"image": ["3g", 0], "low_threshold": 0.10,
                          "high_threshold": 0.25, "resolution": 1024}}
    g["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_CANNY}}
    pos0, neg0 = ["pg", 0], ["ng", 0]
    if canny_s > 0:
        g["22"] = {"class_type": "ControlNetApplyAdvanced",
                   "inputs": {"positive": pos0, "negative": neg0, "control_net": ["21", 0],
                              "image": ["20", 0], "strength": canny_s,
                              "start_percent": 0.0, "end_percent": canny_e}}
        pos0, neg0 = ["22", 0], ["22", 1]
    comb = _add_regions(g, pos0, region_scale, pose)
    g["comb"] = {"class_type": "RegionalListCombine", "inputs": comb}
    g["10"] = {"class_type": "KSampler",
               "inputs": {"model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
                          "latent_image": ["4m", 0], "seed": seed, "steps": steps, "cfg": 6.5,
                          "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": DENOISE}}
    g["30"] = {"class_type": "LatentUpscale",
               "inputs": {"samples": ["10", 0], "upscale_method": "bilinear",
                          "width": tw, "height": th, "crop": "disabled"}}
    g["30m"] = {"class_type": "SetLatentNoiseMask",
                "inputs": {"samples": ["30", 0], "mask": ["2mm", 0]}}
    g["33"] = {"class_type": "CannyEdgePreprocessor",
               "inputs": {"image": ["32", 0], "low_threshold": 0.10,
                          "high_threshold": 0.25, "resolution": 1024}}
    p2, n2_ = ["comb", 0], ["ng", 0]
    if hc_s > 0:
        g["34"] = {"class_type": "ControlNetApplyAdvanced",
                   "inputs": {"positive": p2, "negative": n2_, "control_net": ["21", 0],
                              "image": ["33", 0], "strength": hc_s,
                              "start_percent": 0.0, "end_percent": hc_e}}
        p2, n2_ = ["34", 0], ["34", 1]
    if ht_s > 0:
        g["36"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_TILE}}
        g["37"] = {"class_type": "ControlNetApplyAdvanced",
                   "inputs": {"positive": p2, "negative": n2_, "control_net": ["36", 0],
                              "image": ["32", 0], "strength": ht_s,
                              "start_percent": 0.0, "end_percent": ht_e}}
        p2, n2_ = ["37", 0], ["37", 1]
    g["40"] = {"class_type": "KSampler",
               "inputs": {"model": ["7", 0], "positive": p2, "negative": n2_,
                          "latent_image": ["30m", 0], "seed": seed + 7, "steps": steps2, "cfg": 6.5,
                          "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": hires_dn}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["40", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "SaveImage",
               "inputs": {"images": ["12", 0], "filename_prefix": f"v411_{pose}_{seed}"}}
    return g


def _crop_box(mask: np.ndarray, W: int, H: int, margin: int = 90):
    ys, xs = np.where(mask)
    x0 = max(0, xs.min() - margin); x1 = min(W, xs.max() + 1 + margin)
    y0 = max(0, ys.min() - margin); y1 = min(H, ys.max() + 1 + margin)
    x0 -= x0 % 8; y0 -= y0 % 8
    x1 = min(W, x1 + ((8 - (x1 - x0) % 8) % 8))
    y1 = min(H, y1 + ((8 - (y1 - y0) % 8) % 8))
    return x0, y0, x1, y1


def rebirth(img: Image.Image, seed: int, pose: str, wing=26.0, skull=14.0, *,
            margin=90, color_alpha=0.85, verbose=True, **kw) -> Image.Image:
    W, H = img.size
    subj = body_mask(img)
    canvas = make_skeleton(img, subj, wing, skull)
    gain = skel_mask(canvas) | ndi.binary_dilation(mask_p6(img), _disk(MASK_GROW))
    x0, y0, x1, y1 = _crop_box(gain, W, H, margin)
    crop = img.crop((x0, y0, x1, y1))
    guide = Image.fromarray(canvas[y0:y1, x0:x1])
    mcut = gain[y0:y1, x0:x1]
    ts = str(int(time.time() * 1000))
    mp = f"v411_mask_{seed}_{ts}.png"
    Image.fromarray((mcut * 255).astype(np.uint8), "L").convert("RGB").save(str(CUI_IN / mp))
    if verbose:
        print(f"[v411] pose={pose} wing={wing} skull={skull} crop={crop.size} "
              f"mask={int(mcut.sum())} 骨架墨量={int((_lum(canvas.astype(np.float32)) > 18).sum())}")
    t0 = time.time()
    outs = comfy_submit(build_wf(crop, guide, mp, seed, pose, **kw))
    res = None
    for _n, o in outs.items():
        if "images" in o:
            res = _fetch(o); break
    if res is None:
        raise RuntimeError("no comfy output")
    res = res.resize(crop.size, Image.LANCZOS)
    if verbose:
        print(f"[v411] comfy {time.time() - t0:.0f}s -> {res.size}")

    oc = np.asarray(crop, np.float32)
    rc = np.asarray(res, np.float32)
    if color_alpha:
        sel = ndi.binary_dilation(mcut, iterations=12)
        if int(sel.sum()) > 50:
            rc = _match_region(rc, oc, sel, alpha=float(color_alpha))
    rc = _bg_protect(rc, oc)
    tb = (y0 + np.arange(crop.height)) < TITLE_LINE
    if tb.any():
        rc[tb, :] = oc[tb, :]
    full = np.array(img).copy()
    full[y0:y1, x0:x1] = np.clip(rc, 0, 255).astype(np.uint8)
    return Image.fromarray(full)


def run(seed: int, pose: str, wing=26.0, skull=14.0, out_dir=None, tag=None, **kw) -> Path:
    out_dir = Path(out_dir) if out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or f"{pose}_w{int(wing)}_k{int(skull)}_s{seed}"
    src = Image.open(SRC).convert("RGB")
    t = time.time()
    full = rebirth(src, seed, pose, wing, skull, **kw)
    full = snap_p6(src, full, mask_p6(src))
    dst = out_dir / f"p6_{tag}.jpg"
    full.save(dst, quality=94)
    print(f"[v411] → {dst.name} ({time.time() - t:.0f}s)", flush=True)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default="V1_wingV", choices=sorted(VARIANTS))
    ap.add_argument("--wing", type=float, default=26.0)
    ap.add_argument("--skull", type=float, default=14.0)
    ap.add_argument("--seeds", default="888")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--canny", type=float, default=CN_S)
    ap.add_argument("--canny-end", type=float, default=CN_E)
    ap.add_argument("--region-scale", type=float, default=REGION_SCALE)
    ap.add_argument("--only-sketch", action="store_true", help="只出骨架图，不跑 ComfyUI")
    a = ap.parse_args()
    if a.only_sketch:
        img = Image.open(SRC).convert("RGB")
        c = make_skeleton(img, body_mask(img), a.wing, a.skull)
        vis = np.asarray(img).copy()
        vis[c.max(axis=2) == 0] = (vis[c.max(axis=2) == 0] * 0.25).astype(np.uint8)
        vis = np.maximum(vis, c)
        # 把残影压暗，突出骨架
        Image.fromarray(vis).save(OUT / f"sketch_w{int(a.wing)}_k{int(a.skull)}.jpg", quality=92)
        Image.fromarray(c).save(OUT / f"sketchonly_w{int(a.wing)}_k{int(a.skull)}.png")
        print("sketch ok")
        return
    for s in [int(x) for x in a.seeds.split(",") if x.strip()]:
        run(s, a.pose, a.wing, a.skull, out_dir=a.out_dir,
            tag=(f"{a.tag}_s{s}" if a.tag else None),
            canny_s=a.canny, canny_e=a.canny_end, region_scale=a.region_scale)


if __name__ == "__main__":
    main()
