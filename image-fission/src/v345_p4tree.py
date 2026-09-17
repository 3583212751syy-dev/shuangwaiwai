"""v345: p4 棕榈树「蝙蝠式」逐棵重生 —— 每棵树单独裁块、在 1024 尺度单独重生。

用户第 14 轮：
  「裂变没看到明显变化，并且比原图丑，这种手法严格禁止」
  「蝙蝠设计可以，树木的处理按照蝙蝠的裂变去试试」

对照 6978 蝙蝠为什么成功（jobs/v342_rebirth/6978_rebirth_v344.jpg vs 原图）：
  · 蝙蝠的裁块**只包含蝙蝠本身**，画幅 ~700px → SDXL 在原生长边附近工作 → 硬边干净；
  · 掩膜是**实心块**（碟内暗块），模型有整块空白可以重新设计剪影，而不是往细缝里塞纹理；
  · prompt 死咬 "flat matte vector, bold clean outline"。

p4 旧做法踩的三条坑（本脚本逐条修）：
  ① 整图一次重生：1242x1754 → 降采样到 1536 长边（0.876x）→ 升回 → 线稿发虚；
     改成**逐棵树裁块**（每块 ~400~600px，上采样到 1024 长边 → 生成 → 缩回），
     SDXL 在 1024 甜点尺寸作画 → 硬边。
  ② 掩膜 = 细墨线 + dilate 12 = **细带**；模型只能往带里塞细节 → 软气刷绒毛。
     改成**实心包络**（本树 ink ∪ dilate10 → closing16 → fill_holes）→ 整棵实心区
     → 模型重新设计整棵棕榈剪影。
  ③ 后处理只做二值化，绒毛被原样留下。改成**密度匹配阈值**（新墨像素数 == 原墨像素数）
     + 去碎点 + 硬边纯黑。

输出 = **新的墨迹掩膜**（H×W bool）。由 camo_palm_pattern.fission 用
`tree_ink_override` 消费：out = 重 blob 迷彩底 ⊕ 纯黑新剪影（迷彩/版式/文字管线不变）。
"""
import io
import json
import sys
import time
import urllib.request as ur
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from styles import camo_palm_pattern as cpp            # noqa: E402
import v342_rebirth as R                               # noqa: E402

COMFY = "http://127.0.0.1:8188"
CUI_IN = ROOT / "ComfyUI" / "input"
OUT = ROOT / "jobs" / "v345_p4tree"
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("E:/Desktop/图裂变测试图/Pinterest (4).jpg")
BLACK = np.array([12.0, 10.0, 9.0], np.float32)

CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CKPT = "ProteusV0.4.safetensors"

PALM_POS = ("a single black palm tree silhouette, flat vector stencil, crisp hard edges, "
            "pure black ink, high contrast two-tone screen print, "
            "thin curving trunk with fine horizontal ring texture, "
            "wide open crown made of separate long fronds with clear empty gaps between them, "
            "minimal interior detail, elegant, clean")
PALM_NEG = ("soft, airbrush, watercolor, gradient, shading, grey, gray, blurry, out of focus, "
            "fuzzy, hairy, feathery, painterly, sketch, noise, grain, smudge, blotch, "
            "multiple trees, forest, jungle, dense, cluttered, text, letters, words, "
            "color photo, realistic, 3d render, white background")


# ------------------------------------------------------------------ 工具
def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def keep_big(m, min_px):
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    if n == 0:
        return m
    sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    k = np.zeros(n + 1, bool)
    k[1:] = sz >= min_px
    return k[lab]


def inpaint_crop(crop, mask, prompt, neg, target=1024, denoise=0.88,
                 cn_strength=0.35, cn_end=0.45, cn_low=0.35, cn_high=0.75,
                 seed=7, steps=30, tag="t", ckpt=CKPT):
    """把 crop（PIL）内 mask（bool）区域送 ComfyUI 重生；crop 长边缩放到 target 再生成。
    返回与 crop 同尺寸的生成结果（未合成）。"""
    cw, ch = crop.size
    # 长边一律归到 target（小树上采样 / 大树降采样）→ SDXL 始终在甜点尺寸作画。
    # ⚠️ 之前这里写成分支：小树上采样、大树保持原尺寸 → 1000x1754 的巨块直接进
    # KSampler（1.8MP），单张 185s 且质量反而更差。改成统一归一到 target。
    sc = target / float(max(cw, ch))
    iw = max(8, (int(round(cw * sc)) // 8) * 8)
    ih = max(8, (int(round(ch * sc)) // 8) * 8)
    inj = crop.resize((iw, ih), Image.LANCZOS)
    # 掩膜：inpaint 区 = mask 膨胀 6px（给模型留余量），上采样用 NEAREST
    mp = ndi.binary_dilation(mask, structure=_disk(6))
    mp_img = Image.fromarray((mp.astype(np.uint8) * 255), "L").resize((iw, ih), Image.NEAREST)

    ts = str(int(time.time() * 1000)) + tag
    sn, mn = f"v345_s_{ts}.png", f"v345_m_{ts}.png"
    inj.save(str(CUI_IN / sn))
    mp_img.save(str(CUI_IN / mn))
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "LoadImage", "inputs": {"image": sn}},
        "3": {"class_type": "LoadImageMask", "inputs": {"image": mn, "channel": "red"}},
        "4": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
        "5": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["4", 0], "mask": ["3", 0]}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["1", 1]}},
        "20": {"class_type": "Canny", "inputs": {"image": ["2", 0], "low_threshold": cn_low,
                                                 "high_threshold": cn_high}},
        "21": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CANNY}},
        "22": {"class_type": "ControlNetApplyAdvanced",
               "inputs": {"positive": ["9", 0], "negative": ["10", 0],
                          "control_net": ["21", 0], "image": ["20", 0],
                          "strength": cn_strength, "start_percent": 0.0,
                          "end_percent": cn_end}},
        "11": {"class_type": "KSampler",
               "inputs": {"model": ["1", 0], "positive": ["22", 0], "negative": ["22", 1],
                          "latent_image": ["5", 0], "seed": seed, "steps": steps,
                          "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
                          "denoise": denoise}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0],
                                                     "filename_prefix": f"v345_{ts}"}},
    }
    outs = R.comfy_submit(wf)
    res = None
    for _, o in outs.items():
        if "images" in o:
            res = R._fetch(o)
            break
    # ⚠️ 不删临时文件：本环境有批量删除闸门（同一轮 >50 次 unlink 会被拦截并终止进程），
    # 逐树 2 个临时文件 × 34 棵必被拦。临时文件留在 ComfyUI/input，收尾统一清一次。
    if res is None:
        return None
    return res.resize((cw, ch), Image.LANCZOS)


# ------------------------------------------------------------------ 树提取
def tree_groups(ink, W, H, min_area=1200, min_h=0.075):
    """墨迹 → 空间邻接聚合成"元素部落" → 挑出**棕榈树**（够高够大、纵向为主）。"""
    lab, n = ndi.label(ink, structure=np.ones((3, 3), bool))
    g = ndi.binary_dilation(ink, structure=_disk(12))
    glab, gn = ndi.label(g, structure=np.ones((3, 3), bool))
    from collections import defaultdict
    buckets = defaultdict(list)
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < 25:
            continue
        buckets[int(np.bincount(glab[m]).argmax())].append(m)
    groups = []
    for gid, ms in buckets.items():
        mm = np.zeros((H, W), bool)
        for x in ms:
            mm |= x
        ys, xs = np.where(mm)
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        groups.append(dict(mask=mm, n=len(ms), area=int(mm.sum()),
                           x0=int(xs.min()), x1=int(xs.max()),
                           y0=int(ys.min()), y1=int(ys.max()),
                           bw=int(bw), bh=int(bh),
                           cx=float(xs.mean()), cy=float(ys.mean()),
                           tall=bh >= min_h * H and bh >= 1.02 * bw and mm.sum() >= min_area))
    groups.sort(key=lambda d: -d["area"])
    return groups


def tree_envelopes(groups, W, H):
    """每棵树的实心包络（crown 缝补成整块 + 树干成带），重叠像素判给最近树。"""
    envs = []
    for g in groups:
        d = ndi.binary_dilation(g["mask"], structure=_disk(10))
        e = ndi.binary_closing(d, structure=_disk(16))
        e = ndi.binary_fill_holes(e)
        e = keep_big(e, 800)
        envs.append(e)
    stack = np.stack(envs) if envs else None
    return envs, stack


def dedup_envs(envs, groups):
    """把重叠像素判给质心最近的树，保证每棵树的生成结果只被贴一次。"""
    H, W = envs[0].shape
    if len(envs) == 1:
        return envs
    cost = np.full((len(envs), H, W), 1e18, np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for i, g in enumerate(groups):
        cost[i] = (xx - g["cx"]) ** 2 + (yy - g["cy"]) ** 2
    cost = np.where(np.stack(envs), cost, 1e18)
    win = cost.argmin(0)
    out = []
    for i in range(len(envs)):
        out.append(envs[i] & (win == i))
    return out


# ------------------------------------------------------------------ 主流程
def build(margin=40, target=1024, denoise=0.88, cn_strength=0.35, cn_end=0.45,
          seed=7, styles_used=None, limit=None, tcap=52.0, gain=1.0):
    img = Image.open(SRC).convert("RGB")
    W, H = img.size
    o = np.asarray(img, np.float32)
    ink0 = cpp.tree_ink_mask(img, lum_thr=55.0, sat_thr=14.0, min_px=25, thin_only=False)
    groups = tree_groups(ink0, W, H)
    trees = [g for g in groups if g["tall"]]
    print(f"[p4tree] 元素部落={len(groups)} 判为棕榈={len(trees)} 图={W}x{H} "
          f"原墨={100*ink0.mean():.1f}%")
    if limit:
        trees = trees[:limit]
    for g in trees[:8]:
        print(f"   tree area={g['area']:>6d} bbox={g['bw']}x{g['bh']} at "
              f"({g['x0']},{g['y0']}) parts={g['n']}")
    envs, _ = tree_envelopes(trees, W, H)
    envs = dedup_envs(envs, trees)
    env_all = np.zeros((H, W), bool)
    for e in envs:
        env_all |= e
    print(f"[p4tree] 包络合计={100*env_all.mean():.1f}%")

    # 逐棵重生
    gen = o.copy()
    t0 = time.time()
    for i, (g, e) in enumerate(zip(trees, envs)):
        x0 = max(0, g["x0"] - margin); x1 = min(W, g["x1"] + margin)
        y0 = max(0, g["y0"] - margin); y1 = min(H, g["y1"] + margin)
        ec = e[y0:y1, x0:x1]
        if ec.sum() < 500:
            continue
        crop = img.crop((x0, y0, x1, y1))
        try:
            r = inpaint_crop(crop, ec, PALM_POS, PALM_NEG, target=target,
                             denoise=denoise, cn_strength=cn_strength, cn_end=cn_end,
                             seed=seed + i * 13, tag=f"t{i}")
        except Exception as ex:
            print(f"   [{i}] FAIL {ex}")
            continue
        if r is None:
            continue
        ra = np.asarray(r, np.float32)
        ee = e[y0:y1, x0:x1, None]
        gen[y0:y1, x0:x1] = gen[y0:y1, x0:x1] * (1 - ee) + ra * ee
        if (i + 1) % 5 == 0:
            print(f"   [p4tree] {i+1}/{len(trees)} {time.time()-t0:.0f}s")
    print(f"[p4tree] 重生完成 {time.time()-t0:.0f}s")
    Image.fromarray(np.clip(gen, 0, 255).astype(np.uint8), "RGB").save(
        str(OUT / "_p4tree_gen.jpg"), quality=93)

    # 阈值：包络内 新墨量 == 原墨量（密度匹配），硬上限 tcap 防吃迷彩暗块
    lg = R._lum(gen)
    sel = ndi.binary_erosion(env_all, structure=_disk(6))
    n_in = float((ink0 & sel).sum())
    frac = min(0.9, max(0.02, n_in / float(max(1, sel.sum())))) * gain
    t = float(np.percentile(lg[sel], 100.0 * min(0.9, frac))) if sel.any() else 55.0
    t = float(min(max(t, 12.0), tcap))
    new_ink = (lg < t) & sel
    new_ink = keep_big(ndi.binary_closing(new_ink, structure=_disk(2)), 70)
    new_ink = ndi.binary_opening(new_ink, structure=_disk(1))
    ink_out = (ink0 & ~env_all) | new_ink
    print(f"[p4tree] thr={t:.1f} 新墨={100*ink_out.mean():.1f}% (原 {100*ink0.mean():.1f}%)")

    # 预览：干净迷彩底（nn 回填原墨）⊕ 纯黑新墨
    idx = ndi.distance_transform_edt(ink0, return_distances=False, return_indices=True)
    base = o[idx[0], idx[1]]
    a = ink_out.astype(np.float32)[..., None]
    prev = base * (1 - a) + BLACK[None, None, :] * a
    Image.fromarray(np.clip(prev, 0, 255).astype(np.uint8), "RGB").save(
        str(OUT / "_p4tree_preview.jpg"), quality=94)
    Image.fromarray((ink_out.astype(np.uint8) * 255), "L").save(
        str(OUT / "_p4tree_ink.png"))
    # 并排对照（原图 | 新）
    cb = Image.new("RGB", (W * 2 + 12, H), (20, 20, 20))
    cb.paste(img, (0, 0))
    cb.paste(Image.fromarray(np.clip(prev, 0, 255).astype(np.uint8), "RGB"), (W + 12, 0))
    cb.resize(((W * 2 + 12) // 2, H // 2), Image.LANCZOS).save(
        str(OUT / "_p4tree_cmp.jpg"), quality=92)
    return OUT / "_p4tree_ink.png"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--target", type=int, default=1024)
    ap.add_argument("--denoise", type=float, default=0.88)
    ap.add_argument("--cn", type=float, default=0.35)
    ap.add_argument("--cne", type=float, default=0.45)
    ap.add_argument("--tcap", type=float, default=52.0)
    ap.add_argument("--gain", type=float, default=1.0)
    a = ap.parse_args()
    build(limit=a.limit, target=a.target, denoise=a.denoise, cn_strength=a.cn,
          cn_end=a.cne, tcap=a.tcap, gain=a.gain)
