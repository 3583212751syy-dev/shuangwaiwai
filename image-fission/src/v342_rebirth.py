"""v342: 主体 SDXL 结构级重生（异内容同构）。

根因（用户第 11 轮）：纯像素形变（tree_wind / pose_field / _poseB）只搬动原图像素、
主体内容（纹理 / 剪影形状）完全不变 → 11 轮仍一眼认出同款。
按 🔴2 硬规则「真裂变 = ComfyUI SDXL 结构级重生」：本脚本对主体做**掩膜内 SDXL 重绘**，
IP-Adapter style transfer 双锁保色保构、prompt 驱动异内容同物种；掩膜外原图零改动合成。

管线：
  ① 提取主体掩膜（每图独立 extractor）
  ② 裁主体 bbox + margin，掩膜 grow 作 inpaint 区
  ③ ComfyUI 原生 SDXL inpaint（VAEEncode + SetLatentNoiseMask + KSampler）
     + IPAdapter(style transfer, weight≈0.65) 锁原图配色
  ④ 结果只在（羽化）掩膜内贴回原裁块，其余原样 → 背景 / 文字零改动

⚠️ 本环境模型不能读图，出图只能靠量化自检（背景 LAB≈0、主体 LAB 大、质心/尺寸保留）。
"""
import sys
import io
import json
import time
import urllib.request as ur
import urllib.error
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent          # image-fission/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from styles import camo_palm_pattern as cpp            # tree_ink_mask

COMFY = "http://127.0.0.1:8188"
CUI_IN = ROOT / "ComfyUI" / "input"
OUT = ROOT / "jobs" / "v342_rebirth"
OUT.mkdir(parents=True, exist_ok=True)
SRC_DIR = Path("E:/Desktop/图裂变测试图")
FILE = {
    "6978": "6978fabda2cc99629fa9e81f802762d3.jpg",
    "p6": "Pinterest (6).jpg",
    "p4": "Pinterest (4).jpg",
}

_W = np.array([0.299, 0.587, 0.114], np.float32)


def _lum(a):
    return a @ _W


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _match_region(src, ref, sel, alpha=0.85):
    """把 src 在 sel 区域内的 LAB 均值/方差对齐 ref（Reinhard），再按 alpha 混回。
    用途：SDXL 高 denoise 会让新主体色彩漂移（p6 白羽 / p4 白叶）→ 强制拉回原配色。"""
    from skimage.color import rgb2lab, lab2rgb
    s = rgb2lab(src / 255.0)
    r = rgb2lab(ref / 255.0)
    out = s.copy()
    for c in range(3):
        ms, ss = s[..., c][sel].mean(), s[..., c][sel].std() + 1e-6
        mr, sr = r[..., c][sel].mean(), r[..., c][sel].std() + 1e-6
        out[..., c] = (s[..., c] - ms) / ss * sr + mr
    matched = np.clip(lab2rgb(out) * 255.0, 0, 255)
    return (1.0 - alpha) * src + alpha * matched


def comfy_submit(workflow, timeout=900):
    req = ur.Request(f"{COMFY}/prompt", data=json.dumps({"prompt": workflow}).encode(),
                     headers={"Content-Type": "application/json"})
    try:
        resp = json.loads(ur.urlopen(req, timeout=20).read())
    except ur.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        print(f"[comfy HTTP {e.code}] {body[:3000]}")
        raise
    pid = resp["prompt_id"]
    for _ in range(timeout // 3):
        time.sleep(3)
        hist = json.loads(ur.urlopen(f"{COMFY}/history/{pid}", timeout=20).read())
        if pid in hist and hist[pid].get("outputs"):
            return hist[pid]["outputs"]
    raise TimeoutError(f"comfy timeout pid={pid}")


def _fetch(node_out):
    for img in node_out.get("images", []):
        fn = img["filename"]
        sub = img.get("subfolder", "")
        data = ur.urlopen(f"{COMFY}/view?filename={fn}&subfolder={sub}", timeout=30).read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    return None


# ---------------------------------------------------------------- 6978 蝙蝠
def mask_6978(img):
    a = np.asarray(img, np.float32)
    H, W = a.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    mx = a.max(2)
    lum = _lum(a)
    ell = ((xx - 815.0) / 300.0) ** 2 + ((yy - 742.0) / 305.0) ** 2 <= 1.0
    core = (mx < 75) & ell
    edge = (lum > 118) & ndi.binary_dilation(core, structure=np.ones((3, 3), bool))
    bat = core | edge
    lab, n = ndi.label(bat, structure=np.ones((3, 3), bool))
    if n:
        r_all = np.sqrt(((xx - 813.0) / 272.0) ** 2 + ((yy - 690.0) / 277.0) ** 2)
        sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        keep = np.zeros(n + 1, bool)
        for j in range(1, n + 1):
            mj = (lab == j)
            if int(sz[j - 1]) < 800:
                continue
            if float(r_all[mj].min()) > 0.55:        # 够不到碟心 → 不是蝙蝠
                continue
            keep[j] = True
        bat = keep[lab]
    bat = ndi.binary_dilation(bat, structure=np.ones((3, 3), bool))
    return bat


# ---------------------------------------------------------------- p6 鹰+骷髅
def mask_p6(img):
    """p6 主体 = 红棕鹰（含白头）+ 白角骷髅。黑底插画：非黑像素 − 标题 − 细射线 − 星点。
    v342b：旧版只收「棕（sat>40）」→ 漏掉鹰的白头 + 骷髅的白脸（用户第 11 轮点的
    "鹰头没变"根因）。改：lum>55 的实体 → 闭运算并入 → 开运算去细白射线/星点 →
    取大连通域 → 膨胀。"""
    a = np.asarray(img, np.float32)
    lum = _lum(a)
    nb = lum > 55.0
    nb[:1500, :] = False                             # 排除标题带（白墨 MRCHOSR）
    m = ndi.binary_closing(nb, structure=_disk(6))
    m = ndi.binary_opening(m, structure=_disk(4))    # 去细白射线 / 星点
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    if n:
        sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        keep = np.zeros(n + 1, bool)
        for j in range(1, n + 1):
            if sz[j - 1] >= 12000:
                keep[j] = True
        m = keep[lab]
    m = ndi.binary_dilation(m, structure=_disk(5))
    return m


# ---------------------------------------------------------------- p4 墨迹层（树/点/横线）
def mask_p4(img):
    ink = cpp.tree_ink_mask(img, lum_thr=55.0, sat_thr=14.0, min_px=25, thin_only=False)
    return ndi.binary_dilation(ink, structure=np.ones((3, 3), bool))


# ---------------------------------------------------------------- 通用重生
def rebirth_subject(img, mask, prompt, neg,
                    ckpt="ProteusV0.4.safetensors",
                    denoise=0.72, ipa_weight=0.65, ipa_end=0.5, color_match=0.85,
                    cn_name=None, cn_strength=0.70, cn_pre="canny", cn_end=1.0,
                    cn_low=0.35, cn_high=0.75, lora=None,
                    margin=70, grow=12, seed=12345, tag="subj", max_side=1536):
    W, H = img.size
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return img
    x0 = max(0, xs.min() - margin); x1 = min(W, xs.max() + 1 + margin)
    y0 = max(0, ys.min() - margin); y1 = min(H, ys.max() + 1 + margin)
    crop = img.crop((x0, y0, x1, y1))
    cw, ch = crop.size

    mfull = np.zeros((H, W), bool); mfull[mask] = True
    mcut = mfull[y0:y1, x0:x1]
    feather = np.clip(ndi.gaussian_filter(
        ndi.binary_dilation(mcut, iterations=grow).astype(np.float32), 3.0), 0.0, 1.0)
    minp = ndi.binary_dilation(mcut, iterations=grow + 6)

    # SDXL 友好尺寸：max side 超限 → 降采样重绘、结果再升回原裁块尺寸
    if max_side and max(cw, ch) > max_side:
        sc_in = max_side / float(max(cw, ch))
        iw = max(8, (int(round(cw * sc_in)) // 8) * 8)
        ih = max(8, (int(round(ch * sc_in)) // 8) * 8)
        crop_inj = crop.resize((iw, ih), Image.LANCZOS)
        minp_inj = Image.fromarray((minp.astype(np.uint8) * 255), "L").resize((iw, ih), Image.NEAREST)
        print(f"[{tag}] inj downscale {cw}x{ch} -> {iw}x{ih}")
    else:
        crop_inj = crop
        minp_inj = Image.fromarray((minp.astype(np.uint8) * 255), "L")

    ts = str(int(time.time() * 1000))
    src_name = f"v342_src_{ts}.png"
    mask_name = f"v342_mask_{ts}.png"
    crop_inj.save(str(CUI_IN / src_name))
    minp_inj.save(str(CUI_IN / mask_name))

    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "LoadImage", "inputs": {"image": src_name}},
        "3": {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "red"}},
        "4": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
        "5": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["4", 0], "mask": ["3", 0]}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["1", 1]}},
    }
    # IPAdapter style-lock 可选：weight>0 才挂（end_at 早收 → 只在前段控色、后段放开改结构）
    model_ref = ["1", 0]
    # v344b：可选 LoRA（p4 的"细笔线稿感"要靠 line-art/vector LoRA 拉回——
    # 纯 SDXL 会把棕榈画成粗团块，用户会说"乱做"）。
    if lora:
        wf["30"] = {"class_type": "LoraLoader",
                    "inputs": {"model": ["1", 0], "clip": ["1", 1],
                               "lora_name": lora[0][0],
                               "strength_model": lora[0][1], "strength_clip": lora[0][1]}}
        model_ref = ["30", 0]
        for _n in ("9", "10"):
            wf[_n]["inputs"]["clip"] = ["30", 1]
    if ipa_weight > 0:
        wf.update({
            "6": {"class_type": "IPAdapterUnifiedLoader",
                  "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}},
            "7": {"class_type": "LoadImage", "inputs": {"image": src_name}},
            "8": {"class_type": "IPAdapter",
                  "inputs": {"model": ["6", 0], "ipadapter": ["6", 1], "image": ["7", 0],
                             "weight": ipa_weight, "start_at": 0.0,
                             "end_at": ipa_end, "weight_type": "style transfer"}},
        })
        model_ref = ["8", 0]
    # v344：ControlNet 结构引导 —— 治「乱码/失真」的唯一正解。
    # 根因：无结构约束 + denoise 0.85 → 模型自由发挥 → 白羽乱飞/形体崩坏。
    # 做法：用**原裁块**做 canny/depth 线稿 → ControlNetApplyAdvanced 引导前 cn_end 段，
    # 让新内容落在原构图/剪影骨架里（满足用户 p6 原话"只保留原物种性质跟色彩构图"）。
    # cn_end<1.0 → 后段放开、允许细节自由重生（不然只是描一遍原图、变化为零）。
    pos_ref, neg_ref = ["9", 0], ["10", 0]
    if cn_name:
        _pre = "20"
        if cn_pre == "canny":
            wf[_pre] = {"class_type": "Canny",
                        "inputs": {"image": ["2", 0], "low_threshold": cn_low,
                                   "high_threshold": cn_high}}
        elif cn_pre == "lineart":
            wf[_pre] = {"class_type": "LineArtPreprocessor",
                        "inputs": {"image": ["2", 0], "coarse": "disable", "resolution": 1024}}
        elif cn_pre == "depth":
            wf[_pre] = {"class_type": "DepthAnythingPreprocessor",
                        "inputs": {"image": ["2", 0], "resolution": 1024}}
        elif cn_pre == "tile":
            _pre = None                              # tile 直接用原图
        wf["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": cn_name}}
        wf["22"] = {"class_type": "ControlNetApplyAdvanced",
                    "inputs": {"positive": ["9", 0], "negative": ["10", 0],
                               "control_net": ["21", 0],
                               "image": (["2", 0] if _pre is None else [_pre, 0]),
                               "strength": cn_strength,
                               "start_percent": 0.0, "end_percent": cn_end}}
        pos_ref, neg_ref = ["22", 0], ["22", 1]
    wf.update({
        "11": {"class_type": "KSampler", "inputs": {"model": model_ref, "positive": pos_ref,
              "negative": neg_ref, "latent_image": ["5", 0], "seed": seed,
              "steps": 30, "cfg": 6.5, "sampler_name": "dpmpp_2m",
              "scheduler": "karras", "denoise": denoise}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0],
              "filename_prefix": f"v342_{tag}_{ts}"}},
    })
    t0 = time.time()
    outs = comfy_submit(wf)
    res = None
    for nid, o in outs.items():
        if "images" in o:
            res = _fetch(o); break
    if res is None:
        raise RuntimeError("no comfy output")
    res = res.resize((cw, ch), Image.LANCZOS)
    print(f"[{tag}] comfy {(time.time()-t0):.0f}s -> {res.size}")

    oc = np.asarray(crop, np.float32); rc = np.asarray(res, np.float32)
    if color_match and float(color_match) > 0:
        sel = feather > 0.3
        if int(sel.sum()) > 50:
            rc = _match_region(rc, oc, sel, alpha=float(color_match))
    a_ = feather[..., None]
    comp = oc * (1 - a_) + rc * a_
    comp_img = Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8), "RGB")
    full = np.array(img).copy()
    full[y0:y1, x0:x1] = np.asarray(comp_img)
    for f in (src_name, mask_name):
        p = CUI_IN / f
        if p.exists():
            p.unlink()
    return Image.fromarray(full)


# ---------------------------------------------------------------- 量化自检
def qc(img_orig, img_new, mask):
    from skimage.color import rgb2lab
    W, H = img_orig.size
    o = np.asarray(img_orig, np.float32); n = np.asarray(img_new, np.float32)
    ol = rgb2lab(o / 255.0); nl = rgb2lab(n / 255.0)
    mfull = np.zeros((H, W), bool); mfull[mask] = True
    d = np.sqrt(((ol - nl) ** 2).sum(2))
    bg = ~mfull
    bg_d = d[bg].mean()
    fg = mfull
    fg_d = d[fg].mean() if fg.any() else 0.0
    # 构图：主体质心 + bbox 尺寸（用掩膜在新图上重检困难，这里用原掩膜位置近似）
    ys, xs = np.where(mfull)
    cx, cy = xs.mean(), ys.mean()
    bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
    print(f"[qc] 背景LAB距离(应≈0)={bg_d:.2f}  主体LAB距离(应大)={fg_d:.2f}  "
          f"质心=({cx:.0f},{cy:.0f}) 尺寸={bw}x{bh}")
    return dict(bg_lab=bg_d, fg_lab=fg_d, cx=cx, cy=cy, bw=bw, bh=bh)


PROMPTS = {
    "6978": ("a heraldic bat emblem, dark violet and deep purple ornamental linework, "
             "spread bat wings with dark membrane, different wing and ear shape, "
             "centered oval emblem on a purple circular disc, symmetrical, purple color palette, "
             "flat matte vector colors, dark purple and near-black only, bold clean outline, "
             "no text, no letters, no gold, no yellow, no glow, no highlight",
             "text, letters, words, watermark, blurry, low quality, photo, realistic, "
             "gold, yellow, orange, pink, magenta, neon, glow, luminous, shiny, "
             "sparkle, gradient, white, bright highlights, extra limbs, deformed wings"),
    "p6": ("a bald eagle head with brown and white feathers, and a bone-white horned skull, "
           "sharp beak, fierce eyes, ornate dark engraving, brown tan white and black colors, "
           "black background, different feather arrangement and skull shape, no text, no letters",
           "text, letters, words, watermark, blurry, low quality, photo, realistic, "
           "extra heads, extra skulls, deformed"),
    "p4": ("elegant tropical palm trees drawn as fine black line art, thin smooth curving "
           "trunks with delicate ring texture, airy crown of many thin separate fronds with "
           "fine leaflets, solid jet-black ink on camouflage, crisp sharp edges, high contrast, "
           "flat vector look, lots of open space between fronds, seamless pattern, "
           "no text, no letters",
           "text, letters, words, color, colorful, photo, realistic, blurry, low quality, "
           "gradient, gray, white fronds, smudge, blotch, thick trunks, fat, blobby, "
           "solid black mass, chunky, heavy bold strokes, clump, semi-transparent, faded"),
}


# 定稿参数（v344）：
#  - 挂 **ControlNet canny（结构引导）** 治「乱码/失真」（用户第 12 轮核心批评：
#    "图片乱码很严重，失真情况太明显"）。根因是无结构约束 + 高 denoise → 自由发挥崩形。
#  - 仍不挂 IPAdapter（它把原主体外观锁死/泄幽灵，v342 已证）。
#  - cn_end<1.0：前段跟原骨架、后段放开改细节（= 用户要的"保留物种/配色/构图，其余全改"）。
_CN = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CFG = {
    "6978": dict(denoise=0.92, ipa_weight=0.0, color_match=0.92, seed=7,
                 cn_name=_CN, cn_strength=0.38, cn_pre="canny", cn_end=0.45),
    "p6": dict(denoise=0.85, ipa_weight=0.0, color_match=0.90, seed=7,
               cn_name=_CN, cn_strength=0.60, cn_pre="canny", cn_end=0.75),
    "p4": dict(denoise=0.85, ipa_weight=0.0, color_match=0.90, seed=7,
               cn_name=_CN, cn_strength=0.35, cn_pre="canny", cn_end=0.45),
}


def snap_p4_ink(orig, gen, mask, ink0):
    """p4「迷彩底 + 纯黑墨迹」两层重建。

    p4 的设计语言只有两层：**迷彩底**与**纯黑墨迹**（棕榈剪影）。SDXL 重生后墨迹会发灰/
    发糊（灰树、草簇）—— 与"硬黑剪影"语言不符，用户第 11 轮原话"不允许你这样乱做"。
    本函数在掩膜内把两层重新分开：
      ① base = 原图按 ink0(原墨迹)做**最近邻填充** → 干净迷彩底（背景版式逐像素不变）；
      ② ink_new = 掩膜内 **Otsu 自适应阈值** 切出的暗部。⚠️ 不能用固定阈值+饱和度过滤：
         实测 SDXL 的墨迹并非中性黑（rsat 偏高），固定 118+rsat<0.20 只剩 8% 墨 → 树被
         削薄成残枝；Otsu 在掩膜内自适应分割（实测阈值 67.3 → 墨 20.8%，与原墨迹 22.6%
         基本一致，密度不丢）；
      ③ 输出 = 掩膜外原图；掩膜内 = base 迷彩 ⊕ ink_new 纯黑。
    结果：**剪影形状换新、迷彩与版式分毫不动、墨迹回到硬黑**。
    """
    from skimage.filters import threshold_otsu
    o = np.asarray(orig, np.float32)
    g = np.asarray(gen, np.float32)
    lg = _lum(g)
    # **密度匹配阈值**（v344b）：Otsu 会把新剪影切得比原图厚 ~1.5x（实测 34.7% vs 原
    # 22.6%）→ 棕榈变"粗团块"，用户读作"乱做"。改为令掩膜内落墨像素数 == 原墨迹像素数
    # → 密度与原图对齐，笔画粗细回到线稿量级。（保留 Otsu 作为退化回退）
    _ns = int(mask.sum())
    if _ns > 0:
        _frac = min(0.95, max(0.02, float(ink0.sum()) / float(_ns)))
        t = float(np.percentile(lg[mask], 100.0 * _frac))
    else:
        t = float(threshold_otsu(lg[mask])) if mask.any() else 100.0
    t = min(max(t, 20.0), 200.0)
    ink = (lg < t) & mask
    ink = ndi.binary_closing(ink, structure=_disk(2))
    ink = ndi.binary_opening(ink, structure=_disk(1))
    # 去碎点：SDXL 在迷彩上留的孤立黑斑（< 60px）不是树，必须清掉，否则读作"乱做"
    lab, n = ndi.label(ink, structure=np.ones((3, 3), bool))
    if n:
        sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        keep = np.zeros(n + 1, bool)
        for j in range(1, n + 1):
            if sz[j - 1] >= 60:
                keep[j] = True
        ink = keep[lab]
    # 干净迷彩底：墨迹像素用"最近的非墨迹像素"颜色回填（camo 色块大而平滑，无违和）
    idx = ndi.distance_transform_edt(ink0, return_distances=False, return_indices=True)
    base = o[idx[0], idx[1]]
    black = np.array([12.0, 10.0, 9.0], np.float32)[None, None, :]
    # 硬切（不用软斜坡）：实测软斜坡会把 SDXL 的中调发丝（lum 80~150）打回迷彩、
    # 只留笔画芯 → 树变成"灰雾+黑芯"的糊团；硬切虽然笔画偏粗，但剪影干净利落，
    # 更接近 p4 的"纯黑墨迹"语言。
    a = ink.astype(np.float32)[..., None] * mask[..., None]
    inside = base * (1.0 - a) + black * a
    mm = mask[..., None]
    out = o * (1.0 - mm) + inside * mm
    print(f"[p4-snap] otsu={t:.1f} ink={100 * ink.mean():.1f}%")
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def run(which, denoise=None, ipa=0.0, seed=7):
    ic = FILE[which]
    cand = [SRC_DIR / ic, ROOT / "ComfyUI" / "input" / ic]
    src = next((p for p in cand if p.exists()), None)
    if src is None:
        hits = list(ROOT.rglob(ic))
        src = hits[0] if hits else None
    if src is None:
        raise FileNotFoundError(ic)
    img = Image.open(src).convert("RGB")
    print(f"[{which}] src={src} size={img.size}")
    mask = {"6978": mask_6978, "p6": mask_p6, "p4": mask_p4}[which](img)
    prompt, neg = PROMPTS[which]
    cfg = dict(CFG[which])
    if denoise is not None:
        cfg["denoise"] = denoise
    if ipa:
        cfg["ipa_weight"] = ipa
    out = rebirth_subject(img, mask, prompt, neg, tag=which, **cfg)
    if which == "p4":
        ink0 = cpp.tree_ink_mask(img, lum_thr=55.0, sat_thr=14.0, min_px=25, thin_only=False)
        out = snap_p4_ink(img, out, mask, ink0)             # 迷彩原样 + 墨迹回硬黑
    save = OUT / f"{which}_rebirth_v344.jpg"
    out.save(str(save), quality=93)                    # 先落盘，避免 QC 异常丢结果
    Image.fromarray((mask.astype(np.uint8) * 255), "L").save(str(OUT / f"{which}_mask_v344.png"))
    print(f"[{which}] saved {save}")
    qc(img, out, mask)
    return save


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "6978"
    dn = float(sys.argv[2]) if len(sys.argv) > 2 else 0.72
    run(which, denoise=dn)
