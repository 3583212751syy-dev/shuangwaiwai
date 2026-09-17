"""v344: ControlNet 结构引导参数扫描 —— 专治主体重生的「乱码/失真」。

用法: venv/Scripts/python.exe src/v344_cn_sweep.py p6  < 多个配置在 CFGS 里改 >

对每套配置：跑一次重生 → 存 jpg → 出「原图 | 结果」主体区对比条 → 汇总成一张 sweep 图。
"""
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import v342_rebirth as v

OUT = ROOT / "jobs" / "v342_rebirth"
OUT.mkdir(parents=True, exist_ok=True)

# (tag, kwargs)
CFGS = {
    "p6": [
        ("cnE_cannyS45_e55_d88", dict(denoise=0.88, cn_name="controlnet-canny-sdxl-1.0.fp16.safetensors",
                                      cn_strength=0.45, cn_pre="canny", cn_end=0.55)),
        ("cnF_cannyS60_e75_d85", dict(denoise=0.85, cn_name="controlnet-canny-sdxl-1.0.fp16.safetensors",
                                      cn_strength=0.60, cn_pre="canny", cn_end=0.75)),
        ("cnG_cannyS50_e60_d90", dict(denoise=0.90, cn_name="controlnet-canny-sdxl-1.0.fp16.safetensors",
                                      cn_strength=0.50, cn_pre="canny", cn_end=0.60)),
    ],
    "p4": [
        ("p4A_dm_noLora", dict(denoise=0.85, cn_name="controlnet-canny-sdxl-1.0.fp16.safetensors",
                               cn_strength=0.35, cn_pre="canny", cn_end=0.45)),
        ("p4B_dm_vecLora", dict(denoise=0.85, cn_name="controlnet-canny-sdxl-1.0.fp16.safetensors",
                                cn_strength=0.35, cn_pre="canny", cn_end=0.45,
                                lora=[("DD-vector-v2.safetensors", 0.6)])),
    ],
}
MASKF = {"6978": v.mask_6978, "p6": v.mask_p6, "p4": v.mask_p4}


def sheet(which, items, mask):
    src = Image.open(v.SRC_DIR / v.FILE[which]).convert("RGB")
    ys, xs = np.where(mask)
    pad = 60
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1, y1 = min(src.width, xs.max() + pad), min(src.height, ys.max() + pad)
    box = (x0, y0, x1, y1)
    oc = src.crop(box)
    TW = 340
    sc = TW / oc.width
    oc2 = oc.resize((TW, int(oc.height * sc)), Image.LANCZOS)
    tiles = [("ORIGINAL", oc2)]
    for tag, im in items:
        t = im.crop(box).resize(oc2.size, Image.LANCZOS)
        tiles.append((tag, t))
    try:
        fnt = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 14)
    except Exception:
        fnt = ImageFont.load_default()
    LBL, GAP = 22, 10
    W = oc2.width * len(tiles) + GAP * (len(tiles) + 1)
    H = oc2.height + LBL + GAP * 2
    sh = Image.new("RGB", (W, H), (24, 24, 27))
    dr = ImageDraw.Draw(sh)
    for k, (tag, t) in enumerate(tiles):
        x = GAP + k * (TW + GAP)
        dr.text((x, GAP - 2), tag, fill=(235, 235, 235), font=fnt)
        sh.paste(t, (x, LBL + GAP - 4))
    p = OUT / f"_sweep_{which}.jpg"
    sh.save(p, quality=90)
    print(f"[sheet] {p} {sh.size}")
    return p


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "p6"
    src = Image.open(v.SRC_DIR / v.FILE[which]).convert("RGB")
    mask = MASKF[which](src)
    prompt, neg = v.PROMPTS[which]
    items = []
    for tag, kw in CFGS[which]:
        print(f"===== {which} {tag} =====")
        try:
            out = v.rebirth_subject(src, mask, prompt, neg, color_match=0.90, seed=7,
                                    ipa_weight=0.0, tag=tag, **kw)
        except Exception as e:                     # 单配置失败不连坐（depth 预处理会超时）
            print(f"[skip] {tag}: {type(e).__name__} {e}")
            continue
        out.save(str(OUT / f"_{which}_{tag}.jpg"), quality=93)
        if which == "p4":                       # p4 看的是"墨迹回硬"后的成品
            from styles import camo_palm_pattern as cpp
            ink0 = cpp.tree_ink_mask(src, lum_thr=55.0, sat_thr=14.0, min_px=25,
                                     thin_only=False)
            out = v.snap_p4_ink(src, out, mask, ink0)
            out.save(str(OUT / f"_{which}_{tag}_snap.jpg"), quality=93)
        items.append((tag, out))
    if items:
        sheet(which, items, mask)
