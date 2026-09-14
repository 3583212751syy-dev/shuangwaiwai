"""实验：跳过 SDXL 全图重生，直接在原图上做「保色形变 + 原位换字」，验证能否干净无残影。
仅用于诊断，不写入正式管线。"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image
from styles import base

OUT = Path(__file__).resolve().parent / "jobs" / "router_out_v326" / "_exp"
OUT.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(Path(__file__).resolve().parent / "regression_set" / "set.json", encoding="utf-8"))
BY_ID = {i["id"]: i for i in cfg["images"]}


def rotate_region(img, center, radius, angle, feather=60):
    """以 center 为圆心、radius 为半径的圆形区域旋转 angle 度（羽化边界），used for 主体「改角度」。"""
    W, H = img.size
    cx, cy = center
    box = (int(cx - radius), int(cy - radius), int(cx + radius), int(cy + radius))
    patch = img.crop(box)
    rot = patch.rotate(angle, resample=Image.BICUBIC, center=(radius, radius), fillcolor=None)
    # 圆形羽化 mask
    m = Image.new("L", patch.size, 0)
    from PIL import ImageDraw, ImageFilter
    ImageDraw.Draw(m).ellipse([0, 0, patch.size[0], patch.size[1]], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(feather))
    out = img.copy()
    out.paste(rot, (box[0], box[1]), m)
    return out


def run(iid, morph_strength=0.0, rotate=None, seed=2026, tag="", margin=None, dilate=None):
    ic = BY_ID[iid]
    im = Image.open(ic["path"]).convert("RGB")
    W, H = im.size
    plan = [dict(it) for it in (ic.get("text_plan") or [])]
    if margin is not None:
        for it in plan:
            it["margin"] = margin
    if dilate is not None:
        for it in plan:
            it["dilate"] = dilate
    print(f"[{iid}] size={W}x{H} plan_items={len(plan)} margin={margin} dilate={dilate}")
    if rotate:
        im = rotate_region(im, rotate["center"], rotate["radius"], rotate["angle"])
        print(f"   rotated region center={rotate['center']} r={rotate['radius']} a={rotate['angle']}")
    if morph_strength > 0:
        ex = [it["bbox"] for it in plan]
        im = base.camo_blob_morph(im, seed=seed, strength=morph_strength, exclude_bboxes=ex)
        print(f"   morph strength={morph_strength}")
    if plan:
        im = base.replace_text_plan_v2(im, plan)
        print(f"   text swapped: {[it['word'] for it in plan]}")
    p = OUT / f"{iid}{tag}_swap.jpg"
    im.save(p, quality=92)
    print("   saved", p)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("iid")
    ap.add_argument("--morph", type=float, default=0.0)
    ap.add_argument("--rotate", type=str, default="")   # "cx,cy,radius,angle"
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--margin", type=int, default=None)
    ap.add_argument("--dilate", type=int, default=None)
    a = ap.parse_args()
    rot = None
    if a.rotate:
        cx, cy, r, ang = [float(v) for v in a.rotate.split(",")]
        rot = {"center": (cx, cy), "radius": r, "angle": ang}
    run(a.iid, morph_strength=a.morph, rotate=rot, tag=a.tag,
        margin=a.margin, dilate=a.dilate)
