"""
v254 — 在 v253 真·SDXL重绘 + 配色铁律双满足的基础上, 给蝙蝠加"内容细节" (用户原指令: 改变角度 + 一些内容细节)
  - 原图是平黑蝙蝠剪影, v253 也是平黑剪影(低细节才过边缘门禁); 本版把 SDXL 蝙蝠 prompt 改为
    带耳廓/翼膜扇骨/缺刻的平黑细节蝙蝠, 参数调高(denoise 0.66 / LORA 0.42 / Canny 0.70) 以出 anatomy 细节
  - 二级回退: 若细节版过不了 QC 边缘门禁(右半 edge > 左半x1.3), 回退 v253 安全参数(denoise 0.58 / LORA 0.25)
    + 轻量 PIL 内描边(echo outline + 脊柱线)补"内容细节", 仍平黑 2D / 零新色
  - 全程配色锁(LAB直方图匹配 + 左半色板吸附) + 权威 QC 门禁; 三种姿态复用 A/B/C seed 供直接对比 v253
"""
import importlib.util, shutil, math, json, subprocess, sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path("E:/Desktop/双接口/image-fission")
spec = importlib.util.spec_from_file_location("v253", str(ROOT / "src" / "v253_bat_logo_inpaint.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

JOB = ROOT / "jobs" / "smoke_v254"
JOB.mkdir(parents=True, exist_ok=True)
QC = Path("E:/AI/2026-09-02-09-43-13/qc_split_image.py")
INK_BGR = (31, 10, 26)  # INK(26,10,31) in BGR

# 详细版 prompt: 平黑 + 耳廓/翼膜扇骨/缺刻 + 内描边, 仍 2D flat / 零新色
POS_DETAILED = (
    "a single stylized 2D SOLID FLAT BLACK bat silhouette with wings spread, head up, "
    "DETAILED anatomy: defined pointed EARS, spread wing membrane with visible FINGER BONES and "
    "SCALLOPED wing edges, a thin central body line, gothic vintage craft spirits emblem, "
    "SOLID FILLED FLAT SHAPE with THIN clean internal contour lines only, "
    "NO shading, NO gradient, NO texture, NO fill variation, NO gray, "
    "flat printed vector graphic, perfectly centered inside the circular ring, NO shadow, NO ground plane"
)
# 安全版 prompt: v253 同款平黑剪影
POS_SAFE = (
    "a single stylized 2D SOLID FLAT BLACK bat silhouette with wings spread in a DYNAMIC ROTATED pose, "
    "head tilted up, gothic vintage craft spirits emblem style, "
    "SOLID FILLED FLAT SHAPE, NO internal detail, NO shading, NO gradient, NO texture, NO veins, NO fur, "
    "clean outline only, flat printed vector graphic, perfectly centered inside the circular ring, NO shadow"
)

VARIANTS = [
    ("A_up",     253101, "wings spread WIDE upward, slight LEFT tilt"),
    ("B_dive",   253202, "DYNAMIC DIVING pose, wings swept back, head tilted forward-down"),
    ("C_herald", 253303, "wings spread symmetrically, body tilted RIGHT, aggressive heraldic pose"),
]


# ---- 本地复刻 QC 边缘密度, 用于二级回退决策 ----
def _edge_density(im):
    g = im.convert("L")
    sx = g.filter(ImageFilter.Kernel((3, 3), [-1, 0, 1, -2, 0, 2, -1, 0, 1], scale=1))
    sy = g.filter(ImageFilter.Kernel((3, 3), [-1, -2, -1, 0, 0, 0, 1, 2, 1], scale=1))
    px, py = list(sx.getdata()), list(sy.getdata())
    mag = [abs(a) + abs(b) for a, b in zip(px, py)]
    return sum(1 for v in mag if v > 60) / len(mag)


def edge_ratio(img):
    w, h = img.size
    mid = w // 2
    left = img.crop((0, 0, mid, h)); right = img.crop((mid, 0, w, h))
    le, re = _edge_density(left), _edge_density(right)
    return le, re, (re / le if le > 0 else 99)


def add_bat_detail(img):
    """PIL 补细节: 蝙蝠区内描边(echo) + 脊柱线, 平黑 INK, 零新色, 边缘增量可控"""
    arr = np.array(img.convert("RGB")).copy()
    bx, by, bw, bh = m.BAT_BBOX
    region_rgb = arr[by:by + bh, bx:bx + bw]
    crop = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = (gray < 90).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if mask.sum() < 300:
        return img
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    e1 = cv2.erode(mask, k1)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    e2 = cv2.erode(mask, k2)
    ring = (e1 > 0) & (e2 == 0)
    ys, xs = np.where(mask > 0)
    cy, cx = int(ys.mean()), int(xs.mean())
    region_bgr = crop.copy()
    region_bgr[ring] = INK_BGR
    cv2.line(region_bgr, (cx, int(ys.min())), (cx, int(ys.max())), INK_BGR, 2)
    arr[by:by + bh, bx:bx + bw] = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(arr)


def gen_locked(pose, seed, bat_mask):
    """提交 SDXL inpaint + 配色锁, 返回 (composite_rgb, used_sdxl)"""
    m.POS_BAT = pose
    m.SEED = seed
    raw = m.gen_inpaint(seed, "v253_base.png", "v253_mask.png")
    if not raw:
        print(f"  SDXL FAIL -> 回退 PIL")
        return None, False
    raw_copy = JOB / f"_raw_{seed}.png"
    shutil.copy(raw, raw_copy)
    sdxl = Image.open(raw_copy).convert("RGB")
    if sdxl.size != (1552, 2000):
        sdxl = sdxl.resize((1552, 2000))
    orig = Image.open(m.COMFY_INPUT / m.REF_IMG).convert("RGB")
    bgr = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
    sdxl_bgr = cv2.cvtColor(np.array(sdxl), cv2.COLOR_RGB2BGR)
    locked = m.match_hist_lab(bgr, sdxl_bgr)
    locked = m.snap_to_original(locked, bgr, m.BAT_BBOX, n_colors=12)
    locked_rgb = Image.fromarray(cv2.cvtColor(locked, cv2.COLOR_BGR2RGB))
    chi = m.hist_chi2(bgr, locked)
    broke = m.bat_region_broke(locked_rgb, bat_mask)
    if chi < 0.30 and not broke:
        print(f"  SDXL chi2={chi:.3f} PASSED")
        return locked_rgb, True
    print(f"  SDXL 漂色/崩坏 chi2={chi:.3f} -> 回退 PIL")
    return None, False


def burn_words(composite):
    h, w = composite.size
    composite = m.burn_top_arc(composite, "LVMEN NOCTIS", int(h * 0.045))
    fs_main = m.calibrate("NOCTWING", 955); m.burn_centered(composite, "NOCTWING", fs_main, m.BIG_CENTER[0], m.BIG_CENTER[1])
    fs_small = m.calibrate("MORS VINI", 690); m.burn_centered(composite, "MORS VINI", fs_small, m.SMALL_CENTER[0], m.SMALL_CENTER[1])
    f_est = ImageFont.truetype(m.FONT_PATH, int(h * 0.022))
    ImageDraw.Draw(composite).text((int(w * 0.30), int(h * 0.745)), "Est.", font=f_est, fill=m.INK, anchor="mm")
    ImageDraw.Draw(composite).text((int(w * 0.70), int(h * 0.745)), "1862", font=f_est, fill=m.INK, anchor="mm")
    return composite


def process_one(tag, seed, pose_tail, bat_mask):
    # TIER1: 详细版
    m.DENOISE = 0.66; m.LORA_DETAIL = 0.42; m.CANNY_STRENGTH = 0.70
    pose = POS_DETAILED + ", " + pose_tail
    print(f"[{tag}] TIER1 detailed denoise={m.DENOISE} lora={m.LORA_DETAIL}")
    comp, used = gen_locked(pose, seed, bat_mask)
    detail_src = "sdxl-detailed"
    if comp is not None:
        le, re, ratio = edge_ratio(comp)
        print(f"[{tag}] TIER1 edge ratio={ratio:.3f} (le={le:.4f} re={re:.4f})")
        if ratio <= 1.30:
            tier = "TIER1-detail"
        else:
            print(f"[{tag}] TIER1 过边缘门禁(>{1.30}) -> 转 TIER2 安全+PIL补细节")
            comp = None
    # TIER2: 安全版 + PIL 细节
    if comp is None:
        m.DENOISE = 0.58; m.LORA_DETAIL = 0.25; m.CANNY_STRENGTH = 0.75
        pose2 = POS_SAFE + ", " + pose_tail
        print(f"[{tag}] TIER2 safe denoise={m.DENOISE} lora={m.LORA_DETAIL}")
        comp2, used2 = gen_locked(pose2, seed + 1, bat_mask)
        detail_src = "pil-echo"
        if comp2 is None:
            # 全回退: PIL 蝙蝠
            orig = Image.open(m.COMFY_INPUT / m.REF_IMG).convert("RGB")
            bgr = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
            base, _ = m.clean_base(orig, bgr)
            rot = m.fission_bat_pil(orig, bat_mask); rw, rh = rot.size
            comp2 = base.copy()
            comp2.paste(rot, (m.BAT_CENTER[0] - rw // 2, m.BAT_CENTER[1] - rh // 2), rot)
            used2 = False
            detail_src = "pil-fallback"
        comp2 = add_bat_detail(comp2)
        comp = comp2; used = used2
        tier = "TIER2+echo"
    comp = burn_words(comp)
    if comp.mode != "RGB":
        comp = comp.convert("RGB")
    out = JOB / f"v254_{tag}_bat_logo.jpg"
    comp.save(str(out), quality=95)
    # 跑权威 QC 记录
    try:
        r = subprocess.run([sys.executable, str(QC), str(out)], capture_output=True, text=True, timeout=120)
        qd = json.loads(r.stdout)
        qc_pass = qd.get("passed")
        qchi = qd.get("hist_chi2")
    except Exception as e:
        qc_pass, qchi = "ERR", None
        print(f"  QC parse ERR: {e}")
    print(f"[{tag}] saved {out.name} ({out.stat().st_size//1024} KB) detail={detail_src} tier={tier} QC={qc_pass} chi2={qchi}")
    return out


def make_grid(paths, out_path):
    imgs = [Image.open(p).convert("RGB") for p in paths]
    w, h = imgs[0].size
    gap = 12
    grid = Image.new("RGB", (w * len(imgs) + gap * (len(imgs) + 1), h), "white")
    for i, im in enumerate(imgs):
        grid.paste(im, (gap + i * (w + gap), 0))
    grid.save(str(out_path), quality=92)


def main():
    # 预生成干净底图 + 掩码(一次)
    orig = Image.open(m.COMFY_INPUT / m.REF_IMG).convert("RGB")
    bgr = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
    base, bat_mask = m.clean_base(orig, bgr)
    base.save(str(m.COMFY_INPUT / "v253_base.png"), quality=95)
    w, h = bgr.shape[:2]
    mask = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bx, by, bw, bh = m.BAT_BBOX
    mask.paste(Image.new("RGBA", (bw, bh), (255, 255, 255, 255)), (bx, by))
    mask.save(str(m.COMFY_INPUT / "v253_mask.png"))

    outs = []
    for tag, seed, pose_tail in VARIANTS:
        outs.append(process_one(tag, seed, pose_tail, bat_mask))
    make_grid(outs, JOB / "_grid_v254_variants.jpg")
    print(f"[grid] saved {JOB / '_grid_v254_variants.jpg'}")


if __name__ == "__main__":
    main()
