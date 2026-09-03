"""
v253_variants — 生成 3 个蝙蝠姿态变体供用户挑选 (复用 v253 管线: PIL擦背景/字 + SDXL inpaint真重绘 + LAB锁色 + 左半色板吸附)

每个变体: 不同 seed + 不同 pose prompt -> 不同蝙蝠姿态/角度 (真·AI裂变)
全部颜色锁死 + 过 QC 边缘门禁(平黑剪影); 用户目检挑最好看的蝙蝠姿态
"""
import importlib.util, shutil, math
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("E:/Desktop/双接口/image-fission")
spec = importlib.util.spec_from_file_location("v253", str(ROOT / "src" / "v253_bat_logo_inpaint.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

JOB = ROOT / "jobs" / "smoke_v253"
JOB.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    ("A_up",      253101, "a single stylized 2D SOLID FLAT BLACK bat silhouette with wings spread WIDE upward, head up, "
                         "slight LEFT tilt, gothic vintage craft spirits emblem, SOLID FILLED FLAT SHAPE, NO internal detail, "
                         "NO shading, NO gradient, NO texture, clean outline only, perfectly centered, NO shadow, NO ground plane"),
    ("B_dive",    253202, "a single stylized 2D SOLID FLAT BLACK bat silhouette in a DYNAMIC DIVING pose, wings swept back, "
                         "head tilted forward-down, gothic vintage craft spirits emblem, SOLID FILLED FLAT SHAPE, NO internal detail, "
                         "NO shading, NO gradient, NO texture, clean outline only, perfectly centered, NO shadow, NO ground plane"),
    ("C_herald",  253303, "a single stylized 2D SOLID FLAT BLACK bat silhouette with wings spread symmetrically, body tilted RIGHT, "
                         "aggressive heraldic pose, gothic vintage craft spirits emblem, SOLID FILLED FLAT SHAPE, NO internal detail, "
                         "NO shading, NO gradient, NO texture, clean outline only, perfectly centered, NO shadow, NO ground plane"),
]


def process_one(tag, seed, pose):
    # 复用 v253 的底图/掩码(一次即可)
    orig = Image.open(m.COMFY_INPUT / m.REF_IMG).convert("RGB")
    bgr = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    if not (JOB.parent / "smoke_v253" / "v253_base.png").exists():
        pass
    base, bat_mask = m.clean_base(orig, bgr)
    base.save(str(m.COMFY_INPUT / "v253_base.png"), quality=95)
    mask = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bx, by, bw, bh = m.BAT_BBOX
    mask.paste(Image.new("RGBA", (bw, bh), (255, 255, 255, 255)), (bx, by))
    mask.save(str(m.COMFY_INPUT / "v253_mask.png"))

    m.POS_BAT = pose
    m.SEED = seed
    raw = m.gen_inpaint(seed, "v253_base.png", "v253_mask.png")
    if not raw:
        print(f"[{tag}] SDXL FAIL -> 回退 PIL")
        composite = base.copy()
        rot = m.fission_bat_pil(orig, bat_mask); rw, rh = rot.size
        composite.paste(rot, (m.BAT_CENTER[0] - rw // 2, m.BAT_CENTER[1] - rh // 2), rot)
        used = False
    else:
        # 复制 raw 防覆盖
        raw_copy = JOB / f"v253_{tag}_raw.png"
        shutil.copy(raw, raw_copy)
        sdxl = Image.open(raw_copy).convert("RGB")
        if sdxl.size != (w, h):
            sdxl = sdxl.resize((w, h))
        sdxl_bgr = cv2.cvtColor(np.array(sdxl), cv2.COLOR_RGB2BGR)
        locked = m.match_hist_lab(bgr, sdxl_bgr)
        locked = m.snap_to_original(locked, bgr, m.BAT_BBOX, n_colors=12)
        locked_rgb = Image.fromarray(cv2.cvtColor(locked, cv2.COLOR_BGR2RGB))
        chi = m.hist_chi2(bgr, locked)
        broke = m.bat_region_broke(locked_rgb, bat_mask)
        if chi < 0.30 and not broke:
            composite = locked_rgb; used = True
            print(f"[{tag}] SDXL bat chi2={chi:.3f} PASSED")
        else:
            print(f"[{tag}] SDXL 漂色/崩坏 chi2={chi:.3f} -> 回退 PIL")
            composite = base.copy()
            rot = m.fission_bat_pil(orig, bat_mask); rw, rh = rot.size
            composite.paste(rot, (m.BAT_CENTER[0] - rw // 2, m.BAT_CENTER[1] - rh // 2), rot)
            used = False

    # 烧字
    composite = m.burn_top_arc(composite, "LVMEN NOCTIS", int(h * 0.045))
    fs_main = m.calibrate("NOCTWING", 955); m.burn_centered(composite, "NOCTWING", fs_main, m.BIG_CENTER[0], m.BIG_CENTER[1])
    fs_small = m.calibrate("MORS VINI", 690); m.burn_centered(composite, "MORS VINI", fs_small, m.SMALL_CENTER[0], m.SMALL_CENTER[1])
    f_est = ImageFont.truetype(m.FONT_PATH, int(h * 0.022))
    ImageDraw.Draw(composite).text((int(w * 0.30), int(h * 0.745)), "Est.", font=f_est, fill=m.INK, anchor="mm")
    ImageDraw.Draw(composite).text((int(w * 0.70), int(h * 0.745)), "1862", font=f_est, fill=m.INK, anchor="mm")
    if composite.mode != "RGB":
        composite = composite.convert("RGB")
    out = JOB / f"v253_{tag}_bat_logo.jpg"
    composite.save(str(out), quality=95)
    print(f"[{tag}] saved {out} ({out.stat().st_size//1024} KB) used_sdxl={used}")
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
    outs = []
    for tag, seed, pose in VARIANTS:
        outs.append(process_one(tag, seed, pose))
    make_grid(outs, JOB / "_grid_v253_variants.jpg")
    print(f"[grid] saved {JOB / '_grid_v253_variants.jpg'}")


if __name__ == "__main__":
    main()
