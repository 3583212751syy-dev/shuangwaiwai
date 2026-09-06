"""v272 — mask 局部 inpaint 清字 + AbrilFatface 矢量烧字

目标：BACARDÍ bat_logo (6978...jpg)
策略：
  Stage B: 用慷慨文字 mask（整圈弧带+主副字+Est/年份）做 VAEEncodeForInpaint 局部重画
          —— 非 mask 区（蝙蝠/圆环/ribbon 背景）完全保留，只把文字区重画为无字背景
          —— 后置 Reinhard LAB 配色迁移锁紫底
  Stage C: PIL AbrilFatface 矢量原位烧新字（顶弧品牌+中心大字+底部副字+两侧 EST/年份）
           矢量渲染天然不糊，文本写死核对无错字

用法：
  --stage-a   只跑清字（mask inpaint），输出 v272_clean_lab.png，OCR 自检旧字残留
  --full      在最新 clean 上烧新字，输出终图 + 对照网格
"""
import argparse
import json
import math
import sys
import time
import uuid
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

PROJECT = Path("E:/Desktop/双接口/image-fission")
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
COMFY_OUTPUT = PROJECT / "ComfyUI" / "output"
COMFY_URL = "http://127.0.0.1:8188"
JOB = PROJECT / "jobs" / "v272"
JOB.mkdir(parents=True, exist_ok=True)

ORIG = "6978fabda2cc99629fa9e81f802762d3.jpg"
TARGET_W, TARGET_H = 1024, 1280

CKPT = "ProteusV0.4.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
LORA = "add-detail-xl.safetensors"
FONT = str(PROJECT / "ComfyUI" / "models" / "fonts" / "AbrilFatface-Regular.ttf")

DENOISE = 0.92
IPA = 0.32
CANNY_S = 0.50

POS = (
    "continue the vintage purple background and circular emblem ring, clean seamless area, "
    "plain decorative geometric ornaments only, NO text, NO letters, NO words, NO brand name, "
    "vintage craft spirits label art, sharp clean edges, saturated purple #6B2C8C deep violet #2A0A3F"
)
NEG = (
    "text, letters, words, readable text, legible text, brand name, BACARDI, logo, "
    "deformation, distorted, mutated bat, extra wings, blur, soft focus, watercolor, smudge, "
    "color shift, green, brown, gray, beige, desaturated, duplicate of source"
)


def make_text_mask():
    """1024x1280 文字区 mask：白=重绘，黑=保留。覆盖整圈弧带+主副字+Est/年份。"""
    mask = Image.new("L", (TARGET_W, TARGET_H), 0)
    d = ImageDraw.Draw(mask)
    SX, SY = TARGET_W / 1552, TARGET_H / 2000

    # 整圈 ribbon（弧字带）
    outer = [(x * SX, y * SY) for x, y in [
        (120, 380), (120, 450), (140, 600), (180, 750), (250, 880),
        (450, 950), (700, 980), (850, 980), (1100, 950), (1300, 880),
        (1370, 750), (1410, 600), (1430, 450), (1430, 380),
        (1330, 310), (1100, 270), (776, 255), (450, 270), (220, 310)]]
    inner = [(x * SX, y * SY) for x, y in [
        (600, 400), (776, 400), (950, 400),
        (1100, 520), (1100, 720), (1000, 800),
        (900, 850), (776, 900), (650, 850),
        (550, 800), (400, 720), (400, 520)]]
    d.polygon(outer + inner[::-1], fill=255)
    # 中心大字
    d.rectangle([(160 * SX, 980 * SY), (1390 * SX, 1170 * SY)], fill=255)
    # 底部副字
    d.rectangle([(360 * SX, 1180 * SY), (1195 * SX, 1340 * SY)], fill=255)
    # 两侧 Est / 年份
    d.rectangle([(350 * SX, 740 * SY), (560 * SX, 860 * SY)], fill=255)
    d.rectangle([(1000 * SX, 740 * SY), (1220 * SX, 860 * SY)], fill=255)
    mask = mask.filter(ImageFilter.MaxFilter(13))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=6))
    return mask


def node(id_, class_type, inputs):
    return {id_: {"class_type": class_type, "inputs": inputs}}


def build(seed):
    g = {}
    g.update(node("1", "CheckpointLoaderSimple", {"ckpt_name": CKPT}))
    g.update(node("2", "LoadImage", {"image": ORIG}))
    g.update(node("3", "ImageScale", {"image": ["2", 0], "width": TARGET_W, "height": TARGET_H,
                                      "upscale_method": "lanczos", "crop": "disabled"}))
    g.update(node("m", "LoadImage", {"image": "v272_text_mask.png"}))
    g.update(node("4", "VAEEncodeForInpaint", {"pixels": ["3", 0], "vae": ["1", 2],
                                              "mask": ["m", 1], "grow_mask_by": 12}))
    g.update(node("5", "IPAdapterUnifiedLoader", {"model": ["1", 0], "preset": "PLUS (high strength)"}))
    g.update(node("6", "IPAdapterAdvanced", {"model": ["1", 0], "ipadapter": ["5", 1], "image": ["3", 0],
                                             "weight": IPA, "weight_type": "style transfer",
                                             "combine_embeds": "average", "start_at": 0.0, "end_at": 0.85,
                                             "noise": 0.05, "embeds_scaling": "V only"}))
    g.update(node("7", "LoraLoader", {"model": ["6", 0], "clip": ["1", 1], "lora_name": LORA,
                                     "strength_model": 1.0, "strength_clip": 1.0}))
    g.update(node("20", "CannyEdgePreprocessor", {"image": ["3", 0], "low_threshold": 0.10,
                                                 "high_threshold": 0.25, "resolution": 1024}))
    g.update(node("21", "ControlNetLoader", {"control_net_name": CN_CANNY}))
    g.update(node("22", "ControlNetApply", {"conditioning": ["pg", 0], "control_net": ["21", 0],
                                           "image": ["20", 0], "strength": CANNY_S}))
    g.update(node("pg", "CLIPTextEncode", {"clip": ["7", 1], "text": POS}))
    g.update(node("ng", "CLIPTextEncode", {"clip": ["7", 1], "text": NEG}))
    g.update(node("10", "KSampler", {"model": ["7", 0], "positive": ["22", 0], "negative": ["ng", 0],
                                    "latent_image": ["4", 0], "seed": seed, "steps": 28, "cfg": 7.0,
                                    "sampler_name": "euler", "scheduler": "normal", "denoise": DENOISE}))
    g.update(node("12", "VAEDecode", {"samples": ["10", 0], "vae": ["1", 2]}))
    g.update(node("15", "SaveImage", {"images": ["12", 0], "filename_prefix": "v272_clean"}))
    return g


def color_transfer(src_bgr, dst_bgr, alpha=1.0):
    src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst = cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = dst.copy()
    for i in range(3):
        s_mean, s_std = src[:, :, i].mean(), src[:, :, i].std() + 1e-6
        d_mean, d_std = dst[:, :, i].mean(), dst[:, :, i].std() + 1e-6
        out[:, :, i] = (dst[:, :, i] - d_mean) * (s_std / d_std) + s_mean
    out = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    if alpha >= 1.0:
        return out
    blended = dst.astype(np.float32) * (1 - alpha) + out.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def ocr_check(path):
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        res = reader.readtext(np.array(Image.open(path).convert("RGB")))
        return [(t, round(c, 2)) for _, t, c in res]
    except Exception as e:
        return [("OCR_FAIL", str(e))]


def submit(wf):
    data = json.dumps({"prompt": wf, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def poll(pid, timeout_s=900):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(4)
        try:
            with urllib.request.urlopen(f"{COMFY_URL}/history/{pid}", timeout=20) as r:
                h = json.loads(r.read())
        except Exception:
            continue
        if pid in h:
            e = h[pid]
            if e.get("status", {}).get("completed"):
                return e
            err = e.get("status", {}).get("error")
            if err:
                raise RuntimeError(str(err))
    raise TimeoutError("timeout")


def collect_out(entry):
    outs = []
    for nid, node in entry.get("outputs", {}).items():
        if "images" in node:
            for im in node["images"]:
                p = Path(COMFY_OUTPUT) / im["filename"]
                if not p.exists():
                    p = Path(im.get("abs_path", p))
                if p.exists():
                    outs.append(p)
    return outs


def calibrate_font(text, font_path, start_size, max_w):
    lo, hi, best = 8, start_size, start_size
    while lo <= hi:
        mid = (lo + hi) // 2
        w = ImageFont.truetype(font_path, mid).getlength(text)
        if w <= max_w:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def burn_text(img, big, arc, sub, color=(28, 18, 30)):
    sys.path.insert(0, str(PROJECT / "src"))
    import arc_text
    img = img.convert("RGB")
    w, h = img.width, img.height
    draw = ImageDraw.Draw(img)

    cy = int(h * 0.41)
    radius = int(h * 0.26)
    fs_arc = int(h * 0.067)
    while fs_arc > 18:
        arc_len = arc_text.fit_arc_text_width(arc, FONT, fs_arc, radius, char_spacing_px=2)
        total_deg = math.degrees(arc_len / radius)
        if total_deg <= 100:
            break
        fs_arc = int(fs_arc * 0.92)
    img = arc_text.draw_arc_text(img, arc, FONT, fs_arc, color,
                                 center=(w // 2, cy), radius=radius,
                                 start_angle_deg=225, end_angle_deg=315,
                                 char_spacing_px=2, flip_180=False)

    fs_big = calibrate_font(big, FONT, int(h * 0.16), max_w=int(w * 0.80))
    font = ImageFont.truetype(FONT, fs_big)
    draw = ImageDraw.Draw(img)
    draw.text((w // 2, int(h * 0.555)), big, font=font, fill=color, anchor="mm")

    fs_sub = calibrate_font(sub, FONT, int(h * 0.085), max_w=int(w * 0.62))
    font = ImageFont.truetype(FONT, fs_sub)
    draw.text((w // 2, int(h * 0.655)), sub, font=font, fill=color, anchor="mm")

    f_side = ImageFont.truetype(FONT, int(h * 0.032))
    draw.text((int(w * 0.305), int(h * 0.415)), "EST.", font=f_side, fill=color, anchor="mm")
    draw.text((int(w * 0.695), int(h * 0.415)), "1862", font=f_side, fill=color, anchor="mm")
    return img


TEXT = dict(big="NOCTAVEN", arc="MIDNIGHT BAT DISTILLERY", sub="SHADOW OF THE WING")


def stage_b_cv2():
    """本地 cv2.inpaint 清字：从周围像素修补，绝不重建旧字。"""
    orig = np.array(Image.open(COMFY_INPUT / ORIG).convert("RGB"))
    h, w = orig.shape[:2]
    SX, SY = w / 1552.0, h / 2000.0
    mask = np.zeros((h, w), np.uint8)
    d = ImageDraw.Draw(Image.fromarray(mask))  # 仅用于矩形坐标

    def poly(pts):
        return np.array([(int(x * SX), int(y * SY)) for x, y in pts], np.int32)

    # 整圈 ribbon（弧字带）
    outer = poly([(120, 380), (120, 450), (140, 600), (180, 750), (250, 880),
                  (450, 950), (700, 980), (850, 980), (1100, 950), (1300, 880),
                  (1370, 750), (1410, 600), (1430, 450), (1430, 380),
                  (1330, 310), (1100, 270), (776, 255), (450, 270), (220, 310)])
    inner = poly([(600, 400), (776, 400), (950, 400), (1100, 520), (1100, 720),
                  (1000, 800), (900, 850), (776, 900), (650, 850), (550, 800),
                  (400, 720), (400, 520)])
    cv2.fillPoly(mask, [np.concatenate([outer, inner[::-1]], axis=0)], 255)
    # 中心大字 + 底部副字 + 两侧 Est/年份
    for (x0, y0, x1, y1) in [
        (160, 980, 1390, 1170), (360, 1180, 1195, 1340),
        (350, 740, 560, 860), (1000, 740, 1220, 860)]:
        cv2.rectangle(mask, (int(x0 * SX), int(y0 * SY)), (int(x1 * SX), int(y1 * SY)), 255, -1)

    # EasyOCR 兜底：补捡可能漏掉的小字
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        res = reader.readtext(orig)
        for poly4, txt, conf in res:
            if conf < 0.2:
                continue
            pts = np.array(poly4, np.int32)
            cv2.fillPoly(mask, [pts], 255)
    except Exception as e:
        print("  [warn] EasyOCR 失败，仅用几何 mask:", e)

    # 大膨胀 + 羽化，确保覆盖抗锯齿/细笔画
    mask = cv2.dilate(mask, np.ones((17, 17), np.uint8), iterations=2)
    mask = cv2.GaussianBlur(mask, (0, 0), 12)
    mask_bin = (mask > 127).astype(np.uint8) * 255

    clean = cv2.inpaint(orig, mask_bin, 9, cv2.INPAINT_NS)
    return clean, mask_bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a", action="store_true", help="只跑清字")
    ap.add_argument("--full", action="store_true", help="在 clean 上烧字")
    ap.add_argument("--seed", type=int, default=272201)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.stage_a:
        print("[v272 STAGE-A] cv2.inpaint 清字（不重建旧字）...")
        clean, mask_bin = stage_b_cv2()
        clean_bgr = cv2.cvtColor(clean, cv2.COLOR_RGB2BGR)
        # LAB 锁紫：把清字图配色迁到原图
        src_bgr = cv2.cvtColor(np.array(Image.open(COMFY_INPUT / ORIG).convert("RGB")), cv2.COLOR_RGB2BGR)
        matched = color_transfer(src_bgr, clean_bgr, alpha=1.0)
        lab = JOB / "v272_clean_lab.png"
        Image.fromarray(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB)).save(lab, quality=95)
        # 存一份 1024x1280 供烧字
        small = JOB / "v272_clean_1024.png"
        Image.fromarray(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB)).resize((TARGET_W, TARGET_H), Image.LANCZOS).save(small, quality=95)
        maskp = JOB / "v272_text_mask.png"
        Image.fromarray(mask_bin).save(maskp, quality=98)
        print(f"  clean -> {lab}")
        txt = ocr_check(str(lab))
        print(f"  OCR 旧字残留: {txt}")
        return

    if args.full:
        cands = sorted(JOB.glob("v272_clean_1024.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            print("no clean_1024, 先跑 --stage-a"); return
        base = Image.open(cands[0]).convert("RGB")
        final = burn_text(base, TEXT["big"], TEXT["arc"], TEXT["sub"])
        out = JOB / "v272_final.png"
        final.save(out, quality=95)
        print(f"  final -> {out}")
        txt = ocr_check(str(out))
        print(f"  OCR 终图可读字(应为新字): {txt}")
        grid = Image.new("RGB", (TARGET_W * 2 + 24, TARGET_H), "white")
        grid.paste(Image.open(COMFY_INPUT / ORIG).convert("RGB").resize((TARGET_W, TARGET_H)), (0, 0))
        grid.paste(final, (TARGET_W + 24, 0))
        gp = JOB / "_grid_v272.png"
        grid.save(gp, quality=92)
        print(f"  grid -> {gp}")
        return

    print("用法: --stage-a 或 --full")


if __name__ == "__main__":
    main()
