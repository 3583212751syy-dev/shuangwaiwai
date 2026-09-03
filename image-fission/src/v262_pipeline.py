"""v262_pipeline — 主体原位裂变 v2 (解决 v261 主体仍是蝙蝠剪影)

对比两种主体重画架构:
  [inpaint]  沿用 v261 的 IPAdapter+Canny+VAEEncodeForInpaint, 但把
             IPAdapter 0.35->0.15, denoise 0.92->0.95 (用户选项 A)
  [brushnet] 换用本地 SDXL BrushNet(down=18 检测为 XL) 替代 Canny 构图锁,
             主体区用「bbox 矩形 mask」让模型自由重画物种;
             IPAdapter 默认关(用无字紫底做参考锁色/风格), 可选 0.2 (用户选项 C)

主体区 mask 策略差异:
  inpaint 模式仍用蝙蝠剪影(仅重噪蝙蝠轮廓区域) -> 偏保守
  brushnet 模式用 bbox 矩形(给新鸟自由成形空间) -> 偏大胆

Stage3 复合: brushnet 输出已逐像素保留紫底/圆环, 仅主体区贴回, 高斯模糊 3px 去缝
             inpaint 模式同理用剪影 mask 复合

复用: load_layout / burn_text / qc_palette / submit / poll / collect_out
"""
import json
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
JOB = PROJECT / "jobs" / "v262"
JOB.mkdir(parents=True, exist_ok=True)
SRC = "test_6978fabda2cc99629fa9e81f802762d3.jpg"

sys.path.insert(0, str(PROJECT / "src"))
import importlib.util
spec = importlib.util.spec_from_file_location("v253", str(PROJECT / "src" / "v253_bat_logo_inpaint.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

INK = m.INK
FONT_PATH = m.FONT_PATH
RING = {"cx": 776, "cy": 744, "outer_r": 427, "inner_r": 413}

# 新物种 + 配套新词
SUBJECT_VARIANTS = [
    ("raven", "a single stylized 2D SOLID FLAT BLACK raven, wings spread in symmetric emblem pose, "
              "head turned to side, sharp beak, gothic vintage craft-spirits emblem, "
              "SATURATED PURPLE #6B2C8C and deep violet #2A0A3F and black #1A0A1F, flat printed vector, "
              "NO internal detail, NO shading, NO gradient, clean outline, perfectly centered inside the "
              "circular ring, symmetric composition",
     "NOCTWING", "REALM OF NIGHT WINGS", "RAVEN'S OATH"),
    ("owl", "a single stylized 2D SOLID FLAT BLACK great horned owl, wings spread, symmetric emblem pose, "
            "stern forward gaze, two ear tufts, gothic vintage craft-spirits emblem, "
            "SATURATED PURPLE #6B2C8C and deep violet #2A0A3F and black #1A0A1F, flat printed vector, "
            "NO internal detail, NO shading, NO gradient, clean outline, perfectly centered inside the "
            "circular ring, symmetric composition",
     "NIGHTOWL", "GUARDIAN OF DUSK", "WISE WATCH"),
    ("falcon", "a single stylized 2D SOLID FLAT BLACK peregrine falcon, wings swept in symmetric emblem pose, "
               "head turned, hooked beak, gothic vintage craft-spirits emblem, "
               "SATURATED PURPLE #6B2C8C and deep violet #2A0A3F and black #1A0A1F, flat printed vector, "
               "NO internal detail, NO shading, NO gradient, clean outline, perfectly centered inside the "
               "circular ring, symmetric composition",
     "SKYREAVER", "DOMAIN OF THE SKY", "STORM CLAW"),
]
POS = {v[0]: v[1] for v in SUBJECT_VARIANTS}

NEG = ("3d, 3d render, metallic, glossy, jewelry, gradient shading, smooth shading, soft airbrush, "
       "photorealistic, realistic, hyperrealistic, watercolor, "
       "text, letters, words, readable text, brand name, BACARDI, logo, monogram, "
       "multiple birds, second bird, extra wings, extra limbs, deformed, mutated, malformed, "
       "blurry, soft focus, low quality, jagged, noise, grain, "
       "gray, grayish, brown, green, blue, cyan, beige, tan, desaturated, washed out, off-palette, "
       "background change, new colors, color bleed")


def load_layout():
    return json.loads((PROJECT / "jobs" / "v261" / "layout.json").read_text(encoding="utf-8"))


def build_masks(layout):
    bgr = cv2.imdecode(np.fromfile(str(COMFY_INPUT / SRC), dtype=np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    subj = layout["subject"]
    sx, sy, sbw, sbh = subj["bbox"]

    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((xx - RING["cx"]) ** 2 + (yy - RING["cy"]) ** 2)
    ring_band = (dist > (RING["inner_r"] - 10)) & (dist < (RING["outer_r"] + 10))

    # 擦除 mask (同 v261)
    strokes = (gray < 140).astype(np.uint8) * 255
    text_mask = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    text_mask = cv2.dilate(text_mask, np.ones((4, 4), np.uint8), 1).astype(bool)
    text_mask[ring_band] = 0
    text_mask[sy - 16:sy + sbh + 16, sx - 16:sx + sbw + 16] = 0
    tmask_rgba = np.zeros((h, w, 4), np.uint8)
    tmask_rgba[..., 3] = text_mask.astype(np.uint8) * 255
    Image.fromarray(tmask_rgba, "RGBA").save(str(COMFY_INPUT / "v262_text_mask.png"))

    # 剪影 mask (inpaint 模式用)
    inside = np.zeros((h, w), bool); inside[sy - 14:sy + sbh + 14, sx - 14:sx + sbw + 14] = True
    blob = (gray < 70) & inside
    blob = cv2.morphologyEx(blob.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(blob, 8)
    if n > 1:
        sm_full = (lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))).astype(np.uint8) * 255
    else:
        sm = (gray[sy:sy + sbh, sx:sx + sbw] < 70).astype(np.uint8) * 255
        sm_full = np.zeros((h, w), np.uint8); sm_full[sy:sy + sbh, sx:sx + sbw] = sm
    sm = cv2.dilate(sm_full, np.ones((6, 6), np.uint8), 1)
    smask_rgba = np.zeros((h, w, 4), np.uint8); smask_rgba[..., 3] = sm
    Image.fromarray(smask_rgba, "RGBA").save(str(COMFY_INPUT / "v262_subj_mask.png"))

    # bbox 矩形 mask (brushnet 模式用: 给新鸟自由成形空间)
    pad = 16
    x0, y0 = max(0, sx - pad), max(0, sy - pad)
    x1, y1 = min(w, sx + sbw + pad), min(h, sy + sbh + pad)
    bbox = np.zeros((h, w), np.uint8); bbox[y0:y1, x0:x1] = 255
    bbox_rgba = np.zeros((h, w, 4), np.uint8); bbox_rgba[..., 3] = bbox
    Image.fromarray(bbox_rgba, "RGBA").save(str(COMFY_INPUT / "v262_subj_bbox_mask.png"))

    # masked canny (inpaint 模式用)
    canny = cv2.Canny(gray, int(255 * 0.10), int(255 * 0.25))
    canny[sy - 20:sy + sbh + 20, sx - 20:sx + sbw + 20] = 0
    canny[text_mask] = 0
    Image.fromarray(np.stack([canny] * 3, -1), "RGB").save(str(COMFY_INPUT / "v262_canny_struct.png"))

    # textless base: 径向渐变重建
    clean = (gray > 150) & (~ring_band) & (~text_mask) & (~strokes)
    clean[sy - 16:sy + sbh + 16, sx - 16:sx + sbw + 16] = False
    cys, cxs = np.nonzero(clean)
    dd = dist[cys, cxs].astype(float)
    rgb_f = rgb[cys, cxs].astype(float)
    coef = [np.polyfit(dd, rgb_f[:, c], 2) for c in range(3)]
    out = rgb.astype(float).copy()
    tys, txs = np.nonzero(text_mask)
    for c in range(3):
        out[tys, txs, c] = np.polyval(coef[c], dist[tys, txs])
    textless = Image.fromarray(out.astype(np.uint8), "RGB")
    textless.save(str(COMFY_INPUT / "v262_textless_base.png"), quality=98)
    textless.save(str(JOB / "v262_textless_base.png"), quality=98)

    return {
        "text_mask": "v262_text_mask.png",
        "subj_mask": "v262_subj_mask.png",
        "subj_bbox": "v262_subj_bbox_mask.png",
        "canny": "v262_canny_struct.png",
        "textless": "v262_textless_base.png",
    }


# ---------------------------------------------------------------------------
def build_inpaint(tag, seed, masks, ipa=0.15, denoise=0.95):
    """模式 A: IPAdapter(低)+Canny+VAEEncodeForInpaint, 主体剪影 mask"""
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": m.CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": masks["textless"]}}
    g["3"] = {"class_type": "LoadImage", "inputs": {"image": masks["subj_mask"]}}
    prev = "1"
    if ipa and ipa > 0:
        g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
        g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
            "model": ["1", 0], "ipadapter": ["5", 1], "image": ["2", 0],
            "weight": ipa, "weight_type": "style transfer", "combine_embeds": "average",
            "start_at": 0.0, "end_at": 0.85, "noise": 0.05, "embeds_scaling": "V only"}}
        prev = "6"
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": [prev, 0], "clip": ["1", 1], "lora_name": m.LORA,
        "strength_model": m.LORA_DETAIL, "strength_clip": m.LORA_DETAIL}}
    g["20"] = {"class_type": "LoadImage", "inputs": {"image": masks["canny"]}}
    g["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": m.CN_CANNY}}
    g["22"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["pg", 0], "control_net": ["21", 0], "image": ["20", 0], "strength": 0.75}}
    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": POS[tag]}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG}}
    g["4"] = {"class_type": "VAEEncodeForInpaint", "inputs": {
        "pixels": ["2", 0], "vae": ["1", 2], "mask": ["3", 1], "grow_mask_by": 10}}
    g["41"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["4", 0], "mask": ["3", 1]}}
    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["22", 0], "negative": ["ng", 0],
        "latent_image": ["41", 0], "seed": seed, "steps": 30, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": denoise}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": f"v262_{tag}"}}
    return g


def build_brushnet(tag, seed, masks, ipa=0.0, bn_scale=1.0):
    """模式 C: BrushNet(inpaint 专用) 替代 Canny; 主体 bbox 矩形 mask; IPAdapter 默认关"""
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": m.CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": masks["textless"]}}
    g["3"] = {"class_type": "LoadImage", "inputs": {"image": masks["subj_bbox"]}}
    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": POS[tag]}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": NEG}}
    g["8"] = {"class_type": "BrushNetLoader", "inputs": {
        "brushnet": "diffusion_pytorch_model.safetensors", "dtype": "float16"}}
    prev = "1"
    if ipa and ipa > 0:
        g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
        g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
            "model": ["1", 0], "ipadapter": ["5", 1], "image": ["2", 0],
            "weight": ipa, "weight_type": "style transfer", "combine_embeds": "average",
            "start_at": 0.0, "end_at": 0.85, "noise": 0.05, "embeds_scaling": "V only"}}
        prev = "6"
    g["9"] = {"class_type": "BrushNet", "inputs": {
        "model": [prev, 0], "vae": ["1", 2], "image": ["2", 0], "mask": ["3", 1],
        "brushnet": ["8", 0], "positive": ["pg", 0], "negative": ["ng", 0],
        "scale": bn_scale, "start_at": 0, "end_at": 10000}}
    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["9", 0], "positive": ["9", 1], "negative": ["9", 2],
        "latent_image": ["9", 3], "seed": seed, "steps": 30, "cfg": 7.0,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": f"v262_{tag}"}}
    return g


def submit(wf):
    data = json.dumps({"prompt": wf, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def poll(pid, timeout_s=900):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(4)
        try:
            with urllib.request.urlopen(f"{COMFY_URL}/history/{pid}", timeout=15) as r:
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
                outs.append(Path(COMFY_OUTPUT) / im["filename"])
    return outs


# ---------------------------------------------------------------------------
def _fit_font(text, target_len, lo=4, hi=400):
    while hi - lo > 1:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(FONT_PATH, mid)
        bb = ImageDraw.Draw(Image.new("RGB", (10, 10))).textbbox((0, 0), text, font=f)
        if (bb[2] - bb[0]) < target_len:
            lo = mid
        else:
            hi = mid
    return lo


def burn_text(img, big, arc, sub):
    import importlib
    spec_a = importlib.util.spec_from_file_location("arc_text", str(PROJECT / "src" / "arc_text.py"))
    at = importlib.util.module_from_spec(spec_a); spec_a.loader.exec_module(at)
    img = img.convert("RGB")
    cx, cy = RING["cx"], RING["cy"]
    lo, hi = 4, 400
    while hi - lo > 1:
        mid = (lo + hi) // 2
        L = at.fit_arc_text_width(arc, FONT_PATH, mid, 371, char_spacing_px=8)
        if L < 880:
            lo = mid
        else:
            hi = mid
    img = at.draw_arc_text(img, arc, FONT_PATH, lo, INK, (cx, cy), 371, 200, 340, char_spacing_px=8, flip_180=False)
    fs_big = m.calibrate(big, int(958 * 0.60))
    m.burn_centered(img, big, fs_big, cx, 1075)
    fs_sub = m.calibrate(sub, int(693 * 0.60))
    m.burn_centered(img, sub, fs_sub, cx, 1256)
    return img


def qc_palette(img):
    rgb = np.array(img.convert("RGB")).reshape(-1, 3).astype(int)
    bright = rgb[(rgb[:, 0] > 45) & (rgb[:, 1] > 45) & (rgb[:, 2] > 45)]
    if len(bright) == 0:
        return 0.0
    r, g, b = bright[:, 0], bright[:, 1], bright[:, 2]
    off = ((r - b) < -25) | ((g - b) > 25) | ((abs(r - g) < 25) & (abs(g - b) < 25) & (abs(r - b) < 25) & (r > 90))
    return round(float(off.mean() * 100), 3)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["inpaint", "brushnet"], default="brushnet")
    ap.add_argument("--subject", choices=["raven", "owl", "falcon", "all"], default="all")
    ap.add_argument("--ipa", type=float, default=None, help="IPAdapter weight (默认 brushnet=0, inpaint=0.15)")
    ap.add_argument("--denoise", type=float, default=None)
    ap.add_argument("--bn-scale", type=float, default=1.0)
    ap.add_argument("--reburn", action="store_true")
    args = ap.parse_args()

    layout = load_layout()
    variants = [v for v in SUBJECT_VARIANTS if args.subject in ("all", v[0])]

    if not args.reburn:
        print("[Stage1] build masks ...")
        masks = build_masks(layout)
    else:
        masks = {"subj_mask": "v262_subj_mask.png", "subj_bbox": "v262_subj_bbox_mask.png"}

    finals = []
    for i, (tag, _pos, big, arc, sub) in enumerate(variants):
        cand = sorted(COMFY_OUTPUT.glob(f"v262_{tag}*.png"))
        if args.reburn and cand:
            raw = cand[-1]
            print(f"\n=== reburn [{tag}] <- {raw.name} ===")
        else:
            seed = 262000 + i * 37
            if args.mode == "brushnet":
                ipa = args.ipa if args.ipa is not None else 0.0
                wf = build_brushnet(tag, seed, masks, ipa=ipa, bn_scale=args.bn_scale)
            else:
                ipa = args.ipa if args.ipa is not None else 0.15
                den = args.denoise if args.denoise is not None else 0.95
                wf = build_inpaint(tag, seed, masks, ipa=ipa, denoise=den)
            print(f"\n=== Stage2 [{tag}] mode={args.mode} ipa={ipa} seed={seed} ===")
            try:
                r = submit(wf)
            except Exception as e:
                print(f"  submit FAIL: {e}"); continue
            if "error" in r:
                print(f"  COMFY ERROR: {r['error']}"); continue
            pid = r["prompt_id"]
            try:
                entry = poll(pid)
            except Exception as e:
                print(f"  poll FAIL: {e}"); continue
            outs = collect_out(entry)
            if not outs:
                print("  no output"); continue
            raw = outs[0]
            print(f"  raw -> {raw}")

        # Stage3: 复合 + 烧字
        raw = Image.open(raw).convert("RGB")
        if raw.size != (1552, 2000):
            raw = raw.resize((1552, 2000), Image.LANCZOS)
        tl = Image.open(COMFY_INPUT / "v262_textless_base.png").convert("RGB")
        comp_mask_file = masks["subj_bbox"] if args.mode == "brushnet" else masks["subj_mask"]
        sm = Image.open(COMFY_INPUT / comp_mask_file).convert("RGBA")
        alpha = sm.split()[3].filter(ImageFilter.GaussianBlur(3))
        comp = tl.convert("RGBA").copy()
        comp.paste(raw.convert("RGBA"), (0, 0), alpha)
        img = burn_text(comp.convert("RGB"), big, arc, sub)
        off = qc_palette(img)
        out = JOB / f"v262_{tag}_final.png"
        img.save(str(out), quality=95)
        print(f"  final -> {out}  (非紫家族像素={off}%)  词: {big} / {arc} / {sub}")
        finals.append(out)

    if finals:
        imgs = [Image.open(p).convert("RGB") for p in finals]
        w, h = imgs[0].size
        gap = 14
        grid = Image.new("RGB", (w * len(imgs) + gap * (len(imgs) + 1), h), "white")
        for i, im in enumerate(imgs):
            grid.paste(im, (gap + i * (w + gap), 0))
        grid.save(str(JOB / "_grid_v262.png"), quality=92)
        print(f"\n[OK] grid -> {JOB / '_grid_v262.png'}")


if __name__ == "__main__":
    main()
