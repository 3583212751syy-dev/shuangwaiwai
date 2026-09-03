"""v264_pipeline — 蝙蝠差异化裂变 v2 (修 v263 两个缺陷: 旧字 ghost + 主体裂变不开)

基于 v253 已验证函数重写:
  - 用 v253.clean_base 生成「紫圆环+紫底、蝙蝠与字都清掉」的风格参考图
    -> 该图同时作为 IPAdapter 风格参考(高权重锁配色/徽章质感/排版气质)
       与 inpaint 初始底图(蝙蝠区已无形状, latent 不被原蝙蝠锚定)
  - cv2.inpaint 真填补文字 -> 旧字 100% 干净, 无 ghost
  - IPAdapter 提到 0.5: 因参考图里没有蝙蝠, 高权重只锁风格不锁主体形状
  - denoise 0.80: 蝙蝠区 80% 重噪, 从 prompt 重画 -> 主体明显裂变
  - Canny 0.6 锁圆环/背景结构(参考图本身无蝙蝠, canny 自然无蝙蝠边)
  - 复用 v253 的 match_hist_lab + snap_to_original 锁色, 烧字复用 v253 精确排版

主体三变体(只变姿态/轮廓/细节, 不换物种):
  up     翼尖上扬成 V
  spread 翼展加宽 + 扇贝膜边
  fold   收翼下垂贴近身体
"""
import json
import sys
import time
import uuid
import urllib.request
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path("E:/Desktop/双接口/image-fission")
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
COMFY_OUTPUT = PROJECT / "ComfyUI" / "output"
COMFY_URL = "http://127.0.0.1:8188"
JOB = PROJECT / "jobs" / "v264"
JOB.mkdir(parents=True, exist_ok=True)
REF = "test_6978fabda2cc99629fa9e81f802762d3.jpg"

sys.path.insert(0, str(PROJECT / "src"))
import importlib.util
spec = importlib.util.spec_from_file_location("v253", str(PROJECT / "src" / "v253_bat_logo_inpaint.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

from arc_text import draw_arc_text, fit_arc_text_width  # noqa

INK = m.INK
FONT_PATH = m.FONT_PATH
RING = m.RING
BAT_BBOX = m.BAT_BBOX

# ---- 三变体: 共享风格, 只变蝙蝠姿态 ----
BAT_STYLE = ("a single stylized 2D SOLID FLAT BLACK bat silhouette, gothic vintage craft-spirits emblem, "
             "SOLID FILLED FLAT SHAPE, NO internal detail, NO shading, NO gradient, NO texture, "
             "clean outline only, flat printed vector graphic, "
             "saturated purple #6B2C8C and deep violet #2A0A3F and black #1A0A1F, "
             "perfectly centered inside the circular ring, NO shadow, NO ground plane, ")

SUBJECT_VARIANTS = [
    ("up", BAT_STYLE + "wings RAISED UPWARD into a sharp V shape with pointed angular wing tips, "
                       "head tilted slightly to the left, long pointed ears, compact fierce silhouette",
     "NIGHTBAT", "SHADOW OF THE WING", "ECHO HUNT"),
    ("spread", BAT_STYLE + "wings spread WIDE and horizontal with SCALLOPED membrane edges, "
                           "broad sturdy body, short ears pointed forward, powerful silhouette",
     "DUSKBAT", "WINGS OF TWILIGHT", "SILENT FLIGHT"),
    ("fold", BAT_STYLE + "wings FOLDED DOWNWARD close to the body like a resting bat, "
                         "slender body, tail curled under, calm elegant silhouette",
     "MOONBAT", "GUARDIAN OF THE DARK", "SONAR OATH"),
]
POS = {v[0]: v[1] for v in SUBJECT_VARIANTS}

NEG = ("3d, 3d render, metallic, glossy, jewelry, gradient shading, smooth shading, soft airbrush, "
       "photorealistic, realistic, hyperrealistic, watercolor, "
       "text, letters, words, readable text, brand name, BACARDI, logo, monogram, "
       "multiple bats, second bat, extra wings, extra limbs, deformed, mutated, malformed, "
       "blurry, soft focus, low quality, jagged, noise, grain, "
       "gray, grayish, brown, green, blue, cyan, beige, tan, desaturated, washed out, off-palette, "
       "background change, new colors, color bleed")


def make_style_ref():
    """v253.clean_base: 紫圆环+紫底, 蝙蝠与字都清掉 -> 风格参考图 & inpaint 底图."""
    orig = Image.open(COMFY_INPUT / REF).convert("RGB")
    bgr = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
    base, bat_mask = m.clean_base(orig, bgr)
    base.save(str(COMFY_INPUT / "v264_style_ref.png"), quality=98)
    base.save(str(JOB / "v264_style_ref.png"), quality=98)
    print(f"[v264] style_ref -> {COMFY_INPUT / 'v264_style_ref.png'}")
    return base, bat_mask


def make_masks(bat_mask):
    h, w = bat_mask.shape[:2]
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 3] = (bat_mask.astype(np.uint8) * 255)
    Image.fromarray(rgba, "RGBA").save(str(COMFY_INPUT / "v264_bat_mask.png"))
    return "v264_bat_mask.png"


def build_workflow(tag, seed, ipa=0.5, denoise=0.80, canny=0.6):
    ref = "v264_style_ref.png"
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": m.CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
    g["3"] = {"class_type": "LoadImage", "inputs": {"image": "v264_bat_mask.png"}}
    g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["2", 0],
        "weight": ipa, "weight_type": "style transfer", "combine_embeds": "average",
        "start_at": 0.0, "end_at": 0.82, "noise": 0.05, "embeds_scaling": "V only"}}
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["6", 0], "clip": ["1", 1], "lora_name": m.LORA,
        "strength_model": m.LORA_DETAIL, "strength_clip": m.LORA_DETAIL}}
    g["20"] = {"class_type": "CannyEdgePreprocessor", "inputs": {
        "image": ["2", 0], "low_threshold": 0.10, "high_threshold": 0.25, "resolution": 1024}}
    g["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": m.CN_CANNY}}
    g["22"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["pg", 0], "control_net": ["21", 0], "image": ["20", 0], "strength": canny}}
    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": POS[tag]}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG}}
    g["4"] = {"class_type": "VAEEncodeForInpaint", "inputs": {
        "pixels": ["2", 0], "vae": ["1", 2], "mask": ["3", 1], "grow_mask_by": 12}}
    g["41"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["4", 0], "mask": ["3", 1]}}
    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["22", 0], "negative": ["ng", 0],
        "latent_image": ["41", 0], "seed": seed, "steps": 30, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": denoise}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": f"v264_{tag}"}}
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


def qc_palette(img):
    rgb = np.array(img.convert("RGB")).reshape(-1, 3).astype(int)
    bright = rgb[(rgb[:, 0] > 45) & (rgb[:, 1] > 45) & (rgb[:, 2] > 45)]
    if len(bright) == 0:
        return 0.0
    r, g, b = bright[:, 0], bright[:, 1], bright[:, 2]
    off = ((r - b) < -25) | ((g - b) > 25) | ((abs(r - g) < 25) & (abs(g - b) < 25) & (abs(r - b) < 25) & (r > 90))
    return round(float(off.mean() * 100), 3)


def burn_text(img, big, arc, sub):
    img = img.convert("RGB")
    w, h = img.width, img.height
    # 顶弧(复用 v253 精确排版)
    img = m.burn_top_arc(img, arc, int(h * 0.045))
    # 中心大词
    fs_big = m.calibrate(big, 955)
    m.burn_centered(img, big, fs_big, m.BIG_CENTER[0], m.BIG_CENTER[1])
    # 底部副词
    fs_sub = m.calibrate(sub, 690)
    m.burn_centered(img, sub, fs_sub, m.SMALL_CENTER[0], m.SMALL_CENTER[1])
    # 原图招牌式侧栏文字(参考原图排版)
    f_est = ImageFont.truetype(FONT_PATH, int(h * 0.022))
    ImageDraw.Draw(img).text((int(w * 0.30), int(h * 0.745)), "EST.", font=f_est, fill=INK, anchor="mm")
    ImageDraw.Draw(img).text((int(w * 0.70), int(h * 0.745)), "1862", font=f_est, fill=INK, anchor="mm")
    return img


def composite_bat(style_ref, raw, bat_mask):
    """颜色锁 + 贴回风格参考图."""
    h, w = bat_mask.shape[:2]
    raw_bgr = cv2.cvtColor(np.array(raw.convert("RGB").resize((w, h), Image.LANCZOS)), cv2.COLOR_RGB2BGR)
    orig_bgr = cv2.cvtColor(np.array(Image.open(COMFY_INPUT / REF).convert("RGB")), cv2.COLOR_RGB2BGR)
    locked = m.match_hist_lab(orig_bgr, raw_bgr)
    locked = m.snap_to_original(locked, orig_bgr, BAT_BBOX, n_colors=12)
    locked_rgb = Image.fromarray(cv2.cvtColor(locked, cv2.COLOR_BGR2RGB))
    alpha = Image.fromarray((bat_mask.astype(np.uint8) * 255), "L")
    comp = style_ref.convert("RGBA").copy()
    comp.paste(locked_rgb.convert("RGBA"), (0, 0), alpha)
    return comp.convert("RGB")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipa", type=float, default=0.5)
    ap.add_argument("--denoise", type=float, default=0.80)
    ap.add_argument("--canny", type=float, default=0.6)
    ap.add_argument("--only", default="all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print("[v264] make style ref + masks ...")
    style_ref, bat_mask = make_style_ref()
    make_masks(bat_mask)

    finals = []
    for i, (tag, _pos, big, arc, sub) in enumerate(SUBJECT_VARIANTS):
        if args.only != "all" and tag != args.only:
            continue
        seed = 264000 + i * 41
        # resume: 已有 raw 且非 force 则复用
        existing = sorted(COMFY_OUTPUT.glob(f"v264_{tag}*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if existing and not args.force:
            raw = existing[0]
            print(f"\n=== [v264 {tag}] reuse raw {raw.name} ===")
        else:
            wf = build_workflow(tag, seed, ipa=args.ipa, denoise=args.denoise, canny=args.canny)
            print(f"\n=== [v264 {tag}] submit seed={seed} ipa={args.ipa} denoise={args.denoise} ===")
            try:
                r = submit(wf)
            except Exception as e:
                print(f"  submit FAIL: {e}"); continue
            if "error" in r:
                print(f"  COMFY ERROR: {r['error']}"); continue
            try:
                entry = poll(r["prompt_id"])
            except Exception as e:
                print(f"  poll FAIL: {e}"); continue
            outs = collect_out(entry)
            if not outs:
                print("  no output"); continue
            raw = outs[0]
            print(f"  raw -> {raw}")

        comp = composite_bat(style_ref, Image.open(raw), bat_mask)
        img = burn_text(comp, big, arc, sub)
        off = qc_palette(img)
        out = JOB / f"v264_{tag}_final.png"
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
        grid.save(str(JOB / "_grid_v264.png"), quality=92)
        print(f"\n[OK] grid -> {JOB / '_grid_v264.png'}")


if __name__ == "__main__":
    main()
