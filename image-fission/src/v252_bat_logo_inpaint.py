"""
v252 — bat_logo 混合升级 (PIL 精确擦背景 + SDXL inpaint 真·AI 重绘蝙蝠)

回应 2026-09-03 用户指令 "去 git 找模型技能升级 ai 裂变图效果":
  - 背景/圆环/弯月/文字: 复用 v251 已验证的 PIL 精确擦除 (配色锁死, 永远稳定)
  - 蝙蝠主体: SDXL 原生 inpaint (VAEEncodeForInpaint) 只重绘蝙蝠区, 不用整图 SDXL
       -> 真·AI 裂变 (换姿态/细节) 而非 v250/v251 的 cut-paste 仿射
       -> IPAdapter PLUS(high) 锁风格 + ControlNet Canny 锁结构, 避免 v243-v249 整图失控崩坏
  - 自动回退: 若 SDXL 蝙蝠区检出 3D/金属/新色/模糊等崩坏, 自动改用 v251 PIL 蝙蝠, 保证不出烂图

本地模型 (无需下载): ProteusV0.4 + IPAdapter PLUS + controlnet-canny-sdxl + add-detail-xl + 4x_NMKD
"""
import os, sys, json, time, shutil, uuid, urllib.request, math
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path("E:/Desktop/双接口/image-fission")
COMFY_INPUT = PROJECT / "ComfyUI" / "input"
COMFY_OUTPUT = PROJECT / "ComfyUI" / "output"
COMFY_URL = "http://127.0.0.1:8188"
JOB = PROJECT / "jobs" / "smoke_v252"
JOB.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT / "src"))
from arc_text import draw_arc_text, fit_arc_text_width

CKPT = "ProteusV0.4.safetensors"
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
LORA = "add-detail-xl.safetensors"
UPSCALER = "4x_NMKD-Siax_200k.pth"
REF_IMG = "test_6978fabda2cc99629fa9e81f802762d3.jpg"
FONT_PATH = str(PROJECT / "fonts" / "PirataOne-Regular.ttf")

RING = dict(cx=776, cy=745, outer_r=421, inner_r=407)
BAT_BBOX = (522, 518, 508, 456)
BAT_CENTER = (776, 746)
BIG_CENTER = (776, 1070)
SMALL_CENTER = (776, 1285)
INK = (26, 10, 31)

SEED = 252001
DENOISE = 0.72
IPA_WEIGHT = 0.32
CANNY_STRENGTH = 0.60
LORA_DETAIL = 0.60

# ---- PIL 擦除用的常量 ----
BAT_MASK_BBOX = (522, 518, 508, 430)

POS_BAT = (
    "a single stylized 2D flat black bat silhouette with wings spread in a DYNAMIC ROTATED pose, "
    "head tilted up, gothic vintage craft spirits emblem style, sharp clean edges, bold emblem, "
    "flat printed vector graphic, saturated purple #6B2C8C and deep violet #2A0A3F and black #1A0A1F, "
    "perfectly centered inside the circular ring, symmetric composition, NO shadow, NO ground plane"
)
NEG_BAT = (
    "3d, 3d rendered, metallic, glossy, jewelry, necklace, pendant, gem, crystal, chandelier, "
    "photorealistic, realistic photo, hyperrealistic, gradient shading, smooth shading, soft airbrush, "
    "watercolor, text, letters, words, readable text, brand name, BACARDI, logo, monogram, "
    "multiple bats, second bat, extra wings, extra limbs, deformed, mutated, malformed, "
    "blurry, soft focus, low quality, jagged, noise, grain, "
    "gray, grayish, brown, green, blue, cyan, beige, tan, desaturated, washed out, off-palette"
)


def sample_bg(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark = gray < 90
    h, w = bgr.shape[:2]
    ys, xs = np.ogrid[:h, :w]
    dd = ((xs - RING['cx']) ** 2 + (ys - RING['cy']) ** 2) ** 0.5
    ring = (dd >= RING['inner_r'] - 8) & (dd <= RING['outer_r'] + 8)
    bx, by, bw, bh = BAT_BBOX
    batbox = np.zeros_like(dark); batbox[by:by + bh, bx:bx + bw] = True
    mask = (~dark) & (~ring) & (~batbox)
    med = np.median(bgr[mask], axis=0).astype(int)
    return (int(med[2]), int(med[1]), int(med[0]))


def extract_bat(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark = gray < 90
    bx, by = 522, 518
    bw, bh = 508, 430
    mask = np.zeros_like(dark); mask[by:by + bh, bx:bx + bw] = True
    bat = (dark & mask).astype(np.uint8)
    bat = cv2.morphologyEx(bat, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return bat.astype(bool)


def find_text_bands(cand, min_h=18, min_px=800, gap=45):
    rows = np.where(cand.any(axis=1))[0]
    if len(rows) == 0:
        return []
    segs = []; s = rows[0]; prev = rows[0]
    for y in rows[1:]:
        if y - prev > gap:
            segs.append((s, prev)); s = y
        prev = y
    segs.append((s, prev))
    out = []
    for a, b in segs:
        if (b - a + 1) >= min_h and int(cand[a:b + 1].sum()) >= min_px:
            out.append((a, b))
    return out


def clean_base(orig_rgb, bgr):
    """擦原字 + 擦原蝙蝠 -> 干净底图 (bg/ring/moon 保留)"""
    h, w = bgr.shape[:2]
    bat_mask = extract_bat(bgr)
    ys, xs = np.ogrid[:h, :w]
    dd = ((xs - RING['cx']) ** 2 + (ys - RING['cy']) ** 2) ** 0.5
    ring = (dd >= RING['inner_r'] - 6) & (dd <= RING['outer_r'] + 6)
    BG = sample_bg(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark_w = gray < 140
    cand = dark_w & (~bat_mask) & (~ring)
    bands = find_text_bands(cand, min_h=18, min_px=800, gap=45)
    erase = np.zeros((h, w), bool)
    for a, b in bands:
        erase[a:b + 1, :] |= cand[a:b + 1, :]
    base_np = np.array(orig_rgb).copy()
    base_np[bat_mask] = BG
    if erase.any():
        base_np = cv2.inpaint(base_np, erase.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    return Image.fromarray(base_np), bat_mask


def fission_bat_pil(orig_rgb, bat_mask):
    """PIL 回退蝙蝠: 镜像 + 1.12x 展翼 + 旋 -32 (v251 同款)"""
    bx, by, bw, bh = BAT_BBOX
    crop = orig_rgb.crop((bx, by, bx + bw, by + bh))
    cmask = bat_mask[by:by + bh, bx:bx + bw]
    pa = np.array(crop.convert("RGBA")); pa[..., 3] = (cmask * 255).astype(np.uint8)
    m = Image.fromarray(pa).transpose(Image.FLIP_LEFT_RIGHT)
    m = m.resize((int(m.width * 1.12), m.height), Image.LANCZOS)
    ys, xs = np.where(cmask); cxc, cyc = int(xs.mean()), int(ys.mean())
    return m.rotate(-32, expand=True, center=(cxc, cyc))


def burn_top_arc(img, text, font_size, color=INK):
    w, h = img.width, img.height
    radius = int(w * 0.45)
    arc_len = fit_arc_text_width(text, FONT_PATH, font_size, radius, char_spacing_px=8)
    total_deg = math.degrees(arc_len / radius)
    start = 270 - total_deg / 2; end = 270 + total_deg / 2
    return draw_arc_text(img, text, FONT_PATH, font_size, color,
                         (w // 2, int(h * 0.13) + radius), radius, start, end,
                         char_spacing_px=8, flip_180=False)


def calibrate(text, target_w, lo=4, hi=400):
    while hi - lo > 1:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(FONT_PATH, mid)
        bb = ImageDraw.Draw(Image.new("RGB", (10, 10))).textbbox((0, 0), text, font=f)
        if (bb[2] - bb[0]) < target_w: lo = mid
        else: hi = mid
    return lo


def burn_centered(img, text, font_size, cx, cy, color=INK):
    ImageDraw.Draw(img).text((cx, cy), text,
                             font=ImageFont.truetype(FONT_PATH, font_size), fill=color, anchor="mm")


# ===== SDXL inpaint 工作流 =====
def build_inpaint(seed, base_name, mask_name):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": base_name}}
    g["3"] = {"class_type": "LoadImage", "inputs": {"image": mask_name}}
    g["4"] = {"class_type": "VAEEncodeForInpaint", "inputs": {
        "pixels": ["2", 0], "vae": ["1", 2], "mask": ["3", 1], "grow_mask_by": 12}}
    g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["2", 0],
        "weight": IPA_WEIGHT, "weight_type": "style transfer",
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.80,
        "noise": 0.05, "embeds_scaling": "V only"}}
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["6", 0], "clip": ["1", 1], "lora_name": LORA,
        "strength_model": LORA_DETAIL, "strength_clip": LORA_DETAIL}}
    g["20"] = {"class_type": "CannyEdgePreprocessor", "inputs": {
        "image": ["2", 0], "low_threshold": 0.10, "high_threshold": 0.25, "resolution": 1024}}
    g["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_CANNY}}
    g["22"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["pg", 0], "control_net": ["21", 0], "image": ["20", 0], "strength": CANNY_STRENGTH}}
    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": POS_BAT}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG_BAT}}
    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["22", 0], "negative": ["ng", 0],
        "latent_image": ["4", 0], "seed": seed, "steps": 28, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": DENOISE}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": "v252_bat_inpaint"}}
    return g


def submit(prompt):
    data = json.dumps({"prompt": prompt, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def history(pid):
    with urllib.request.urlopen(f"{COMFY_URL}/history/{pid}", timeout=15) as r:
        return json.loads(r.read())


def gen_inpaint(seed, base_name, mask_name):
    g = build_inpaint(seed, base_name, mask_name)
    try:
        r = submit(g)
    except Exception as e:
        print(f"[v252] submit FAIL: {e}")
        return None
    if "error" in r:
        print(f"[v252] COMFY ERROR: {r['error']}")
        return None
    pid = r["prompt_id"]
    print(f"[v252] submitted pid={pid[:8]} denoise={DENOISE} ipa={IPA_WEIGHT} canny={CANNY_STRENGTH}")
    for _ in range(120):
        time.sleep(5)
        try:
            h = history(pid)
        except Exception:
            continue
        if pid in h:
            break
    else:
        print("[v252] TIMEOUT")
        return None
    for node_id, info in h[pid].get("outputs", {}).items():
        for img in info.get("images", []):
            src = COMFY_OUTPUT / img["filename"]
            if not src.exists():
                src = Path(img.get("abs_path", src))
            if src.exists():
                return str(src)
    return None


def bat_region_broke(sdxl_rgb, bat_mask_full):
    """自动崩坏检测: SDXL 蝙蝠区若出现过量灰/金属/新色或过度模糊 -> True(回退)"""
    bx, by, bw, bh = BAT_BBOX
    crop = np.array(sdxl_rgb.crop((bx, by, bx + bw, by + bh)).convert("RGB"))
    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    cm = bat_mask_full[by:by + bh, bx:bx + bw]
    if cm.sum() == 0:
        return True
    region = bgr[cm]
    if len(region) == 0:
        return True
    # 灰度占比 (金属/灰 = 去饱和) -> 原图蝙蝠是纯黑, 灰像素应极少
    hsv = cv2.cvtColor(region.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    grayish = (hsv[:, 1] < 25) & (np.abs(hsv[:, 2].astype(int) - 90) < 70)  # 低饱和中灰
    gray_ratio = grayish.mean()
    # 清晰度 (拉普拉斯) -> 模糊即崩
    lap = cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    print(f"[v252] bat-region gray_ratio={gray_ratio:.3f} lap_var={lap:.1f}")
    return gray_ratio > 0.45 or lap < 30


def color_transfer(src_bgr, dst_bgr, alpha=1.0):
    """Reinhard LAB 配色迁移: 把 dst 的 L 通道均值/标准差拉向 src (锁原图紫调)"""
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


def hist_chi2(src_bgr, dst_bgr, bins=32):
    inter = 0.0
    for c in range(3):
        hs = cv2.calcHist([src_bgr], [c], None, [bins], [0, 256])
        hd = cv2.calcHist([dst_bgr], [c], None, [bins], [0, 256])
        cv2.normalize(hs, hs); cv2.normalize(hd, hd)
        inter += cv2.compareHist(hs, hd, cv2.HISTCMP_INTERSECT)
    return 1.0 - inter / 3.0


def side_by_side(orig_path, gen_path, out_path):
    o = Image.open(orig_path).convert("RGB"); gimg = Image.open(gen_path).convert("RGB")
    if o.size != gimg.size: gimg = gimg.resize(o.size)
    out = Image.new("RGB", (o.width * 2 + 30, o.height), "white")
    out.paste(o, (0, 0)); out.paste(gimg, (o.width + 30, 0))
    out.save(out_path, quality=95)


def main():
    import math
    orig = Image.open(COMFY_INPUT / REF_IMG).convert("RGB")
    bgr = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    base, bat_mask = clean_base(orig, bgr)
    base.save(str(COMFY_INPUT / "v252_base.png"), quality=95)

    # inpaint 掩码: 蝙蝠区 alpha=255(白), 其余透明 -> ComfyUI MASK 读 alpha, 白=重绘区
    mask = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bx, by, bw, bh = BAT_BBOX
    mask.paste(Image.new("RGBA", (bw, bh), (255, 255, 255, 255)), (bx, by))
    mask.save(str(COMFY_INPUT / "v252_mask.png"))

    raw = gen_inpaint(SEED, "v252_base.png", "v252_mask.png")
    used_sdxl = False
    composite = base.copy()
    if raw:
        sdxl = Image.open(raw).convert("RGB")
        if sdxl.size != (w, h):
            sdxl = sdxl.resize((w, h))
        sdxl_bgr = cv2.cvtColor(np.array(sdxl), cv2.COLOR_RGB2BGR)
        # 配色锁: Reinhard LAB 拉回原图紫调 (修 SDXL 漂色)
        locked = color_transfer(bgr, sdxl_bgr, alpha=0.9)
        locked_rgb = Image.fromarray(cv2.cvtColor(locked, cv2.COLOR_BGR2RGB))
        chi = hist_chi2(bgr, locked)
        broke = bat_region_broke(locked_rgb, bat_mask)
        print(f"[v252] color-lock chi2={chi:.3f} bat_broke={broke}")
        if chi < 0.30 and not broke:
            composite = locked_rgb
            used_sdxl = True
            print("[v252] SDXL bat + 配色锁 PASSED -> 使用 AI 重绘蝙蝠")
        else:
            print("[v252] SDXL 配色漂移/崩坏 -> 自动回退 PIL 蝙蝠")
            rot = fission_bat_pil(orig, bat_mask)
            rw, rh = rot.size
            composite.paste(rot, (BAT_CENTER[0] - rw // 2, BAT_CENTER[1] - rh // 2), rot)
    else:
        print("[v252] SDXL 生成失败 -> 回退 PIL 蝙蝠")
        rot = fission_bat_pil(orig, bat_mask)
        rw, rh = rot.size
        composite.paste(rot, (BAT_CENTER[0] - rw // 2, BAT_CENTER[1] - rh // 2), rot)

    # PIL 矢量烧字
    composite = burn_top_arc(composite, "LVMEN NOCTIS", int(h * 0.045))
    fs_main = calibrate("NOCTWING", 955)
    burn_centered(composite, "NOCTWING", fs_main, BIG_CENTER[0], BIG_CENTER[1])
    fs_small = calibrate("MORS VINI", 690)
    burn_centered(composite, "MORS VINI", fs_small, SMALL_CENTER[0], SMALL_CENTER[1])
    f_est = ImageFont.truetype(FONT_PATH, int(h * 0.022))
    ImageDraw.Draw(composite).text((int(w * 0.30), int(h * 0.745)), "Est.", font=f_est, fill=INK, anchor="mm")
    ImageDraw.Draw(composite).text((int(w * 0.70), int(h * 0.745)), "1862", font=f_est, fill=INK, anchor="mm")

    if composite.mode != "RGB":
        composite = composite.convert("RGB")
    out = JOB / "v252_bat_logo.jpg"
    composite.save(str(out), quality=95)
    side_by_side(COMFY_INPUT / REF_IMG, out, JOB / "_compare_v252.jpg")
    print(f"[v252] saved {out} ({out.stat().st_size//1024} KB)  used_sdxl={used_sdxl}")


if __name__ == "__main__":
    main()
