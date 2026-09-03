"""v261_pipeline — 元素原位裂变主流程 (修复 v260 三大失败)

架构:
  Stage0 layout.json (已由 v261_detect_layout.py 产出, 唯一坐标真值源)
  Stage1 精确擦字: 用 text-bands 的「笔画级暗像素」做 mask(排除主体), cv2.inpaint 填紫
          -> 无字底图 (构图/渐变 100% 保留原图像素, 无幽灵字)
  Stage2 主体原位重画: textless底图 送入 ComfyUI,
          - IPAdapter weight=0.35 (只锁色/风格, 不再焊死 bat形态)
          - ControlNet Canny 作用在「masked-canny」(主体区+文字区清零, 只锁圆环/背景)
          - VAEEncodeForInpaint + SetLatentNoiseMask 只在主体 bbox 加噪 -> 非遮罩区像素级不动
          - prompt = 新物种(raven/owl/falcon) 强制对称展翼+居中+不超出内圈+紫
  Stage3 文字 PIL 精确烧录: 按 layout.json 实测坐标, 顶弧/大字/副字 三类,
          新词与主体元素隐喻同构(raven->夜/翼/誓; owl->守夜/智; falcon->猎/风暴)

依赖: ComfyUI 在 http://127.0.0.1:8188 已加载 ProteusV0.4 + IPAdapter PLUS + canny-sdxl
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
JOB = PROJECT / "jobs" / "v261"
JOB.mkdir(parents=True, exist_ok=True)
SRC = "test_6978fabda2cc99629fa9e81f802762d3.jpg"

sys.path.insert(0, str(PROJECT / "src"))
import importlib.util
spec = importlib.util.spec_from_file_location("v253", str(PROJECT / "src" / "v253_bat_logo_inpaint.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

INK = m.INK
FONT_PATH = m.FONT_PATH
RING = {"cx": 776, "cy": 744, "outer_r": 427, "inner_r": 413}  # 来自 layout.json (==v253 共识)

# ---- 新物种 + 配套新词 (词与主体隐喻同构) ----
SUBJECT_VARIANTS = [
    ("raven", "a single stylized 2D SOLID FLAT BLACK raven, wings spread in symmetric emblem pose, "
              "head turned to side, gothic vintage craft-spirits emblem, SATURATED PURPLE #6B2C8C and "
              "deep violet #2A0A3F and black #1A0A1F, flat printed vector, NO internal detail, NO shading, "
              "NO gradient, clean outline, perfectly centered inside the circular ring, symmetric composition",
     "NOCTWING", "REALM OF NIGHT WINGS", "RAVEN'S OATH"),
    ("owl", "a single stylized 2D SOLID FLAT BLACK great horned owl, wings spread, symmetric emblem pose, "
            "stern forward gaze, gothic vintage craft-spirits emblem, SATURATED PURPLE #6B2C8C and deep "
            "violet #2A0A3F and black #1A0A1F, flat printed vector, NO internal detail, NO shading, NO gradient, "
            "clean outline, perfectly centered inside the circular ring, symmetric composition",
     "NIGHTOWL", "GUARDIAN OF DUSK", "WISE WATCH"),
    ("falcon", "a single stylized 2D SOLID FLAT BLACK peregrine falcon, wings swept in symmetric emblem pose, "
               "head turned, gothic vintage craft-spirits emblem, SATURATED PURPLE #6B2C8C and deep violet "
               "#2A0A3F and black #1A0A1F, flat printed vector, NO internal detail, NO shading, NO gradient, "
               "clean outline, perfectly centered inside the circular ring, symmetric composition",
     "SKYREAVER", "DOMAIN OF THE SKY", "STORM CLAW"),
]

NEG = ("3d, 3d render, metallic, glossy, jewelry, gradient shading, smooth shading, soft airbrush, "
       "photorealistic, realistic, hyperrealistic, watercolor, "
       "text, letters, words, readable text, brand name, BACARDI, logo, monogram, "
       "multiple birds, second bird, extra wings, extra limbs, deformed, mutated, malformed, "
       "blurry, soft focus, low quality, jagged, noise, grain, "
       "gray, grayish, brown, green, blue, cyan, beige, tan, desaturated, washed out, off-palette, "
       "background change, new colors, color bleed")


# ---------------------------------------------------------------------------
def load_layout():
    return json.loads((JOB / "layout.json").read_text(encoding="utf-8"))


def build_masks(layout):
    """生成三类 mask 并落盘到 ComfyUI/input:
       text_mask  (RGBA, alpha=字笔画)  -> Stage1 擦字
       subj_mask  (RGBA, alpha=主体)    -> Stage2 inpaint
       canny_mask (RGB 黑底白边, 主体+文字区清零) -> ControlNet 只锁结构
       返回 dict of 文件名
    """
    bgr = cv2.imdecode(np.fromfile(str(COMFY_INPUT / SRC), dtype=np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    subj = layout["subject"]
    sx, sy, sbw, sbh = subj["bbox"]

    # --- 极坐标 + 圆环带 ---
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((xx - RING["cx"]) ** 2 + (yy - RING["cy"]) ** 2)
    ring_band = (dist > (RING["inner_r"] - 10)) & (dist < (RING["outer_r"] + 10))

    # --- 擦除 mask: 描边 + 闭运算填实字母肚子, 排除主体 bbox 与圆环带 ---
    strokes = (gray < 140).astype(np.uint8) * 255
    text_mask = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    text_mask = cv2.dilate(text_mask, np.ones((4, 4), np.uint8), 1).astype(bool)
    text_mask[ring_band] = 0
    text_mask[sy - 16:sy + sbh + 16, sx - 16:sx + sbw + 16] = 0   # 绝不擦蝙蝠
    tmask_rgba = np.zeros((h, w, 4), np.uint8)
    tmask_rgba[..., 3] = text_mask.astype(np.uint8) * 255
    Image.fromarray(tmask_rgba, "RGBA").save(str(COMFY_INPUT / "v261_text_mask.png"))

    # --- subject mask (bbox 内暗色连通域, 略羽化) ---
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
    smask_rgba = np.zeros((h, w, 4), np.uint8)
    smask_rgba[..., 3] = sm
    Image.fromarray(smask_rgba, "RGBA").save(str(COMFY_INPUT / "v261_subj_mask.png"))

    # --- masked canny: 主体+文字区清零, 只留圆环/背景结构 ---
    canny = cv2.Canny(gray, int(255 * 0.10), int(255 * 0.25))
    canny[sy - 20:sy + sbh + 20, sx - 20:sx + sbw + 20] = 0  # 主体清零
    canny[text_mask] = 0                                    # 文字清零
    canny_rgb = np.stack([canny] * 3, -1)
    Image.fromarray(canny_rgb, "RGB").save(str(COMFY_INPUT / "v261_canny_struct.png"))

    # --- textless base: 径向渐变重建填字 (确定性, 不受圆环污染, 零残留) ---
    # 干净背景像素: 紫(灰>150) 且 非环 非文字mask 非原字 且 非主体
    clean = (gray > 150) & (~ring_band) & (~text_mask) & (~strokes)
    clean[sy - 16:sy + sbh + 16, sx - 16:sx + sbw + 16] = False
    cys, cxs = np.nonzero(clean)
    dd = dist[cys, cxs].astype(float)
    rgb_f = rgb[cys, cxs].astype(float)
    coef = [np.polyfit(dd, rgb_f[:, c], 2) for c in range(3)]   # color = a+b*d+c*d^2
    out = rgb.astype(float).copy()
    tys, txs = np.nonzero(text_mask)
    for c in range(3):
        out[tys, txs, c] = np.polyval(coef[c], dist[tys, txs])
    textless = Image.fromarray(out.astype(np.uint8), "RGB")
    textless.save(str(COMFY_INPUT / "v261_textless_base.png"), quality=98)
    textless.save(str(JOB / "v261_textless_base.png"), quality=98)

    return {
        "text_mask": "v261_text_mask.png",
        "subj_mask": "v261_subj_mask.png",
        "canny": "v261_canny_struct.png",
        "textless": "v261_textless_base.png",
    }


# ---------------------------------------------------------------------------
def build_workflow(tag, seed, masks):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": m.CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": masks["textless"]}}   # 无字底图
    g["3"] = {"class_type": "LoadImage", "inputs": {"image": masks["subj_mask"]}}  # 主体 mask(alpha)
    g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["2", 0],
        "weight": 0.35, "weight_type": "style transfer", "combine_embeds": "average",
        "start_at": 0.0, "end_at": 0.85, "noise": 0.05, "embeds_scaling": "V only"}}
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["6", 0], "clip": ["1", 1], "lora_name": m.LORA,
        "strength_model": m.LORA_DETAIL, "strength_clip": m.LORA_DETAIL}}
    # ControlNet 作用在 "只锁结构" 的 masked-canny
    g["20"] = {"class_type": "LoadImage", "inputs": {"image": masks["canny"]}}
    g["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": m.CN_CANNY}}
    g["22"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["pg", 0], "control_net": ["21", 0], "image": ["20", 0], "strength": 0.75}}
    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": POS[tag]}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG}}
    # 只在主体 mask 加噪 -> 非遮罩区像素级不动
    g["4"] = {"class_type": "VAEEncodeForInpaint", "inputs": {
        "pixels": ["2", 0], "vae": ["1", 2], "mask": ["3", 1], "grow_mask_by": 10}}
    g["41"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["4", 0], "mask": ["3", 1]}}
    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["22", 0], "negative": ["ng", 0],
        "latent_image": ["41", 0], "seed": seed, "steps": 30, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.92}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": f"v261_{tag}"}}
    return g


POS = {v[0]: v[1] for v in SUBJECT_VARIANTS}


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
    """二分搜索 font_size 使弧上/直排总宽 ≈ target_len"""
    while hi - lo > 1:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(FONT_PATH, mid)
        bb = ImageDraw.Draw(Image.new("RGB", (10, 10))).textbbox((0, 0), text, font=f)
        if (bb[2] - bb[0]) < target_len:
            lo = mid
        else:
            hi = mid
    return lo


def burn_text(img, layout, big, arc, sub):
    """按 layout 实测坐标精确烧字:
       顶弧: 圆心=徽章中心(776,744) radius=371 上半弧 200°→340° (v253 的 radius=698 是错位, 已弃)
       大字: 中心=(776,1075) 宽≈原 BACARDÍ 字宽
       副字: 中心=(776,1256) 宽≈原 WHEART 字宽
    """
    import importlib
    spec_a = importlib.util.spec_from_file_location("arc_text", str(PROJECT / "src" / "arc_text.py"))
    at = importlib.util.module_from_spec(spec_a); spec_a.loader.exec_module(at)

    img = img.convert("RGB")
    cx, cy = RING["cx"], RING["cy"]
    # 顶弧: 居中分布, 总弧长限制在 ~880px (200°→340° 实测弧)
    arc_len = at.fit_arc_text_width(arc, FONT_PATH, 60, 371, char_spacing_px=8)
    fs_arc = _fit_font(arc, int(arc_len), lo=4, hi=400) if arc_len > 8 else 60
    # 二次校准: 用 fit 反推合适字号使弧长≈880
    lo, hi = 4, 400
    while hi - lo > 1:
        mid = (lo + hi) // 2
        L = at.fit_arc_text_width(arc, FONT_PATH, mid, 371, char_spacing_px=8)
        if L < 880:
            lo = mid
        else:
            hi = mid
    fs_arc = lo
    img = at.draw_arc_text(img, arc, FONT_PATH, fs_arc, INK,
                           (cx, cy), 371, 200, 340, char_spacing_px=8, flip_180=False)
    # 大字 / 副字: 用原字宽比例校准
    fs_big = m.calibrate(big, int(958 * 0.60))
    m.burn_centered(img, big, fs_big, cx, 1075)
    fs_sub = m.calibrate(sub, int(693 * 0.60))
    m.burn_centered(img, sub, fs_sub, cx, 1256)
    return img


def qc_palette(img, layout):
    """检查输出是否出现非紫家族的新颜色 (灰/绿/蓝/棕 等), 排除暗墨色与背景紫"""
    rgb = np.array(img.convert("RGB")).reshape(-1, 3).astype(int)
    # 非暗像素(亮于40)中, 判断是否在紫/品红家族: 紫= R>=G>=B 近似 且 不过灰
    bright = rgb[(rgb[:, 0] > 45) & (rgb[:, 1] > 45) & (rgb[:, 2] > 45)]
    if len(bright) == 0:
        return 0.0
    r, g, b = bright[:, 0], bright[:, 1], bright[:, 2]
    off_family = ((r - b) < -25) | ((g - b) > 25) | ((abs(r - g) < 25) & (abs(g - b) < 25) & (abs(r - b) < 25) & (r > 90))
    return round(float(off_family.mean() * 100), 3)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reburn", action="store_true", help="只重烧已生成的 raw 图(跳过 ComfyUI)")
    args = ap.parse_args()

    layout = load_layout()
    masks = None
    if not args.reburn:
        print("[Stage1] build masks + erase text ...")
        masks = build_masks(layout)

    finals = []
    for i, (tag, _pos, big, arc, sub) in enumerate(SUBJECT_VARIANTS):
        if args.reburn:
            cand = sorted(COMFY_OUTPUT.glob(f"v261_{tag}*.png"))
            if not cand:
                print(f"[reburn] 缺 v261_{tag}*.png, 跳过"); continue
            raw = cand[-1]
            print(f"\n=== reburn [{tag}] <- {raw.name} ===")
        else:
            seed = 261000 + i * 37
            print(f"\n=== Stage2 [{tag}] seed={seed} ===")
            wf = build_workflow(tag, seed, masks)
            try:
                r = submit(wf)
            except Exception as e:
                print(f"  submit FAIL: {e}"); continue
            if "error" in r:
                print(f"  COMFY ERROR: {r['error']}"); continue
            pid = r["prompt_id"]
            print(f"  pid={pid[:8]} polling ...")
            try:
                entry = poll(pid)
            except Exception as e:
                print(f"  poll FAIL: {e}"); continue
            outs = collect_out(entry)
            if not outs:
                print("  no output"); continue
            raw = outs[0]
            print(f"  raw -> {raw}")

        # Stage3: 复合 (SDXL 只出主体, 背景/圆环/文字区 100% 用原图紫) + 烧字
        raw = Image.open(raw).convert("RGB")
        if raw.size != (1552, 2000):
            raw = raw.resize((1552, 2000), Image.LANCZOS)
        tl = Image.open(COMFY_INPUT / "v261_textless_base.png").convert("RGB")
        sm = Image.open(COMFY_INPUT / "v261_subj_mask.png").convert("RGBA")
        alpha = sm.split()[3].filter(ImageFilter.GaussianBlur(3))
        raw_rgba = raw.convert("RGBA")
        comp = tl.convert("RGBA").copy()
        comp.paste(raw_rgba, (0, 0), alpha)   # 仅主体区替换为 SDXL 输出
        img = comp.convert("RGB")
        img = burn_text(img, layout, big, arc, sub)
        off = qc_palette(img, layout)
        out = JOB / f"v261_{tag}_final.png"
        img.save(str(out), quality=95)
        print(f"  final -> {out}  (非紫家族像素={off}%)  词: {big} / {arc} / {sub}")
        finals.append(out)

    # grid
    if finals:
        imgs = [Image.open(p).convert("RGB") for p in finals]
        w, h = imgs[0].size
        gap = 14
        grid = Image.new("RGB", (w * len(imgs) + gap * (len(imgs) + 1), h), "white")
        for i, im in enumerate(imgs):
            grid.paste(im, (gap + i * (w + gap), 0))
        grid.save(str(JOB / "_grid_v261.png"), quality=92)
        print(f"\n[OK] grid -> {JOB / '_grid_v261.png'}")


if __name__ == "__main__":
    main()
