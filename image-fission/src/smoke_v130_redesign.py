"""v130 强裂变 (用户原话：看不出区别，需要大幅度裂变).

核心修复 v129 失败原因:
  - denoise: 0.55 (v129) → 0.72 (v130, balanced 档) / 0.85 (strong) / 0.55 (mild)
  - 按 docs: demo_pattern_fission.py 第 7 行注释 "denoise 高 (0.75-0.85) 让内容完全重生成"

用户硬约束 (本轮):
  1. 假设 6 张图全侵权 → 所有文字按风格替换为合法 + 内容相符词
  2. 不变色彩 (元素族系符合原图 + 同质感 + 但具体内容不一样)
  3. 不变结构 (构图 / 视角 / 主体位置)
  4. 加大裂变范围 + 加摇杆克制
  5. 跑出来对比给他看

摇杆 (FISSION_STRENGTH):
  mild     = 0.55 (细节级微调, 跟 v129 一样)
  balanced = 0.72 (默认, 内容级裂变)  ← 本轮默认
  strong   = 0.85 (剧烈级裂变)

用法:
  python smoke_v130_redesign.py                       # 默认 balanced 跑 6 张
  python smoke_v130_redesign.py --strength strong    # 试 strong 档
  python smoke_v130_redesign.py --strength mild --only denim_3
"""
import json, time, sys, argparse
from pathlib import Path
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = Path("E:/Desktop/图裂变测试图")
COMFYUI = "http://127.0.0.1:8188"

CKPT = "ProteusV0.4.safetensors"
CONTROL_CKPT = "controlnet-canny-sdxl-1.0.fp16.safetensors"
UPSCALE = "4x_NMKD-Siax_200k.pth"
SEED = 700301

# 摇杆 3 档
STRENGTH_PRESETS = {
    "mild":     {"denoise": 0.55, "cn_weight": 0.60, "ipa_weight": 0.60, "label": "mild (细节微调)"},
    "balanced": {"denoise": 0.72, "cn_weight": 0.50, "ipa_weight": 0.50, "label": "balanced (内容裂变)"},
    "strong":   {"denoise": 0.85, "cn_weight": 0.40, "ipa_weight": 0.45, "label": "strong (剧烈裂变)"},
}

# 6 张图 + 各自 REDESIGNED 内容 (主体换但族系一致) + 强制文字替换策略
# 假设 6 张全部侵权 → 文字统一改为合法短词, 风格保持一致
REFS = [
    {"id": "denim_3", "filename": "pinterest_denim_3.jpg", "cn_weight": 0.50, "ipa_weight": 0.50,
     "subject": "delicate moth with intricate wings resting on a distressed X-shaped denim patch, fabric wear stitch holes threadbare, REDESIGNED arrangement, completely different pose and details",
     "palette": "vintage faded denim texture, indigo blue and white only, rough brushstroke X mark, decorative stitched edges",
     "text_replacement": None},  # 智能: 让 AI 自然出 VINTAGE / CLASSIC 等合法词

    {"id": "camo_4", "filename": "pinterest_camo_4.jpg", "cn_weight": 0.55, "ipa_weight": 0.55,
     "subject": "pine tree silhouettes and snowy mountain peaks in dense military camouflage pattern, dark conifer forest scattered fog, REDESIGNED foliage positioning, completely new tree arrangements",
     "palette": "dark green brown black white camouflage, matte, shadowy, dense conifer pattern",
     "text_replacement": None},

    {"id": "eagle_2", "filename": "pinterest_eagle_2.jpg", "cn_weight": 0.45, "ipa_weight": 0.50,
     "subject": "phoenix with sweeping tail feathers rising from stylized flames, symmetrical heraldic composition, REDESIGNED feather arrangement, completely different majestic bird form",
     "palette": "earthy red gold rust palette, weathered vintage tattoo illustration, gold and rust linework",
     "text_replacement": None},

    {"id": "illust_1", "filename": "pinterest_illust_1.jpg", "cn_weight": 0.40, "ipa_weight": 0.50,
     "subject": "black and white ornamental pattern with stylized roses and thorny vines, monochrome damask design, REDESIGNED botanical layout, completely different flower species",
     "palette": "high contrast black and white only, vintage engraving style, intricate linework, ornamental symmetrical border",
     "text_replacement": None},

    {"id": "metal_6", "filename": "pinterest_metal_6.jpg", "cn_weight": 0.55, "ipa_weight": 0.50,
     "subject": "ornate weathered bronze medallion with carved stone relief, classical barbarian metalwork, REDESIGNED carved motif, completely different ornamental pattern",
     "palette": "oxidized bronze patina, aged tarnished metal texture, sepia brown gold antiqued finish",
     "text_replacement": None},

    {"id": "skull_5", "filename": "pinterest_skull_5.jpg", "cn_weight": 0.45, "ipa_weight": 0.50,
     "subject": "dark gothic composition of rose blooms and thorny vines entwined with exposed bones, ornamental skull frame, REDESIGNED arrangement, completely different dark botanical motif",
     "palette": "deep black crimson ivory parchment, ornate damask dark romanticism, gothic engraving aesthetic",
     "text_replacement": None},
]

POS_TEMPLATE = ("masterpiece, best quality, ultra detailed, sharp clean edges, intricate linework, "
                 "{subject}, "
                 "completely redrawn internal details, redesigned color placement, new visual elements throughout, "
                 "framed composition, {palette}, {hard_avoid}")

HARD_AVOID = ("no logos, no celebrities, no national flags, no political figures, "
              "no copyrighted characters, no trademarked text, no readable slogans")

# 禁文字糊在图里 (但允许合法装饰短词自然出现)
NEG = ("blurry, smudged, deformed, disfigured, low quality, bad anatomy, mutation, "
       "cropped, jpeg artifacts, noise, watermark, signature, "
       "illegible text, garbled letters, random text, gibberish words, distorted text")


def get_original_megapixels(filename):
    from PIL import Image
    im = Image.open(INPUT_DIR / filename)
    w, h = im.size
    mp = (w * h) / 1_000_000.0
    # ComfyUI ImageScaleToTotalPixels 上限 16.0, 超出会 400 报错
    return min(mp, 16.0)


def build_graph(ref, mp, strength):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": ref["filename"]}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels",
              "inputs": {"upscale_method": "lanczos", "megapixels": round(mp, 4),
                         "image": ["2", 0], "resolution_steps": 8}}
    g["4"] = {"class_type": "Canny",
              "inputs": {"image": ["3", 0], "low_threshold": 0.08, "high_threshold": 0.18}}
    g["5"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CONTROL_CKPT}}
    pos = POS_TEMPLATE.format(subject=ref["subject"], palette=ref["palette"], hard_avoid=HARD_AVOID)
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": NEG}}
    g["6"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
        "positive": ["8", 0], "negative": ["9", 0], "image": ["4", 0],
        "strength": ref["cn_weight"], "start_percent": 0.0, "end_percent": 0.85,
        "control_net": ["5", 0]}}
    g["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["10"] = {"class_type": "IPAdapterUnifiedLoader",
               "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["11"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["10", 0], "ipadapter": ["10", 1],
        "image": ["2", 0],
        "weight": ref["ipa_weight"], "weight_type": "style transfer",
        "combine_embeds": "average", "embeds_scaling": "V only",
        "start_at": 0.0, "end_at": 0.85, "noise": 0.08}}
    g["12"] = {"class_type": "KSampler", "inputs": {
        "model": ["11", 0], "positive": ["6", 0], "negative": ["6", 1],
        "latent_image": ["7", 0],
        "seed": SEED, "steps": 40, "cfg": 7.0, "sampler_name": "dpmpp_2m",
        "scheduler": "karras", "denoise": strength["denoise"], "noise_mask": None}}
    g["13"] = {"class_type": "KSampler", "inputs": {
        "model": ["11", 0], "positive": ["6", 0], "negative": ["6", 1],
        "latent_image": ["12", 0],
        "seed": SEED + 1, "steps": 28, "cfg": 6.5, "sampler_name": "dpmpp_2m",
        "scheduler": "karras", "denoise": 0.18, "noise_mask": None}}
    g["14"] = {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["1", 2]}}
    g["15"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE}}
    g["16"] = {"class_type": "ImageUpscaleWithModel",
               "inputs": {"upscale_model": ["15", 0], "image": ["14", 0]}}
    g["17"] = {"class_type": "SaveImage",
               "inputs": {"images": ["16", 0], "filename_prefix": f"v130_{ref['id']}"}}
    return g


def run_one(ref, out_dir, strength):
    tag = ref["id"]
    out_path = out_dir / f"v130_{tag}.png"
    if out_path.exists() and out_path.stat().st_size > 100_000:
        print(f"[skip] {tag}: 已存在 {out_path.stat().st_size//1024//1024}MB", flush=True)
        return True
    mp = get_original_megapixels(ref["filename"])
    print(f"[run] {tag}  cn={ref['cn_weight']}  ipa={ref['ipa_weight']}  "
          f"denoise={strength['denoise']}  mp={mp:.4f}", flush=True)
    g = build_graph(ref, mp, strength)
    try:
        r = requests.post(f"{COMFYUI}/prompt",
                          json={"prompt": g, "client_id": f"v130_{int(time.time())}"}, timeout=15)
    except Exception as e:
        print(f"  ERR submit conn: {e}", flush=True); return False
    if r.status_code != 200:
        print(f"  ERR submit: {r.status_code} {r.text[:300]}", flush=True); return False
    pid = r.json().get("prompt_id")
    if not pid:
        print(f"  ERR no prompt_id", flush=True); return False
    print(f"  pid={pid}", flush=True)
    t0 = time.time()
    for _ in range(60):  # 300s 硬超时 (6 张轮跑需要更久)
        time.sleep(5)
        try:
            h = requests.get(f"{COMFYUI}/history/{pid}", timeout=10).json()
            if pid in h:
                status = h[pid].get("status", {})
                if status.get("completed"):
                    imgs = h[pid].get("outputs", {}).get("17", {}).get("images", [])
                    if not imgs:
                        print(f"  ERR no images", flush=True); return False
                    url = f"{COMFYUI}/view?filename={imgs[0]['filename']}&type=output&subfolder={imgs[0].get('subfolder','')}"
                    data = requests.get(url, timeout=120).content
                    out_path.write_bytes(data)
                    print(f"  OK {out_path.stat().st_size//1024//1024}MB (waited {int(time.time()-t0)}s)", flush=True)
                    return True
                if status.get("error"):
                    print(f"  ERR exec: {status.get('error')}", flush=True); return False
        except Exception as e:
            print(f"  poll err: {e}", flush=True)
    print(f"  TIMEOUT after {int(time.time()-t0)}s", flush=True); return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strength", default="balanced",
                        choices=list(STRENGTH_PRESETS.keys()),
                        help="裂变强度摇杆: mild / balanced / strong")
    parser.add_argument("--only", default=None, help="只跑指定 id (如 denim_3)")
    args = parser.parse_args()
    strength = STRENGTH_PRESETS[args.strength]

    out = PROJECT_ROOT / "jobs" / f"smoke_v130_{args.strength}_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== v130 REDESIGN ({strength['label']}) -> {out} ===", flush=True)
    print(f"=== 摇杆参数: denoise={strength['denoise']} cn={strength['cn_weight']} ipa={strength['ipa_weight']} ===", flush=True)
    ok = 0
    targets = [r for r in REFS if (not args.only or r["id"] == args.only)]
    for ref in targets:
        if run_one(ref, out, strength):
            ok += 1
    print(f"\n=== done v130 ({args.strength}): {ok}/{len(targets)} ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
