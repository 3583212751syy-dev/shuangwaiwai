"""v131 大幅裂变 + 颜色严格保 + 文字全替换.

升级点 (vs v130):
  - 颜色严格保 (v130 用户反馈色彩漂移):
      1) prompt 强制 "identical color palette to original, no color shift"
      2) IPA style weight 提到 0.60 (锁配色 + 画风)
      3) 后处理: Reinhard 颜色迁移 (per-channel mean/std) 把出图颜色统计
         强行对齐原图 — 这是决定性保色, 出图再花哨, 颜色统计会与原图匹配
  - 裂变范围加大 (用户反馈看不出区别):
      1) denoise 0.55 -> 0.82 (内容大幅重生成)
      2) Canny weight 0.50 -> 0.35 (松绑结构, 让主体真的换)
      3) prompt 加 "completely different composition, replaced subject with new subject
         in same general placement"
  - 文字全替换 (用户要求原图所有字体都换):
      每张都强写新词 (如 WORN / BRAVE / VENGEFUL / BORN ALWAYS).
      SDXL 文字生成可靠性中等, 但尽力. 失败则用户后续可让 AI 强制修复.

metal_6 17.6MP -> 4MP 基准 + PIL resize 回原图 (与 v130 同逻辑, 内嵌).

用法:
  python smoke_v131_wide.py                 # 默认 wide 档跑 6 张
  python smoke_v131_wide.py --only denim_3  # 单图
"""
import json, time, sys, argparse
from pathlib import Path
import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = Path("E:/Desktop/图裂变测试图")
COMFYUI = "http://127.0.0.1:8188"

CKPT = "ProteusV0.4.safetensors"
CONTROL_CKPT = "controlnet-canny-sdxl-1.0.fp16.safetensors"
UPSCALE = "4x_NMKD-Siax_200k.pth"
SEED = 800131

# 宽档: denoise 0.82 / cn 0.35 (结构松绑) / ipa 0.60 (画风+色锁)
WIDE = {"denoise": 0.82, "cn_weight": 0.35, "ipa_weight": 0.60,
        "label": "wide (大幅裂变 + 颜色锁定)"}

# 6 张图: subject 大改 + 配色 lock + 强制文字替换
REFS = [
    {"id": "denim_3", "filename": "pinterest_denim_3.jpg",
     "subject": ("a moth made of torn denim fabric and silver studs resting on a "
                 "frayed fabric patch, completely different subject matter and pose, "
                 "redesigned arrangement, identical color palette to original (denim "
                 "blue and white only), bold legible chunky text 'WORN' at top in "
                 "the same chunky letter style replacing original text"),
     "palette": "same exact colors as original: faded indigo blue and white only, "
                "rough denim texture"},

    {"id": "camo_4", "filename": "pinterest_camo_4.jpg",
     "subject": ("dense military camouflage pattern with overlapping pine tree "
                 "silhouettes and snow patches, completely different foliage layout "
                 "from original (no palm trees), redesigned distribution, "
                 "identical color palette to original (dark green brown black tan)"),
     "palette": "same exact colors as original: dark green brown black tan camouflage, "
                "matte flat military pattern"},

    {"id": "eagle_2", "filename": "pinterest_eagle_2.jpg",
     "subject": ("a majestic hawk with spread wings perched on a weathered shield, "
                 "fire in background, completely different bird and shield motif "
                 "than original, redesigned arrangement, identical color palette "
                 "to original (black and red-orange), bold legible text 'BRAVE' at "
                 "top and 'WILD' at bottom in the same gothic letter style"),
     "palette": "same exact colors as original: deep black and fiery red-orange, "
                "high contrast tattoo illustration style"},

    {"id": "illust_1", "filename": "pinterest_illust_1.jpg",
     "subject": ("black and white ornamental pattern with stylized magnolia blooms "
                 "and curving vines, completely different botanical species and "
                 "scrollwork than original, redesigned symmetrical layout, "
                 "identical color palette to original (black and white only)"),
     "palette": "same exact colors as original: pure black and white only, "
                "intricate engraving style, ornate linework"},

    {"id": "metal_6", "filename": "pinterest_metal_6.jpg",
     "subject": ("bronze carved eagle emblem on a shield with cracked stone, "
                 "completely different motif than original death metal logo, "
                 "redesigned layout, identical color palette to original (bronze "
                 "gold white black), bold legible text 'VENGEFUL' at top in the "
                 "same jagged gothic metal letter style"),
     "palette": "same exact colors as original: bronze gold white on black, "
                "weathered metal engraving aesthetic"},

    {"id": "skull_5", "filename": "pinterest_skull_5.jpg",
     "subject": ("ornate skull with thorny vines and roses, blood drips, "
                 "completely different dark motif and arrangement than original, "
                 "redesigned symmetrical composition, identical color palette to "
                 "original (deep black crimson ivory), bold legible gothic text "
                 "'BORN' at top and 'ALWAYS' at bottom in the same dark fantasy "
                 "letter style"),
     "palette": "same exact colors as original: deep black crimson ivory parchment, "
                "gothic dark romanticism"},
]

POS_TEMPLATE = ("masterpiece, best quality, ultra detailed, sharp clean edges, "
                "intricate linework, "
                "{subject}, "
                "completely redrawn internal details, new visual elements throughout, "
                "framed composition, {palette}, {hard_avoid}")

# 不再禁 readable slogans (因为我们要放替换词); 但仍禁 trademarked text
HARD_AVOID = ("no logos, no celebrities, no national flags, no political figures, "
              "no copyrighted characters, no trademarked text, no brand names")

NEG = ("blurry, smudged, deformed, disfigured, low quality, bad anatomy, mutation, "
       "cropped, jpeg artifacts, noise, watermark, signature, "
       "garbled letters, distorted text, gibberish words, overlapping text, "
       "text bleeding, ugly text")


def get_original_dimensions(filename):
    im = Image.open(INPUT_DIR / filename)
    return im.size  # (W, H)


def get_base_megapixels(filename):
    """metal_6 (17.6MP) 走 4MP 基准 + 后 resize 回原图; 其他走原图 MP."""
    w, h = get_original_dimensions(filename)
    mp = (w * h) / 1_000_000.0
    if mp > 8.0:
        return min(4.0, 16.0)  # 4MP 基准
    return min(mp, 16.0)


def reinhard_color_transfer(src_pil, ref_pil):
    """把 src 的颜色统计 (RGB 三通道 mean/std) 强行对齐 ref.

    Reinhard et al. 2001 简化版 (per-channel linear normalization).
    结果: 颜色统计与 ref 完全一致, 但空间结构保留.
    """
    import numpy as np
    src = np.asarray(src_pil.convert("RGB"), dtype=np.float32)
    ref = np.asarray(ref_pil.convert("RGB"), dtype=np.float32)
    out = np.zeros_like(src)
    for c in range(3):
        s_m, s_s = src[..., c].mean(), src[..., c].std() + 1e-6
        r_m, r_s = ref[..., c].mean(), ref[..., c].std() + 1e-6
        out[..., c] = (src[..., c] - s_m) / s_s * r_s + r_m
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB")


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
    pos = POS_TEMPLATE.format(subject=ref["subject"], palette=ref["palette"],
                              hard_avoid=HARD_AVOID)
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": NEG}}
    g["6"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
        "positive": ["8", 0], "negative": ["9", 0], "image": ["4", 0],
        "strength": strength["cn_weight"], "start_percent": 0.0, "end_percent": 0.85,
        "control_net": ["5", 0]}}
    g["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["10"] = {"class_type": "IPAdapterUnifiedLoader",
               "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["11"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["10", 0], "ipadapter": ["10", 1],
        "image": ["2", 0],
        "weight": strength["ipa_weight"], "weight_type": "style transfer",
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
               "inputs": {"images": ["16", 0], "filename_prefix": f"v131_{ref['id']}"}}
    return g


def run_one(ref, out_dir, strength):
    tag = ref["id"]
    out_path = out_dir / f"v131_{tag}.png"
    if out_path.exists() and out_path.stat().st_size > 100_000:
        print(f"[skip] {tag}: 已存在 {out_path.stat().st_size//1024//1024}MB", flush=True)
        return True
    mp = get_base_megapixels(ref["filename"])
    orig_w, orig_h = get_original_dimensions(ref["filename"])
    print(f"[run] {tag}  cn={strength['cn_weight']}  ipa={strength['ipa_weight']}  "
          f"denoise={strength['denoise']}  base_mp={mp:.4f}  "
          f"orig={orig_w}x{orig_h} ({orig_w*orig_h/1e6:.1f}MP)", flush=True)
    g = build_graph(ref, mp, strength)
    try:
        r = requests.post(f"{COMFYUI}/prompt",
                          json={"prompt": g, "client_id": f"v131_{int(time.time())}"}, timeout=15)
    except Exception as e:
        print(f"  ERR submit conn: {e}", flush=True); return False
    if r.status_code != 200:
        print(f"  ERR submit: {r.status_code} {r.text[:300]}", flush=True); return False
    pid = r.json().get("prompt_id")
    if not pid:
        print(f"  ERR no prompt_id", flush=True); return False
    print(f"  pid={pid}", flush=True)
    t0 = time.time()
    for _ in range(90):  # 450s 硬超时
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
                    raw_path = out_dir / f"v131_{tag}_raw.png"
                    raw_path.write_bytes(data)
                    # 后处理: 颜色迁移 + resize 回原图
                    out_im = Image.open(raw_path).convert("RGB")
                    orig_im = Image.open(INPUT_DIR / ref["filename"]).convert("RGB")
                    # 1) resize 回原图尺寸 (保 AR — 用 target size 而不是 ratio)
                    out_im = out_im.resize((orig_w, orig_h), Image.LANCZOS)
                    # 2) Reinhard 颜色迁移对齐原图配色
                    out_im = reinhard_color_transfer(out_im, orig_im)
                    out_im.save(out_path, "PNG", optimize=True)
                    print(f"  OK {out_path.stat().st_size//1024//1024}MB "
                          f"(raw {raw_path.stat().st_size//1024//1024}MB, "
                          f"color-locked + resized to {orig_w}x{orig_h}, "
                          f"waited {int(time.time()-t0)}s)", flush=True)
                    return True
                if status.get("error"):
                    print(f"  ERR exec: {status.get('error')}", flush=True); return False
        except Exception as e:
            print(f"  poll err: {e}", flush=True)
    print(f"  TIMEOUT after {int(time.time()-t0)}s", flush=True); return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="只跑指定 id (如 denim_3)")
    args = parser.parse_args()

    out = PROJECT_ROOT / "jobs" / f"smoke_v131_wide_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== v131 WIDE ({WIDE['label']}) -> {out} ===", flush=True)
    print(f"=== denoise={WIDE['denoise']}  cn={WIDE['cn_weight']}  ipa={WIDE['ipa_weight']} ===", flush=True)
    print(f"=== post: Reinhard 颜色迁移 (src stats → ref stats) + resize 回原图 ===", flush=True)
    ok = 0
    targets = [r for r in REFS if (not args.only or r["id"] == args.only)]
    for ref in targets:
        if run_one(ref, out, WIDE):
            ok += 1
    print(f"\n=== done v131 (wide): {ok}/{len(targets)} ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())