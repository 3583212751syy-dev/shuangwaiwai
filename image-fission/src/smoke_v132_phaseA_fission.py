"""v132 Phase A: 纯内容裂变 (无 CN 无 IPA, 高 denoise).

策略 (按用户新指令 '分多步多模型'):
  - Phase A (本脚本): 纯内容重画, 让 SDXL 自由生成新主体
       * sd_xl_base_1.0 (最灵活, 无偏置)
       * no ControlNet, no IPAdapter (这两会强行锁住原图)
       * denoise 0.88 (大幅重生成)
       * prompt 强写 new content, negative 禁 old content
       * base 仍是原图 latent (保留大尺度构图/色彩基线)
  - Phase B (下一脚本): 用 anytext_v2.0 模型做文字替换, 对准原文字区域

Phase A 输出会保留颜色/构图大关系但主体内容会大幅改变.

用法:
  python smoke_v132_phaseA_fission.py              # 跑所有 6 张
  python smoke_v132_phaseA_fission.py --only camo_4  # 只跑 camo_4 (用户先看效果)
"""
import json, time, sys, argparse
from pathlib import Path
import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = Path("E:/Desktop/图裂变测试图")
COMFYUI = "http://127.0.0.1:8188"

# Phase A 改用 sd_xl_base_1.0 (最自由, 无风格偏置)
CKPT = "sd_xl_base_1.0.safetensors"
SEED = 132001

# 6 张图: Phase A 内容重写 prompt (不要锁文字, 文字留给 Phase B)
# 注意: 这阶段不强制文字替换, 让 SDXL 自然生成
REFS = [
    {"id": "denim_3", "filename": "pinterest_denim_3.jpg",
     "subject": "a vintage patchwork composition of repurposed denim patches forming an abstract moth-like shape, indigo blue and white tones, embroidery stitches, retro textile collage",
     "neg": "smooth surface, modern, photograph, single solid color, blurry text, watermark",
     "base_mp": 1.0},

    {"id": "camo_4", "filename": "pinterest_camo_4.jpg",
     "subject": "dense military camouflage pattern featuring overlapping pine tree silhouettes and snowy mountain peaks, dark green brown black tan colors, matte flat military textile pattern, scattered conifer forest",
     "neg": "palm trees, tropical, beach, summer, smooth surface, photograph, watermark, blurry text",
     "base_mp": 2.0},

    {"id": "eagle_2", "filename": "pinterest_eagle_2.jpg",
     "subject": "a dark heraldic composition of a phoenix rising from flames with shield and crossed bones below, dramatic fire background, deep black and fiery red-orange, gothic tattoo illustration, symmetrical emblem",
     "neg": "eagle, photograph, realistic photo, modern, soft colors, watermark, blurry text",
     "base_mp": 1.2},

    {"id": "illust_1", "filename": "pinterest_illust_1.jpg",
     "subject": "intricate black and white ornamental pattern with stylized magnolia blooms and curving vines, symmetrical damask design, ornate engraving style, pure monochrome linework",
     "neg": "color, gray, photograph, modern, soft edges, watermark, blurry text",
     "base_mp": 0.5},

    {"id": "metal_6", "filename": "pinterest_metal_6.jpg",
     "subject": "weathered bronze carved emblem of an eagle perched on a cracked skull, bronze gold white on black, jagged lightning rays behind, classical metal engraving aesthetic, ornamental medallion",
     "neg": "modern logo, smooth surface, photograph, bright colors, watermark, blurry text",
     "base_mp": 4.0},  # 17.6MP -> 4MP 基准

    {"id": "skull_5", "filename": "pinterest_skull_5.jpg",
     "subject": "gothic composition of a cracked skull entwined with thorny vines and blood-red roses, blood drips, deep black crimson ivory parchment, dark romanticism, symmetrical ornamental frame",
     "neg": "modern, photograph, soft, cute, watermark, blurry text",
     "base_mp": 1.0},
]

POS_TEMPLATE = ("masterpiece, best quality, ultra detailed, sharp clean edges, "
                "intricate linework, {subject}, "
                "redesigned composition, completely new internal details, "
                "{hard_avoid}")

HARD_AVOID = ("no logos, no celebrities, no national flags, no political figures, "
              "no copyrighted characters, no brand names, no readable slogans")


def build_graph(ref, mp, seed):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": ref["filename"]}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels",
              "inputs": {"upscale_method": "lanczos", "megapixels": round(mp, 4),
                         "image": ["2", 0], "resolution_steps": 8}}
    pos = POS_TEMPLATE.format(subject=ref["subject"], hard_avoid=HARD_AVOID)
    neg = ref["neg"]
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": neg}}
    g["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["12"] = {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "positive": ["8", 0], "negative": ["9", 0],
        "latent_image": ["7", 0],
        "seed": seed, "steps": 40, "cfg": 7.5, "sampler_name": "dpmpp_2m",
        "scheduler": "karras", "denoise": 0.88, "noise_mask": None}}
    g["14"] = {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}}
    g["17"] = {"class_type": "SaveImage",
               "inputs": {"images": ["14", 0], "filename_prefix": f"v132A_{ref['id']}"}}
    return g


def run_one(ref, out_dir):
    tag = ref["id"]
    out_path = out_dir / f"v132A_{tag}.png"
    if out_path.exists() and out_path.stat().st_size > 100_000:
        print(f"[skip] {tag}: 已存在 {out_path.stat().st_size//1024//1024}MB", flush=True)
        return True
    orig_im = Image.open(INPUT_DIR / ref["filename"])
    orig_w, orig_h = orig_im.size
    print(f"[run] {tag}  ckpt={CKPT}  denoise=0.88  "
          f"base_mp={ref['base_mp']:.4f}  orig={orig_w}x{orig_h}", flush=True)
    g = build_graph(ref, ref["base_mp"], SEED + hash(tag) % 9999)
    try:
        r = requests.post(f"{COMFYUI}/prompt",
                          json={"prompt": g, "client_id": f"v132A_{int(time.time())}"}, timeout=15)
    except Exception as e:
        print(f"  ERR submit conn: {e}", flush=True); return False
    if r.status_code != 200:
        print(f"  ERR submit: {r.status_code} {r.text[:300]}", flush=True); return False
    pid = r.json().get("prompt_id")
    if not pid:
        print(f"  ERR no prompt_id", flush=True); return False
    print(f"  pid={pid}", flush=True)
    t0 = time.time()
    for _ in range(60):  # 300s
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
                    raw_path = out_dir / f"v132A_{tag}_raw.png"
                    raw_path.write_bytes(data)
                    # resize 回原图 (保 AR 等原图尺寸)
                    out_im = Image.open(raw_path).convert("RGB")
                    out_im = out_im.resize((orig_w, orig_h), Image.LANCZOS)
                    out_im.save(out_path, "PNG", optimize=True)
                    print(f"  OK {out_path.stat().st_size//1024//1024}MB "
                          f"(raw {raw_path.stat().st_size//1024//1024}MB, "
                          f"resized to {orig_w}x{orig_h}, "
                          f"waited {int(time.time()-t0)}s)", flush=True)
                    return True
                if status.get("error"):
                    print(f"  ERR exec: {status.get('error')}", flush=True); return False
        except Exception as e:
            print(f"  poll err: {e}", flush=True)
    print(f"  TIMEOUT after {int(time.time()-t0)}s", flush=True); return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="只跑指定 id (如 camo_4)")
    args = parser.parse_args()

    out = PROJECT_ROOT / "jobs" / f"smoke_v132_phaseA_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== v132 Phase A 纯内容裂变 (sd_xl_base, denoise=0.88, no CN/IPA) -> {out} ===", flush=True)
    print(f"=== 注意: 文字留给 Phase B 用 anytext_v2.0 模型处理 ===", flush=True)
    ok = 0
    targets = [r for r in REFS if (not args.only or r["id"] == args.only)]
    for ref in targets:
        if run_one(ref, out):
            ok += 1
    print(f"\n=== done v132 Phase A: {ok}/{len(targets)} ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())