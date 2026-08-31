"""v176b eagle_2 单独补 v147 基线。regions 为空会让 RegionalListCombine 只有 global 输入，
在 ComfyUI 端执行失败。这里给一个满幅 region（100% 覆盖，prompt=原 global_pos）保证 ≥2 输入。
IPA=0.18 冻结。渲染后不接 CT（eagle_2 原图是黑底红橙，裂变需保持该高反差配色，CT 意义不大）。
"""
import sys, time, json, requests, traceback, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smoke_v164_per_image as v164
import color_transfer as ct

OUT = HERE.parent / "outputs" / "v176"
OUT.mkdir(parents=True, exist_ok=True)
COMFY = v164.COMFYUI
LOG = OUT / "v176b_eagle2.log"

def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f: f.write(s + "\n")

try:
    v164.IPA_WEIGHT = 0.18
    v164.LORA_DETAIL = 1.0
    # 构造 eagle_2 ref，含一个满幅 region
    eagle = {
        "id": "eagle_2", "ref_img": "pinterest_eagle_2.jpg",
        "global_pos": ("gothic tattoo illustration, pure black background, "
                       "red and orange flames, white and silver eagle and skull, gray iron, "
                       "bold t-shirt graphic print, high contrast, sharp edges, "
                       "no text, no letters, no words, no banner, no inscription anywhere, "
                       "cohesive composition"),
        "regions": [
            {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "strength": 1.0,
             "prompt": ("gothic tattoo illustration, pure black background, "
                        "red and orange flames, white and silver eagle and skull, gray iron, "
                        "bold t-shirt graphic print, high contrast, sharp edges, "
                        "no text, no letters, no words, no banner, no inscription anywhere")}
        ],
    }
    g = v164.build(eagle, v164.SEED)
    g["15"]["inputs"]["filename_prefix"] = "v176_eagle_2"
    log(f"[v176b] IPA={v164.IPA_WEIGHT} LORA={v164.LORA_DETAIL} DENOISE={v164.DENOISE} 提交 eagle_2")
    r = requests.post(f"{COMFY}/prompt", json={"prompt": g, "client_id": f"v176b_{int(time.time())}"}, timeout=15)
    j = r.json()
    if r.status_code != 200 or "error" in j:
        log("[ERR] submit", r.status_code, json.dumps(j)[:1500]); sys.exit(1)
    pid = j.get("prompt_id"); log("[v176b] pid=", pid)
    done = False
    for i in range(120):
        time.sleep(5)
        h = requests.get(f"{COMFY}/history/{pid}", timeout=10).json()
        if pid not in h:
            continue
        rec = h[pid]; st = rec.get("status", {})
        if st.get("completed"):
            imgs = rec.get("outputs", {}).get("15", {}).get("images", [])
            if imgs:
                fn = imgs[0]["filename"]; sub = imgs[0].get("subfolder", "")
                data = requests.get(f"{COMFY}/view?filename={fn}&type=output&subfolder={sub}", timeout=60).content
                raw = OUT / "v176_eagle_2_raw.png"; raw.write_bytes(data)
                from PIL import Image, ImageFilter
                im = Image.open(raw).convert("RGB")
                im.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)).save(raw, quality=95)
                # 复制到 final + 画廊
                final = HERE.parent / "outputs" / "final" / "eagle_2.jpg"
                final.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(raw, final)
                gal = HERE.parent / "web_gallery" / "img" / "fiss_eagle_2.jpg"
                shutil.copyfile(raw, gal)
                log(f"[v176b] ✓ eagle_2 saved {raw.stat().st_size/1024/1024:.1f}MB -> final + gallery")
                done = True; break
        if "error" in st:
            log("[ERR] exec", st); break
    if not done:
        log("[ERR] timeout")
except Exception:
    log("[FATAL]", traceback.format_exc())
