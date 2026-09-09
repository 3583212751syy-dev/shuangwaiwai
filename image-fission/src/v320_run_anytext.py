"""v320: run AnyText2 text-editing via ComfyUI API for one image.

Usage: python v320_run_anytext.py <image> <ref_name> <mask_name> <out_name> "<text1>|<text2>"
  image: 'bacardi' | 'eagle' | 'denim'
"""
import sys, json, time, urllib.request, urllib.error, os
from PIL import Image

COMFY = "http://127.0.0.1:8188"

CFG = {
    "bacardi": dict(w=1552, h=2000, ref="src_bacardi.jpg", mask="mask_bacardi.png", out="v320_bacardi.png"),
    "eagle":   dict(w=964,  h=1280, ref="src_eagle.jpg",   mask="mask_eagle.png",   out="v320_eagle.png"),
    "denim":   dict(w=736,  h=1308, ref="src_denim.jpg",   mask="mask_denim.png",   out="v320_denim.png"),
}

def post(api, data):
    req = urllib.request.Request(COMFY + api, data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())

def queue(prompt):
    return post("/prompt", {"prompt": prompt, "client_id": "v320"})

def get_history(pid):
    return post("/history/" + pid, {})

def build_prompt(c, texts):
    p = {
        "loader": {"class_type": "UL_AnyTextLoader", "inputs": {
            "ckpt_name": "anytext_v2.0.ckpt", "control_net_name": "None",
            "miaobi_clip": "None", "weight_dtype": "fp16", "init_device": "auto"}},
        "ref": {"class_type": "LoadImage", "inputs": {"image": c["ref"]}},
        "mask": {"class_type": "LoadImage", "inputs": {"image": c["mask"]}},
        "latent": {"class_type": "EmptyLatentImage", "inputs": {"width": c["w"], "height": c["h"], "batch_size": 1}},
        "encoder": {"class_type": "UL_AnyTextEncoder", "inputs": {
            "model": ["loader", 0], "mask": ["mask", 1], "image": ["ref", 0],
            "latent": ["latent", 0],
            "prompt": "", "texts": texts,
            "font_name": "Arial_Unicode.ttf", "mode": False, "sort_radio": False,
            "a_prompt": "best quality, detailed, clear text edges, neat writing, matching original font style, serif elegant",
            "n_prompt": "low-res, distorted text, messy words, watermark, extra digits",
            "random_mask": False, "revise_pos": False}},
        "sampler": {"class_type": "UL_AnyTextSampler", "inputs": {
            "model": ["loader", 0], "positive": ["encoder", 0], "negative": ["encoder", 1],
            "seed": 42, "steps": 25, "cfg": 7.5, "strength": 1.0,
            "attnx_scale": 1.0, "eta": 1.0, "keep_load": True, "keep_device": True}},
        "vae": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["loader", 1]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["vae", 0], "filename_prefix": c["out"].replace(".png", "")}},
    }
    return p

def main():
    image = sys.argv[1]
    c = CFG[image]
    texts = sys.argv[2].split("|") if len(sys.argv) > 2 else ["NOCTAVEN", "DARK RESERVE"]
    print(f"=== AnyText2 text-editing: {image} texts={texts} ===")
    prompt = build_prompt(c, texts)
    r = queue(prompt)
    pid = r.get("prompt_id")
    print("prompt_id:", pid)
    # poll
    for i in range(120):
        time.sleep(5)
        try:
            hist = post("/history", {}) if False else None
            # use /history/<pid>
            h = post("/history/" + pid, {})
        except Exception as e:
            print("poll err", e); continue
        if pid in h:
            outputs = h[pid]["outputs"]
            for node, out in outputs.items():
                if "images" in out:
                    for im in out["images"]:
                        fn = im["filename"]
                        sub = im.get("subfolder", "")
                        url = f"{COMFY}/view?filename={fn}&subfolder={sub}&type=output"
                        # download
                        with urllib.request.urlopen(url, timeout=60) as resp:
                            data = resp.read()
                        dst = os.path.join(r"E:\Desktop\双接口\image-fission\jobs\v320", fn)
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        with open(dst, "wb") as f:
                            f.write(data)
                        print(f"SAVED -> {dst} ({len(data)} bytes)")
                        return dst
            print("prompt done but no image? outputs:", outputs)
            return None
        if i % 6 == 0:
            print(f"  waiting... {i*5}s")
    print("TIMEOUT")
    return None

if __name__ == "__main__":
    main()
