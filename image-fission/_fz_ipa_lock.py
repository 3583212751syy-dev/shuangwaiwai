# -*- coding: utf-8 -*-
"""强裂变引擎 v2：SDXL + IPAdapterPreciseComposition(锁构图/元素大小数量)
+ IPAdapterPreciseStyleTransfer(锁色板/纹理肌理) + 高 denoise img2img(换主体形态)。全本地 ComfyUI。

LOCK（交给 IPAdapter）: 构图布局 / 色板色值 / 文本排版位置 / 纹理肌理 / 元素大小数量
FISSION（交给 prompt+denoise）: 唯一变量=主体元素换成同物种新形态

用法:
  python _fz_ipa_lock.py --num 04 --seeds 701,702
  python _fz_ipa_lock.py --num 04 --seeds 701 --denoise 0.72 --comp-w 1.6 --style-w 1.5
  python _fz_ipa_lock.py --all --seeds 701
"""
import os, sys, io, json, time, uuid, argparse, glob

for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy",
          "HTTP_ALL_PROXY","HTTPS_ALL_PROXY","ALL_PROXY","all_proxy"):
    os.environ.pop(k, None)
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import requests
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fz_comfy as C

HOST = C.HOST
CKPT = "RealVisXL_V4.0.safetensors"        # 写实底模，锁色准
IPA_FILE = "ip-adapter-plus_sdxl_vit-h.safetensors"
CLIPV = "sd1.5model.safetensors"

BASE = r"E:/Desktop/双接口/image-fission"
SRC = r"D:/Users/Administrator/Desktop/双接口/outputs/masks_1_9"
OUT = BASE + r"/outputs/_ipa_lock"
os.makedirs(OUT, exist_ok=True)

# 每张图：主体形态裂变指令（只描述「换成什么新形态」，颜色/构图由 IPAdapter 锁）
FISSION = {
 "01_skeleton": ("a horned skull reimagined as an ornate Day-of-the-Dead sugar skull with intricate "
   "filigree carvings, a rusted dented military helmet with bullet holes, skeletal soldiers reimagined "
   "as armored bone warriors with tattered capes, a torn battle standard with frayed stars"),
 "02_stripes": ("bold stripes reimagined as weathered painted wooden barricade planks, a mountain "
   "silhouette reimagined as jagged cracked rock spires, small graphic symbols reimagined as stamped "
   "industrial stencils"),
 "03_monkey": ("a street-art monkey reimagined as a mechanical simian with visible gears and rivets, "
   "urban props reimagined as a different graffiti alley with new murals and fire escapes"),
 "04_route66_gold": ("the highway shield reimagined as an aged embossed brass road-sign with layered "
   "raised metal plates and deep engraved recesses, the two small badges reimagined as a circular brass "
   "medallion and a striped shield-shaped brass badge, ornate vintage metalwork"),
 "05_route666": ("a devil pin-up woman reimagined with a different contrapposto pose and ornate horned "
   "headdress, a hot-rod coupe reimagined as a chopped custom rat-rod with a different grille and exposed "
   "engine, flames reimagined as layered curling tongues of fire with smoke"),
 "06_route66_rust": ("a rusted road shield reimagined as a corroded cast-iron sign with flaking paint "
   "and oxidized edges, decorative rivets and weathered deep embossing"),
 "07_route66_purple": ("a vintage gas station reimagined as a different Art-Deco roadside filling station "
   "with a rounded canopy and retro pump island, a waitress reimagined with a different pose and uniform, "
   "the tilted road sign reimagined with a different arrow shape"),
 "08_semi_truck": ("a semi-truck reimagined as a different long-haul cab-over rig with chrome stacks, "
   "a sleeper berth and a detailed grille, the background blobs resolved into clear distinct roadside objects"),
 "09_set_goals": ("gym equipment reimagined as modern industrial training rigs with different geometry, "
   "an athlete reimagined in a new dynamic lifting pose"),
 "10_show_time": ("graffiti stickers reimagined with fresh hand-drawn shapes, a yellow skull with a red star "
   "reimagined as a cartoon skull sticker, the ribbon reimagined as a folded banner with different tails, "
   "skateboard, pennants, arrows, stars and shield emblems reimagined as new street-art variants"),
 "11_st_tropez": ("ornate golden emblems reimagined as new baroque filigree badges, a sailboat reimagined "
   "with a different hull and sail shape, palm trees reimagined with different fronds, the small label cloud "
   "reimagined as fresh vintage travel stickers"),
}

NEG = ("text, letters, words, numbers, watermark, brand name, logo, gibberish, fake lettering, "
       "unreadable words, misspelled, random letters, "
       "blurry, smudged, soft edges, melted shapes, deformed, missing limbs, broken parts, "
       "flat colour block, patch boundary, plain gradient background, cheap cartoon, low detail, "
       "frame, paper border, vignette, wrong palette, colour shift, recoloured, hue drift")

def build_graph(ref_name, pos, neg, denoise, seed, steps, cfg,
                comp_w=1.6, comp_boost=1.2, style_w=1.5, style_boost=1.4):
    g = {}
    g["1"] = {"class_type":"CheckpointLoaderSimple", "inputs":{"ckpt_name": CKPT}}
    g["2"] = {"class_type":"LoadImage", "inputs":{"image": ref_name}}
    g["3"] = {"class_type":"CLIPVisionLoader", "inputs":{"clip_name": CLIPV}}
    g["4"] = {"class_type":"IPAdapterModelLoader", "inputs":{"ipadapter_file": IPA_FILE}}
    # 构图锁
    g["10"] = {"class_type":"IPAdapterPreciseComposition",
               "inputs":{"model":["1",0], "ipadapter":["4",0], "image":["2",0],
                         "weight":float(comp_w), "composition_boost":float(comp_boost),
                         "combine_embeds":"concat", "start_at":0.0, "end_at":1.0,
                         "embeds_scaling":"V only", "clip_vision":["3",0]}}
    # 色彩/肌理锁
    g["11"] = {"class_type":"IPAdapterPreciseStyleTransfer",
               "inputs":{"model":["10",0], "ipadapter":["4",0], "image":["2",0],
                         "weight":float(style_w), "style_boost":float(style_boost),
                         "combine_embeds":"concat", "start_at":0.0, "end_at":1.0,
                         "embeds_scaling":"V only", "clip_vision":["3",0]}}
    g["5"] = {"class_type":"VAEEncode", "inputs":{"pixels":["2",0], "vae":["1",2]}}
    g["6"] = {"class_type":"CLIPTextEncode", "inputs":{"text":pos, "clip":["1",1]}}
    g["7"] = {"class_type":"CLIPTextEncode", "inputs":{"text":neg, "clip":["1",1]}}
    g["8"] = {"class_type":"KSampler",
              "inputs":{"model":["11",0], "positive":["6",0], "negative":["7",0],
                        "latent_image":["5",0], "seed":int(seed), "steps":int(steps),
                        "cfg":float(cfg), "sampler_name":"dpmpp_2m", "scheduler":"karras",
                        "denoise":float(denoise)}}
    g["9"] = {"class_type":"VAEDecode", "inputs":{"samples":["8",0], "vae":["1",2]}}
    g["12"] = {"class_type":"SaveImage", "inputs":{"images":["9",0], "filename_prefix":"ipa_lock"}}
    return g

def run(orig_path, pos, neg=NEG, seed=701, denoise=0.72, steps=28, cfg=6.0,
        timeout=1500, **kw):
    img = Image.open(orig_path).convert("RGB")
    canvas, tw, th = C._pad1024(img)
    name = C._up(canvas, "ipal_%s.png" % uuid.uuid4().hex[:12])
    g = build_graph(name, pos, neg, denoise, seed, steps, cfg, **kw)
    r = requests.post(HOST + "/prompt", json={"prompt":g, "client_id":uuid.uuid4().hex}, timeout=120)
    if r.status_code != 200:
        raise RuntimeError("POST /prompt %s %s" % (r.status_code, r.text[:1200]))
    pid = r.json()["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(3)
        try:
            h = requests.get(HOST + "/history/" + pid, timeout=30).json()
        except Exception:
            continue
        if pid in h:
            rec = h[pid]; st = rec.get("status", {})
            if st.get("status_str") == "error" or st.get("completed") is False:
                raise RuntimeError("exec error: " + json.dumps(st.get("messages", []))[:1800])
            for v in rec.get("outputs", {}).values():
                if "images" in v:
                    im = v["images"][0]
                    rr = requests.get(HOST + "/view", params={
                        "filename": im["filename"], "subfolder": im.get("subfolder",""),
                        "type": im.get("type","output")}, timeout=180)
                    rr.raise_for_status()
                    return C.unpad(Image.open(io.BytesIO(rr.content)).convert("RGB"), tw, th)
            raise RuntimeError("no image output")
    raise RuntimeError("timeout " + pid)

def prompt_for(num):
    # num 可能是 "04" 或 "04_route66_gold"，统一取前缀
    key = next((k for k in FISSION if k.startswith(num + "_")), num)
    fis = FISSION.get(key, "the main subject reimagined as a fresh new form of the same kind")
    return ("A vintage screen-print poster illustration. " + fis + ". "
            "Crisp clean hard edges, sharp focus, highly detailed, aged lithograph print texture, "
            "professional graphic design.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seeds", default="701")
    ap.add_argument("--denoise", type=float, default=0.72)
    ap.add_argument("--comp-w", type=float, default=1.6)
    ap.add_argument("--style-w", type=float, default=1.5)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    files = {}
    for p in glob.glob(SRC + "/_fz_in_A_*.jpg"):
        n = os.path.basename(p).split("_")[4]   # "04"
        files[n] = p
    # 支持 "04" 或 "04_route66_gold" 两种写法
    want = args.num.split("_")[0] if args.num else ""
    nums = sorted(files.keys()) if args.all else [want]
    for num in nums:
        if num not in files:
            print("[skip] no source", num, file=sys.stderr); continue
        pos = prompt_for(num)
        for sd in seeds:
            print("[gen]", num, "seed", sd, file=sys.stderr)
            try:
                im = run(files[num], pos, seed=sd, denoise=args.denoise,
                         comp_w=args.comp_w, style_w=args.style_w)
                outp = "%s/%s_s%d.jpg" % (OUT, num, sd)
                im.save(outp, quality=95)
                print("[saved]", outp, file=sys.stderr)
            except Exception as e:
                print("[ERR]", num, sd, str(e)[:400], file=sys.stderr)

if __name__ == "__main__":
    main()
