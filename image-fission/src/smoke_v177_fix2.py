"""v177 修复版：
- camo_4: 换 IPAdapterAdvanced -> IPAdapterPreciseStyleTransfer(style_boost=2.5) 管线内锁色，LoRA=0 圆润，CT s=1.15 后期补色
- denim_3: denoise 0.80->0.70 (保蝴蝶轮廓)，LORA 1.0->0.4 (减弱细节打碎)，改 butterfly 区域提示 (intact symmetrical butterfly)
其余参数 (tile 0.60, canny 0.25, KSampler 24+20, 4x_NMKD-Siax, ProteusV0.4) 全锁死。
"""
import sys, time, json, requests, shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smoke_v164_per_image as v164
import color_transfer as ct
from selfcheck_metrics import color_intersect, ssim, frag_ratio

OUT = HERE.parent / "outputs" / "v177"
OUT.mkdir(parents=True, exist_ok=True)
COMFY = v164.COMFYUI

# 任务配置: (id, mode, overrides, ipa_mode, ipa_kwargs, denoise, lora, butterfly_swap, ct_strengths)
# ipa_mode: "advanced" / "precise"
JOBS = [
    {
        "id": "camo_4",
        "ipa_mode": "precise",
        "ipa_kwargs": {"weight": 0.18, "style_boost": 2.5, "combine_embeds": "average",
                       "start_at": 0.0, "end_at": 0.85},
        "denoise": 0.80,
        "lora": 0.0,
        "butterfly_swap": False,
        "ct_strengths": [0.85, 1.0, 1.15, 1.3],
    },
    {
        "id": "denim_3",
        "ipa_mode": "advanced",
        "ipa_kwargs": {"weight": 0.18, "weight_type": "style transfer",
                       "combine_embeds": "average", "start_at": 0.0, "end_at": 0.85,
                       "noise": 0.05, "embeds_scaling": "V only"},
        "denoise": 0.70,
        "lora": 0.4,
        "butterfly_swap": True,
        "ct_strengths": [0.7, 0.85, 1.0],
    },
]


def patch_butterfly(ref):
    """把 denim_3 主体蝴蝶提示从'破布拼贴'换成'完整对称蝴蝶'。"""
    for r in ref["regions"]:
        p = r["prompt"]
        if "ONE large butterfly" in p:
            r["prompt"] = (
                "ONE large INTACT butterfly with a clean symmetrical shape, "
                "both wings fully formed with rounded teardrop outline, "
                "two slender antennae extending upward from the head, "
                "a slim soft body in the center, "
                "wings show DENIM fabric texture as inner material "
                "(indigo blue denim panel with visible weave) "
                "but the OVERALL butterfly silhouette is INTACT (not torn, not patchwork, not ripped), "
                "delicate stitched veins inside wings, "
                "slight worn wash on wing edges only. " + v164.COHESIVE
            )


def build_overrides(ref, denoise, lora, ipa_mode, ipa_kwargs, butterfly_swap):
    """复刻 v164.build() 但允许覆盖 denoise/lora/ipa 节点。"""
    if butterfly_swap:
        patch_butterfly(ref)
    g = v164.build(ref, v164.SEED)
    # 覆盖 KSampler denoise
    g["11"]["inputs"]["denoise"] = denoise
    # 覆盖 LoRA 强度
    g["7"]["inputs"]["strength_model"] = lora
    g["7"]["inputs"]["strength_clip"] = lora
    # 覆盖 IPA 节点
    if ipa_mode == "precise":
        # IPAdapterPreciseStyleTransfer 必填: weight / style_boost / combine_embeds / start_at / end_at / embeds_scaling
        g["6"] = {
            "class_type": "IPAdapterPreciseStyleTransfer",
            "inputs": {
                "model": ["1", 0],
                "ipadapter": ["5", 1],
                "image": ["3", 0],
                "weight": ipa_kwargs["weight"],
                "style_boost": ipa_kwargs["style_boost"],
                "combine_embeds": ipa_kwargs["combine_embeds"],
                "start_at": ipa_kwargs["start_at"],
                "end_at": ipa_kwargs["end_at"],
                "embeds_scaling": ipa_kwargs.get("embeds_scaling", "V only"),
            },
        }
        # LoraLoader 链上游需变：model from "6" (Precise 输出)
        g["7"]["inputs"]["model"] = ["6", 0]
    else:
        # advanced 模式：v164.build 默认已是 advanced
        for k, v in ipa_kwargs.items():
            g["6"]["inputs"][k] = v
    return g


def render(job):
    rid = job["id"]
    ref = next(r for r in v164.REFS if r["id"] == rid)
    # 读出 ref 是 dict，patch_butterfly 会就地改 regions；要避免污染共享 REFS，传 deep copy
    import copy
    ref = copy.deepcopy(ref)
    g = build_overrides(ref, job["denoise"], job["lora"],
                        job["ipa_mode"], job["ipa_kwargs"], job["butterfly_swap"])
    prefix = f"v177_{rid}"
    g["15"]["inputs"]["filename_prefix"] = prefix
    print(f"[v177] {rid}: DENOISE={job['denoise']} LORA={job['lora']} IPA={job['ipa_mode']} "
          f"{job['ipa_kwargs']}", flush=True)
    r = requests.post(f"{COMFY}/prompt", json={"prompt": g, "client_id": f"v177_{int(time.time())}"}, timeout=15)
    j = r.json()
    if r.status_code != 200 or "error" in j:
        print("[ERR]", r.status_code, json.dumps(j)[:1500]); return None
    pid = j.get("prompt_id"); print(f"[v177] {rid} pid={pid}", flush=True)
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
                raw = OUT / f"{rid}_raw.png"; raw.write_bytes(data)
                im = Image.open(raw).convert("RGB")
                im.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)).save(raw, quality=95)
                print(f"[v177] {rid} raw saved {raw.stat().st_size/1024/1024:.1f}MB")
                return raw
        if "error" in st:
            print("[ERR]", st); return None
    print("[ERR] timeout", rid); return None


def post_ct_select(rid, raw, strengths):
    """raw vs 各 CT strength 取最高 color∩。"""
    orig = HERE.parent / "web_gallery" / "img" / f"orig_{rid}.jpg"
    if not orig.exists():
        print(f"[v177] {rid} 无原图对照，跳过 CT"); return raw
    o = np.array(Image.open(orig).convert("RGB"))
    b = np.array(Image.open(raw).convert("RGB"))
    # 先看 raw 自身
    raw_ci = ct.color_intersect(str(orig), str(raw))
    best = (raw_ci, "raw", raw)
    print(f"  {rid} raw color∩={raw_ci:.3f}")
    for s in strengths:
        out = ct.reinhard(b, o, s)
        p = OUT / f"{rid}_ct_s{s:.2f}.png"
        Image.fromarray(out).save(p)
        ci = ct.color_intersect(str(orig), str(p))
        print(f"  {rid} s={s:.2f} color∩={ci:.3f}")
        if ci > best[0]:
            best = (ci, f"s={s:.2f}", p)
    print(f"[v177] {rid} BEST={best[1]} color∩={best[0]:.3f}")
    final = HERE.parent / "outputs" / "final" / f"{rid}.jpg"
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best[2], final)
    gpath = HERE.parent / "web_gallery" / "img" / f"fiss_{rid}.jpg"
    shutil.copyfile(best[2], gpath)
    print(f"[v177] {rid} FINAL metrics: color∩={ct.color_intersect(str(orig),str(best[2])):.3f} "
          f"ssim={ssim(str(orig),str(best[2])):.3f} fragR={frag_ratio(str(orig),str(best[2])):.3f}")
    return best[2]


results = []
for job in JOBS:
    raw = render(job)
    if raw:
        f = post_ct_select(job["id"], raw, job["ct_strengths"])
        results.append((job["id"], f))

print("[v177] DONE:", [(r, str(f)) for r, f in results])