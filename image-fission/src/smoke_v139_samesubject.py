"""v144 真正大胆重排（本机 ComfyUI / SDXL）：v143 仍像原图的根因修复。
v143 翻车根因：subject 仍描述原图"heraldic crest"（鹰展翅+三骷髅+包裹火焰+垂链+带字盾牌），
即便 variant 写"换单骷髅/对角链"，模型权重仍偏向 subject，把 variant 当噪音；
导致 corr 0.665 > v142 的 0.488（更不像原图的方向走反了），banner 文字也没去掉。
v144 修复：subject 改为直接描述"新设计"（俯冲鹰+单皇冠骷髅+不对称火焰+对角断链+无banner），
配合更弱 CN 0.22 / 更弱 IPA 0.18，让新 prompt 压过 img2img latent，生成真正不同的构图。
保留 v143 的反失真设置：Detail Tweaker 1.0 + 1.2MP + 强 anti-mutation 负向。

用法：python smoke_v139_samesubject.py [ref_id]   (默认 eagle_2)
输出：jobs/smoke_v139/v139_{id}.jpg
"""
import time, requests, sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

# v144 调参（v143 仍像原图的反向修复）：
# - CN 0.35→0.22：极弱边缘约束，让新 prompt 压过 latent（不再锁原图 pose）
# - IPA 0.22→0.18：少锁原图内容，保留"画风配色相关"即可
# - denoise 0.85 保持：高重排空间
# - Detail Tweaker 1.0 + 1.2MP 保持：抗红框失真（v143 这点已 OK，保留）
# - 关键：eagle_2 的 subject 改为直接描述"新设计"（俯冲鹰+单皇冠骷髅+不对称火焰+对角断链+无banner）
SEED = 700401
CKPT = "ProteusV0.4.safetensors"
DENOISE = 0.85
CN_STRENGTH = 0.22
IPA_WEIGHT = 0.18
LORA_DETAIL = 1.0
MEGA_PIXELS = 1.2

# 同主体 prompt：subject 严格描述原图“同一类”元素（鹰还是鹰/棕榈还是棕榈/蝴蝶还是蝴蝶），
# 不替换为别的主体；palette 锁死原图配色，pos/neg 都强调“不加新色、不换主体”。
REFS = [
    {"id": "eagle_2", "ref_img": "pinterest_eagle_2.jpg", "weight": IPA_WEIGHT,
     # v144：subject 直接描述"新设计"，不再提 heraldic crest / banner（v143 仍提了所以模型没逃掉）
     "subject": ("a single dynamic bald eagle in a top-down diving pose viewed from above, "
                 "both wings swept back and talons thrust forward and down, "
                 "a single large cracked human skull wearing an iron spike crown placed prominently "
                 "at the lower center as the dominant focal point, "
                 "three distinct asymmetric flame columns rising from the bottom "
                 "(one tall flame on the left side, one short flame on the right, one diagonal flame ribbon sweeping across), "
                 "one long iron chain that breaks the frame running diagonally from the top-right corner to the bottom-left corner, "
                 "pure decorative iron scrollwork filigree, "
                 "absolutely no banners, no heraldic shields, no inscriptions, no text, no letters, no words, "
                 "no readable characters anywhere in the image, "
                 "dramatic asymmetric dynamic composition"),
     "palette": ("gothic tattoo illustration, pure black background, red and orange flames, "
                 "white and silver eagle and skull and chain, no new colors"),
     "variant": ("RADICAL NEW LAYOUT — completely DIFFERENT from any reference heraldic crest. "
                 "The reference's symmetric spread-wing eagle + three-skull base + wrapping flames + draped chains + inscribed banner "
                 "is REPLACED by: a top-down diving eagle, ONE central iron-crown skull, asymmetric flame columns, "
                 "a diagonal breaking chain, and PURE scrollwork with ZERO text. "
                 "Same black-red-silver-white gothic palette and same eagle/skull/flame/chain element category, "
                 "but the composition is DRAMATICALLY RECOMPOSED — do NOT mirror or redraw any reference arrangement")},
    {"id": "camo_4", "ref_img": "pinterest_camo_4.jpg", "weight": IPA_WEIGHT,
     "subject": "palm trees and tropical fronds arranged in a military camouflage pattern",
     "palette": "green brown black camouflage, matte, no text"},
    {"id": "illust_1", "ref_img": "pinterest_illust_1.jpg", "weight": IPA_WEIGHT,
     "subject": ("ornate black and white scrolling acanthus foliage, flowers and decorative "
                 "leaves, symmetrical border frame"),
     "palette": "monochrome ink line illustration, pure black background, white filigree, no color"},
    {"id": "denim_3", "ref_img": "pinterest_denim_3.jpg", "weight": IPA_WEIGHT,
     "subject": ("a butterfly perched on a distressed X-shaped denim patch with fabric wear "
                 "and stitch holes"),
     "palette": "vintage faded denim texture, indigo blue and white only"},
    {"id": "skull_5", "ref_img": "pinterest_skull_5.jpg", "weight": IPA_WEIGHT,
     "subject": ("a grim human skull with black eye patch, a red snake coiled around it, "
                 "red feathered wings, a red rose"),
     "palette": "gothic tattoo illustration, pure black background, red and white palette"},
    {"id": "metal_6", "ref_img": "pinterest_metal_6.jpg", "weight": IPA_WEIGHT,
     "subject": ("a bald eagle perched on a cracked human skull with horn-like spikes, "
                 "radiating lightning bolts"),
     "palette": "death metal illustration, pure black background, white and bronze palette"},
]

POS_TAIL = (
    "creative alternative interpretation of the reference design, "
    "KEEP the same color palette and overall graphic energy as the reference, "
    "KEEP the same category of elements (same kind of subjects), "
    "but RECOMPOSE with MAJOR RADICAL changes in element ANGLE, SCALE and PROPORTION, "
    "reposition and re-angle the elements dramatically, replace with RELATED same-category elements "
    "in different poses or completely different arrangements, "
    "the result must look like a CLEARLY DIFFERENT DESIGN that is only LOOSELY inspired by the original, "
    "BOLD DYNAMIC ASYMMETRIC composition — DO NOT mirror the reference layout, place subjects in UNEXPECTED positions, "
    "professional high-end commercial apparel graphic with museum-grade craft, "
    "masterpiece best quality ultra detailed, "
    "anatomically correct and physically coherent subjects with CLEAN READABLE FORMS, "
    "intricate craftsmanship and fine engraved details on every element, "
    "every small accessory and detail rendered with PIN-SHARP precision, "
    "ULTRA sharp crisp high-contrast edges on every element, "
    "every element surrounded by a CLEAR solid-black separating outline silhouette, "
    "elements NEVER bleed into neighbors, "
    "bold graphic t-shirt print composition full bleed edge-to-edge, "
    "ABSOLUTELY NO text, NO letters, NO words, NO alphabet shapes, NO banner inscription, NO readable characters anywhere, "
    "no halftone no noise no grain no smudge no watercolor no soft airbrush"
)
NEG_BASE = (
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    "text, letters, words, writing, typography, signature, caption, label, paragraph, alphabet, "
    "banner, banner inscription, engraved lettering, runic text, readable text, glyphs, calligraphy, "
    "3d, photographic, painterly, illustration by child, beginner drawing, "
    "blur, soft focus, smooth shading, smudge, soft airbrush, watercolor, "
    "bleeding borders, fused elements, melted edges, soft halo, gradient transition, "
    "out of focus, dreamy, ethereal, foggy, hazy, low contrast, pastel, "
    "small subject, distant view, zoomed out, far away, miniature, tiny, "
    "noise, grain, pixelated, jagged edges, aliasing, duplicate image, exact copy, watermark, "
    "mutated, malformed, deformed anatomy, broken bones, extra limbs, missing claws, "
    "melted, fused, smudged, bleeding, water damaged, anatomically incorrect, "
    "extra wings, asymmetric error, garbled forms, nonsense, AI artifact, tiling, repeating pattern, "
    "new colors, different color palette, extra colors, color shift"
)


def build(ref, seed):
    variant = ref.get("variant", "")
    pos = (f"t-shirt graphic design, {ref['subject']}, {ref['palette']}, {POS_TAIL} {variant}")
    neg = NEG_BASE
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": ref["ref_img"]}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["2", 0], "upscale_method": "lanczos", "megapixels": MEGA_PIXELS, "resolution_steps": 64}}
    g["4"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}
    g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["3", 0],
        "weight": ref["weight"], "weight_type": "style transfer",
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.85,
        "noise": 0.05, "embeds_scaling": "V only"}}
    # Detail Tweaker XL LoRA：在 IPAdapter 之后加载，注入细节强化能力，防止局部元素失真
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["6", 0], "clip": ["1", 1],
        "lora_name": "add-detail-xl.safetensors",
        "strength_model": LORA_DETAIL, "strength_clip": LORA_DETAIL}}
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": pos}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": neg}}
    # ControlNet Canny：强度提升到 0.50 恢复细节约束，仍允许重排角度/大小占比
    g["16"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "controlnet-canny-sdxl-1.0.safetensors"}}
    g["17"] = {"class_type": "Canny", "inputs": {
        "image": ["2", 0], "low_threshold": 0.06, "high_threshold": 0.20, "resolution": 1024}}
    g["18"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["8", 0],
        "control_net": ["16", 0], "image": ["17", 0],
        "strength": CN_STRENGTH}}
    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["18", 0], "negative": ["9", 0],
        "latent_image": ["4", 0], "seed": seed, "steps": 32, "cfg": 6.0,
        "sampler_name": "euler", "scheduler": "normal", "denoise": DENOISE}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["18", 0], "negative": ["9", 0],
        "latent_image": ["10", 0], "seed": seed + 1, "steps": 28, "cfg": 6.0,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.20}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["13", 0], "image": ["12", 0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0],
                       "filename_prefix": f"v139_{ref['id']}"}}
    return g


def gen(ref, seed, out_base):
    tag = ref["id"]
    out = out_base / f"v139_{tag}.jpg"
    if out.exists() and out.stat().st_size > 100000:
        print(f"  [{tag}] 已存在，跳过", flush=True); return True
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": build(ref, seed),
                    "client_id": f"v139_{int(time.time())}"}, timeout=15)
    try:
        j = r.json()
    except Exception:
        j = {}
    if r.status_code != 200 or "error" in j:
        print(f"[ERR] {tag}: {r.status_code} {str(j)[:300]}", flush=True); return False
    pid = j.get("prompt_id")
    if not pid:
        print(f"[ERR] {tag}: 无 prompt_id {str(j)[:300]}", flush=True); return False
    print(f"  [{tag}] pid={pid}", flush=True)
    for i in range(60):  # 300s 硬超时
        time.sleep(5)
        try:
            h = requests.get(f"{COMFYUI}/history/{pid}", timeout=10).json()
            if pid in h:
                rec = h[pid]
                if rec.get("status", {}).get("completed"):
                    imgs = rec.get("outputs", {}).get("15", {}).get("images", [])
                    if imgs:
                        url = f"{COMFYUI}/view?filename={imgs[0]['filename']}&type=output&subfolder={imgs[0].get('subfolder','')}"
                        try:
                            data = requests.get(url, timeout=60).content
                        except Exception as e:
                            print(f"  [{tag}] 取图失败 {e}", flush=True); return False
                        out.write_bytes(data)
                        # ====== 后期 PIL USM 锐化（补 SDXL 4x 超分后的边缘软化） ======
                        try:
                            from PIL import Image, ImageFilter
                            im = Image.open(out).convert('RGB')
                            sharp = im.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3))
                            sharp.save(out, 'JPEG', quality=95, optimize=True)
                            import os
                            os.replace(out, out)  # 刷新大小
                            print(f"  [{tag}] USM锐化 {out.stat().st_size/1024/1024:.1f}MB", flush=True)
                        except Exception as e:
                            print(f"  [{tag}] USM失败 原图保留 {e}", flush=True)
                        print(f"  [{tag}] OK {out.stat().st_size/1024/1024:.1f}MB", flush=True); return True
                elif rec.get("status", {}).get("error"):
                    print(f"  [{tag}] COMFY错误 {rec['status'].get('error')}", flush=True); return False
        except Exception as e:
            print(f"  [{tag}] 轮询异常 {e}", flush=True)
        if i % 6 == 0: print(f"    [{tag}] {i*5}s...", flush=True)
    print(f"  [{tag}] TIMEOUT 跳过重试", flush=True); return False


def main():
    wants = sys.argv[1:] if len(sys.argv) > 1 else ["eagle_2"]
    out = PROJECT_ROOT / "jobs" / "smoke_v139"
    out.mkdir(parents=True, exist_ok=True)
    for want in wants:
        ref = next((r for r in REFS if r["id"] == want), None)
        if not ref:
            print(f"未知 ref_id={want}，可选: {[r['id'] for r in REFS]}"); continue
        print(f"--- {want} ---", flush=True)
        gen(ref, SEED, out)
    print("ALL done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
