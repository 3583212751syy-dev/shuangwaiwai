"""v142 大裂变+保细节（本机 ComfyUI / SDXL）：img2img(VAEEncode 锁大构图) + IPAdapter style(锁配色画风)
+ ControlNet Canny 0.50(防细节崩) + Detail Tweaker XL LoRA(强化细节) + 双 KSampler(denoise 重排) + USM 后处理。
用户定义"裂变"= 元素角度/大小占比/姿态明显变化，或换成同类相关元素；不是保留原图元素只改纹理。
构图=整体版式能量保留；元素结构=同类型(鹰还是鹰/骷髅还是骷髅)；配色严格锁死；不换物种/不加新色。

新增(08-29)：Detail Tweaker XL LoRA 从 Civitai 下载(add-detail-xl.safetensors 217MB)；
            ControlNet Tile 暂不集成（TTPlanet 文件 1.07GB fp32 会让 12G 显存爆 OOM）。

用法：python smoke_v139_samesubject.py [ref_id]   (默认 eagle_2)
输出：jobs/smoke_v139/v139_{id}.jpg
"""
import time, requests, sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

# v142 调参（修复 eagle_2 v141 局部元素失真：鹰脸/小铁链/徽章中心/小元素崩）：
# - CN 0.20→0.50：恢复轮廓约束但仍允许重排角度/大小占比（不能再弱 Canny，弱了=细节崩）
# - 加 Detail Tweaker XL LoRA（0.7）：专防高细节区域失真
# - denoise 0.78 保持：仍允许元素重排
# - IPA 0.32 保持
SEED = 700401
CKPT = "ProteusV0.4.safetensors"
DENOISE = 0.78
CN_STRENGTH = 0.50
IPA_WEIGHT = 0.32
LORA_DETAIL = 0.7  # Detail Tweaker XL 权重（+加强细节；负向=减少细节）

# 同主体 prompt：subject 严格描述原图“同一类”元素（鹰还是鹰/棕榈还是棕榈/蝴蝶还是蝴蝶），
# 不替换为别的主体；palette 锁死原图配色，pos/neg 都强调“不加新色、不换主体”。
REFS = [
    {"id": "eagle_2", "ref_img": "pinterest_eagle_2.jpg", "weight": IPA_WEIGHT,
     "subject": ("a bald eagle with spread wings and talons, surrounded by red flames, "
                 "three human skulls below, draped iron chains, gothic heraldic crest"),
     "palette": ("gothic tattoo illustration, pure black background, white and silver eagle, "
                 "red and orange flames, gray iron chains"),
     "variant": ("RECOMPOSE the layout with BIG changes in ANGLE and SCALE: "
                 "the eagle is now seen from a SIDE PROFILE diving downward with one wing swept up and one tucked (not symmetric spread), "
                 "the three skulls are REPLACED by ONE large cracked skull placed at the CENTER as the dominant focal element (much larger proportion), "
                 "the red flames now ERUPT from the BOTTOM-LEFT corner as a sweeping vertical column instead of wrapping all around, "
                 "the iron chains are REWOVEN into a DIAGONAL swag across the upper right, "
                 "keep the same gothic black-red-silver palette and heraldic energy, but the elements are clearly rearranged and re-angled")},
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
    "but RECOMPOSE with MAJOR changes in element ANGLE, SCALE and PROPORTION, "
    "reposition and re-angle the elements, replace with RELATED same-category elements "
    "in different poses or different arrangements, "
    "the result must look like a clearly DIFFERENT design that is still obviously related to the original, "
    "masterpiece best quality ultra detailed, "
    "anatomically correct and physically coherent subjects, "
    "intricate craftsmanship and fine engraved details on every element, "
    "ULTRA sharp crisp high-contrast edges, "
    "every element surrounded by a CLEAR solid-black separating outline silhouette, "
    "elements NEVER bleed into neighbors, "
    "bold graphic t-shirt print composition full bleed edge-to-edge, "
    "no halftone no noise no grain no smudge no watercolor no soft airbrush"
)
NEG_BASE = (
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    "text, letters, words, writing, typography, signature, caption, label, paragraph, "
    "3d, photographic, painterly, illustration by child, beginner drawing, "
    "blur, soft focus, smooth shading, smudge, soft airbrush, watercolor, "
    "bleeding borders, fused elements, melted edges, soft halo, gradient transition, "
    "out of focus, dreamy, ethereal, foggy, hazy, low contrast, pastel, "
    "small subject, distant view, zoomed out, far away, miniature, tiny, "
    "noise, grain, pixelated, jagged edges, aliasing, duplicate image, exact copy, watermark, "
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
        "image": ["2", 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 64}}
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
