"""v179 denim_3 回收（按用户 2026-09-01 批准）：

方向：保留 v178 的干净完整蝴蝶形态，把「牛仔布料/斜纹肌理/车缝线/多色阶」拉回原图设计语言。
- 形态：INTACT/CLEAN/COMPLETE 对称蝴蝶（v178 已对，不动）
- 材质：从「蓝色插画」→「真牛仔布料剪成蝴蝶形」
  · 整片 DENIM 单块布剪成蝴蝶（NOT patchwork/拼布片）
  · 可见 diagonal TWILL WEAVE 斜纹肌理
  · 翅膀分 2-3 种 indigo 色阶（深靛 #1B2A4A / 中蓝 #4A6FA5 / 浅洗 #B8C5D6）
  · 色块之间用 STITCHED SEAM 缝合线（NOT frayed/torn 撕裂）
  · 翼脉 = 浅黄车缝线（NOT 画线）
  · 翅膀边缘 = top-stitched hem 车缝包边（NOT raw edge）
  · 显式禁：torn/frayed/patchwork/ripped/loose threads/raw edges

参数锁死（铁律「单变量」）：denoise 0.80 / tile 0.60 / canny 0.25 / IPA advanced 0.18 /
                add-detail-xl LoRA=0 / ProteusV0.4 / KSampler 24+20 / 4x_NMKD-Siax_200k / SEED=700401
单变量改动：仅 denim 蝴蝶区域提示（主体+右上小蝴蝶+左下小蝴蝶+顶部 banner）从「插画」→「真牛仔布料」。
"""
import sys, time, json, requests, shutil, copy
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smoke_v164_per_image as v164
from PIL import Image, ImageFilter

OUT = HERE.parent / "outputs" / "v179"
OUT.mkdir(parents=True, exist_ok=True)
FINAL = HERE.parent / "outputs" / "final"
GALLERY = HERE.parent / "web_gallery" / "img"
COMFY = v164.COMFYUI

# ==========================================================================
# denim_3 — 干净完整蝴蝶形态 + 真牛仔布料材质
# ==========================================================================
DENIM_OVERRIDES = {
    "global_pos": (
        "denim fabric illustration on a clean bright WHITE background, "
        "indigo blue denim and bleached light-wash denim tones only, "
        "ONE large INTACT symmetrical butterfly CUT FROM REAL DENIM FABRIC as the hero subject, "
        "classic denim diagonal twill weave texture visible everywhere, "
        "clean sharp edges, pale-yellow top-stitching detail, "
        "no text, no letters, no words anywhere, "
        "premium fashion textile print quality, real denim appliqué look"
    ),
    "regions": [
        # 主体：完整对称蝴蝶 — 真牛仔布料剪裁 + 车缝包边 + 斜纹肌理 + 多色阶缝合
        {"x": 0.20, "y": 0.40, "w": 0.60, "h": 0.45, "strength": 1.35,
         "prompt": (
            "ONE large INTACT symmetrical butterfly with a CLEAN complete silhouette, "
            "two pairs of wings — upper forewings larger teardrop-shaped, "
            "lower hindwings rounded and smaller, "
            "TWO slender antennae extending upward, a slim soft body in the center, "
            "the entire butterfly is CUT FROM REAL DENIM FABRIC — a single continuous piece of indigo denim "
            "shaped into a perfect butterfly silhouette "
            "(NOT patchwork, NOT fabric scraps, NOT stitched-together pieces), "
            "clear visible diagonal TWILL WEAVE pattern across the wings, classic denim texture, "
            "TONAL DENIM BLOCKING: 2 to 3 distinct shades of indigo on the wings "
            "(deep indigo #1B2A4A on the upper forewings, mid-wash blue #4A6FA5 on the lower hindwings, "
            "and bleached light blue #B8C5D6 as accent panels), "
            "blocks separated by clean pale-yellow STITCHED SEAMS (classic double-needle denim stitching), "
            "NOT separated by torn edges, NOT separated by frayed gaps — seams ONLY, "
            "the wing vein lines are subtle STITCHED SEAM lines in pale yellow thread, not painted lines, "
            "CLEAN smooth wing edges finished with a top-stitched hem (like a real garment hem), "
            "the butterfly is WHOLE and COMPLETE — NOT torn, NOT frayed, NOT patchwork, NOT ripped, "
            "no loose threads, no raw edges, no exposed fraying anywhere on the silhouette, "
            "soft drop shadow under the wings on the white background. " + v164.COHESIVE
         )},
        # 右上小蝴蝶 — 浅洗牛仔 + 一小片深靛点缀
        {"x": 0.65, "y": 0.20, "w": 0.20, "h": 0.20, "strength": 1.10,
         "prompt": (
            "a SECONDARY smaller INTACT butterfly in the upper-right, "
            "about one-third the size of the main butterfly, "
            "diagonally tilted in flight pose, "
            "CUT FROM LIGHTER BLEACHED DENIM with one small deep-indigo accent panel, "
            "visible diagonal twill weave, pale-yellow top-stitching along edges, "
            "complete smooth wings no frays no tears no raw edges, "
            "soft shadow underneath. " + v164.COHESIVE
         )},
        # 左下小蝴蝶 — 中蓝牛仔
        {"x": 0.10, "y": 0.78, "w": 0.20, "h": 0.15, "strength": 1.00,
         "prompt": (
            "a TINY INTACT butterfly in the bottom-left, "
            "smallest of the three, flying in the opposite direction, "
            "CUT FROM mid-wash indigo DENIM with visible diagonal twill weave, "
            "complete smooth wings no frays no tears. " + v164.COHESIVE
         )},
        # 蓝色虚线轨迹（装饰）— 保留 v178
        {"x": 0.30, "y": 0.30, "w": 0.65, "h": 0.50, "strength": 0.95,
         "prompt": (
            "a trailing arc of 12 small indigo dots forming a sweeping S-curve "
            "from the small upper-right butterfly DOWN to the main butterfly "
            "and continuing to the bottom-left tiny butterfly, "
            "dots evenly spaced, soft fabric-ink feel. " + v164.COHESIVE
         )},
        # 顶部 banner — 真牛仔布料 + 斜纹肌理（不放字）
        {"x": 0.30, "y": 0.05, "w": 0.40, "h": 0.10, "strength": 1.00,
         "prompt": (
            "a small horizontal rectangular banner area at the very top center, "
            "CUT FROM indigo DENIM FABRIC with visible diagonal twill weave, "
            "pale-yellow top-stitched border, "
            "EMPTY smooth flat surface (no text, no letters, no marks, no tassels), "
            "just a clean denim zone with rounded stitched corners. " + v164.COHESIVE
         )},
    ],
}


def build_denim():
    """LORA=0 + 干净蝴蝶形态 + 真牛仔布料材质 的 denim_3 管线（其余完全复用 v164.build）。"""
    ref = copy.deepcopy(next(r for r in v164.REFS if r["id"] == "denim_3"))
    ref["global_pos"] = DENIM_OVERRIDES["global_pos"]
    ref["regions"] = DENIM_OVERRIDES["regions"]
    g = v164.build(ref, v164.SEED)
    # 拆 LoRA（去掉高频纹理噪点 → 蝴蝶边缘干净）
    g["7"]["inputs"]["strength_model"] = 0.0
    g["7"]["inputs"]["strength_clip"] = 0.0
    # denoise/tile/canny/IPA 保持 v147 基线不动
    g["15"]["inputs"]["filename_prefix"] = "v179_denim_3"
    return g


print(f"\n[v179] denim_3 参数: LORA=0  DENOISE={v164.DENOISE}  TILE={v164.TILE_STRENGTH}  "
      f"CANNY={v164.CANNY_STRENGTH}  IPA=advanced {v164.IPA_WEIGHT}  upscaler=4x_NMKD-Siax")
print(f"[v179] 单变量改动: 仅 denim 区域提示 — 真牛仔布料剪裁 + 斜纹肌理 + 多色阶车缝")
print(f"[v179] 形态沿用 v178 (CLEAN 完整对称蝴蝶 + 禁 torn/frayed/patchwork)")
print(f"[v179] 提交 denim_3 ...", flush=True)

g = build_denim()
r = requests.post(f"{COMFY}/prompt",
                 json={"prompt": g, "client_id": f"v179_{int(time.time())}"},
                 timeout=15)
j = r.json()
if r.status_code != 200 or "error" in j:
    print("[ERR]", r.status_code, json.dumps(j)[:1500]); sys.exit(1)
pid = j.get("prompt_id")
print(f"[v179] denim_3 pid={pid}", flush=True)

done = False
for i in range(96):
    time.sleep(5)
    h = requests.get(f"{COMFY}/history/{pid}", timeout=10).json()
    if pid not in h:
        if i % 6 == 0: print(f"  ...{i*5}s", flush=True)
        continue
    rec = h[pid]
    st = rec.get("status", {})
    if st.get("completed"):
        imgs = rec.get("outputs", {}).get("15", {}).get("images", [])
        if imgs:
            fn = imgs[0]["filename"]
            sub = imgs[0].get("subfolder", "")
            data = requests.get(f"{COMFY}/view?filename={fn}&type=output&subfolder={sub}", timeout=60).content
            out = OUT / "v179_denim_3.jpg"
            out.write_bytes(data)
            im = Image.open(out).convert("RGB")
            im.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)).save(out, quality=95)
            print(f"[v179] ✓ {out} {out.stat().st_size/1024/1024:.1f}MB")
            done = True
            break
    if "error" in st:
        print("[ERR]", st); break

if not done:
    print("[ERR] timeout"); sys.exit(1)

# 复制到 final + gallery
shutil.copyfile(out, FINAL / "denim_3.jpg")
shutil.copyfile(out, GALLERY / "fiss_denim_3.jpg")
print(f"[v179] ✓ denim_3 → final + gallery")

# 量化自检
from selfcheck_metrics import color_intersect, ssim, frag_ratio
orig = GALLERY / "orig_denim_3.jpg"
ci = color_intersect(str(orig), str(out))
sm = ssim(str(orig), str(out))
fr = frag_ratio(str(orig), str(out))
print(f"[v179] denim_3 metrics: color∩={ci:.3f}  ssim={sm:.3f}  fragR={fr:.3f}")

# camo 仍用 v174
v174_camo = HERE.parent / "outputs" / "v174" / "v174_camo_4.jpg"
shutil.copyfile(v174_camo, FINAL / "camo_4.jpg")
shutil.copyfile(v174_camo, GALLERY / "fiss_camo_4.jpg")
orig_camo = GALLERY / "orig_camo_4.jpg"
ci_c = color_intersect(str(orig_camo), str(v174_camo))
sm_c = ssim(str(orig_camo), str(v174_camo))
fr_c = frag_ratio(str(orig_camo), str(v174_camo))
print(f"[v179] camo_4  metrics: color∩={ci_c:.3f}  ssim={sm_c:.3f}  fragR={fr_c:.3f}  (← v174)")

print("\n[v179] DONE")
print(f"  final/camo_4.jpg   ← v174 (圆润+配色)")
print(f"  final/denim_3.jpg  ← v179 (CLEAN 蝴蝶 + 真牛仔布料剪裁)")