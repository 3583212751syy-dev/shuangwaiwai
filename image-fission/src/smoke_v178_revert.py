"""v178 撤回 + 修复（按用户 2026-09-01 反馈）：

根因诊断：
  camo_4  v177 用 IPAdapterPreciseStyleTransfer + ColorTransfer 把圆润有机迷彩
          变成硬多边形 + 棕褐单色（丢深橄榄绿），违反「配色必须与原图一致」铁律。
          → 直接撤回 v174（LORA=0 + v147 其余），圆润 + 深绿/棕/沙色齐全。

  denim_3 v164/v177 蝴蝶都是破布拼贴。罪魁是 v164 蝴蝶提示自己写了
          "layered DENIM PATCHES, irregular fabric pieces with visible raw frayed edges"
          ——SDXL 忠实地按提示做了破布（不是 AI 翻车，是提示写反了）。
          → 拆 LoRA（去掉高频纹理噪点）+ 蝴蝶提示重写为 INTACT/CLEAN
            （明确禁止 torn/frayed/patchwork/ripped）+ global_pos 去掉 collage/patchwork。

参数锁死：denoise 0.80 / tile 0.60 / canny 0.25 / IPA advanced 0.18 / ProteusV0.4
         KSampler 24+20 / 4x_NMKD-Siax_200k / SEED=700401
单变量原则：camo 改 LoRA(=0)；denim 改 LoRA(=0) + 提示（蝴蝶区域 + global_pos）。
"""
import sys, time, json, requests, shutil, copy
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smoke_v164_per_image as v164
from PIL import Image, ImageFilter

OUT = HERE.parent / "outputs" / "v178"
OUT.mkdir(parents=True, exist_ok=True)
FINAL = HERE.parent / "outputs" / "final"
GALLERY = HERE.parent / "web_gallery" / "img"
COMFY = v164.COMFYUI

# ==========================================================================
# 1) camo_4 — 直接采用 v174（圆润 + v147 配色，已验证符合用户预期）
# ==========================================================================
v174_camo = HERE.parent / "outputs" / "v174" / "v174_camo_4.jpg"
assert v174_camo.exists(), f"v174 camo 不存在: {v174_camo}"
shutil.copyfile(v174_camo, FINAL / "camo_4.jpg")
shutil.copyfile(v174_camo, GALLERY / "fiss_camo_4.jpg")
print(f"[v178] camo_4 ← v174 ({v174_camo.stat().st_size/1024/1024:.1f}MB)  圆润+深绿/棕/沙色")

# ==========================================================================
# 2) denim_3 — LORA=0 + 蝴蝶提示重写为 INTACT/CLEAN
# ==========================================================================
DENIM_OVERRIDES = {
    "global_pos": (
        "denim illustration on a clean bright WHITE background, "
        "indigo blue denim and soft cream tones, "
        "ONE large clean INTACT symmetrical butterfly as the hero subject, "
        "high contrast, clean sharp edges, soft natural shadows, "
        "no text, no letters, no words anywhere, "
        "fashion textile print quality"
    ),
    "regions": [
        # 主体：完整干净蝴蝶（关键 — 禁所有破布词）
        {"x": 0.20, "y": 0.40, "w": 0.60, "h": 0.45, "strength": 1.35,
         "prompt": (
            "ONE large INTACT symmetrical butterfly with a CLEAN complete silhouette, "
            "two pairs of wings — upper forewings larger teardrop-shaped, "
            "lower hindwings rounded and smaller, "
            "TWO slender antennae extending upward, a slim soft body in the center, "
            "wings rendered in smooth indigo blue DENIM with subtle diagonal twill weave texture, "
            "delicate pale-white stitching forming wing vein patterns, "
            "CLEAN smooth wing edges, "
            "the butterfly is WHOLE and COMPLETE — NOT torn, NOT frayed, NOT patchwork, NOT ripped, "
            "no loose threads escaping the wing edges, "
            "soft drop shadow under the wings on the white background. " + v164.COHESIVE
         )},
        # 右上小蝴蝶（干净）
        {"x": 0.65, "y": 0.20, "w": 0.20, "h": 0.20, "strength": 1.10,
         "prompt": (
            "a SECONDARY smaller INTACT butterfly in the upper-right, "
            "about one-third the size of the main butterfly, "
            "diagonally tilted in flight pose, "
            "lighter bleached denim tone with one small indigo accent, "
            "CLEAN complete wings with no frays no tears, "
            "soft shadow underneath. " + v164.COHESIVE
         )},
        # 左下小蝴蝶
        {"x": 0.10, "y": 0.78, "w": 0.20, "h": 0.15, "strength": 1.00,
         "prompt": (
            "a TINY INTACT butterfly in the bottom-left, "
            "smallest of the three, flying in the opposite direction, "
            "clean indigo denim with thin white stitching, "
            "complete smooth wings no frays. " + v164.COHESIVE
         )},
        # 蓝色虚线轨迹（装饰）
        {"x": 0.30, "y": 0.30, "w": 0.65, "h": 0.50, "strength": 0.95,
         "prompt": (
            "a trailing arc of 12 small indigo dots forming a sweeping S-curve "
            "from the small upper-right butterfly DOWN to the main butterfly "
            "and continuing to the bottom-left tiny butterfly, "
            "dots evenly spaced, soft fabric-ink feel. " + v164.COHESIVE
         )},
        # 顶部：干净色块（不要 tassel 避免碎感，给文字留位）
        {"x": 0.30, "y": 0.05, "w": 0.40, "h": 0.10, "strength": 1.00,
         "prompt": (
            "a small clean horizontal indigo denim banner area at the very top center, "
            "EMPTY smooth flat surface (no text, no letters, no marks, no tassels), "
            "just a clean colored zone with soft rounded edges. " + v164.COHESIVE
         )},
    ],
}


def build_denim():
    """LORA=0 + 干净蝴蝶提示 的 denim_3 管线（其余完全复用 v164.build）。"""
    ref = copy.deepcopy(next(r for r in v164.REFS if r["id"] == "denim_3"))
    ref["global_pos"] = DENIM_OVERRIDES["global_pos"]
    ref["regions"] = DENIM_OVERRIDES["regions"]
    g = v164.build(ref, v164.SEED)
    # 拆 LoRA（去掉高频纹理噪点 → 蝴蝶边缘干净）
    g["7"]["inputs"]["strength_model"] = 0.0
    g["7"]["inputs"]["strength_clip"] = 0.0
    # denoise 保持 v147 = 0.80
    g["15"]["inputs"]["filename_prefix"] = "v178_denim_3"
    return g


print(f"\n[v178] denim_3 参数: LORA=0  DENOISE={v164.DENOISE}  TILE={v164.TILE_STRENGTH}  "
      f"CANNY={v164.CANNY_STRENGTH}  IPA=advanced {v164.IPA_WEIGHT}  upscaler=4x_NMKD-Siax")
print(f"[v178] 关键改动: 蝴蝶提示重写为 INTACT/CLEAN（明确禁 torn/frayed/patchwork/ripped）")
print(f"[v178] 提交 denim_3 ...", flush=True)

g = build_denim()
r = requests.post(f"{COMFY}/prompt",
                 json={"prompt": g, "client_id": f"v178_{int(time.time())}"},
                 timeout=15)
j = r.json()
if r.status_code != 200 or "error" in j:
    print("[ERR]", r.status_code, json.dumps(j)[:1500]); sys.exit(1)
pid = j.get("prompt_id")
print(f"[v178] denim_3 pid={pid}", flush=True)

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
            out = OUT / "v178_denim_3.jpg"
            out.write_bytes(data)
            im = Image.open(out).convert("RGB")
            im.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)).save(out, quality=95)
            print(f"[v178] ✓ {out} {out.stat().st_size/1024/1024:.1f}MB")
            done = True
            break
    if "error" in st:
        print("[ERR]", st); break

if not done:
    print("[ERR] timeout"); sys.exit(1)

# 复制到 final + gallery
shutil.copyfile(out, FINAL / "denim_3.jpg")
shutil.copyfile(out, GALLERY / "fiss_denim_3.jpg")
print(f"[v178] ✓ denim_3 → final + gallery")

# 量化自检（color∩ / ssim / frag_ratio）
from selfcheck_metrics import color_intersect, ssim, frag_ratio
orig = GALLERY / "orig_denim_3.jpg"
ci = color_intersect(str(orig), str(out))
sm = ssim(str(orig), str(out))
fr = frag_ratio(str(orig), str(out))
print(f"[v178] denim_3 metrics: color∩={ci:.3f}  ssim={sm:.3f}  fragR={fr:.3f}")

# camo 指标（用 v174）
orig_camo = GALLERY / "orig_camo_4.jpg"
ci_c = color_intersect(str(orig_camo), str(v174_camo))
sm_c = ssim(str(orig_camo), str(v174_camo))
fr_c = frag_ratio(str(orig_camo), str(v174_camo))
print(f"[v178] camo_4  metrics: color∩={ci_c:.3f}  ssim={sm_c:.3f}  fragR={fr_c:.3f}  (← v174)")

print("\n[v178] DONE")
print(f"  final/camo_4.jpg   ← v174 (圆润+配色)")
print(f"  final/denim_3.jpg  ← v178 (LORA=0 + INTACT 蝴蝶)")
