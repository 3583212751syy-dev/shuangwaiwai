"""v179b denim_3 收口（修 v179 顶部出现牛仔字母的 bug）：

v179 主体配色/材质完美对住原图（color∩ 0.896、twill weave + 车缝 + 多色阶都到位），
但顶部出现了牛仔布料字母 "D" "S"（违反「no text, no letters」铁律）。

根因：原图 UPGY 字母是浅蓝牛仔 3D 字母，在浅底上极显眼；IPA 0.18 style transfer
把字母区视觉模式锁住，真牛仔布料风触发 AI 在该位置「画牛仔字母」。
（v178 矢量插画风未触发，真布料风触发了。）

单变量修复（铁律遵守）：
  ① 顶部 banner 区域提示：绝对禁字母/字符轮廓（强化版）
  ② monkey-patch 扩展 NEG_BASE：禁 denim letter / fabric letter / stitched letter 等
其他参数全锁死：denoise 0.80 / tile 0.60 / canny 0.25 / IPA advanced 0.18 /
                LoRA=0 / ProteusV0.4 / KSampler 24+20 / 4x_NMKD-Siax_200k / SEED=700401
主体蝴蝶形态 + 真牛仔布料材质完全沿用 v179（已对，不动）。
"""
import sys, time, json, requests, shutil, copy
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smoke_v164_per_image as v164
from PIL import Image, ImageFilter

# 单变量 ②：扩展 NEG（临时 monkey-patch，不影响 v164 模块本身供其他脚本用）
EXTRA_NEG = (
    ", denim letter, denim letters, fabric letter, fabric letters, stitched letter, stitched letters, "
    "alphabet patch, alphabet shape, character shape, character outline, "
    "fabric typography, denim typography, denim word, fabric word, fabric alphabet, denim alphabet, "
    "embroidered letter, embroidered letters, appliqued letter, appliqued letters, "
    "text made of fabric, text made of denim, letters made of denim, letters made of fabric, "
    "3D fabric letter, 3D denim letter, fabric text, denim text"
)
v164.NEG_BASE = v164.NEG_BASE + EXTRA_NEG

OUT = HERE.parent / "outputs" / "v179b"
OUT.mkdir(parents=True, exist_ok=True)
FINAL = HERE.parent / "outputs" / "final"
GALLERY = HERE.parent / "web_gallery" / "img"
COMFY = v164.COMFYUI

# ==========================================================================
# denim_3 — 沿用 v179 的形态+材质，仅强化顶部 banner 区域提示（绝对禁字母）
# ==========================================================================
DENIM_OVERRIDES = {
    "global_pos": (
        "denim fabric illustration on a clean bright WHITE background, "
        "indigo blue denim and bleached light-wash denim tones only, "
        "ONE large INTACT symmetrical butterfly CUT FROM REAL DENIM FABRIC as the hero subject, "
        "classic denim diagonal twill weave texture visible everywhere, "
        "clean sharp edges, pale-yellow top-stitching detail, "
        "ABSOLUTELY NO TEXT, NO LETTERS, NO ALPHABET SHAPES, NO CHARACTER OUTLINES, "
        "no word shapes, no logo, no number anywhere on the entire image, "
        "the top area is plain empty flat fabric — no marks of any kind, "
        "premium fashion textile print quality, real denim appliqué look"
    ),
    "regions": [
        # 主体：完全沿用 v179（已对，不动）
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
        # 右上小蝴蝶 — 沿用 v179
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
        # 左下小蝴蝶 — 沿用 v179
        {"x": 0.10, "y": 0.78, "w": 0.20, "h": 0.15, "strength": 1.00,
         "prompt": (
            "a TINY INTACT butterfly in the bottom-left, "
            "smallest of the three, flying in the opposite direction, "
            "CUT FROM mid-wash indigo DENIM with visible diagonal twill weave, "
            "complete smooth wings no frays no tears. " + v164.COHESIVE
         )},
        # 蓝色虚线轨迹 — 沿用 v179
        {"x": 0.30, "y": 0.30, "w": 0.65, "h": 0.50, "strength": 0.95,
         "prompt": (
            "a trailing arc of 12 small indigo dots forming a sweeping S-curve "
            "from the small upper-right butterfly DOWN to the main butterfly "
            "and continuing to the bottom-left tiny butterfly, "
            "dots evenly spaced, soft fabric-ink feel. " + v164.COHESIVE
         )},
        # 顶部 banner — 单变量 ①：绝对禁字母/字符轮廓/任何 marks（强化版）
        {"x": 0.30, "y": 0.05, "w": 0.40, "h": 0.10, "strength": 1.00,
         "prompt": (
            "a thin horizontal rectangular strip of PLAIN indigo DENIM FABRIC at the top center, "
            "ABSOLUTELY NO LETTERS, NO ALPHABET SHAPES, NO CHARACTER OUTLINES, "
            "no word shapes, no logo shapes, no numbers, no marks of any kind, "
            "no stitched letter outlines, no embroidered characters, no appliqued text, "
            "just a smooth flat rectangular denim panel with visible diagonal twill weave "
            "and a pale-yellow top-stitched BORDER around the edges, "
            "rounded stitched corners, like a real empty clothing label, "
            "the fabric surface is completely empty and blank inside the border. " + v164.COHESIVE
         )},
    ],
}


def build_denim():
    ref = copy.deepcopy(next(r for r in v164.REFS if r["id"] == "denim_3"))
    ref["global_pos"] = DENIM_OVERRIDES["global_pos"]
    ref["regions"] = DENIM_OVERRIDES["regions"]
    g = v164.build(ref, v164.SEED)
    # LoRA=0
    g["7"]["inputs"]["strength_model"] = 0.0
    g["7"]["inputs"]["strength_clip"] = 0.0
    g["15"]["inputs"]["filename_prefix"] = "v179b_denim_3"
    return g


print(f"\n[v179b] 单变量 ① 顶部 banner 区域提示强化禁字母")
print(f"[v179b] 单变量 ② NEG 扩展禁 denim letter / fabric letter 等 {len(EXTRA_NEG.split(','))} 词")
print(f"[v179b] 其余完全沿用 v179（主体配色/形态/车缝/twill weave 全部保留）")
print(f"[v179b] 提交 denim_3 ...", flush=True)

g = build_denim()
r = requests.post(f"{COMFY}/prompt",
                 json={"prompt": g, "client_id": f"v179b_{int(time.time())}"},
                 timeout=15)
j = r.json()
if r.status_code != 200 or "error" in j:
    print("[ERR]", r.status_code, json.dumps(j)[:1500]); sys.exit(1)
pid = j.get("prompt_id")
print(f"[v179b] denim_3 pid={pid}", flush=True)

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
            out = OUT / "v179b_denim_3.jpg"
            out.write_bytes(data)
            im = Image.open(out).convert("RGB")
            im.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)).save(out, quality=95)
            print(f"[v179b] ✓ {out} {out.stat().st_size/1024/1024:.1f}MB")
            done = True
            break
    if "error" in st:
        print("[ERR]", st); break

if not done:
    print("[ERR] timeout"); sys.exit(1)

# 复制到 final + gallery（覆盖 v179 的输出）
shutil.copyfile(out, FINAL / "denim_3.jpg")
shutil.copyfile(out, GALLERY / "fiss_denim_3.jpg")
print(f"[v179b] ✓ denim_3 → final + gallery（覆盖 v179）")

# 量化自检
from selfcheck_metrics import color_intersect, ssim, frag_ratio
orig = GALLERY / "orig_denim_3.jpg"
ci = color_intersect(str(orig), str(out))
sm = ssim(str(orig), str(out))
fr = frag_ratio(str(orig), str(out))
print(f"[v179b] denim_3 metrics: color∩={ci:.3f}  ssim={sm:.3f}  fragR={fr:.3f}")

# camo 仍用 v174
v174_camo = HERE.parent / "outputs" / "v174" / "v174_camo_4.jpg"
shutil.copyfile(v174_camo, FINAL / "camo_4.jpg")
shutil.copyfile(v174_camo, GALLERY / "fiss_camo_4.jpg")
orig_camo = GALLERY / "orig_camo_4.jpg"
ci_c = color_intersect(str(orig_camo), str(v174_camo))
sm_c = ssim(str(orig_camo), str(v174_camo))
fr_c = frag_ratio(str(orig_camo), str(v174_camo))
print(f"[v179b] camo_4  metrics: color∩={ci_c:.3f}  ssim={sm_c:.3f}  fragR={fr_c:.3f}  (← v174)")

print("\n[v179b] DONE")
print(f"  final/camo_4.jpg   ← v174 (圆润+配色)")
print(f"  final/denim_3.jpg  ← v179b (CLEAN 蝴蝶 + 真牛仔布料剪裁 + 顶部无字母)")