"""
批量风格裂变：6 张 Pinterest 原图 × 4 主题 = 24 张，全部加 vector LoRA。
治"糊"用 Doctor Diffusion Vector Art XL LoRA（trigger word: vector + black line art）。
"""
import os
import sys
import time
import shutil
import base64

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.comfy_client import ComfyClient
from pipelines.build import build_mode1
import config as _cfg

_cfg.SDXL_CHECKPOINT = "juggernautXL_ragnarokBy.safetensors"

COMFYUI_INPUT = r"E:\Desktop\双接口\image-fission\ComfyUI\input"
JOBS_BASE = r"E:\Desktop\双接口\image-fission\jobs"

# 4 个"完全同源"主题变体：保持原图色调+构图+画风，只换具体元素
# 关键：prompt 强调"keep the same color palette and composition" 让 IP-Adapter 0.75 锁住原图
SUBJECTS = [
    ("botanical",  "vector, black line art, keep the same color palette and composition as the original, an intricate ornamental pattern with different botanical motifs and stylized plants, monochrome, decorative symmetrical, no text"),
    ("creature",   "vector, black line art, keep the same color palette and composition as the original, an intricate ornamental pattern with heraldic creature and detailed beast elements, monochrome, decorative symmetrical, no text"),
    ("geometry",   "vector, black line art, keep the same color palette and composition as the original, an intricate ornamental pattern with sacred geometry stars circles and crosses, monochrome, decorative symmetrical, no text"),
    ("ornamental", "vector, black line art, keep the same color palette and composition as the original, an ornate ornamental pattern with corner flourishes and scrollwork, monochrome, decorative symmetrical, no text"),
]

# 6 张原图
ORIGINALS = [
    ("pinterest_illust_1", "illust_1"),
    ("pinterest_eagle_2",  "eagle_2"),
    ("pinterest_denim_3",  "denim_3"),
    ("pinterest_camo_4",   "camo_4"),
    ("pinterest_skull_5",  "skull_5"),
    ("pinterest_metal_6",  "metal_6"),
]

LORA = "DD-vector-v2.safetensors"
LORA_STRENGTH = 0.85
SIM = 0.75  # 提高到 0.75，强锁原图风格/色调/构图
IPA_WEIGHT_TYPE = "style transfer"   # SDXL 风格锁（IPAdapter Plus 1.5+ 推荐）
IPA_NOISE = 0.10                     # 加噪防止完全照搬
IPA_END = 0.85                       # 前 85% 强影响，后 15% 让模型自由发挥


def img_to_b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def main():
    out_dir = os.path.join(JOBS_BASE, f"batch_6x4_vectorlora_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")
    print(f"[cfg] LoRA={LORA}@{LORA_STRENGTH}  sim={SIM}")

    client = ComfyClient()
    base_seed = 8888
    results = []  # (orig_label, sub_label, path)
    t_start = time.time()

    for orig_seed, orig_label in ORIGINALS:
        seed_src = os.path.join(COMFYUI_INPUT, f"{orig_seed}.jpg")
        if not os.path.exists(seed_src):
            print(f"[WARN] 跳过: {seed_src} 不存在")
            continue
        # 复制 seed 到 input（用带时间戳名字，避免冲突）
        seed_name = f"batch_{orig_label}_{int(time.time()*1000)}.jpg"
        shutil.copy(seed_src, os.path.join(COMFYUI_INPUT, seed_name))

        for sub_label, sub_prompt in SUBJECTS:
            params = {
                "similarity": SIM,
                "lora_name": LORA,
                "lora_strength": LORA_STRENGTH,
                "ipadapter_weight_type": IPA_WEIGHT_TYPE,
                "ipadapter_noise": IPA_NOISE,
                "ipadapter_end": IPA_END,
                "usdu_model": "4x_NMKD-Siax_200k.pth",  # 治糊：4x 真实超分
                "negative_prompt": ("color, colorful, photorealistic, photo, photography, "
                                     "text, watermark, signature, blurry, deformed, low quality, "
                                     "3d, render, gradient, soft, smooth shading, "
                                     "person, face, human, animal, building, vehicle, "
                                     "different style, different color palette"),
                "width": 768, "height": 1344, "batch_per_run": 1,
                "steps": 30, "cfg": 5.0,
                "hires_scale": 1.5, "hires_denoise": 0.40, "hires_steps": 20,
                "seed": base_seed + hash((orig_label, sub_label)) % 9999,
                "style_prompt": sub_prompt,
            }
            g = build_mode1(seed_name, params, f"batch_{orig_label}_{sub_label}")
            t0 = time.time()
            try:
                res = client.run(g, timeout=300)
                data = next(iter(res.values()))[0]
                out_path = os.path.join(out_dir, f"{orig_label}_{sub_label}.jpg")
                with open(out_path, "wb") as f:
                    f.write(data)
                dt = time.time() - t0
                print(f"[OK ] {orig_label} × {sub_label}  {dt:.0f}s  {len(data)} B")
                results.append((orig_label, sub_label, out_path))
            except Exception as e:
                print(f"[FAIL] {orig_label} × {sub_label}: {repr(e)}")

    total = time.time() - t_start
    print(f"\n[DONE] {len(results)}/24  总耗时 {total:.0f}s")

    if not results:
        return

    # 画廊：6 行（原图）× 4 列（主题）+ 顶部原图
    rows_html = []
    for orig_seed, orig_label in ORIGINALS:
        orig_path = os.path.join(COMFYUI_INPUT, f"{orig_seed}.jpg")
        if not os.path.exists(orig_path):
            continue
        seed_b64 = img_to_b64(orig_path)
        cells = ""
        for sub_label, _ in SUBJECTS:
            img = next((p for o, s, p in results if o == orig_label and s == sub_label), None)
            if img:
                cells += f'<td><img src="data:image/jpeg;base64,{img_to_b64(img)}"><div class="cap">{sub_label}</div></td>'
            else:
                cells += f'<td class="empty">—</td>'
        rows_html.append(f"""
        <tr>
          <th>{orig_label}</th>
          <td class="seed"><img src="data:image/jpeg;base64,{seed_b64}"><div class="cap">原图</div></td>
          {cells}
        </tr>""")

    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>6x4 批量风格裂变</title>
<style>
body{{font-family:system-ui;background:#fafafa;padding:24px;margin:0;}}
h1{{margin:0 0 8px;font-size:22px;}}.sub{{color:#666;margin-bottom:24px;}}
.params{{background:#222;color:#fff;padding:10px 14px;border-radius:6px;display:inline-block;font-family:monospace;font-size:12px;margin-bottom:18px;}}
table{{border-collapse:collapse;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.1);width:100%;}}
th,td{{padding:6px;text-align:center;vertical-align:top;border-bottom:1px solid #eee;}}
th{{background:#f4f4f4;font-size:13px;text-align:left;padding-left:12px;}}
td img{{max-width:200px;display:block;margin:0 auto;}}
td.seed img{{max-width:120px;}}
.cap{{font-size:11px;color:#666;padding-top:4px;}}
.empty{{color:#aaa;}}
</style></head><body>
<h1>图裂变 · 6 原图 × 4 主题批量（Vector LoRA 治糊）</h1>
<div class="sub">6 张 Pinterest 设计稿 → 4 个新主题 = 24 张同画风不同内容</div>
<div class="params">LoRA = {LORA} (strength {LORA_STRENGTH}) | similarity = {SIM} | prompt trigger: vector, black line art, white background</div>
<table>
  <tr><th>原图</th><th>原图</th><th>damask</th><th>creature</th><th>geometry</th><th>frame</th></tr>
  {''.join(rows_html)}
</table>
</body></html>"""
    html_path = os.path.join(out_dir, "gallery.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[gallery] {html_path}")


if __name__ == "__main__":
    main()