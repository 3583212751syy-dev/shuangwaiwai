"""v114 双锁裂变：颜色锁(style transfer) + 构图锁(composition) 同时挂原图，
内容由不同 subject prompt 变化 → 同色同构、异内容的裂变图。

用户要求：裂变图「有区别」，但「保留原图颜色与构图参考」。
- 颜色锁：IPAdapterAdvanced weight_type="style transfer"（锁配色/材质/笔触）
- 构图锁：IPAdapterAdvanced weight_type="composition"（锁布局/结构）
- 内容：每个变体不同 subject_word（裂变图因此有区别）
- ipadapter_noise=0.05：给内容留自由空间，避免完全照搬
- 可选 --controlnet >0：叠加 Canny 硬构图锁，进一步钉死原图边缘

底模 Proteus v0.4（与裂变管线一致），真实 4x 超分。

用法：
  python src/smoke_v114_duallock.py
  python src/smoke_v114_duallock.py --ref 我的图.png --count 6 --color 0.65 --comp 0.6
"""
import argparse
import os
import sys
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as _cfg
from pipelines.build import build_mode1
from demo_pattern_fission import img_to_b64
from engine.comfy_client import ComfyClient

COMFYUI_INPUT = _cfg.COMFYUI_INPUT if hasattr(_cfg, "COMFYUI_INPUT") else \
    r"E:\Desktop\双接口\image-fission\ComfyUI\input"
JOBS_BASE = _cfg.JOBS_DIR if hasattr(_cfg, "JOBS_DIR") else \
    r"E:\Desktop\双接口\image-fission\jobs"

# 默认黑白色调装饰图案风（与 demo_pattern_fission 同源），4 个不同 subject 演示「有区别」
SUBJECTS = [
    ("lily_vines",
     "an intricate black and white ornamental pattern with stylized lilies and winding vines, "
     "monochrome vector illustration, decorative symmetrical border, elegant linework, "
     "intricate botanical damask, no text"),
    ("rose_thorns",
     "a detailed black and white damask pattern with roses and thorny vines, "
     "monochrome high contrast, symmetrical composition, intricate linework, "
     "ornamental botanical engraving, no text"),
    ("chrysanthemum_foliage",
     "an ornamental black and white pattern with chrysanthemums and curling foliage, "
     "intricate symmetrical border, monochrome engraving style, "
     "decorative botanical motif, no text"),
    ("iris_flowing",
     "a decorative black and white pattern with irises and flowing botanical motifs, "
     "intricate symmetrical composition, monochrome, elegant linework, "
     "ornamental damask design, no text"),
]

NEG = (
    "color, colorful, photorealistic, photo, photography, "
    "text, watermark, signature, blurry, deformed, low quality, "
    "3d, render, gradient, soft, smooth shading, "
    "person, face, human, animal, building, vehicle"
)


def main():
    p = argparse.ArgumentParser(description="v114 双锁裂变验证（颜色+构图锁原图，内容变化）")
    p.add_argument("--ref", default=None, help="参考图文件名（在 ComfyUI/input 下，默认 pinterest_illust_1_hd.png）")
    p.add_argument("--count", "-n", type=int, default=4)
    p.add_argument("--color", type=float, default=0.6, help="颜色锁强度（默认 0.6）")
    p.add_argument("--comp", type=float, default=0.55, help="构图锁强度（默认 0.55）")
    p.add_argument("--noise", type=float, default=0.05, help="颜色锁噪点（默认 0.05）")
    p.add_argument("--end", type=float, default=0.85, help="IPAdapter 结束步（默认 0.85）")
    p.add_argument("--controlnet", type=float, default=0.0, help="Canny 硬构图锁强度（默认 0，>0 开启）")
    p.add_argument("--steps", type=int, default=45)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=100301)
    args = p.parse_args()

    ref = args.ref or "pinterest_illust_1_hd.png"
    ref_path = os.path.join(COMFYUI_INPUT, ref)
    if not os.path.exists(ref_path):
        print(f"[ERROR] 参考图不存在: {ref_path}")
        print(f"        可用参考图示例：将 pinterest_illust_1.jpg 先用 enhance_references.py 高清化得到 _hd.png")
        return 1

    subjects = SUBJECTS[:args.count]
    out_dir = os.path.join(JOBS_BASE, f"smoke_v114_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)

    base_params = {
        "color_strength": args.color,
        "composition_strength": args.comp,
        "ipadapter_noise": args.noise,
        "ipadapter_end": args.end,
        "negative_prompt": NEG,
        "width": 1024, "height": 1024,
        "batch_per_run": 1,
        "steps": args.steps, "cfg": args.cfg,
        "hires_scale": 1.5, "hires_denoise": 0.40, "hires_steps": 30,
    }
    if args.controlnet and args.controlnet > 0:
        base_params["controlnet_name"] = "controlnet-canny-sdxl-1.0.fp16.safetensors"
        base_params["controlnet_strength"] = args.controlnet
        base_params["controlnet_end"] = 0.9

    print(f"[v114] DUAL-LOCK  color={args.color}  comp={args.comp}  noise={args.noise}  end={args.end}")
    print(f"[ref]  {ref}")
    print(f"[out]  {out_dir}")

    client = ComfyClient()
    results = []
    for i, (name, prompt) in enumerate(subjects, 1):
        params = dict(base_params)
        params["seed"] = args.seed + i * 313
        params["style_prompt"] = prompt
        g = build_mode1(ref, params, f"v114_{name}")
        print(f"\n[{i}/{len(subjects)}] {name}")
        print(f"       {prompt[:80]}...")
        t0 = time.time()
        try:
            res = client.run(g, timeout=600)
            node_id, imgs = next(iter(res.items()))
            data = imgs[0]
            out_path = os.path.join(out_dir, f"{i:02d}_{name}.jpg")
            with open(out_path, "wb") as f:
                f.write(data)
            print(f"[OK] {time.time()-t0:.0f}s -> {os.path.basename(out_path)} ({len(data)} bytes)")
            results.append((name, out_path))
        except Exception as e:
            print(f"[FAIL] {name}: {repr(e)}")

    if not results:
        print("[ERROR] 没有任何图片生成")
        return 2

    seed_b64 = img_to_b64(ref_path)
    cards = "".join(
        f'<div class="card"><img src="data:image/jpeg;base64,{img_to_b64(p)}">'
        f'<div class="cap">{n}</div></div>'
        for n, p in results
    )
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>v114 双锁裂变</title>
<style>
body{{font-family:system-ui;background:#fafafa;padding:24px;margin:0;}}
h1{{margin:0 0 8px;font-size:22px;}}
.sub{{color:#666;margin-bottom:16px;}}
.lock{{background:#0a7;color:#fff;padding:6px 12px;border-radius:4px;display:inline-block;font-size:13px;font-weight:600;margin-bottom:20px;}}
.seed{{background:#fff;padding:16px;border-radius:8px;margin-bottom:24px;text-align:center;}}
.seed img{{max-width:300px;border:1px solid #ddd;}}
.grid{{display:grid;grid-template-columns:repeat({min(4, len(results))},1fr);gap:16px;}}
.card{{background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.1);}}
.card img{{width:100%;display:block;}}
.cap{{padding:8px;font-size:13px;color:#333;text-align:center;font-weight:500;}}
</style></head><body>
<h1>v114 · 双锁裂变（同色同构·异内容）</h1>
<div class="sub">颜色锁({args.color}) + 构图锁({args.comp}) 锁定原图 → 内容由 prompt 变化 → 裂变图有区别但同源</div>
<p><span class="lock">颜色锁 {args.color} · 构图锁 {args.comp} · noise {args.noise} · end {args.end}</span></p>
<div class="seed"><img src="data:image/jpeg;base64,{seed_b64}"><div class="cap">原图（参考）</div></div>
<div class="grid">{cards}</div>
</body></html>"""
    html_path = os.path.join(out_dir, "gallery.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[DONE] 画廊: {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
