"""
图裂变 CLI - 画面参考强度可调
用法：
  python fission.py --input 原图.jpg --similarity 0.55 --count 4
  python fission.py --input 原图.jpg -s 0.3 -p "百合" "玫瑰" "菊花" "鸢尾"
  python fission.py --input 原图.jpg -s 0.7 --count 6 --out ./my_out

参数：
  --input / -i     必需：原图路径
  --similarity / -s  画面参考强度 0-1（旧单 IPAdapter linear 模式，仅显式传此参时启用；
                    启用后关闭下方颜色/构图双锁，回退到单一权重）
  --color-strength   颜色/材质锁强度 0-1（IP-Adapter style transfer，默认 0.6；锁原图配色）
  --composition-strength  构图锁强度 0-1（IP-Adapter composition，默认 0.55；锁原图布局）
                    默认「颜色锁 + 构图锁」同时开 → 裂变图同色同构、异内容。
  --prompts / -p   新内容主题（不传则用默认花卉变体）
  --count / -n     生成张数（默认 4，受 prompts 数量限制）
  --out / -o       输出目录（默认 ./jobs/fission_<时间戳>）
  --seed           随机种子（默认 8888）
  --steps          采样步数（默认 35）
  --cfg            CFG 强度（默认 5.0）
"""
import argparse
import os
import sys
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.comfy_client import ComfyClient
from pipelines.build import build_mode1
from demo_pattern_fission import img_to_b64

# 默认 8 个花卉/植物变体（适合黑白矢量装饰图案风格）
DEFAULT_VARIANTS = [
    ("lily_vines", "an intricate black and white ornamental pattern with stylized lilies and winding vines, "
                  "monochrome vector illustration, decorative symmetrical border, elegant linework, "
                  "intricate botanical damask, no text"),
    ("rose_thorns", "a detailed black and white damask pattern with roses and thorny vines, "
                    "monochrome high contrast, symmetrical composition, intricate linework, "
                    "ornamental botanical engraving, no text"),
    ("chrysanthemum_foliage", "an ornamental black and white pattern with chrysanthemums and curling foliage, "
                              "intricate symmetrical border, monochrome engraving style, "
                              "decorative botanical motif, no text"),
    ("iris_flowing", "a decorative black and white pattern with irises and flowing botanical motifs, "
                     "intricate symmetrical composition, monochrome, elegant linework, "
                     "ornamental damask design, no text"),
    ("peony_lush", "a lush black and white peony pattern with full blooms and layered petals, "
                   "monochrome engraving, symmetrical composition, decorative botanical motif, no text"),
    ("vine_leaves", "a black and white ornamental vine pattern with grape leaves and curling tendrils, "
                    "monochrome decorative border, symmetrical, intricate linework, no text"),
    ("fern_spiral", "a decorative black and white fern and fiddlehead spiral pattern, "
                    "monochrome symmetrical, organic curves, intricate botanical design, no text"),
    ("tulip_spring", "an ornamental black and white tulip pattern with stylized spring blooms, "
                     "monochrome symmetrical border, decorative damask, no text"),
]

COMFYUI_INPUT = r"E:\Desktop\双接口\image-fission\ComfyUI\input"
JOBS_BASE = r"E:\Desktop\双接口\image-fission\jobs"

# 通用 negative prompt（治 SDXL 写实底子乱入装饰图）
NEG = ("color, colorful, photorealistic, photo, photography, "
       "text, watermark, signature, blurry, deformed, low quality, "
       "3d, render, gradient, soft, smooth shading, "
       "person, face, human, animal, building, vehicle")


def main():
    p = argparse.ArgumentParser(description="图裂变 - 风格裂变 CLI")
    p.add_argument("--input", "-i", required=True, help="原图路径")
    p.add_argument("--similarity", "-s", type=float, default=None,
                   help="画面参考强度 0-1（旧单 linear 模式；显式传此参会关闭颜色/构图双锁）")
    p.add_argument("--prompts", "-p", nargs="*", default=None,
                   help="新内容主题字符串列表（不传则用默认 8 个花卉变体）")
    p.add_argument("--count", "-n", type=int, default=4,
                   help="生成张数（默认 4）")
    p.add_argument("--out", "-o", default=None, help="输出目录")
    p.add_argument("--seed", type=int, default=8888)
    p.add_argument("--steps", type=int, default=35)
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=1344)
    p.add_argument("--lora", default=None, help="LoRA 文件名（ComfyUI/models/loras/ 下）")
    p.add_argument("--lora-strength", type=float, default=0.85, help="LoRA 强度 0-1（默认 0.85）")
    # IPAdapter 三通道：颜色锁 + 构图锁（默认同时开） + 内容(prompt)
    p.add_argument("--color-strength", type=float, default=0.6,
                   help="颜色/材质锁强度（IP-Adapter style transfer，默认 0.6；0=关闭）")
    p.add_argument("--style-strength", type=float, default=None,
                   help="(兼容别名) 等同 --color-strength")
    p.add_argument("--composition-strength", type=float, default=0.55,
                   help="构图锁强度（IP-Adapter composition，默认 0.55；0=关闭）")
    p.add_argument("--ipadapter-noise", type=float, default=0.05,
                   help="颜色锁噪点 0-0.5（防完全照搬、给内容留空间，推荐 0.05-0.15）")
    p.add_argument("--ipadapter-end", type=float, default=0.85,
                   help="IPAdapter 结束步 0-1（默认 0.85，末段放开让内容自由）")
    p.add_argument("--controlnet-strength", type=float, default=0.0,
                   help="可选 Canny 硬构图锁 0-1（默认 0 关闭；>0 时叠加，强锁原图边缘布局）")
    args = p.parse_args()

    # 校验
    if not (0.0 <= args.similarity <= 1.0):
        print(f"[ERROR] similarity 必须在 0-1 之间（当前 {args.similarity}）")
        return
    if not os.path.exists(args.input):
        print(f"[ERROR] 原图不存在: {args.input}")
        return

    # 复制原图到 ComfyUI input
    os.makedirs(COMFYUI_INPUT, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    ts = int(time.time())
    seed_name = f"fission_{base}_{ts}.jpg"
    seed_path = os.path.join(COMFYUI_INPUT, seed_name)
    shutil.copy(args.input, seed_path)
    print(f"[seed] {args.input} -> {seed_path}")

    # 选 prompts
    if args.prompts:
        prompts = [(f"custom_{i+1}", p) for i, p in enumerate(args.prompts)]
    else:
        prompts = DEFAULT_VARIANTS
    prompts = prompts[:args.count]
    if not prompts:
        print("[ERROR] 没有可用 prompt")
        return

    # 输出目录
    if args.out:
        out_dir = args.out
    else:
        out_dir = os.path.join(JOBS_BASE, f"fission_{base}_s{args.similarity:.2f}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out]  {out_dir}")
    print(f"[cfg] color={color_w}  comp={comp_w}  count={len(prompts)}  steps={args.steps}  cfg={args.cfg}  size={args.width}x{args.height}")

    # 跑
    base_params = {
        "similarity": args.similarity,
        "negative_prompt": NEG,
        "width": args.width,
        "height": args.height,
        "batch_per_run": 1,
        "steps": args.steps,
        "cfg": args.cfg,
        "hires_scale": 1.5,
        "hires_denoise": 0.40,
        "hires_steps": 25,
    }
    if args.lora:
        base_params["lora_name"] = args.lora
        base_params["lora_strength"] = args.lora_strength
        print(f"[lora] {args.lora} strength={args.lora_strength}")
    # ---- 三通道参考控制：颜色锁 + 构图锁（默认同时开）----
    # 用户要求：裂变图「有区别」但「保留原图颜色与构图参考」。
    # 默认 color_strength=0.6(style transfer 锁色) + composition_strength=0.55(锁构图)，
    # 内容由每个 prompt 变化 → 同色同构、异内容。
    # 仅当用户显式传 --similarity 时，回退到旧单 IPAdapter linear 模式（关闭双锁）。
    color_w = args.color_strength
    if args.style_strength is not None:
        color_w = args.style_strength
    comp_w = args.composition_strength
    if args.similarity is not None:
        # 旧模式：单 IPAdapter linear（兼容旧用法），关闭双锁
        base_params["similarity"] = args.similarity
        color_w = 0.0
        comp_w = 0.0
        print(f"[ipa] LINEAR(similarity) weight={args.similarity} "
              f"noise={args.ipadapter_noise} end={args.ipadapter_end}")
    else:
        base_params["color_strength"] = color_w
        base_params["composition_strength"] = comp_w
        base_params["ipadapter_noise"] = args.ipadapter_noise
        base_params["ipadapter_end"] = args.ipadapter_end
        print(f"[ipa] DUAL-LOCK  color(style transfer)={color_w}  "
              f"composition={comp_w}  noise={args.ipadapter_noise}  end={args.ipadapter_end}")
    # 可选 Canny 硬构图锁（进一步钉死原图边缘布局）
    if args.controlnet_strength and args.controlnet_strength > 0:
        base_params["controlnet_name"] = "controlnet-canny-sdxl-1.0.fp16.safetensors"
        base_params["controlnet_strength"] = args.controlnet_strength
        base_params["controlnet_end"] = 0.9
        print(f"[cn] Canny hard-comp lock strength={args.controlnet_strength}")
    client = ComfyClient()
    results = []
    for i, (name, prompt) in enumerate(prompts, 1):
        params = dict(base_params)
        params["seed"] = args.seed + i * 313
        params["style_prompt"] = prompt
        g = build_mode1(seed_name, params, f"fission_{name}")
        print(f"\n[{i}/{len(prompts)}] {name}")
        print(f"        {prompt[:80]}...")
        t0 = time.time()
        try:
            res = client.run(g, timeout=300)
            node_id, imgs = next(iter(res.items()))
            data = imgs[0]
            out_path = os.path.join(out_dir, f"{i:02d}_{name}_s{args.similarity:.2f}.jpg")
            with open(out_path, "wb") as f:
                f.write(data)
            print(f"[OK] {time.time()-t0:.0f}s -> {os.path.basename(out_path)} ({len(data)} bytes)")
            results.append((name, out_path))
        except Exception as e:
            print(f"[FAIL] {name}: {repr(e)}")

    if not results:
        print("[ERROR] 没有任何图片生成")
        return

    # 画廊
    seed_b64 = img_to_b64(seed_path)
    cards = "".join(
        f'<div class="card"><img src="data:image/jpeg;base64,{img_to_b64(p)}">'
        f'<div class="cap">{n}</div></div>'
        for n, p in results
    )
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>图裂变</title>
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
.params{{background:#222;color:#fff;padding:8px 12px;border-radius:4px;display:inline-block;font-family:monospace;font-size:12px;}}
</style></head><body>
<h1>图裂变 · 同色同构裂变</h1>
<div class="sub">原图 → {len(results)} 张「同色同构·异内容」裂变图（颜色/构图锁定原图，内容由 prompt 变化）</div>
<p><span class="lock">颜色锁 {color_w} · 构图锁 {comp_w}</span></p>
<p><span class="params">steps={args.steps}  cfg={args.cfg}  size={args.width}x{args.height}  noise={args.ipadapter_noise}  end={args.ipadapter_end}</span></p>
<div class="seed"><img src="data:image/jpeg;base64,{seed_b64}"><div class="cap">原图（参考）</div></div>
<div class="grid">{cards}</div>
</body></html>"""
    html_path = os.path.join(out_dir, "gallery.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[DONE] 画廊: {html_path}")


if __name__ == "__main__":
    main()