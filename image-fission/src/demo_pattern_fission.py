"""
图案裂变 demo：保留画风、换掉内容、不侵权、高清。
针对 6 张 Pinterest 设计稿（黑白色调、机车/牛仔/迷彩/哥特朋克风格）。

参数策略（与 mode1 产品裂变完全不同）：
  - IP-Adapter similarity 低（0.30-0.40）：让"形"传过去、内容换
  - denoise 高（0.75-0.85）：让"内容"完全重生成
  - prompt 写**新内容**（不是同一个花卉/骷髅）
  - SDXL 写实基底 + 强烈风格关键词 + 高 negative 防退化
"""
import os
import sys
import time
import shutil
import base64
from io import BytesIO
from engine.comfy_client import ComfyClient
from pipelines.build import build_mode1
import config as _cfg

JOBS_DIR = r"E:\Desktop\双接口\image-fission\jobs"
SEED_SRC = r"E:\Desktop\双接口\image-fission\ComfyUI\input\pinterest_illust_1.jpg"
SEED_NAME = "pinterest_illust_1.jpg"
OUT_DIR = os.path.join(JOBS_DIR, sys.argv[1] if len(sys.argv) > 1 else "demo_pattern_fission")
os.makedirs(OUT_DIR, exist_ok=True)

# 4 个新内容（保持同画风——黑白矢量装饰花卉藤蔓——但换具体植物种类）
NEW_CONTENTS = [
    (
        "lily_vines",
        "an intricate black and white ornamental pattern with stylized lilies and winding vines, "
        "monochrome vector illustration, decorative symmetrical border, elegant linework, "
        "intricate botanical damask, no text",
    ),
    (
        "rose_thorns",
        "a detailed black and white damask pattern with roses and thorny vines, "
        "monochrome high contrast, symmetrical composition, intricate linework, "
        "ornamental botanical engraving, no text",
    ),
    (
        "chrysanthemum_foliage",
        "an ornamental black and white pattern with chrysanthemums and curling foliage, "
        "intricate symmetrical border, monochrome engraving style, "
        "decorative botanical motif, no text",
    ),
    (
        "iris_flowing",
        "a decorative black and white pattern with irises and flowing botanical motifs, "
        "intricate symmetrical composition, monochrome, elegant linework, "
        "ornamental damask design, no text",
    ),
]

# 风格裂变参数（贴合原图画风、换具体内容）
BASE_PARAMS = {
    "similarity": 0.55,           # 调高（0.35→0.55），让画风强传承
    "negative_prompt": (
        "color, colorful, photorealistic, photo, photography, "
        "text, watermark, signature, blurry, deformed, low quality, "
        "3d, render, gradient, soft, smooth shading, "
        "person, face, human, animal, building, vehicle"
    ),
    "width": 768,                  # 原图比例
    "height": 1344,
    "batch_per_run": 1,
    "steps": 35,
    "cfg": 5.0,
    "hires_scale": 1.5,
    "hires_denoise": 0.40,
    "hires_steps": 25,
}


def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def main():
    client = ComfyClient()
    results = []
    base_seed = 8888
    for i, (name, prompt) in enumerate(NEW_CONTENTS, 1):
        params = dict(BASE_PARAMS)
        params["seed"] = base_seed + i * 313
        params["style_prompt"] = prompt
        g = build_mode1(SEED_NAME, params, f"pattern_{name}")
        print(f"\n[{i}/{len(NEW_CONTENTS)}] {name}")
        print(f"       {prompt[:80]}...")
        t0 = time.time()
        try:
            res = client.run(g, timeout=300)
            node_id, imgs = next(iter(res.items()))
            data = imgs[0]
            out_path = os.path.join(OUT_DIR, f"pattern_fission_{i}_{name}.jpg")
            with open(out_path, "wb") as f:
                f.write(data)
            print(f"[OK] {time.time()-t0:.0f}s -> {os.path.basename(out_path)} ({len(data)} bytes)")
            results.append((name, out_path))
        except Exception as e:
            print(f"[FAIL] {name}: {repr(e)}")

    if not results:
        print("[ERROR] 没有生成任何图片")
        return

    seed_b64 = img_to_b64(SEED_SRC)
    cards = "".join(
        f'<div class="card"><img src="data:image/jpeg;base64,{img_to_b64(p)}"><div class="cap">{n}</div></div>'
        for n, p in results
    )
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>图案风格裂变 demo</title>
<style>
body{{font-family:system-ui;background:#fafafa;padding:24px;margin:0;}}
h1{{margin:0 0 8px;font-size:22px;}}
.sub{{color:#666;margin-bottom:24px;}}
.seed{{background:#fff;padding:16px;border-radius:8px;margin-bottom:24px;text-align:center;}}
.seed img{{max-width:300px;border:1px solid #ddd;}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;}}
.card{{background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.1);}}
.card img{{width:100%;display:block;}}
.cap{{padding:8px;font-size:13px;color:#333;text-align:center;}}
</style></head><body>
<h1>图裂变 · 图案风格裂变 demo</h1>
<div class="sub">输入：黑白花卉藤蔓设计稿 · 输出：4 张同画风不同内容的新图案（不侵权）</div>
<div class="seed"><img src="data:image/jpeg;base64,{seed_b64}"><div class="cap">原图（用户输入）</div></div>
<div class="grid">{cards}</div>
</body></html>"""
    html_path = os.path.join(OUT_DIR, "gallery.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[DONE] 画廊: {html_path}")


if __name__ == "__main__":
    main()