"""
图裂变 背景替换（bgswap）演示：BiRefNet 显式锁主体，换 4 种背景。
链路：BiRefNet 抠主体 -> InvertMask 背景蒙版 -> SetLatentNoiseMask 只重绘背景
      -> KSampler 换新场景，主体像素级保留。
输出：对照画廊 HTML（内含 base64 图片）。
"""
import os
import sys
import shutil
import base64
import time

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from config import JOBS_DIR, SDXL_CHECKPOINT
from pipelines.build import build_bgswap
from engine.comfy_client import ComfyClient

SEED_SRC = r"E:\Desktop\双接口\image-fission\jobs\demo_fission_v3\demo_fission_v3_暖阳窗边.jpg"
SEED_IN_INPUT = r"E:\Desktop\双接口\image-fission\ComfyUI\input\bgswap_seed.jpg"
JOB_ID = sys.argv[1] if len(sys.argv) > 1 else "demo_bgswap"
OUT_DIR = os.path.join(JOBS_DIR, JOB_ID)

SCENES = [
    (
        "白底商业图",
        "a clean pure white studio background, seamless white, soft diffused lighting, "
        "professional e-commerce product photography, subtle shadow under the tea set",
    ),
    (
        "山野茶园",
        "blurred green mountain tea garden background, soft bokeh, fresh outdoor morning light, "
        "misty hills, natural atmosphere",
    ),
    (
        "大理石台面",
        "luxury black marble table surface, dark elegant studio background, "
        "dramatic soft spotlight, high-end commercial photography",
    ),
    (
        "原木餐桌",
        "warm wooden dining table, cozy restaurant interior background, soft warm ambient light, "
        "inviting tea house atmosphere, blurred shelves with teaware in background",
    ),
]

BASE_PARAMS = {
    "negative_prompt": "low quality, blurry, deformed, watermark, text, extra fingers",
    "width": 1024,
    "height": 1024,
    "batch_per_run": 1,
    "steps": 32,
    "cfg": 5.0,
    "matting_model": "BiRefNet-matting",
}


def _img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _build_html(seed_path: str, results: list) -> str:
    seed_b64 = _img_to_base64(seed_path)
    rows = []
    for name, path in results:
        b64 = _img_to_base64(path)
        rows.append(
            f"""
            <div class="card">
                <img src="data:image/jpeg;base64,{b64}" alt="{name}" />
                <div class="caption">{name}</div>
            </div>"""
        )
    cards_html = "\n".join(rows)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>图裂变 背景替换演示</title>
    <style>
        :root {{
            --bg: #f7f8fa; --card: #ffffff; --text: #1a1a1a;
            --muted: #666666; --accent: #B83A2B;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif;
            background: var(--bg); color: var(--text);
        }}
        header {{ text-align: center; padding: 48px 24px 24px; }}
        h1 {{ margin: 0 0 8px; font-size: 32px; }}
        p.sub {{ margin: 0; color: var(--muted); font-size: 16px; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
        .section-title {{ font-size: 20px; font-weight: 600; margin: 32px 0 16px; }}
        .seed-card {{
            background: var(--card); border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06); overflow: hidden;
            max-width: 520px; margin: 0 auto;
        }}
        .seed-card img {{ width: 100%; display: block; }}
        .seed-caption {{ padding: 14px 18px; font-size: 15px; color: var(--muted); text-align: center; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 24px; }}
        .card {{
            background: var(--card); border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06); overflow: hidden;
        }}
        .card img {{ width: 100%; display: block; }}
        .caption {{ padding: 14px 18px; font-size: 15px; font-weight: 600; text-align: center; }}
        footer {{ text-align: center; padding: 40px 24px; color: var(--muted); font-size: 13px; }}
    </style>
</head>
<body>
    <header>
        <h1>图裂变 · 背景替换演示（显式主体锁定）</h1>
        <p class="sub">BiRefNet 抠主体 → 只重绘背景 → 同一茶器放进 4 种场景</p>
    </header>
    <div class="container">
        <div class="section-title">种子图</div>
        <div class="seed-card">
            <img src="data:image/jpeg;base64,{seed_b64}" alt="seed" />
            <div class="seed-caption">输入：mode1 裂变产物「暖阳窗边」（主体为茶器组合）</div>
        </div>
        <div class="section-title">背景替换结果（主体像素级保留）</div>
        <div class="grid">
            {cards_html}
        </div>
    </div>
    <footer>
        图裂变 image-fission · bgswap · 基底 {SDXL_CHECKPOINT} · BiRefNet-matting 主体锁定 · denoise=1.0(仅背景区)
    </footer>
</body>
</html>"""


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    shutil.copy(SEED_SRC, SEED_IN_INPUT)
    print(f"[seed] 已复制种子图到 {SEED_IN_INPUT}")

    client = ComfyClient()
    results = []

    for i, (name, scene_prompt) in enumerate(SCENES, 1):
        params = dict(BASE_PARAMS)
        params["seed"] = 998877 + i * 223
        params["style_prompt"] = scene_prompt
        prompt = build_bgswap("bgswap_seed.jpg", params, JOB_ID)
        print(f"\n[{i}/{len(SCENES)}] 场景: {name}")
        print(f"       prompt: {scene_prompt[:60]}...")
        start = time.time()
        try:
            out = client.run(prompt, timeout=600)
        except Exception as e:
            print(f"[FAIL] {name}: {repr(e)}")
            continue
        elapsed = time.time() - start
        node_id, imgs = next(iter(out.items()))
        data = imgs[0]
        out_path = os.path.join(OUT_DIR, f"{JOB_ID}_{name}.jpg")
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"[OK] {elapsed:.1f}s -> {out_path} ({len(data)} bytes)")
        results.append((name, out_path))

    if not results:
        print("[ERROR] 没有生成任何图片")
        return

    html_path = os.path.join(OUT_DIR, "gallery.html")
    html = _build_html(SEED_SRC, results)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[DONE] 画廊已生成: {html_path}")
    print(f"       共生成 {len(results)} 张")


if __name__ == "__main__":
    main()
