"""
图裂变效果演示：一张种子图 -> 4 种风格/场景裂变。
输出：对照画廊 HTML（内含 base64 图片，可单独打开）。
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

from config import JOBS_DIR, DEFAULTS, SDXL_CHECKPOINT
from pipelines.build import build_mode1
from engine.comfy_client import ComfyClient

SEED_SRC = r"E:\Desktop\双接口\image-fission\jobs\test_mode1\test_mode1_v2_0_node10_0.jpg"
SEED_IN_INPUT = r"E:\Desktop\双接口\image-fission\ComfyUI\input\demo_seed.jpg"
# 任务名可用 argv[1] 指定（如 demo_fission_v3），不覆盖旧结果
JOB_ID = sys.argv[1] if len(sys.argv) > 1 else "demo_fission"
OUT_DIR = os.path.join(JOBS_DIR, JOB_ID)

STYLES = [
    (
        "暖阳窗边",
        "a premium tea product photo, warm afternoon sunlight streaming through a large window, "
        "cozy wooden table, soft shadows, porcelain teacup, elegant lifestyle, "
        "hyperdetailed photography, shot on Canon EOS R5, 85mm f/1.4, shallow depth of field, "
        "natural color, sharp focus, 8k, ultra detailed",
    ),
    (
        "暗调高级",
        "a luxury tea product photo, dark moody studio, black marble background, dramatic side lighting, "
        "porcelain teacup, high-end commercial photography, cinematic lighting, "
        "hyperdetailed photography, shot on Canon EOS R5, 85mm f/1.4, shallow depth of field, "
        "sharp focus, 8k, ultra detailed",
    ),
    (
        "山野自然",
        "a natural tea product photo, outdoor mountain meadow, fresh spring morning sunlight, "
        "green grass and wildflowers, porcelain teacup on a wooden tray, cinematic nature, "
        "hyperdetailed photography, shot on Canon EOS R5, 85mm f/1.4, shallow depth of field, "
        "natural color, sharp focus, 8k, ultra detailed",
    ),
    (
        "杂志大理石",
        "an editorial magazine cover style tea product photo, minimalist white marble surface, "
        "soft pastel props, modern aesthetic, porcelain teacup, high fashion product photography, "
        "hyperdetailed photography, shot on Canon EOS R5, 85mm f/1.4, shallow depth of field, "
        "sharp focus, 8k, ultra detailed",
    ),
]

BASE_PARAMS = {
    "similarity": 0.50,
    "negative_prompt": (
        "low quality, blurry, deformed, watermark, text, extra fingers"
    ),
    "width": 1024,
    "height": 1024,
    "batch_per_run": 1,
    "steps": 32,
    "cfg": 5.0,
    "hires_scale": 1.5,
    "hires_denoise": 0.40,
    "hires_steps": 25,
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
    <title>图裂变效果演示</title>
    <style>
        :root {{
            --bg: #f7f8fa;
            --card: #ffffff;
            --text: #1a1a1a;
            --muted: #666666;
            --accent: #B83A2B;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif;
            background: var(--bg);
            color: var(--text);
        }}
        header {{
            text-align: center;
            padding: 48px 24px 24px;
        }}
        h1 {{ margin: 0 0 8px; font-size: 32px; }}
        p.sub {{ margin: 0; color: var(--muted); font-size: 16px; }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 24px;
        }}
        .section-title {{
            font-size: 20px;
            font-weight: 600;
            margin: 32px 0 16px;
        }}
        .seed-card {{
            background: var(--card);
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
            overflow: hidden;
            max-width: 520px;
            margin: 0 auto;
        }}
        .seed-card img {{ width: 100%; display: block; }}
        .seed-caption {{
            padding: 14px 18px;
            font-size: 15px;
            color: var(--muted);
            text-align: center;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 24px;
        }}
        .card {{
            background: var(--card);
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
            overflow: hidden;
        }}
        .card img {{ width: 100%; display: block; }}
        .caption {{
            padding: 14px 18px;
            font-size: 15px;
            font-weight: 600;
            text-align: center;
        }}
        footer {{
            text-align: center;
            padding: 40px 24px;
            color: var(--muted);
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <header>
        <h1>图裂变效果演示</h1>
        <p class="sub">一张种子图 → 4 种不同场景/风格（mode1 换景换风格）</p>
    </header>
    <div class="container">
        <div class="section-title">种子图</div>
        <div class="seed-card">
            <img src="data:image/jpeg;base64,{seed_b64}" alt="seed" />
            <div class="seed-caption">输入：茶器产品图（1024×1024 经 hires fix 至 1536×1536）</div>
        </div>
        <div class="section-title">裂变结果</div>
        <div class="grid">
            {cards_html}
        </div>
    </div>
    <footer>
        图裂变 image-fission · mode1 换景换风格 · 基底模型 {SDXL_CHECKPOINT} · similarity=0.5 · hires fix 1.5x
    </footer>
</body>
</html>"""


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    shutil.copy(SEED_SRC, SEED_IN_INPUT)
    print(f"[seed] 已复制种子图到 {SEED_IN_INPUT}")

    client = ComfyClient()
    results = []

    for i, (name, style_prompt) in enumerate(STYLES, 1):
        params = dict(BASE_PARAMS)
        params["seed"] = 20240821 + i * 111
        params["style_prompt"] = style_prompt
        prompt = build_mode1("demo_seed.jpg", params, JOB_ID)
        print(f"\n[{i}/{len(STYLES)}] 生成风格: {name}")
        print(f"       prompt: {style_prompt[:60]}...")
        start = time.time()
        try:
            out = client.run(prompt, timeout=600)
        except Exception as e:
            print(f"[FAIL] {name}: {repr(e)}")
            continue
        elapsed = time.time() - start
        # Save first image from first output node
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
    print(f"       共裂变 {len(results)} 张")


if __name__ == "__main__":
    main()
