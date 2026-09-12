#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_regression.py — image-fission 5 图回归基准 runner

用法:
    cd E:/Desktop/双接口/image-fission
    venv/Scripts/python.exe src/run_regression.py

产物:
    jobs/regression_v324_baseline_<ts>/
      stageA/<img_id>/01_custom_1.jpg      # 元素裂变（fission.py mode1/mode3）
      full_text/                             # Pinterest (3) 完整文本裂变（text_fission_v8）
      gallery.html                           # before/after 内嵌 base64 对照

说明:
    - 无主体图案类（佩斯利/迷彩）走 fission.py --mode mode1（装饰 prompt）。
    - 有主体类（蝙蝠/蝴蝶/鹰骷髅）走 fission.py --mode mode3 --redraw-amount 0.45。
    - Pinterest (3) 额外跑 text_fission_v8.py --test 做完整 Stage A~E 文本裂变示范。
    - 当前 v324 baseline 仅做 Stage A 元素裂变；其余 4 张图的完整文字原位替换需
      在后续升级中按各图字体/版式补齐（弧形 Didone、BlackOpsOne、金属尖刺字体等）。
"""
import argparse
import base64
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

BASE = Path(__file__).resolve().parents[1]
REG = BASE / "regression_set" / "set.json"
JOBS = BASE / "jobs"
SRC_IMG = Path(r"E:/Desktop/图裂变测试图")
VENV_PYTHON = BASE / "venv" / "Scripts" / "python.exe"


def _img_to_data_uri(p: Path, max_h: int = 360) -> str:
    img = Image.open(p).convert("RGB")
    w, h = img.size
    if h > max_h:
        nw = int(w * max_h / h)
        img = img.resize((nw, max_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def run_fission(input_path: str, mode: str, prompts: list, out_dir: Path,
                redraw: float = None, similarity: float = None, seed: int = 8888):
    cmd = [
        str(VENV_PYTHON), str(BASE / "src" / "fission.py"),
        "--input", str(input_path),
        "--mode", mode,
        "--prompts", *prompts,
        "--count", str(len(prompts)),
        "--seed", str(seed),
        "--out", str(out_dir),
        "--color-strength", "0.6",
        "--composition-strength", "0.55",
        "--ipadapter-noise", "0.10",
        "--steps", "28",
        "--cfg", "5.0",
    ]
    if mode == "mode3":
        cmd += ["--redraw-amount", str(redraw or 0.45)]
    elif similarity is not None:
        cmd += ["--similarity", str(similarity)]
    print(f"\n[RUN] fission {' '.join(cmd[-6:])}", flush=True)
    r = subprocess.run(cmd, cwd=BASE, check=False)
    if r.returncode != 0:
        print(f"[WARN] fission returned {r.returncode}", flush=True)


def run_text_fission_v8(job_dir: Path):
    # text_fission_v8.py 写死读取 SRC/pinterest_denim_3.jpg
    alias = SRC_IMG / "pinterest_denim_3.jpg"
    original = SRC_IMG / "Pinterest (3).jpg"
    if original.exists() and not alias.exists():
        shutil.copy2(original, alias)
        print(f"[COPY] {original.name} -> pinterest_denim_3.jpg", flush=True)

    cmd = [str(VENV_PYTHON), str(BASE / "src" / "text_fission_v8.py"), "--test"]
    print(f"\n[RUN] text_fission_v8.py --test", flush=True)
    r = subprocess.run(cmd, cwd=BASE, check=False)
    if r.returncode != 0:
        print(f"[WARN] text_fission_v8 returned {r.returncode}", flush=True)

    # 把最新产物复制到本 job 目录
    dirs = sorted(JOBS.glob("text_fission_v8_*"), key=lambda p: p.stat().st_mtime)
    if dirs:
        latest = dirs[-1]
        dest = job_dir / "full_text"
        if latest.exists():
            shutil.copytree(latest, dest, dirs_exist_ok=True)
            print(f"[COPY] {latest.name} -> {dest}", flush=True)


def make_gallery(job_dir: Path, cfg: dict):
    rows = []
    for img_cfg in cfg["images"]:
        img_id = img_cfg["id"]
        orig = Path(img_cfg["path"])
        stage_dir = job_dir / "stageA" / img_id
        variants = []
        if stage_dir.exists():
            variants = sorted(stage_dir.glob("*.jpg"))
        orig_uri = _img_to_data_uri(orig) if orig.exists() else ""
        var_cards = ""
        for v in variants:
            var_cards += (
                f'<div class="v"><img src="{_img_to_data_uri(v)}" loading="lazy">'
                f'<div class="cap">{v.name}</div></div>'
            )
        rows.append(
            f'<section><div class="head">'
            f'<div class="src"><img src="{orig_uri}" loading="lazy"></div>'
            f'<div class="meta"><h3>{img_id}</h3>'
            f'<div class="prof">style=<b>{img_cfg.get("style", "?")}</b> · '
            f'category=<b>{img_cfg.get("category", "?")}</b> · '
            f'strategy={img_cfg.get("strategy", "?")}</div></div></div>'
            f'<div class="variants">{var_cards}</div></section>'
        )

    # full_text 区域
    ft_html = ""
    ft_dir = job_dir / "full_text"
    if ft_dir.exists():
        cards = []
        for img in sorted(ft_dir.glob("*_compare.png")):
            cards.append(
                f'<div class="v"><img src="{_img_to_data_uri(img)}" loading="lazy">'
                f'<div class="cap">{img.name}</div></div>'
            )
        if cards:
            ft_html = (
                '<section class="ft"><h2>text_fission_v8 完整文本裂变（Pinterest 3 denim）</h2>'
                '<div class="variants">' + "".join(cards) + '</div></section>'
            )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>image-fission v324 5 图回归基准</title>
<style>
:root {{ --bg:#f5f6f8; --card:#fff; --text:#1a1a1a; --muted:#555; --accent:#b83a2b; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif; background:var(--bg); color:var(--text); }}
header {{ text-align:center; padding:36px 20px 24px; }}
h1 {{ margin:0 0 8px; font-size:30px; }}
p.sub {{ margin:0; color:var(--muted); }}
section {{ background:var(--card); margin:20px; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,.06); padding:20px; }}
section.ft {{ background:#1a1a1a; color:#eee; }}
.head {{ display:flex; gap:18px; flex-wrap:wrap; align-items:flex-start; }}
.src img {{ max-height:320px; border-radius:8px; border:1px solid #ddd; }}
.meta h3 {{ margin:0 0 8px; font-size:20px; }}
.prof {{ color:var(--muted); font-size:13px; line-height:1.5; }}
.variants {{ display:flex; flex-wrap:wrap; gap:14px; margin-top:14px; }}
.v {{ background:#f9f9f9; border-radius:8px; overflow:hidden; border:1px solid #e5e5e5; }}
.ft .v {{ background:#2a2a2a; border-color:#444; }}
.v img {{ display:block; height:320px; }}
.cap {{ padding:8px 10px; font-size:12px; color:var(--muted); }}
.ft .cap {{ color:#bbb; }}
</style>
</head>
<body>
<header>
<h1>image-fission v324 5 图回归基准</h1>
<p class="sub">version = {cfg.get('version','?')} · {cfg.get('created','?')} · Stage A 元素裂变 + Pinterest(3) 完整文本裂变</p>
</header>
{''.join(rows)}
{ft_html}
</body>
</html>"""

    gallery_path = job_dir / "gallery.html"
    gallery_path.write_text(html, encoding="utf-8")
    print(f"\n[GALLERY] {gallery_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-text", action="store_true", help="跳过 text_fission_v8 完整文本裂变")
    args = ap.parse_args()

    if not REG.exists():
        print(f"[ERROR] 回归集配置不存在: {REG}")
        return 1

    cfg = json.loads(REG.read_text(encoding="utf-8"))
    ts = int(time.time())
    job_dir = JOBS / f"regression_v324_baseline_{ts}"
    (job_dir / "stageA").mkdir(parents=True, exist_ok=True)
    print(f"=== regression job: {job_dir} ===", flush=True)

    for img_cfg in cfg["images"]:
        img_id = img_cfg["id"]
        out_dir = job_dir / "stageA" / img_id
        out_dir.mkdir(parents=True, exist_ok=True)
        cat = img_cfg["category"]
        if cat == "no_subject_pattern":
            mode, similarity, redraw = "mode1", 0.65, None
        else:
            mode, similarity, redraw = "mode3", None, 0.45
        prompts = img_cfg.get("prompts", [])[:1]  # baseline 只跑 1 个 prompt 求快
        if not prompts:
            prompts = ["seamless pattern variation, same style, no text"]
        run_fission(img_cfg["path"], mode, prompts, out_dir,
                    redraw=redraw, similarity=similarity, seed=8888)

    if not args.skip_text:
        run_text_fission_v8(job_dir)

    make_gallery(job_dir, cfg)
    print(f"\n=== DONE: {job_dir} ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
