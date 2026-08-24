"""
生成「轻量版」画廊：用相对路径引用图片，不内联 base64。
解决 demo_batch / smoke 生成的 gallery.html 内联 base64 导致 700MB+ 打不开的问题。

用法：
    python make_gallery.py <job_dir> [输出名 gallery_lean.html]

图片按 <原图>_<主题>.jpg 命名，画廊自动按原图分组展示。
"""
import os
import sys
import glob
import re
import html


def parse_name(fn):
    base = os.path.splitext(os.path.basename(fn))[0]
    # 形如 eagle_2_eagle_alt  -> 原图=eagle_2, 主题=eagle_alt
    parts = base.split("_")
    if len(parts) >= 2:
        original = parts[0] + "_" + parts[1]
        subject = "_".join(parts[2:])
    else:
        original = base
        subject = ""
    return original, subject


def build(job_dir, out_name="gallery_lean.html"):
    jpgs = sorted(glob.glob(os.path.join(job_dir, "*.jpg")))
    if not jpgs:
        print("no jpgs found in", job_dir)
        return None

    groups = {}
    for p in jpgs:
        original, subject = parse_name(p)
        groups.setdefault(original, []).append((subject, os.path.basename(p)))

    cards = []
    for original, items in groups.items():
        cards.append(f'<div class="grp-head">{html.escape(original)}</div>')
        cards.append('<div class="row">')
        for subject, fname in sorted(items, key=lambda x: x[0]):
            cards.append(
                f'''<figure>
  <a href="{html.escape(fname)}" target="_blank">
    <img loading="lazy" src="{html.escape(fname)}" alt="{html.escape(subject)}">
  </a>
  <figcaption>{html.escape(subject or "—")}</figcaption>
</figure>'''
            )
        cards.append("</div>")

    total = len(jpgs)
    doc = f'''<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>风格裂变画廊 · {total} 张</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin:0; font-family: system-ui,-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;
         background:#f5f6f8; color:#1a1a1a; }}
  header {{ position:sticky; top:0; background:#fff; border-bottom:1px solid #e5e7eb; padding:14px 20px; }}
  header h1 {{ margin:0; font-size:18px; }}
  header .meta {{ font-size:13px; color:#6b7280; margin-top:4px; }}
  main {{ padding:18px 20px 60px; }}
  .grp-head {{ font-weight:600; margin:22px 0 10px; font-size:15px; color:#374151;
               border-left:4px solid #6366f1; padding-left:10px; }}
  .row {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:14px; }}
  figure {{ margin:0; background:#fff; border:1px solid #eceef1; border-radius:10px; overflow:hidden;
            box-shadow:0 1px 4px rgba(0,0,0,.06); }}
  figure a {{ display:block; }}
  figure img {{ width:100%; height:300px; object-fit:cover; display:block; background:#eee; }}
  figcaption {{ padding:8px 10px; font-size:13px; color:#4b5563; word-break:break-word; }}
</style>
</head>
<body>
<header>
  <h1>风格裂变画廊</h1>
  <div class="meta">{total} 张 · 点击任意图放大（新标签页打开原图）</div>
</header>
<main>
{chr(10).join(cards)}
</main>
</body>
</html>'''

    out_path = os.path.join(job_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"[gallery] {out_path} ({size_kb:.1f} KB, {total} 图)")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python make_gallery.py <job_dir> [out_name]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else "gallery_lean.html"
    build(sys.argv[1], out)
