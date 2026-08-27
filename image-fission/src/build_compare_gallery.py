"""生成 v126(LoRA乱字) vs v127(PIL干净字) 对比画廊，便于肉眼验收修复效果。"""
import shutil
from pathlib import Path

V126 = Path(r"E:\Desktop\双接口\image-fission\jobs\smoke_v126_1787808430")
V127 = Path(r"C:\Users\lenovo\WorkBuddy\2026-08-24-16-39-13\outputs\v127")
OUT = Path(r"C:\Users\lenovo\WorkBuddy\2026-08-24-16-39-13\outputs\v127_compare")
OUT.mkdir(parents=True, exist_ok=True)

PAIRS = [
    ("eagle_2", "eagle_2_s700302.jpg", "eagle_2_final.jpg", "DOMINION", "LoRA:'^ > Do' 乱码", "PIL:PirataOne 清晰"),
    ("denim_3", "denim_3_s700302.jpg", "denim_3_final.jpg", "UPCY", "LoRA:'4 3' 乱码", "PIL:Rye 清晰"),
    ("skull_5", "skull_5_s700302.jpg", "skull_5_final.jpg", "VENOM", "LoRA:'VEN VEXOM' 乱码", "PIL:PirataOne 清晰"),
    ("metal_6", "metal_6_s700302.jpg", "metal_6_final.jpg", "MRCHGSR", "LoRA:'1 V7 A' 乱码", "PIL:MetalMania 清晰"),
]

for pre, v126f, v127f, word, v126note, v127note in PAIRS:
    shutil.copy(V126 / v126f, OUT / f"a_{pre}.jpg")
    shutil.copy(V127 / v127f, OUT / f"b_{pre}.jpg")

html = ["<html><head><meta charset='utf-8'><title>v126 vs v127 文字修复对比</title>",
        "<style>body{background:#111;color:#eee;font-family:sans-serif;margin:0;padding:24px}",
        "h1{font-size:20px}.row{display:flex;gap:16px;margin:24px 0;align-items:flex-start}",
        ".col{flex:1}.col img{width:100%;border:1px solid #444;border-radius:8px}",
        ".cap{font-size:13px;margin-top:8px;color:#aaa}.word{color:#6cf;font-weight:bold}",
        ".bad{color:#f66}.good{color:#6f6}</style></head><body>",
        "<h1>v126 (Harrlogos LoRA 画字 → 乱码) &nbsp;vs&nbsp; v127 (PIL 后期烧字 → 清晰)</h1>"]

for pre, v126f, v127f, word, v126note, v127note in PAIRS:
    html.append(f"<div class='row'><div class='col'><img src='a_{pre}.jpg'>"
                f"<div class='cap'><b>{pre}</b> · 目标词 <span class='word'>{word}</span><br>"
                f"<span class='bad'>v126: {v126note}</span></div></div>"
                f"<div class='col'><img src='b_{pre}.jpg'>"
                f"<div class='cap'><span class='good'>v127: {v127note}</span></div></div></div>")

html.append("</body></html>")
(OUT / "index.html").write_text("\n".join(html), encoding="utf-8")
print(f"画廊已生成: {OUT / 'index.html'}")
print(f"含 {len(PAIRS)} 组对比 + 无字图 illust_1/camo_4 见 outputs/v127/")
