"""
build_real_gallery.py -- 把 jobs/fission_real/ 下的真裂变结果拼成
"原图 vs 3 变体" 对照画廊（解决"不知道做的是啥、跟原图啥关联"）。
"""
import base64
from pathlib import Path
from PIL import Image

JOBS = Path("E:/Desktop/双接口/image-fission/jobs/fission_real")
SRC  = Path("E:/Desktop/图裂变测试图")
OUT  = JOBS / "gallery.html"


def b64(p: Path) -> str:
    img = Image.open(p).convert("RGB")
    # 缩到长边 720 降 HTML 体积
    img.thumbnail((720, 720))
    buf = __import__("io").BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


# (label, original_path, variants_dir, mode)
GROUPS = [
    {
        "label": "b78e60 军徽+狗牌/链 → 真裂变 mode3 img2img+双锁(denoise 0.45, 军事主题词)",
        "src_name": "b78e60de8dfdf44acda99395326a7298.jpg",
        "variants_dir": JOBS / "b78e60_mil",
        "note": "军徽主体没出（变成抽象灰斑），狗牌/链变薄变淡——v147 把元素都裂变了但元素本身糊。需换 v241/v242 或加 Canny 硬锁。自评 5/10。",
    },
    {
        "label": "13c8b7 红黑佩斯利头巾 → 真裂变 mode1 IPAdapter双锁(红黑装饰主题词)",
        "src_name": "13c8b7bf8dae757e6c2d4b3d6a860f9d.jpg",
        "variants_dir": JOBS / "13c8b7",
        "note": "3 张全新红黑/白装饰图案（damask 花章 / 哥特玫瑰 / 满铺藤蔓），颜色锁住红黑，内容完全不同——这才是真裂变。WILDHUNT 文字没了（mode1 从空白画布重画），可接受/需 text_fission_v8 补。自评 8/10。",
    },
]


def main():
    parts = ["""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>真裂变对照</title>
<style>
body{font-family:system-ui;background:#fafafa;padding:24px;margin:0;color:#222;}
h1{margin:0 0 8px;font-size:22px;}
h2{margin:24px 0 8px;font-size:17px;color:#0a7;}
.note{background:#fff8e1;border-left:4px solid #f90;padding:10px 14px;border-radius:4px;
       margin:8px 0 18px;font-size:13px;color:#444;}
.row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:32px;}
.card{background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.1);}
.card img{width:100%;display:block;background:#eee;}
.cap{padding:8px;font-size:12px;color:#333;text-align:center;font-weight:500;line-height:1.4;}
.cap.orig{background:#0a7;color:#fff;}
.tag{display:inline-block;background:#222;color:#fff;padding:3px 8px;border-radius:3px;
     font-size:11px;margin-left:6px;font-weight:600;}
</style></head><body>
<h1>真裂变 · 原图 vs 变体 对照</h1>
<p>ComfyUI SDXL 真重生（IPAdapter 颜色锁 + 构图锁）——不是 PIL HSV 改色。每行：左=原图，右=3 张 AI 裂变图。</p>"""]

    for g in GROUPS:
        orig = SRC / g["src_name"]
        vdir: Path = g["variants_dir"]
        variants = sorted(vdir.glob("*.jpg"))
        if not orig.exists():
            print(f"原图缺失: {orig}"); continue
        cards = [f'<div class="card"><img src="data:image/jpeg;base64,{b64(orig)}">'
                 f'<div class="cap orig">原图（{g["src_name"][:12]}…）</div></div>']
        for v in variants:
            cards.append(f'<div class="card"><img src="data:image/jpeg;base64,{b64(v)}">'
                         f'<div class="cap">{v.name}</div></div>')
        parts.append(f'<h2>{g["label"]}</h2>')
        parts.append(f'<div class="note">{g["note"]}</div>')
        parts.append(f'<div class="row">{"".join(cards)}</div>')

    parts.append("</body></html>")
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"[OK] {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
