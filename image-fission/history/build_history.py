# -*- coding: utf-8 -*-
"""build_history.py —— image-fission 进度历史记录（可回滚查看）

用途
----
把「每一次对话做的修改」固化成一个个**节点**：每个节点记录
  · 版本号 / 日期 / 轮次 / 标题 / 状态（baseline | superseded | rejected）
  · 改动的代码文件、关键参数、备注
  · **该节点代码处理后的图片**（原图 vs 该节点产出）
然后生成一个**离线自包含**的 `index.html`：
  · 左侧时间线：所有节点一览，点一下切换
  · 右侧主区：原图 ↔ 该节点产出的**拖动滑块对比**（可直接对比）
  · 底部总览：所有节点缩略图网格（直观看到每个节点的图长什么样）
  · 顶部：当前基线 + 一键回滚命令

用法
----
    python history/build_history.py                     # 重建 index.html
    python history/build_history.py --rollback v395     # 把基线回滚到 v395 并重建
    python history/build_history.py --add-node node.json  # 追加一个节点再重建

约定（每轮对话结束时执行）
------------------------
每轮做完图，往 `history/registry.json` 的 nodes 里**追加一个节点**（或调 --add-node），
然后跑一次 build_history.py —— 这样历史记录自动同步到最新。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # image-fission/history
PROJ = ROOT.parent                              # image-fission
REG = ROOT / "registry.json"
ASSETS = ROOT / "assets"
OUT = ROOT / "index.html"

try:
    from PIL import Image
except Exception:                               # pragma: no cover
    Image = None

MAX_W = 1100          # 展示图默认最大宽度（控制体积）
THUMB_W = 380         # 总览缩略图宽度
WIDE_W = 1700         # 宽幅对比图（多格拼版）用的展示宽度：shot 里写 "width": 1700


# ---------------------------------------------------------------- 工具
def _resolve(p: str) -> Path:
    """相对路径按 image-fission 解析；绝对路径原样。"""
    q = Path(p)
    return q if q.is_absolute() else (PROJ / q)


def _copy_scaled(src: Path, dst: Path, max_w: int) -> bool:
    """把 src 缩放到 max_w 宽后存成 jpg。返回是否成功。"""
    if not src.exists():
        return False
    if Image is None:
        shutil.copy2(src, dst)
        return True
    try:
        im = Image.open(src)
        im.load()
        im = im.convert("RGB")
        if im.width > max_w:
            h = max(1, int(round(im.height * max_w / im.width)))
            im = im.resize((max_w, h), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "JPEG", quality=88)
        return True
    except Exception as e:                       # pragma: no cover
        print(f"    ! 缩放失败 {src.name}: {e}")
        return False


def _load() -> dict:
    if not REG.exists():
        raise SystemExit(f"缺少 {REG}")
    return json.loads(REG.read_text(encoding="utf-8"))


def _save(cfg: dict) -> None:
    cfg["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    REG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- 资源
def prepare_assets(cfg: dict) -> dict:
    """把每个节点的 before/after 图缩放复制进 assets/，返回 node_id -> 展示信息。"""
    ASSETS.mkdir(parents=True, exist_ok=True)
    srcs = cfg.get("sources", {})
    pack: dict[str, list] = {}

    for nd in cfg["nodes"]:
        nid = nd["id"]
        shots_out = []
        for k, sh in enumerate(nd.get("shots", [])):
            img_key = sh.get("img", "")
            mw = int(sh.get("width") or (WIDE_W if sh.get("wide") else MAX_W))
            before_p = _resolve(sh["before"]) if sh.get("before") else (
                _resolve(srcs[img_key]) if img_key in srcs else None)
            after_p = _resolve(sh["after"])

            a_name = f"{nid}_{k}_after.jpg"
            a_thumb = f"{nid}_{k}_after_t.jpg"
            ok_a = _copy_scaled(after_p, ASSETS / a_name, mw)
            ok_at = _copy_scaled(after_p, ASSETS / a_thumb, THUMB_W)

            b_name = f"{nid}_{k}_before.jpg"
            b_thumb = f"{nid}_{k}_before_t.jpg"
            ok_b_thumb = _copy_scaled(before_p, ASSETS / b_thumb, THUMB_W) if (before_p and before_p.exists()) else False
            ok_b = _copy_scaled(before_p, ASSETS / b_name, mw) if (before_p and before_p.exists()) else False

            if not ok_a:
                print(f"  ! 缺图 [{nid}] {sh.get('after')}")

            shots_out.append({
                "img": img_key,
                "label": sh.get("label", img_key),
                "note": sh.get("note", ""),
                "w": mw,
                "after": a_name if ok_a else "",
                "after_thumb": a_thumb if ok_at else "",
                "before": b_name if ok_b else "",
                "before_thumb": b_thumb if ok_b_thumb else "",
                "full": str(after_p) if after_p else "",
            })
        pack[nid] = shots_out
    return pack


# ---------------------------------------------------------------- HTML
CSS = """
*{box-sizing:border-box}
body{margin:0;background:#0f1115;color:#e8eaf0;
 font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif;font-size:14px}
a{color:#7cc7ff;text-decoration:none}
header{position:sticky;top:0;z-index:20;background:#151922;border-bottom:1px solid #262c38;
 padding:12px 20px;display:flex;align-items:center;gap:18px;flex-wrap:wrap}
header h1{font-size:16px;margin:0;font-weight:600;letter-spacing:.5px}
.badge{display:inline-block;padding:2px 9px;border-radius:10px;font-size:12px;line-height:18px}
.b-base{background:#0b3d66;color:#9fd8ff;border:1px solid #1d6fa8}
.b-sup{background:#2b2f3a;color:#a8b0c0;border:1px solid #3a4150}
.b-rej{background:#4a1520;color:#ff9aa8;border:1px solid #7a2432}
.b-new{background:#12402a;color:#8ff0bd;border:1px solid #1e7a4c}
.wrap{display:flex;gap:0;align-items:flex-start}
nav{width:290px;flex:0 0 290px;border-right:1px solid #262c38;height:calc(100vh - 57px);
 overflow-y:auto;padding:12px}
nav .grp{color:#6d7688;font-size:12px;margin:14px 0 6px;letter-spacing:1px}
.node{border:1px solid #262c38;border-radius:8px;padding:9px 10px;margin-bottom:8px;cursor:pointer;
 background:#151922;transition:.15s}
.node:hover{background:#1b2130;border-color:#3a4658}
.node.on{background:#12283c;border-color:#2b7fbf}
.node .t{font-weight:600;font-size:13px;margin-bottom:3px}
.node .m{color:#8d97a8;font-size:11.5px;display:flex;gap:6px;flex-wrap:wrap;align-items:center}
main{flex:1;padding:18px 22px;height:calc(100vh - 57px);overflow-y:auto}
.card{background:#151922;border:1px solid #262c38;border-radius:10px;padding:14px 16px;margin-bottom:16px}
.card h2{margin:0 0 10px;font-size:15px}
.kv{display:grid;grid-template-columns:78px 1fr;gap:4px 10px;font-size:13px;color:#c6cdda}
.kv b{color:#7f8a9c;font-weight:400}
code{background:#0b0e14;border:1px solid #262c38;border-radius:5px;padding:1px 6px;
 font-family:Consolas,monospace;font-size:12.5px;color:#9fe0b0}
pre{background:#0b0e14;border:1px solid #262c38;border-radius:8px;padding:10px 12px;overflow-x:auto;
 font-family:Consolas,monospace;font-size:12.5px;color:#b9c4d4;margin:8px 0 0}
.shot{margin-bottom:22px}
.shot .cap{font-size:13px;color:#9aa5b6;margin-bottom:6px}
.shot .cap b{color:#e8eaf0}
/* 拖动滑块对比 */
.slider{position:relative;width:100%;max-width:1100px;overflow:hidden;border-radius:8px;
 border:1px solid #2c3442;background:#000;user-select:none;touch-action:none}
.slider img{display:block;width:100%;height:auto;pointer-events:none}
.slider .top{position:absolute;left:0;top:0;width:100%;overflow:hidden}
.slider .hd{position:absolute;top:0;bottom:0;width:3px;background:#4ea8ff;box-shadow:0 0 8px #4ea8ff;
 transform:translateX(-1px);pointer-events:none}
.slider .tag{position:absolute;top:8px;background:rgba(10,12,18,.78);border:1px solid #2c3442;
 border-radius:5px;padding:2px 8px;font-size:12px;color:#dfe6f2;pointer-events:none}
.slider .tag.l{left:8px} .slider .tag.r{right:8px}
.range{width:100%;max-width:1100px;margin:6px 0 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px}
.gitem{border:1px solid #262c38;border-radius:8px;overflow:hidden;background:#111520;cursor:pointer}
.gitem:hover{border-color:#2b7fbf}
.gitem img{width:100%;display:block}
.gitem .gl{padding:6px 8px;font-size:11.5px;color:#9aa5b6;display:flex;justify-content:space-between;gap:6px}
.cmd{background:#0b0e14;border:1px dashed #3a4658;border-radius:8px;padding:9px 12px;
 font-family:Consolas,monospace;font-size:12.5px;color:#9fe0b0;margin-top:10px}
.legend{font-size:12px;color:#7f8a9c}
"""


def build_html(cfg: dict, pack: dict) -> str:
    nodes = cfg["nodes"]
    baseline = cfg.get("baseline", "")
    by_id = {n["id"]: n for n in nodes}

    # ---- 左侧时间线（按 date 正序）
    ordered = sorted(nodes, key=lambda n: (n.get("date", ""), n["id"]))
    nav = []
    last_date = None
    for nd in ordered:
        d = (nd.get("date") or "")[:10]
        if d != last_date:
            nav.append(f"<div class='grp'>{d or '—'}</div>")
            last_date = d
        st = nd.get("status", "superseded")
        cls = {"baseline": "b-base", "rejected": "b-rej", "new": "b-new"}.get(st, "b-sup")
        stxt = {"baseline": "基线", "rejected": "已否决", "new": "本轮新"}.get(st, "已被取代")
        n_img = len(nd.get("shots", []))
        nav.append(
            f"<div class='node' data-id='{nd['id']}' onclick=\"sel('{nd['id']}')\">"
            f"<div class='t'>{nd.get('version', nd['id'])} · {nd.get('title', '')}</div>"
            f"<div class='m'><span class='badge {cls}'>{stxt}</span>"
            f"<span>{nd.get('round', '')}</span><span>{n_img} 图</span></div></div>")
    nav_html = "".join(nav)

    # ---- 主区：每个节点一块（默认只显示被选中的）
    main = []
    for nd in ordered:
        nid = nd["id"]
        shots = pack.get(nid, [])
        codes = " ".join(f"<code>{c}</code>" for c in nd.get("code", []))
        params = nd.get("params", "")
        if isinstance(params, dict):
            params = " · ".join(f"{k}={v}" for k, v in params.items())
        shots_html = []
        for k, sh in enumerate(shots):
            if not sh["after"]:
                continue
            sid = f"sl_{nid}_{k}"
            mw = int(sh.get("w") or MAX_W)
            style_w = f"max-width:{mw}px"
            if sh["before"]:
                cmp_html = (
                    f"<div class='slider' id='{sid}' style='{style_w}'>"
                    f"<img src='assets/{sh['after']}' alt='after'>"
                    f"<div class='top' style='width:50%'>"
                    f"<img src='assets/{sh['before']}' style='width:{mw}px;max-width:none' alt='before'></div>"
                    f"<div class='hd' style='left:50%'></div>"
                    f"<div class='tag l'>原图</div><div class='tag r'>{nid} 产出</div></div>"
                    f"<input class='range' type='range' min='0' max='100' value='50' style='{style_w}' "
                    f"oninput=\"slide('{sid}', this.value)\">")
            else:
                cmp_html = (f"<img src='assets/{sh['after']}' "
                            f"style='max-width:{mw}px;width:100%;border-radius:8px'>")
            shots_html.append(
                f"<div class='shot'><div class='cap'><b>{sh['label']}</b>"
                + (f" —— {sh['note']}" if sh.get('note') else "") + "</div>"
                + cmp_html + "</div>")
        main.append(
            f"<section class='pane' id='pane_{nid}' style='display:none'>"
            f"<div class='card'><h2>{nd.get('version', nid)} · {nd.get('title', '')}</h2>"
            f"<div class='kv'>"
            f"<b>日期</b><span>{nd.get('date', '')}</span>"
            f"<b>轮次</b><span>{nd.get('round', '')}</span>"
            f"<b>状态</b><span>{nd.get('status', '')}</span>"
            f"<b>代码</b><span>{codes}</span>"
            f"<b>参数</b><span>{params}</span>"
            f"<b>备注</b><span>{nd.get('note', '')}</span>"
            f"</div>"
            f"<div class='cmd'>回滚到本节点：python history/build_history.py --rollback {nid}</div>"
            f"</div>" + "".join(shots_html) + "</section>")
    main_html = "".join(main)

    # ---- 总览网格
    grid = []
    for nd in ordered:
        for sh in pack.get(nd["id"], []):
            if not sh["after_thumb"]:
                continue
            grid.append(
                f"<div class='gitem' onclick=\"sel('{nd['id']}')\">"
                f"<img src='assets/{sh['after_thumb']}' loading='lazy'>"
                f"<div class='gl'><span>{nd.get('version', nd['id'])}</span>"
                f"<span>{sh['label']}</span></div></div>")
    grid_html = "".join(grid)

    order_ids = json.dumps([n["id"] for n in ordered], ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>image-fission 进度历史记录</title><style>{CSS}</style></head><body>
<header>
  <h1>image-fission · 进度历史记录</h1>
  <span class='badge b-base'>当前基线：{baseline or '—'}</span>
  <span class='legend'>左侧点节点 → 右侧看「原图 ↔ 该节点产出」拖动对比 · 底部是全部节点总览</span>
</header>
<div class="wrap">
  <nav>
    <div class="grp">节点时间线（共 {len(nodes)}）</div>
    {nav_html}
    <div class="grp">总览</div>
    <div class="node" onclick="sel('__grid__')"><div class="t">全部节点缩略图</div>
      <div class="m"><span class="badge b-sup">总览</span></div></div>
  </nav>
  <main>
    <section class="pane" id="pane___grid__" style="display:none">
      <div class="card"><h2>全部节点总览 —— 每个节点的代码处理出来的图长什么样</h2>
      <div class="grid">{grid_html}</div></div>
    </section>
    {main_html}
  </main>
</div>
<script>
var ORDER = {order_ids};
function sel(id){{
  ORDER.concat(['__grid__']).forEach(function(k){{
    var p=document.getElementById('pane_'+k); if(p) p.style.display = (k===id?'block':'none');
  }});
  document.querySelectorAll('nav .node').forEach(function(el){{
    el.classList.toggle('on', el.dataset.id===id);
  }});
  window.scrollTo(0,0);
  var m=document.querySelector('main'); if(m) m.scrollTop=0;
}}
function slide(sid, v){{
  var s=document.getElementById(sid); if(!s) return;
  var top=s.querySelector('.top'), hd=s.querySelector('.hd');
  top.style.width=v+'%'; hd.style.left=v+'%';
}}
/* 拖动：按住图片左右滑 */
document.querySelectorAll('.slider').forEach(function(s){{
  function mv(e){{
    var r=s.getBoundingClientRect();
    var x=(e.touches?e.touches[0].clientX:e.clientX)-r.left;
    var v=Math.max(0,Math.min(100,x/r.width*100));
    var top=s.querySelector('.top'), hd=s.querySelector('.hd');
    top.style.width=v+'%'; hd.style.left=v+'%';
    var rg=s.nextElementSibling;
    if(rg&&rg.classList.contains('range')) rg.value=v;
  }}
  s.addEventListener('mousemove',function(e){{ if(e.buttons!==1) return; mv(e); }});
  s.addEventListener('touchmove',mv,{{passive:true}});
}});
sel(ORDER.length?ORDER[ORDER.length-1]:'__grid__');
</script>
</body></html>"""


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollback", metavar="NODE_ID", help="把基线设为该节点")
    ap.add_argument("--add-node", metavar="JSON", help="追加节点（json 文件或 json 字符串）")
    args = ap.parse_args()

    cfg = _load()

    if args.add_node:
        raw = Path(args.add_node).read_text(encoding="utf-8") if Path(args.add_node).exists() \
            else args.add_node
        nd = json.loads(raw)
        cfg["nodes"] = [n for n in cfg["nodes"] if n["id"] != nd["id"]] + [nd]
        print(f"+ 追加节点 {nd['id']}")

    if args.rollback:
        ids = [n["id"] for n in cfg["nodes"]]
        if args.rollback not in ids:
            raise SystemExit(f"未知节点 {args.rollback}；可选：{', '.join(ids)}")
        cfg["baseline"] = args.rollback
        for n in cfg["nodes"]:
            if n["id"] == args.rollback:
                n["status"] = "baseline"
            elif n.get("status") == "baseline":
                n["status"] = "superseded"
        (ROOT / "BASELINE.txt").write_text(args.rollback, encoding="utf-8")
        print(f"↩ 基线回滚 -> {args.rollback}")

    _save(cfg)

    if ASSETS.exists():
        shutil.rmtree(ASSETS, ignore_errors=True)
    print("准备图片资源 ...")
    pack = prepare_assets(cfg)

    OUT.write_text(build_html(cfg, pack), encoding="utf-8")
    n_shots = sum(len(v) for v in pack.values())
    print(f"OK  {OUT}   （节点 {len(cfg['nodes'])} 个 / 对比图 {n_shots} 张）")


if __name__ == "__main__":
    main()
