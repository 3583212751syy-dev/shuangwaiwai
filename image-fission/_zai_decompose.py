# -*- coding: utf-8 -*-
"""多模型 VLM 精细拆解引擎：把原图拆成结构化 spec（色板/构图/元素/文本区/肌理/工艺）。
语义交给 VLM，像素交给 _zai_pxmeasure.py 实测。

用法:
  python _zai_decompose.py --img <原图> --model gemma3:12b --out <json>
  python _zai_decompose.py --img <原图> --model all --out-dir <目录>   # 四模型都跑
  python _zai_decompose.py --ensemble <目录> --out <融合json>

四模型: gemma3:12b / qwen3-vl:8b / qwen2.5vl:7b / minicpm-v:8b
"""
import argparse, base64, json, os, sys, time, urllib.request, urllib.error

MODELS = ["gemma3:12b", "qwen3-vl:8b", "qwen2.5vl:7b", "minicpm-v:8b"]
HOST = "http://127.0.0.1:11434"

DECOMPOSE_PROMPT = r"""你是一位顶级海报视觉拆解分析师。请对这张图做【显微镜级】的完整拆解，目的是让另一个 AI 生成模型能据此 1:1 复刻这张图的色彩、构图、纹理肌理、元素种类数量大小、文字排版位置（但主体元素本身会换成新形态）。

请逐项精确观察并输出严格 JSON（不要输出任何多余文字，不要 markdown 代码块）：

{
 "canvas": {"orientation":"portrait/landscape", "aspect_ratio":"约值如 0.78"},
 "background": {
   "base_color_rgb": [r,g,b] 估计值, "color_name":"中文名",
   "is_flat_or_gradient":"纯色/渐变/噪点纹理/纸张纹理",
   "vignette":"有无暗角", "empty_margin":"四边留白比例描述",
   "texture_detail":"底纹颗粒/划痕/网点/污渍的具体描述"
 },
 "palette": [
   {"role":"主色/辅色/点缀色", "rgb":[r,g,b]估计, "hex":"#RRGGBB",
    "name":"中文色名", "approx_share_pct":数字, "where":"出现在哪些元素上"}
 ],
 "composition": {
   "layout":"整体布局一句话（如：上标题带+中央主体+下标语带）",
   "zones":[
     {"zone":"顶部/中上/中央/中下/底部/左/右",
      "y_range_pct":[起点,终点], "content":"该区内容", "occupies_width_pct":数字}
   ],
   "main_subject":"主体是什么", "main_width_pct":数字, "main_center":"x%,y%",
   "overlap_relation":"谁在谁前面/遮挡关系", "symmetry":"对称/不对称"
 },
 "elements": [
   {"type":"元素类别（人物/车辆/盾牌/火焰/缎带/徽章/星星/箭矢/字母牌...）",
    "count":数量, "size_pct":"占画布宽x高百分比 如 30x25",
    "center":"x%,y%", "shape":"具体形状描述",
    "color":"颜色", "detail":"细节特征（工艺/纹样/朝向/姿态）",
    "zf_layer":"背景层/中层/前景层"}
 ],
 "text_zones": [
   {"content":"读到的文字（读不出写 unreadable）",
    "position":"语义方位（如 盾牌上部铭牌/底部横带/左圆章内）",
    "center_pct":"x%,y%", "box_wh_pct":"约宽x高百分比",
    "arrangement":"水平/弧形/倾斜/竖排", "tilt_deg":数字或0,
    "font_feel":"字体感觉（粗黑无衬线/花体/做旧衬线/徽章体）",
    "weight":"字重（细/中/粗/特粗）", "color":"字色",
    "has_plate":"是否有底板/铭牌/描边", "plate_color":"底板色"}
 ],
 "texture_craft": {
   "craft":"工艺（丝网印刷/石版/金属浮雕/木刻/做旧照片）",
   "surface":"表面肌理（颗粒/网点/划痕/磨损/凸凹/光泽）",
   "edge_quality":"边缘（硬切/柔和/毛边）",
   "aging":"做旧程度与表现", "signature_technique":"最独特的技法特征"
 },
 "style_keywords": ["用于生成 prompt 的英文风格关键词，10-20个"],
 "fidelity_warning": ["最容易做错、必须严格保留的3-5个点（中文）"]
}"""

def b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()

def _extract_content(msg):
    """qwen3-vl 有 thinking 模式：内容可能落在 thinking 字段，content 为空。"""
    content = (msg.get("content") or "").strip()
    if content:
        return content
    return (msg.get("thinking") or "").strip()

def _parse_json(text):
    """从可能带 markdown 包裹/前后缀的文本里抠出 JSON。"""
    t = text.strip()
    if not t:
        return None, "EMPTY_RESPONSE"
    # 去 markdown 代码块
    if "```" in t:
        parts = t.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                t = p
                break
    # 直接解析
    try:
        return json.loads(t), None
    except Exception:
        pass
    # 截取首个 { 到末个 }
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(t[i:j+1]), None
        except Exception as e:
            return None, "PARSE_ERR:" + str(e)[:120]
    return None, "NO_JSON:" + t[:80]

def decompose(img, model="gemma3:12b", host=HOST, retries=2, timeout=1500):
    """单模型拆解，返回 (spec_dict, meta)。带重试 + format=json + thinking 兜底。"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": DECOMPOSE_PROMPT,
                      "images": [b64(img)]}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.15, "num_ctx": 8192},
    }
    r = urllib.request.Request(host + "/api/chat",
                               data=json.dumps(payload).encode("utf-8"),
                               headers={"Content-Type": "application/json"})
    last_raw, last_err = "", ""
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            raw = _extract_content(data.get("message", {}))
            last_raw = raw
            spec, err = _parse_json(raw)
            if spec is not None:
                return spec, {"model": model, "ok": True, "attempt": attempt + 1,
                              "chars": len(raw), "eval_count": data.get("eval_count", 0)}
            last_err = err
        except Exception as e:
            last_err = "REQ_ERR:" + str(e)[:160]
        if attempt < retries:
            time.sleep(2)
    return None, {"model": model, "ok": False, "error": last_err,
                  "chars": len(last_raw), "raw_head": last_raw[:300]}

def _num(x):
    try:
        return float(x)
    except Exception:
        return None

def ensemble(specs):
    """多模型融合：元素取并集去重、色板按角色聚类取平均 RGB、其余取字段最全者。"""
    ok = {m: s for m, s in specs.items() if s}
    if not ok:
        return None
    # 基底：取字段最全（JSON 字符串最长）的一份
    base = max(ok.items(), key=lambda kv: len(json.dumps(kv[1], ensure_ascii=False)))[1]
    out = json.loads(json.dumps(base))  # deep copy

    # palette 合并：VLM 提供语义色名；占比/RGB 一律以程序实测为准（见 spec2prompt）
    # 只保留 votes>=2 的共识色 + 各模型主色，剔除 votes=1 的零星垃圾色
    pal, seen = [], {}
    for m, s in ok.items():
        for idx, c in enumerate(s.get("palette", []) or []):
            name = (c.get("name") or c.get("hex") or "").strip()
            if not name:
                continue
            key = name
            if key not in seen:
                seen[key] = {"c": dict(c), "models": set(), "rank": idx}
            seen[key]["models"].add(m)
    for name, rec in seen.items():
        if len(rec["models"]) < 2 and rec["rank"] > 0:
            continue  # 非主色且只有单模型提及 → 丢弃
        c0 = rec["c"]
        c0["_voted_by"] = len(rec["models"])
        pal.append(c0)
    pal.sort(key=lambda c: (-c.get("_voted_by", 0), c.get("approx_share_pct") or 0))
    if pal:
        out["palette"] = pal[:8]

    # elements 并集：同义归一 + 占位符过滤 + 数量投票
    SYN = {"盾牌状徽章":"盾牌", "盾牌徽章":"盾牌", "盾": "盾牌", "圆形徽章":"圆形徽章",
           "圆章":"圆形徽章", "徽章":"徽章", "文字带":"缎带", "横带":"缎带"}
    PLACEHOLDER = ("元素类别", "type", "元素", "object", "element", "...")
    elems, eseen = [], {}
    for m, s in ok.items():
        for e in s.get("elements", []) or []:
            key = (e.get("type") or "").strip()
            if not key or any(p in key for p in PLACEHOLDER) or len(key) > 14:
                continue
            key = SYN.get(key, key)
            if key not in eseen:
                e2 = dict(e)
                e2["type"] = key
                e2["_models"] = set()
                eseen[key] = e2
            eseen[key]["_models"].add(m)
            cnt = _num(e.get("count"))
            if cnt is not None:
                eseen[key].setdefault("_counts", []).append(cnt)
    for key, e2 in eseen.items():
        e2["_voted_by"] = len(e2.pop("_models"))
        cs = e2.pop("_counts", [])
        if cs:
            e2["count"] = int(round(sum(cs) / len(cs)))
        elems.append(e2)
    elems.sort(key=lambda e: -e.get("_voted_by", 0))
    if elems:
        out["elements"] = elems

    # text_zones 并集
    tz, tseen = [], set()
    for m, s in ok.items():
        for t in s.get("text_zones", []) or []:
            key = (t.get("position") or t.get("content") or "").strip()
            if not key or key in tseen:
                continue
            tseen.add(key)
            tz.append(t)
    if tz:
        out["text_zones"] = tz

    out["_ensemble_from"] = list(ok.keys())
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default="")
    ap.add_argument("--model", default="gemma3:12b", help="模型名或 all")
    ap.add_argument("--out", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--ensemble", default="", help="对目录下 *_spec.json 做融合")
    a = ap.parse_args()

    if a.ensemble:
        specs = {}
        for fn in sorted(os.listdir(a.ensemble)):
            if fn.endswith("_spec.json"):
                with open(os.path.join(a.ensemble, fn), "r", encoding="utf-8") as f:
                    d = json.load(f)
                specs[d.get("meta", {}).get("model", fn.replace("_spec.json", ""))] = d.get("spec")
        merged = ensemble(specs)
        print(json.dumps(merged, ensure_ascii=False, indent=2))
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
        return

    if not a.img:
        ap.error("需要 --img 或 --ensemble")
    if a.out:
        outdir = os.path.dirname(a.out) or "."
    else:
        outdir = a.out_dir or "."
    os.makedirs(outdir, exist_ok=True)
    models = MODELS if a.model == "all" else [a.model]
    all_specs = {}
    for m in models:
        t0 = time.time()
        spec, meta = decompose(a.img, m)
        meta["elapsed_s"] = round(time.time() - t0, 1)
        print(f"[{m}] {'OK' if spec else 'FAIL'} {meta}", file=sys.stderr)
        all_specs[m] = spec
        outp = a.out if (a.out and len(models) == 1) else os.path.join(outdir, f"{m.replace(':','_')}_spec.json")
        with open(outp, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "spec": spec}, f, ensure_ascii=False, indent=2)
    if len(models) > 1:
        merged = ensemble(all_specs)
        if merged:
            with open(os.path.join(outdir, "ensemble_spec.json"), "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            print(f"[ensemble] saved {len(merged.get('elements',[]))} elements, "
                  f"{len(merged.get('palette',[]))} colors", file=sys.stderr)

if __name__ == "__main__":
    main()
