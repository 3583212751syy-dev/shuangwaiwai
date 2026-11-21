# -*- coding: utf-8 -*-
"""本地识图 AI 审图：qwen3-vl 对比 原图 vs 成品，输出结构化裁决。
用法:
  python _zai_vreview.py --orig <原图> --final <成品> --req <逐图要求文本> [--model qwen3-vl:8b] [--out <json>]
"""
import argparse, base64, json, sys, io, os

PROMPT_TMPL = """你是苛刻的海报质检员。第1张图是【原图】（只作色彩/构图/主题/工艺参考），第2张图是【成品】（AI裂变产物，要求主体元素全换新形态但风格回归原图）。

逐图硬要求：
{req}

通用铁律：
1. 色板/色值/色调必须与原图一致，任何改色、色族迁移=FAIL
2. 主体元素（人物/车/火焰/盾/装饰）必须是「同物种不同形态」的新图，只是改字=FAIL
3. 构图布局（主体位置/占比/朝向/文字区域分布）必须贴近原图
4. 工艺质感（做旧印刷感/描边/阴影/字重）必须回归原图水准，出现平涂色块背景、补丁边界=FAIL
5. 无乱码伪字、无残缺元素、无糊团融块，精美度不得低于原图
6. 有文字处拼写必须正确

请严格对比两张图，输出JSON（不要多余文字）：
{{
 "color_match": "PASS/FAIL+一句话证据（给出成品与原图主色调偏差描述）",
 "composition_match": "PASS/FAIL+一句话证据（主体位置占比布局对比）",
 "element_fission": "PASS/FAIL+一句话证据（主体元素是否为新形态而非照搬/改字）",
 "craft_quality": "PASS/FAIL+一句话证据（工艺质感字重对比，指名具体缺陷）",
 "defects": ["缺陷1（具体到位置，如：右下角伪字/盾面乱码/某元素残缺/某区域平涂色块）", ...],
 "spell_check": "PASS/FAIL+列出所有拼写错误词",
 "verdict": "PASS/FAIL",
 "fix_hint": "若FAIL，给一句最关键的生成层修改建议（prompt层面，中文）"
}}"""

REQS = {
    "04": "深棕金盾：排版靠原图（盾+弧形环带+左右两枚圆形徽章的组合几何、文字位置照原图风格），徽章内容换词，深棕底+金色盾工艺，字重回原图",
    "05": "恶魔女郎黑车：人车四肢不缺块，构图比例贴原图（盾约1/3宽居中、女郎在盾左、黑车在盾右、火焰左下），砖红主色调，文字区域分布照原图",
    "06": "上黑下锈徽：盾亮黑金珐琅质感（非铜绿古旧），上黑下锈橙麻布背景，背景构图与颜色照原图，环带弧形文字",
    "07": "紫调加油站：低饱和灰紫暗调（非鲜紫），印刷做旧质感关联原图，无纸张边缘，斜置路牌上的文字沿牌面倾斜",
    "08": "蓝半挂卡车：上部白块+黑色斜条带+圆形徽章布局，糊团必须变成清晰合理个体，卡车细节精美",
    "10": "涂鸦show time：元素竖列分散排版（非环形围一圈），小元素多参考原图种类与画法，多彩涂鸦风",
    "11": "黑底金字st-tropez：元素群密集加大、黑金配色大气，小元素画法参考原图，装饰华丽",
}

def b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()

def review(orig, final, req, model="qwen3-vl:8b", host="http://127.0.0.1:11434"):
    import urllib.request
    prompt = PROMPT_TMPL.replace("{req}", req)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt,
                      "images": [b64(orig), b64(final)]}],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 8192},
    }
    r = urllib.request.Request(host + "/api/chat",
                               data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=1200) as resp:
        data = json.loads(resp.read().decode())
    return data["message"]["content"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--req", default="")
    ap.add_argument("--model", default="qwen3-vl:8b")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    txt = review(a.orig, a.final, a.req, a.model)
    print(txt)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(txt)

if __name__ == "__main__":
    main()
