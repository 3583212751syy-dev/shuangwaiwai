# -*- coding: utf-8 -*-
"""补写 C5/D5，并对 11/1-11/4 全部 12 格做独立只读复核。

方向键失效修复：不再依赖单个缓存容器；每次按键前重新扫描，
对「名称框之前」所有可聚焦网格容器依次 SetFocus 重试，直到名称框真的变化。
全程后台，只在开头激活一次窗口。
"""
import ctypes, sys, time, json
from runtime_config import state_file
import write_return_code as w
import write_ads_cost as a

u = ctypes.windll.user32
REMAIN = [("C5", 442), ("D5", 186)]
EXPECT = {
    "B2": "450", "C2": "40", "D2": "54",
    "B3": "1159", "C3": "546", "D3": "210",
    "B4": "1258", "C4": "546", "D4": "281",
    "B5": "1262", "C5": "442", "D5": "186",
}


def containers(out):
    """名称框之前的所有可聚焦网格容器（近的优先）"""
    p = None
    for k, (i, ct, nm, e) in enumerate(out):
        if a.CELLREF.match(nm):
            p = k
            break
    if p is None:
        return []
    res = []
    for k in range(p - 1, -1, -1):
        i, ct, nm, e = out[k]
        if ct in (50026, 50030) and e is not None:
            try:
                if e.CurrentIsKeyboardFocusable:
                    res.append(e)
            except Exception:
                continue
    return res[:8]


def press(vk, cur):
    """按键直到名称框变化；返回新名称框或原值"""
    for pass_ in range(3):
        for e in containers(a.scan()):
            try:
                e.SetFocus()
            except Exception:
                continue
            time.sleep(0.12)
            w.key(vk)
            time.sleep(0.45)
            new = a.get_ref(a.scan())
            if new and new != cur:
                return new
    return cur


def goto(target):
    if w.goto_by_name_box(target):
        return True
    for step in range(40):
        out = a.scan()
        cur = a.get_ref(out)
        if not cur:
            return False
        if cur == target:
            return True
        c, r = a.split(cur)
        tc, tr = a.split(target)
        vk = 0x27 if c < tc else 0x25 if c > tc else 0x28 if r < tr else 0x26
        new = press(vk, cur)
        if new == cur:
            a.log(f"  方向键仍失效: {cur} -> {target}")
            return False
    return a.get_ref(a.scan()) == target


def write_cell(target, value):
    for attempt in range(3):
        if not goto(target):
            continue
        if not w.focus_grid_for_cell(target):
            return False, '已定位单元格但网格输入焦点未验证，未粘贴'
        time.sleep(0.2)
        if not a.paste_value(value):
            continue
        time.sleep(0.8)
        if not goto(target):
            continue
        v = a.get_formula(a.scan())
        if v is not None and v.strip() == str(value):
            return True, v
        a.log(f"  {target} 第{attempt+1}轮 真值={v!r}")
    return False, a.get_formula(a.scan())


def main():
    a.main = w.ding_main_hwnd()
    w.focus_hwnd(a.main, settle=2.0)
    if not a.ensure_sheet():
        raise RuntimeError("「广告费消耗」表格未就绪")

    for target, value in REMAIN:
        ok, v = write_cell(target, value)
        a.log(f"  {target} <- {value}: {'OK' if ok else 'FAIL'} 真值={v!r}")
        if not ok:
            print("FAIL", target, v, flush=True)
            return 1

    # 全量独立复核：逐格跳过去读公式栏真值（不写入）
    verify = {}
    for target in EXPECT:
        if not goto(target):
            verify[target] = {"ok": False, "note": "无法定位"}
            continue
        v = a.get_formula(a.scan())
        verify[target] = {"ok": v is not None and v.strip() == EXPECT[target],
                          "真值": v, "期望": EXPECT[target]}
    print("\n=== 12 格复核 ===", flush=True)
    for k, v in verify.items():
        print(f"  {k}: {'OK' if v['ok'] else 'FAIL'} 真值={v.get('真值')!r} 期望={v.get('期望')}", flush=True)
    ok_all = all(v["ok"] for v in verify.values())
    print("ALL_OK=" + str(ok_all), flush=True)
    with open(state_file("ads_cost_verify.json"),
              "w", encoding="utf-8") as f:
        json.dump(verify, f, ensure_ascii=False, indent=2)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit("升级版只允许从 daily_701_oneflow.py 启动；本文件仅提供辅助函数。")

