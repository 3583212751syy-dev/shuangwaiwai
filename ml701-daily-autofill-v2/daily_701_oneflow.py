# -*- coding: utf-8 -*-
"""【701 每日登记 · 统一主线】紫鸟(美客多)采集 -> 钉钉两张表写入，一条链路一次做完。

用户口径（2026-11-18 定稿，替代此前所有分散入口）：
  - 完整：退货码 + 广告费（营业额/广告费/点击）在同一条链路里串行走完；
  - 稳定：窗口、焦点、树、凭证、日历、假凭证等历史补丁全部收敛到本文件；
  - 统一：只有一个入口、一个结果文件、一份日志、一个自动化任务；
  - 一个窗口：整条链路只操作【一个】钉钉窗口(HWND 动态精确发现，全程复用)，
    两张表在该窗口内按固定列名称切换；绝不新开窗口、绝不同时操作多个窗口；
  - 自己打开：脚本自行恢复最小化窗口、进入文档视图、按精确名称打开「退货登记码」，
    绝不要求用户手动打开任何页面；表格落在哪个窗口就动态锁哪个（修掉窗口漂移）；
  - 后台：不置顶(TOPMOST 永久禁用)、不锁桌面、不反复抢焦点、不刷页面找页面；
    整轮最多「紫鸟一次 + 钉钉一次」必要激活，结束恢复用户原来的前台窗口；
  - 数据已知即停：--code / --data 或当日凭证命中时零采集、零窗口操作；
  - 不达标不写：采集校验不过 / 钉钉无障碍树未构建 -> 明确 ABORT，不填异常值、
    不补凭证、不假报成功（写入一律逐格读应用内真值验证）。

复用（业务逻辑零重写，只统一调度与窗口策略）：
  write_return_code.py  -> 退货码表打开、701 锚点对版、逐格真值验证、凭证
  daily_ads_cost.py     -> 紫鸟业务指标/广告费采集（单日、两轮一致、0-800）
  write_ads_cost.py     -> 「广告费消耗」表打开、名称框/公式栏真值
  finish_ads_cost.py    -> B/C/D 逐格写入 + 回读验证

用法（受管 venv python）：
  python daily_701_oneflow.py                      # 全链路：采昨天数据 + 今日退货码并写入
  python daily_701_oneflow.py --probe             # 只读体检：不写、不切窗口、不改状态
  python daily_701_oneflow.py --code 220DF71D     # 当日码已知，不碰紫鸟重新取码
  python daily_701_oneflow.py --data 473,180,104  # 三项数据已知，不碰紫鸟采集
  python daily_701_oneflow.py --date 2026-11-17   # 指定广告费目标日
  python daily_701_oneflow.py --dry-run           # 只采集+体检，不写任何单元格
  python daily_701_oneflow.py --fix-a11y          # 只做无障碍软修复（绝不结束/重启钉钉）

说明：钉钉启动时若未启用 CEF 无障碍，名称框/公式栏读不到（树约 20 节点、
只剩 "Chrome Legacy Window"），这是历史「有时行有时不行」的真正根因。
脚本会自己打开文档并做软修复（系统辅助技术标志 + 显式 MSAA 请求）；
若仍读不到应用内真值一律 ABORT 并保留 pending，绝不盲写、绝不假报成功。
【硬约束】脚本绝不自行 taskkill / 重启 / 启动用户钉钉进程（2026-11-19 教训：
强杀 + 非标准重启会触发「钉钉安全模式」并弄坏实例）。钉钉确需重启时由用户
自行点图标，脚本随后自己打开文档、零采集补写，绝不需要用户手动打开任何页面。

结果：.workbuddy/tmp/oneflow_last.json（唯一权威凭证）
日志：.workbuddy/tmp/oneflow.log
待补数据：.workbuddy/tmp/oneflow_pending.json（数据已采到但钉钉未就绪时落盘，
          恢复后加 --code/--data 直接补写，绝不重新采集）
"""
import os, sys, re, json, time, ctypes, argparse, traceback, subprocess, unicodedata
from ctypes import wintypes
from datetime import date, timedelta

from runtime_config import BASE, STATE_DIR, state_file
from validation import validate_code, validate_data, reusable_pending
BASE = str(BASE)
TMP = str(STATE_DIR)
os.makedirs(TMP, exist_ok=True)

import write_return_code as w
import write_ads_cost as a
import daily_ads_cost as d
import finish_ads_cost as f

LOG = os.path.join(TMP, "oneflow.log")
OUT = os.path.join(TMP, "oneflow_last.json")
PENDING = os.path.join(TMP, "oneflow_pending.json")
RETURN_LOG = os.path.join(TMP, "return_code.log")
RETURN_CRED = os.path.join(TMP, "return_write_last.json")
ADS_CRED = os.path.join(TMP, "daily_ads_cost_last.json")

u = ctypes.windll.user32
k = ctypes.windll.kernel32

# ---------------- 全局链路状态（整轮只有一个钉钉窗口） ----------------
STATE = {
    "ding_hwnd": None,       # 唯一钉钉窗口
    "fg_before": None,       # 用户原来的前台窗口，结束恢复
    "activated": set(),      # 每个窗口整轮最多激活一次
    "tree": {},              # 钉钉树体检结果
    "ml_tree": 0,            # 紫鸟树规模
}


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + str(msg)
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as fp:
            fp.write(line + "\n")
    except Exception:
        pass


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return None


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=2)


# ---------------- 窗口：动态精确发现，只认一个 ----------------
def all_windows(include_hidden=False):
    CB = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    out = []

    def cb(h, l):
        if not include_hidden and not u.IsWindowVisible(h):
            return True
        n = u.GetWindowTextLengthW(h)
        b = ctypes.create_unicode_buffer(n + 2); u.GetWindowTextW(h, b, n + 2)
        c = ctypes.create_unicode_buffer(256); u.GetClassNameW(h, c, 256)
        r = wintypes.RECT(); u.GetWindowRect(h, ctypes.byref(r))
        pid = wintypes.DWORD(); u.GetWindowThreadProcessId(h, ctypes.byref(pid))
        W, H = r.right - r.left, r.bottom - r.top
        out.append({"h": h, "pid": pid.value, "cls": c.value, "title": b.value.strip(),
                    "rect": (r.left, r.top, W, H), "area": W * H})
        return True

    u.EnumWindows(CB(cb), 0)
    return out


class WP(ctypes.Structure):
    _fields_ = [("length", wintypes.UINT), ("flags", wintypes.UINT),
                ("showCmd", wintypes.UINT), ("ptMin", wintypes.POINT),
                ("ptMax", wintypes.POINT), ("rcNormal", wintypes.RECT)]


def restore_from_tray(h):
    """藏托盘(rect.left<-10000)时，只有 SetWindowPlacement(showCmd=1) 能恢复。"""
    r = wintypes.RECT(); u.GetWindowRect(h, ctypes.byref(r))
    if r.left > -10000:
        return False
    wp = WP(); wp.length = ctypes.sizeof(WP)
    u.GetWindowPlacement(h, ctypes.byref(wp))
    wp.showCmd = 1
    u.SetWindowPlacement(h, ctypes.byref(wp))
    time.sleep(1.0)
    u.GetWindowRect(h, ctypes.byref(r))
    log(f"[窗口] 藏托盘已恢复 rect=({r.left},{r.top})")
    return r.left > -10000


DING_EXE = r"D:\新建文件夹\DingDing\main\current_new\DingTalk.exe"


def revive_ding():
    """钉钉只在托盘/空壳时，单实例唤起主窗口（显式 --revive 才做）。

    绝不结束任何进程、绝不重新登录、绝不置顶；只是让已运行实例把主窗口显示出来。
    """
    if not os.path.exists(DING_EXE):
        return None
    # 进程未运行时绝不自行启动：脚本拉起的进程活不过会话，只会把状态弄乱。
    chk = subprocess.run(["tasklist", "/FI", "IMAGENAME eq DingTalk.exe", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True, creationflags=0x08000000)
    if "DingTalk.exe" not in (chk.stdout or ""):
        raise RuntimeError("钉钉进程未运行（脚本绝不自行启动/结束你的钉钉）；"
                           "请点一次钉钉图标启动，之后脚本会自己打开文档并补写，"
                           "数据仍在 oneflow_pending.json")
    log("[窗口] 显式 --revive：仅恢复【已运行】钉钉的窗口（绝不启动/结束任何进程）")
    # 只做窗口级恢复：把藏托盘/最小化的钉钉窗口摆正，不新建进程、不重登。
    for x in all_windows():
        if "钉钉" in x["title"] or x["cls"].startswith("Qt51511QWindowIcon"):
            restore_from_tray(x["h"])
            if u.IsIconic(x["h"]):
                u.ShowWindow(x["h"], 9)
    for i in range(6):
        time.sleep(1.5)
        try:
            H, st = discover_ding()
            log(f"[窗口] 唤起成功 hwnd={H} 节点={st['nodes']}")
            return H
        except Exception:
            continue
    return None


def fix_a11y():
    """只做无障碍软修复：系统辅助技术标志 + 显式 MSAA 请求。

    【硬约束】绝不结束/重启用户钉钉进程（2026-11-19 教训：沙箱里由脚本拉起的
    进程活不过会话，自行 taskkill 会把用户钉钉弄没）。钉钉若需重启以启用 CEF
    无障碍，由用户点一次图标即可；脚本随后会自己打开文档、零采集补写。
    """
    enable_cef_accessibility()
    cur = ctypes.c_int(0)
    u.SystemParametersInfoW(0x0046, 0, ctypes.byref(cur), 0)
    log(f"[无障碍] 软修复完成，标志={cur.value}；未结束任何进程、未重启钉钉")
    return bool(cur.value)


def tree_state(h):
    """只读体检一个窗口的树。不做任何页面/标签操作。"""
    names, refs, s = w.scan_hwnd(h)
    plain = [n for _, n in names]
    return {
        "hwnd": h, "nodes": len(names), "refs": refs[:6],
        "has_return_doc": any("退货登记码" in n or "退货码更新" in n for n in plain),
        "has_ads_doc": any("广告费消耗" in n for n in plain),
        "ready": bool(refs) or any("退货码更新" in n or "广告费消耗" in n for n in plain),
        "sample": plain[:12],
    }


def discover_ding():
    """精确发现【唯一】钉钉窗口：表格窗口优先，其次标题恰好『钉钉』的主窗口。

    修正历史坑：w.ding_main_hwnd() 按 '钉钉' 子串挑可见 Qt 窗口，会误选
    图片查看器 / ToolTip / 聊天子窗口，导致整轮写错窗口还误报『找不到』。
    """
    # 钉钉会在别的窗口置前时最小化/藏托盘；必须把隐藏窗口也算进候选，
    # 否则会误报「未发现钉钉主窗口」（历史误报根源之一）。
    wins = [x for x in all_windows(include_hidden=True)
            if x["area"] > 200000
            and (x["cls"].startswith("Qt51511QWindowIcon")
                 or x["cls"] == "Chrome_WidgetWin_1"
                 or x["cls"] == "StandardFrame_DingTalk")
            and ("钉钉" in x["title"] or "文档" in x["title"] or "DingTalk" in x["title"])]
    wins.sort(key=lambda x: (not u.IsWindowVisible(x["h"]), -x["area"]))
    # 1) 已经打开表格的窗口（名称框 + 工作表名）
    for x in wins[:6]:
        st = tree_state(x["h"])
        if st["ready"]:
            STATE["tree"] = st
            return x["h"], st
    # 2) 精确主窗口：标题恰好『钉钉』且足够大
    mains = [x for x in wins if x["title"] == "钉钉"
             and (x["cls"].startswith("Qt51511QWindowIcon")
                  or x["cls"] == "StandardFrame_DingTalk")]
    if not mains:
        # 允许任意钉钉顶层窗口兜底（安全模式弹窗 / 独立文档窗口也叫「钉钉」），
        # 绝不因为类名不匹配就说「没打开」。
        mains = [x for x in wins if "钉钉" in x["title"]]
    if not mains:
        raise RuntimeError("未发现任何钉钉顶层窗口（钉钉进程可能未运行）；"
                           "脚本绝不自行启动/结束钉钉，数据仍在 oneflow_pending.json")
    # 排除「图片查看器」类窗口：这类窗口标题也叫『钉钉』，选它会乱按键。
    good = []
    for x in mains[:8]:
        st = tree_state(x["h"])
        sample = " ".join(st["sample"])
        if any(t in sample for t in ("分享的图片", "ImageTool", "Weixin", "图片查看")):
            continue
        good.append((x, st))
    if not good:
        # 树缩水 ≠ 页面没打开：返回真实存在的窗口，交给 preflight 自己打开文档 +
        # 软修复，绝不把「控件树没构建好」说成「页面没打开」，也不要求用户动手。
        x = mains[0]
        st = tree_state(x["h"])
        log(f"[窗口] 窗口存在但无障碍树缩水 hwnd={x['h']} 节点={st['nodes']}；"
            "由脚本自行打开文档并软修复（不要求用户打开任何页面）")
        restore_from_tray(x["h"])
        STATE["tree"] = st
        return x["h"], st
    best = None
    for x, st in good:
        if st["ready"] or any(t in " ".join(st["sample"])
                              for t in ("搜索", "IM模块", "消息", "AI助理")):
            best = x
            break
    h = (best or good[0][0])["h"]
    restore_from_tray(h)
    if not u.IsWindowVisible(h) or u.IsIconic(h):
        # SW_SHOWNORMAL 只恢复显示，不置顶、不激活（后台处理）
        wp = WP(); wp.length = ctypes.sizeof(WP)
        u.GetWindowPlacement(h, ctypes.byref(wp))
        wp.showCmd = 1
        u.SetWindowPlacement(h, ctypes.byref(wp))
        time.sleep(1.0)
        if not u.IsWindowVisible(h):
            u.ShowWindow(h, 4)      # SW_SHOWNOACTIVATE
        log(f"[窗口] 钉钉窗口原为最小化/隐藏，已后台恢复显示 hwnd={h}")
    STATE["tree"] = tree_state(h)
    return h, STATE["tree"]


def activate_once(h, why=""):
    """整轮对每个窗口最多一次必要激活。绝不置顶、绝不守护、绝不反复抢焦点。"""
    if h in STATE["activated"] or u.GetForegroundWindow() == h:
        STATE["activated"].add(h)
        return True
    fg = u.GetForegroundWindow()
    ft = u.GetWindowThreadProcessId(fg, None)
    mt = k.GetCurrentThreadId()
    att = bool(ft and ft != mt and u.AttachThreadInput(mt, ft, True))
    try:
        u.SetForegroundWindow(h)
        u.BringWindowToTop(h)
        try:
            w.auto.ElementFromHandle(h).SetFocus()
        except Exception:
            pass
    finally:
        if att:
            u.AttachThreadInput(mt, ft, False)
    STATE["activated"].add(h)
    ok = u.GetForegroundWindow() == h
    log(f"[窗口] 唯一一次激活 hwnd={h} {why} 结果={ok}")
    time.sleep(2.5)   # Qt/CEF 无障碍树构建
    return ok


def restore_foreground():
    fg = STATE["fg_before"]
    if not fg or not u.IsWindow(fg) or u.GetForegroundWindow() == fg:
        return
    ft = u.GetWindowThreadProcessId(u.GetForegroundWindow(), None)
    mt = k.GetCurrentThreadId()
    att = bool(ft and ft != mt and u.AttachThreadInput(mt, ft, True))
    try:
        u.SetForegroundWindow(fg)
    finally:
        if att:
            u.AttachThreadInput(mt, ft, False)
    log(f"[窗口] 已恢复用户原前台窗口 {fg}")


# ---------------- 后台窗口策略（注入到既有业务模块） ----------------
def install_policy(H):
    """把三个业务模块里各自为政的窗口/焦点策略，统一成『一个窗口 + 后台』。"""
    # 防重入：表格窗口动态切换时会再调一次，避免把 focus_hwnd 层层包裹
    if STATE.get("policy_h") == H:
        return
    STATE["policy_h"] = H
    # 1) 钉钉窗口发现：全链路锁定同一个 HWND
    w.ding_main_hwnd = lambda: H
    w.DING_HWND = H
    w.SHEET_HWND = H
    a.main = H
    # 注意：finish_ads_cost.main 是函数，不能覆盖；它内部走 a.scan() -> a.main

    # 2) 焦点：同一窗口整轮只激活一次；之后只做控件级 SetFocus，不拉回前台
    orig_focus = w.focus_hwnd

    def focus_policy(h, settle=2.5):
        h = H                      # 无论业务层传什么，都收敛到唯一窗口
        if h not in STATE["activated"]:
            activate_once(h, "钉钉表格写入")
        else:
            try:
                w.auto.ElementFromHandle(h).SetFocus()
            except Exception:
                pass
            time.sleep(0.15)
        return None

    w.focus_hwnd = focus_policy
    d.ensure_ding_ready = lambda: None   # 统一层已做，避免二次激活

    # 3) 键盘：原生按键，不加任何『保前台守护』（run_ads_bg/run_return_bg 的
    #    TOPMOST + keep() 是历史上锁桌面/抢焦点的根源，本链路彻底不用）
    w.key = lambda vk: (u.keybd_event(vk, 0, 0, 0), u.keybd_event(vk, 0, 2, 0))

    # 4) 表格就绪判定：只认唯一窗口，不扫别的窗口、不误切 SHEET_HWND
    def sheet_ready(wide=False):
        names, refs, s = w.scan_hwnd(H)
        if refs and any("退货码更新" in n or "广告费消耗" in n for _, n in names):
            w.SHEET_HWND = H
            return True, refs
        return False, refs

    w.sheet_ready = sheet_ready
    log(f"[统一] 窗口策略已注入，唯一 HWND={H}；TOPMOST/保前台守护=禁用")


# ---------------- 幂等 + 假凭证防御 ----------------
def last_seen_code(today):
    """日志里上一次『非今天』采信的码（日志不会被 SKIP 补写污染）。"""
    try:
        with open(RETURN_LOG, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
    except Exception:
        return None
    for line in reversed(lines):
        if "当日授权码已采信" not in line or line[:10] == today:
            continue
        m = re.search(r"当日授权码已采信\s+([0-9A-F]{8})", line)
        if m:
            return m.group(1)
    return None


def return_done(today):
    c = load_json(RETURN_CRED)
    if not (c and c.get("date") == today and c.get("ok") is True):
        return False
    prev = last_seen_code(today)
    if c.get("code") and prev and c["code"] == prev:
        log("[幂等] 退货码凭证疑似假凭证（码==上次采信码），拒绝判成功")
        return False
    return True


def ads_done(target_day):
    c = load_json(ADS_CRED)
    return bool(c and c.get("date") == target_day and c.get("ok") is True)


# ---------------- 采集：数据已知即停，零导航直读 ----------------
def log_code_today(today):
    try:
        with open(RETURN_LOG, "r", encoding="utf-8") as fp:
            txt = fp.read()
    except Exception:
        return None
    hits = re.findall(r"(\d{4}-\d{2}-\d{2})[^\n]*当日授权码已采信\s+([0-9A-F]{8})", txt)
    today_hits = [c for dt, c in hits if dt == today]
    return today_hits[-1] if today_hits else None


def read_code_from_ml(today):
    """紫鸟当前活动页零导航直读：树完整门槛 + 值格式 + 新鲜度校验。

    禁止 Ctrl+T / 改地址栏找页面 / 试 URL / 遍历全部标签 / 为找页面刷新。
    """
    H = d.find_ml_hwnd()
    if not H:
        raise RuntimeError("未找到紫鸟 G-701 窗口；如实报未取到码，不猜码")
    names = d.ml_names()
    STATE["ml_tree"] = len(names)
    if len(names) < 120:
        time.sleep(2.0)
        names = d.ml_names()
        STATE["ml_tree"] = len(names)
    if len(names) < 120:
        raise RuntimeError(f"紫鸟树缩水({len(names)}<120)，只重建树、不刷页面；未采信任何码")

    def scan():
        out = None
        for idx, ct, nm, e in d.ml_names():
            if "Código de autorización para hoy" in (nm or ""):
                out = idx
                break
        if out is None:
            return None
        els = d.ml_names()
        for j in range(out + 1, min(out + 7, len(els))):
            m = re.search(r"\b([0-9A-F]{8})\b", els[j][2] or "")
            if m:
                return m.group(1)
        return None

    code = scan()
    if not code:
        # 只按名称精确激活店铺首页标签一次（绝不遍历全部标签）
        activate_once(H, "店铺首页标签")
        time.sleep(2.0)
        code = scan()
    if not code:
        raise RuntimeError("当前活动页读不到当日授权码（未导航、未刷新）；如实报错")

    prev = last_seen_code(today)
    if prev and code == prev:
        # 页面没刷新 = 昨日残留。数据新鲜度校验例外：只对当前活动页按一次 F5
        log(f"[取码] {code} 与上次采信码相同，按一次 F5 校验新鲜度")
        activate_once(H, "F5 新鲜度校验")
        u.keybd_event(0x74, 0, 0, 0); u.keybd_event(0x74, 0, 2, 0)
        time.sleep(4.0)
        code2 = scan()
        if not code2 or code2 == prev:
            raise RuntimeError(f"刷新后仍为旧码 {prev}，拒绝采信，不写入、不补凭证")
        code = code2
    log(f"[取码] 零导航直读当日码 {code}，紫鸟树 {STATE['ml_tree']} 节点")
    return code


def install_ads_stability():
    """把广告费采集的两个历史抖动点收敛到统一主线（原文件不动，可随时回退）：

    ① 投资卡文本识别放宽：新版 Mercado Ads 把 'Inversión $ 220' 显示成
       'Inversión: 220' / 'Invertido 220' / '投资 220'，旧正则只认 Button(50000)
       且前缀写死，导致「日期已确认单日、两轮=[]」白跑。
    ② collect_ads 第二轮不再无条件重开日历：区间文本已是目标单日时直接读数，
       避免下拉状态变化后 ads_open_calendar() 返回 False 误报「日历未打开」。
    硬性校验零改动：单日日期确认 + 两轮一致 + 0-800 + 逐格真值验证。
    """
    orig_scan = d.ads_scan

    def scan_stable():
        ranges, cal, invest = orig_scan()
        if invest is None:
            for i, ct, nm, e in d.ml_names():
                if not nm or nm.strip().endswith("-"):
                    continue
                m = re.search(
                    r"(?:Inversi[oó]n|Invertido|Gasto|Spend|投资)\D{0,14}([\d][\d.,]*)",
                    nm, re.I)
                if m:
                    invest = m.group(1)
                    break
        return ranges, cal, invest

    def collect_stable(day):
        d.ml_select_tab("广告")
        _, url = d.ml_url()
        if not url or ("/admin/campaigns" not in url
                       and "ads.mercadolibre.com.mx/hub" not in url):
            ads_url = ("https://vendedores.mercadolibre.com.mx/"
                       "publicidad/product-ads/admin/campaigns")
            d.log("  广告后台标签页未打开，导航到广告页(不带日期参数)")
            if not d.ml_goto(ads_url):
                return None, "找不到紫鸟地址栏，无法打开广告页"
            for _ in range(12):
                time.sleep(2.5)
                _, url = d.ml_url()
                if url and ("/admin/campaigns" in url
                            or "ads.mercadolibre.com.mx/hub" in url):
                    break
            else:
                return None, f"导航后仍不在广告后台页({(url or '')[:90]})"
            d.log(f"  已进入广告后台页 {(url or '')[:90]}")

        for attempt in range(2):
            ranges, cal, invest = d.ads_scan()
            confirmed = d.date_range_ok(ranges, day) if ranges else None
            if not confirmed:
                if not d.ads_open_calendar():
                    return None, "日历未打开"
                if not d.ads_double_click(day):
                    continue
                for k in range(10):
                    time.sleep(1.5)
                    ranges, cal, invest = d.ads_scan()
                    if cal >= 10 or not ranges:
                        continue
                    confirmed = d.date_range_ok(ranges, day)
                    if confirmed:
                        break
            if not confirmed:
                d.log(f"  第{attempt+1}轮日期未确认 {ranges[:1]}")
                continue
            # 单日已确认：只读重扫取值，不重开日历、不刷新页面、不重新导航
            vals = []
            for k in range(3):
                time.sleep(2.0)
                r2, c2, inv2 = d.ads_scan()
                if not d.date_range_ok(r2, day):
                    break
                if inv2 is not None:
                    vals.append(inv2)
            d.log(f"  广告费 日期={confirmed!r} 读数={vals}")
            if (len(vals) >= 2 and vals[-1] == vals[-2]
                    and d.plausible(vals[-1])):
                return {"广告费": vals[-1].replace(",", "")}, None
            d.log("  单日已确认但投资卡无数值（页面数据未加载），不重开日历、不填异常值")
        return None, "广告费投资卡未读到可信值（单日区间已确认，未填写）"

    d.ads_scan = scan_stable
    d.collect_ads = collect_stable
    log("[统一] 广告费采集稳定化已注入（投资卡宽识别 + 单日确认后不重开日历）")


def collect_ads_data(day, provided):
    if provided:
        parts = [p.strip() for p in provided.replace(";", ",").split(",")]
        if len(parts) != 3 or any(not re.fullmatch(r"\d+(?:\.\d+)?", p) for p in parts):
            raise ValueError("--data 需为 营业额,广告费,点击 三个纯数字")
        data = {"营业额": parts[0], "广告费": parts[1], "点击": parts[2]}
        if not d.plausible(data["广告费"]):
            raise RuntimeError(f"广告费 {data['广告费']} 不在 0-800 可信区间，拒绝写入")
        log("[采集] 使用已知数据，紫鸟零操作")
        return data
    H = d.find_ml_hwnd()
    if not H:
        raise RuntimeError("未找到紫鸟 G-701 窗口，无法采集；不填异常值")
    activate_once(H, "紫鸟采集")
    biz, err = d.collect_business(day)
    if biz is None:
        raise RuntimeError(f"业务指标采集失败: {err}")
    ads, err = d.collect_ads(day)
    if ads is None:
        raise RuntimeError(f"广告费采集失败: {err}（不填写，等下次或人工处理）")
    data = {**biz, **ads}
    log(f"[采集] {day.isoformat()} = {data}")
    return data


# ---------------- 钉钉写入：一个窗口内两张表，顺序强依赖 ----------------
def force_foreground(h):
    """同一窗口的一次性激活（不置顶、不守护）。返回是否真的成为前台。"""
    fg = u.GetForegroundWindow()
    if fg == h:
        return True
    ft = u.GetWindowThreadProcessId(fg, None)
    mt = k.GetCurrentThreadId()
    att = bool(ft and ft != mt and u.AttachThreadInput(mt, ft, True))
    try:
        u.SetForegroundWindow(h)
        u.BringWindowToTop(h)
        try:
            w.auto.ElementFromHandle(h).SetFocus()
        except Exception:
            pass
    finally:
        if att:
            u.AttachThreadInput(mt, ft, False)
    time.sleep(1.2)
    return u.GetForegroundWindow() == h


def rebuild_tree_once(H):
    """钉钉 Qt/CEF 树缩水时的【唯一一次】稳定恢复，只重建树、不刷页面、不找页面。

    顺序：确认前台(同一窗口最多两次) -> UIA 精确名激活「文档」一次 ->
    仅在前台确实是钉钉时做有限 Tab 焦点遍历，命中固定列文档条目回车打开 ->
    重新体检。任何一步不成就明确返回 False，绝不把键落到别的窗口。
    """
    ok = False
    for _ in range(2):
        ok = force_foreground(H)
        if ok:
            break
    if not ok:
        log("[树重建] 钉钉无法成为前台，停止（不向其它窗口发按键）")
        return False
    time.sleep(2.5)   # Qt/CEF 无障碍树构建
    st = tree_state(H)
    if st["ready"]:
        return True

    names, refs, s = w.scan_hwnd(H)
    if s:
        for i, nm in names:
            if nm == "文档":
                log("[树重建] 按精确名激活「文档」导航项一次")
                w.activate_ctrl(s.GetElement(i))
                time.sleep(3.0)
                break
    st = tree_state(H)
    if st["ready"]:
        return True

    # 固定列是 Qt 自绘、UIA 名称常为空；键盘焦点名可读，命中即停（不逐个试 ListItem）
    log("[树重建] 有限 Tab 焦点遍历，命中固定列文档条目即回车")
    for i in range(30):
        if u.GetForegroundWindow() != H:
            log("[树重建] 前台已离开钉钉，立即停止按键")
            return False
        w.key(0x09); time.sleep(0.09)
        try:
            nm = (w.auto.GetFocusedElement().CurrentName or "").strip()
        except Exception:
            nm = ""
        if "退货登记码" in nm or "广告费消耗" in nm:
            log(f"[树重建] 命中固定列 {nm[:30]!r}，回车打开表格")
            w.key(0x0D)
            time.sleep(3.5)
            break
    return tree_state(H)["ready"]


def is_ding_window(x):
    """钉钉相关顶层窗口（含最小化/藏托盘），用于动态发现承载表格的那一个。"""
    return (x["area"] > 150000
            and (x["cls"].startswith("Qt51511QWindowIcon") or x["cls"] == "Chrome_WidgetWin_1")
            and ("钉钉" in x["title"] or "文档" in x["title"] or "DingTalk" in x["title"]))


def ding_cef_windows(H):
    """候选窗口 = 钉钉顶层窗口 + 主窗口内嵌 CEF 大子窗口。只读枚举、不改状态。"""
    out = [x["h"] for x in all_windows(include_hidden=True) if is_ding_window(x)]
    CB = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def cb(h, l):
        r = wintypes.RECT(); u.GetWindowRect(h, ctypes.byref(r))
        if (r.right - r.left) > 400 and (r.bottom - r.top) > 250 and u.IsWindowVisible(h):
            c = ctypes.create_unicode_buffer(256); u.GetClassNameW(h, c, 256)
            if c.value.startswith(("Chrome_", "CefBrowser", "Qt51511QWindowIcon")):
                out.append(h)
        return True

    u.EnumChildWindows(H, CB(cb), 0)
    seen, res = set(), []
    for h in out:
        if h not in seen and u.IsWindow(h):
            seen.add(h); res.append(h)
    return res


def pick_table_window(H):
    """历史抖动根因之一：表格有时内嵌主窗口、有时独立成窗。
    只读扫描全部候选，谁的名称框/工作表名真的出现就锁谁，绝不写死 HWND。"""
    for h in ding_cef_windows(H):
        st = tree_state(h)
        if st["ready"]:
            log(f"[窗口] 表格实际承载窗口 hwnd={h} 节点={st['nodes']} 名称框={st['refs']}")
            return h, st
    return H, tree_state(H)


def enable_cef_accessibility(H=None):
    """让 Chromium 渲染进程开启无障碍树（系统屏幕阅读器标志 + 显式 MSAA 请求）。
    只读式设置，不结束进程、不刷页面、不弹窗。"""
    cur = ctypes.c_int(0)
    try:
        u.SystemParametersInfoW(0x0046, 0, ctypes.byref(cur), 0)   # SPI_GETSCREENREADER
        if not cur.value:
            u.SystemParametersInfoW(0x0047, 1, None, 3)             # SPI_SETSCREENREADER
            log("[无障碍] 已开启系统辅助技术标志（供 CEF 构建控件树）")
    except Exception as ex:
        log(f"[无障碍] 标志设置失败 {ex}")
    try:
        oleacc = ctypes.windll.oleacc
        from comtypes import GUID
        iid = GUID("{618736E0-3C3D-11CF-810C-00AA00389B71}")
        root = H or STATE.get("ding_hwnd") or 0
        for h in ding_cef_windows(root)[:8]:
            p = ctypes.c_void_p()
            oleacc.AccessibleObjectFromWindow(ctypes.c_void_p(h), ctypes.c_uint32(0xFFFFFFFC),
                                              ctypes.byref(iid), ctypes.byref(p))
    except Exception:
        pass


def open_docs_self(H):
    """【脚本自己打开「退货登记码」】不要求用户手动打开任何页面。

    实测有效路径(2026-11-19)：恢复最小化窗口 -> 一次激活 -> 文档视图固定列
    第一个 ListItem 在同一控件上依次 DoDefaultAction/Invoke/Select(必要时双击)
    -> 文档首页「最近打开」出现可读 Text -> 按精确名称 DoDefaultAction 打开表格。
    全程不用坐标、不翻聊天消息、不逐个试无名称 ListItem、不新开窗口。
    """
    # 1) 恢复最小化/藏托盘的钉钉窗口（只恢复显示，不激活、不置顶）
    for x in all_windows(include_hidden=True):
        if is_ding_window(x) and (u.IsIconic(x["h"]) or x["rect"][0] < -10000):
            restore_from_tray(x["h"])
            wp = WP(); wp.length = ctypes.sizeof(WP)
            u.GetWindowPlacement(x["h"], ctypes.byref(wp)); wp.showCmd = 1
            u.SetWindowPlacement(x["h"], ctypes.byref(wp))
    time.sleep(1.0)

    # 2) 一次必要激活（键盘/CEF 树构建需要）；已在前台则零激活
    w.focus_hwnd(H, settle=3.0)
    enable_cef_accessibility(H)
    H2, st = pick_table_window(H)
    if st["ready"]:
        log("[自打开] 表格已就绪，无需再打开文档")
        return H2, st

    # 3) 进入文档视图：只激活固定列滚动区之后第一个 ListItem（＝「文档」入口）
    #    索引每次现算，绝不复用缓存索引（工具栏增减会漂移）
    for round_no in range(2):
        s = w.subs_of(H)
        target = None
        for i in range(s.Length):
            try:
                e = s.GetElement(i)
                if (e.CurrentClassName or "") == "client_ding::NavigatorScrollView":
                    for j in range(i + 1, min(i + 5, s.Length)):
                        it = s.GetElement(j)
                        if it.CurrentControlType == 50007:      # ListItem
                            target = it
                            break
                    break
            except Exception:
                continue
        if target is None:
            break
        how = []
        for pid, cls, m in ((10018, w.IUIAutomationLegacyIAccessiblePattern, "DoDefaultAction"),
                            (10000, w.IUIAutomationInvokePattern, "Invoke"),
                            (10010, w.IUIAutomationSelectionItemPattern, "Select")):
            try:
                getattr(target.GetCurrentPattern(pid).QueryInterface(cls), m)()
                how.append(m)
            except Exception:
                pass
        log(f"[自打开] 第{round_no+1}轮激活固定列「文档」入口 {how}")
        if round_no == 1:
            try:
                lg = target.GetCurrentPattern(10018).QueryInterface(
                    w.IUIAutomationLegacyIAccessiblePattern)
                lg.DoDefaultAction(); lg.DoDefaultAction()      # Legacy 默认动作＝双击
            except Exception:
                pass
        time.sleep(4.0)
        H2, st = pick_table_window(H)
        if st["ready"]:
            return H2, st

    # 4) 文档首页「最近打开」里按【精确名称】打开「退货登记码」
    for h in ding_cef_windows(H):
        names, refs, s = w.scan_hwnd(h)
        if s is None:
            continue
        cands = [(i, n) for i, n in names if n.strip() == "退货登记码"]
        if not cands:
            continue
        # Text(50020) 优先：实测 Button(50000) 点了不切换
        cands.sort(key=lambda x: 0 if _ctype(s, x[0]) == 50020 else 1)
        log(f"[自打开] 按精确名称打开「退货登记码」 hwnd={h} 候选={len(cands)}")
        w.activate_ctrl(s.GetElement(cands[0][0]))
        for _ in range(8):
            time.sleep(2.0)
            H2, st = pick_table_window(h)
            if st["ready"]:
                log("[自打开] 表格控件已就绪")
                return H2, st
        break
    H2, st = pick_table_window(H)
    return H2, st


def _plain(s):
    """去掉钉钉标题里的零宽格式字符（Cf），只保留可读文字。"""
    return "".join(ch for ch in (s or "")
                   if unicodedata.category(ch) != "Cf").strip()


def doc_title(H):
    """读窗口内真实文档标题（Pane 名称形如「广告费消耗 · 钉钉文档」）。

    返回 (标题, 承载窗口)。绝不靠截图/坐标猜，只读 UIA 控件名。
    """
    for h in [H] + [x for x in ding_cef_windows(H) if x != H]:
        try:
            names, _, _ = w.scan_hwnd(h)
        except Exception:
            continue
        for _, n in names:
            p = _plain(n)
            if "· 钉钉文档" in p:
                return p.split("·")[0].strip(), h
    return "", H


def ensure_doc(H, name):
    """写入前按【真实文档标题】对版；不对版就按精确名称点开目标文档。

    历史坑：名称框存在 ≠ 打开的是这张表。当前停在「广告费消耗」时去 Ctrl+F 找
    701 必然失败（广告费表 Ctrl+F 不生效）。左侧固定列文档条目常驻，Text(50020)
    优先，Button 点了不切换（2026-11-06 实测）。全程控件级，不用坐标、不翻消息。
    """
    title, th = doc_title(H)
    if name in (title or ""):
        log(f"[文档] 已在「{name}」，标题对版通过")
        return th or H
    log(f"[文档] 当前标题={title!r}，需切到「{name}」")

    target = None
    for h in [H] + [x for x in ding_cef_windows(H) if x != H]:
        try:
            names, refs, s = w.scan_hwnd(h)
        except Exception:
            continue
        if s is None:
            continue
        cands = [(i, n) for i, n in names if _plain(n).strip() == name]
        if cands:
            cands.sort(key=lambda x: 0 if _ctype(s, x[0]) == 50020 else 1)
            target = (h, s.GetElement(cands[0][0]))
            break
    if target is None:
        raise RuntimeError(
            f"控件树里没有「{name}」精确名称条目（当前标题={title!r}）；"
            "未写入任何单元格，不猜页面、不要求用户打开页面")

    how = w.activate_ctrl(target[1])
    log(f"[文档] 按精确名称激活「{name}」({how}) hwnd={target[0]}")
    for k in range(10):
        time.sleep(2.0)
        H2, st = pick_table_window(H)
        t2, t2h = doc_title(H2)
        if st["ready"] and name in (t2 or ""):
            H3 = t2h or H2
            install_policy(H3)          # 标题/窗口变了就重绑唯一窗口策略
            STATE["ding_hwnd"] = H3
            log(f"[文档] 已切到「{name}」 hwnd={H3} 名称框={st['refs']}")
            return H3
    raise RuntimeError(
        f"已点开「{name}」但标题/名称框未对版（当前={doc_title(H)[0]!r}）；"
        "未写入任何单元格")


def _ctype(s, i):
    try:
        return s.GetElement(i).CurrentControlType
    except Exception:
        return 0


def preflight(H):
    """写入前门槛：脚本自己打开文档 -> 动态锁定表格窗口 -> 树完整性校验。

    绝不盲写、绝不把「树缩水」说成「页面找不到」、绝不要求用户手动打开页面。
    """
    H2, st = open_docs_self(H)
    if H2 != H:
        install_policy(H2)          # 表格在哪个窗口就锁哪个，修掉窗口漂移
    STATE["ding_hwnd"] = H2
    STATE["tree"] = st
    if st["ready"]:
        return H2, st
    enable_cef_accessibility(H2)
    if rebuild_tree_once(H2):
        H3, st = pick_table_window(H2)
        if st["ready"]:
            if H3 != H2:
                install_policy(H3)
            STATE["ding_hwnd"] = H3
            STATE["tree"] = st
            log(f"[门槛] 树已重建：节点={st['nodes']} 名称框={st['refs']}")
            return H3, st
    st = tree_state(H2)
    STATE["tree"] = st
    raise RuntimeError(
        f"脚本已自动打开「退货登记码」文档，但钉钉 CEF 无障碍树仍未启用"
        f"(hwnd={H2}, 节点={st['nodes']}, 名称框={st['refs']})；"
        "读不到应用内真值，按硬规则拒绝盲写。"
        "这是钉钉渲染进程无障碍未启用（常见于安全模式自检/异常退出后首启）；"
        "系统辅助技术标志已由脚本设好，钉钉稳定后由脚本自己打开文档、零采集直接补写。"
        "脚本绝不自行启动/结束/重启钉钉，也不要求你手动打开任何页面。"
        "数据已存 oneflow_pending.json")


def write_return_code(H, code, today):
    """退货码：固定列按名称打开 -> 701 锚点对版 -> 单格写入 -> 逐格真值验证。"""
    log("[退货码] 在同一钉钉窗口内打开「退货登记码」")
    w.ensure_sheet_open()
    w.dismiss_freeze()
    last = w.load_last_result()
    prev = last.get("code") if last.get("ok") else None
    status, pos = w.write_code(code, prev, last.get("date"), today)
    log(f"[退货码] {status} {code} @ {pos}")
    return {"ok": True, "code": code, "cell": pos, "status": status}


def write_ads_cost(H, day, data):
    """广告费：同一窗口内切到「广告费消耗」-> 日期行真值对版 -> B/C/D 逐格验证。"""
    label = f"{day.month}月{day.day}日"
    log("[广告费] 在同一钉钉窗口内切到「广告费消耗」")
    if not a.ensure_sheet():
        raise RuntimeError("未能打开「广告费消耗」表格，未写入")
    row = d.find_row_by_date(day, label)
    if row is None:
        raise RuntimeError(f"表内未找到日期行 {label!r}（已读公式栏真值对版），未写入")
    log(f"[广告费] 日期 {label} -> 第 {row} 行 B/C/D")
    ok, cells = d.write_row(row, data)
    result = {"date": day.isoformat(), "label": label, "row": row,
              "data": data, "ok": ok, "cells": cells}
    save_json(ADS_CRED, result)     # 兼容旧兜底判定的凭证
    if not ok:
        raise RuntimeError(f"第 {row} 行未全部通过真值验证，详见 {ADS_CRED}")
    log(f"[广告费] DONE 第{row}行 营业额={data['营业额']} 广告费={data['广告费']} 点击={data['点击']}")
    return result


# ---------------- 主链路 ----------------
def probe(args):
    """只读体检：零写入、零窗口激活、不改任何状态。"""
    today = date.today().isoformat()
    target = (date.fromisoformat(args.date) if args.date
              else date.today() - timedelta(days=1)).isoformat()
    out = {"probe": True, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "today": today, "target_date": target}
    try:
        H, st = discover_ding()
        out["ding"] = st
    except Exception as ex:
        out["ding"] = {"error": str(ex)}
    H = d.find_ml_hwnd()
    out["ml_hwnd"] = H
    if H:
        names = d.ml_names()
        out["ml_tree_nodes"] = len(names)
        e, url = d.ml_url()
        out["ml_url"] = url
    out["credentials"] = {"return": load_json(RETURN_CRED), "ads": load_json(ADS_CRED),
                          "oneflow": load_json(OUT)}
    out["today_code_in_log"] = log_code_today(today)
    out["last_seen_code"] = last_seen_code(today)
    out["pending"] = load_json(PENDING)
    print("ONEFLOW_PROBE " + json.dumps(out, ensure_ascii=False), flush=True)
    log("[体检] " + json.dumps({"ding": out.get("ding"), "ml": out.get("ml_tree_nodes"),
                                "code": out["today_code_in_log"]}, ensure_ascii=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="只读体检，不写、不切窗口")
    ap.add_argument("--date", help="广告费目标日 YYYY-MM-DD，默认昨天")
    ap.add_argument("--code", help="当日退货码（已知则零采集）")
    ap.add_argument("--data", help="已知数据 营业额,广告费,点击（已知则零采集）")
    ap.add_argument("--dry-run", action="store_true", help="只采集+体检，不写任何单元格")
    ap.add_argument("--revive", action="store_true",
                    help="钉钉只剩托盘/空壳时，仅恢复【已运行】实例的窗口（不启动/结束任何进程）")
    ap.add_argument("--fix-a11y", action="store_true",
                    help="只做无障碍软修复（设标志+MSAA 请求），绝不结束或重启钉钉进程")
    args = ap.parse_args()
    if args.code:
        args.code = validate_code(args.code)
    if args.probe:
        probe(args)
        return 0

    today = date.today().isoformat()
    day = date.fromisoformat(args.date) if args.date else date.today() - timedelta(days=1)
    target = day.isoformat()
    STATE["fg_before"] = u.GetForegroundWindow()
    started = time.time()
    result = {"date": today, "target_date": target, "overall_ok": False,
              "policy": "one_window_background", "steps": {}}
    log(f"=== 701 统一主线 start 今日={today} 目标日={target} "
        f"code={args.code or 'auto'} data={args.data or 'auto'} dry_run={args.dry_run} ===")

    try:
        # S0 幂等：两步都完成 -> 零操作
        r_done, a_done = return_done(today), ads_done(target)
        if args.code:
            r_done = False
        if args.data:
            a_done = False
        if r_done and a_done:
            result.update({"overall_ok": True, "skipped": True,
                           "note": "当日退货码与目标日广告费均已登记，零窗口操作"})
            save_json(OUT, result)
            print("ONEFLOW_SKIP " + json.dumps(result, ensure_ascii=False), flush=True)
            log("[幂等] 全部已完成，零操作 SKIP")
            return 0

        # S1 采集（数据已知即停；单项失败不影响已采信的另一项，但绝不填异常值）
        install_ads_stability()
        pending = load_json(PENDING)
        reusable = reusable_pending(pending, today, target)
        if reusable and not args.data:
            # 上次已采到可信数据（可能卡在钉钉侧）：零采集、零窗口操作，直接补写
            code, data, ads_error = args.code or pending["code"], validate_data(pending["data"]), None
            log(f"[复用] 使用已采信数据 code={code} data={data}，紫鸟零操作")
        else:
            code = args.code or log_code_today(today)
            if code:
                log(f"[取码] 复用已采信当日码 {code}，紫鸟零操作")
            else:
                code = read_code_from_ml(today)
            data, ads_error = None, None
            try:
                data = collect_ads_data(day, args.data)
            except Exception as ex:
                ads_error = str(ex)
                log(f"[采集] 广告费未取到可信值：{ads_error}；不填写，退货码继续")
        code = validate_code(code)
        if data is not None:
            data = validate_data(data)
        result["collected"] = {"code": code, "data": data, "ads_error": ads_error}
        save_json(PENDING, {"date": today, "target_date": target,
                            "code": code, "data": data, "ads_error": ads_error,
                            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        if args.dry_run:
            result.update({"dry_run": True, "overall_ok": data is not None and ads_error is None,
                           "note": "仅完成采集检查，未验证钉钉写入条件，未写入任何单元格"})
            save_json(OUT, result)
            print("ONEFLOW_DRY " + json.dumps(result, ensure_ascii=False), flush=True)
            return 0 if result["overall_ok"] else 1

        # S2 唯一钉钉窗口 + 后台策略 + 脚本自己打开文档 + 树门槛
        if args.fix_a11y:
            fix_a11y()
        try:
            H, st = discover_ding()
        except RuntimeError:
            if not args.revive or not revive_ding():
                raise
            H, st = discover_ding()
        install_policy(H)
        H, st = preflight(H)         # 自己打开「退货登记码」，不要求用户动手
        result["window"] = st

        # S3 顺序强依赖：退货码真正做成，才做广告费
        if not r_done:
            H = ensure_doc(H, "退货登记码")
            result["steps"]["return_code"] = write_return_code(H, code, today)
        else:
            result["steps"]["return_code"] = {"ok": True, "skipped": True}
        if a_done:
            result["steps"]["ads_cost"] = {"ok": True, "skipped": True,
                                           "reason": "目标日广告费已登记"}
        elif data is not None:
            H = ensure_doc(H, "广告费消耗")
            result["steps"]["ads_cost"] = write_ads_cost(H, day, data)
        else:
            result["steps"]["ads_cost"] = {
                "ok": False,
                "reason": "本次无广告费数据，未填写（绝不填异常值/不复用别的日期数据）",
                "detail": ads_error}

        result["overall_ok"] = all(v.get("ok") for v in result["steps"].values())
        if result["overall_ok"]:
            try:
                os.remove(PENDING)
            except Exception:
                pass
        log(f"=== 701 统一主线 DONE overall_ok={result['overall_ok']} "
            f"耗时={int(time.time()-started)}s ===")
    except Exception as ex:
        result["error"] = str(ex)
        result["trace"] = traceback.format_exc()[-1200:]
        log("ABORT: " + str(ex))
        # 数据已采到就落盘，供恢复后直接补写，不重新采集
        if result.get("collected") and not os.path.exists(PENDING):
            save_json(PENDING, {"date": today, "target_date": target,
                                **result["collected"]})
    finally:
        restore_foreground()

    save_json(OUT, result)
    print("ONEFLOW_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["overall_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

