# -*- coding: utf-8 -*-
"""
每日退货码登记 v2 —— 纯应用内控件操作, 零固定坐标 / 零硬编码窗口句柄

v2 修复(2026-11-05 13:30 写入失败复盘):
  1. 去掉写死的 DING=21630518 句柄, 改为运行时枚举窗口动态发现钉钉主窗口;
  2. 去掉 r[0] < 200 这类绝对屏幕坐标判断(窗口一移动就失效), 改为按控件类型/名称枚举;
  3. 表格可能开在主窗口内嵌 WebView, 也可能开在独立文档窗口/浏览器窗口,
     因此就绪判定扫描"所有候选窗口", 找到哪个就把后续控件操作绑定到哪个;
  4. 表格加载慢: 每轮重新枚举窗口, 等待窗口上限提高, 并支持二次激活重试。

流程:
  A. 幂等: 当日已成功登记 -> 零操作跳过(不采集/不碰窗口)
  B. 美客多: 紫鸟应用内 Ventas 标签 + 「重新加载」按钮刷新 -> 读当日授权码
  C. 钉钉: 应用内「退货登记码」条目打开表格(表格窗口动态发现)
  D. 表内 Ctrl+F 找 701(主锚点, 常量) -> 右移一格到 C 列
  E. 读 C 列当前真值 = 上一次登记的码 -> 与凭证对版 -> 只在确认过的单格替换
  F. 写后读公式栏真值验证 == 当日码, 并更新凭证供次日对版

v3 (2026-11-06): 用户提出"读昨天填的码, 搜索替换成今天的码"。
  采纳: 读旧码做写入前对版(直接从表格真值读, 不依赖本地文件)。
  否决: 表格级全局查找替换(Ctrl+H) —— 会跨单元格/跨工作表误伤(历史明细、
        其他店铺行、备注里同码), 且上次写入失败时旧码不在 C 列会静默失败。
  实现: 701 保持主锚点, 旧码仅作第二重对版; 作用范围严格限定 701 右邻一格。
  状态异常(C列当前值 != 上次登记码) -> 中止并如实报错, 绝不盲目覆盖。

v4 (2026-11-07): 修「每日自动化一直出错做不到」的取码环节。
  旧 main() 无条件调 daily_return_code.read_ml_code() 实时刷新紫鸟, 而
  11/07 13:31 两次因「找不到 Ventas 标签页」直接失败; 且当天 13:30 退货码
  与广告费两个任务同时启动, 13:42:29 同一秒互抢钉钉焦点, 双双打不开表格。
  改造: 取码优先级 --data > return_code.log 当天已采集凭证(零采集) > 兜底
  实时采集一次(失败如实报, 不反复刷页面); 支持 --no-collect 严格数据已知即停。
  调度层由自动化错开(退货码 13:30 / 广告费 14:00), 不再同时间抢焦点。
"""
import ctypes, comtypes, time, re, sys, os, json, subprocess
from ctypes import wintypes
from comtypes.client import GetModule
GetModule("UIAutomationCore.dll")
from comtypes.gen.UIAutomationClient import (
    CUIAutomation, IUIAutomation, TreeScope_Subtree,
    IUIAutomationInvokePattern, IUIAutomationLegacyIAccessiblePattern,
    IUIAutomationSelectionItemPattern, IUIAutomationValuePattern)

user32 = ctypes.windll.user32
k32 = ctypes.windll.kernel32
ctypes.windll.ole32.CoInitializeEx(0, 2)
auto = comtypes.client.CreateObject(CUIAutomation, interface=IUIAutomation)

LOG = r"D:\workjiyineiron\2026-11-02-09-03-07\.workbuddy\tmp\return_code.log"
RESULT = r"D:\workjiyineiron\2026-11-02-09-03-07\.workbuddy\tmp\return_write_last.json"
DING_EXE = r"C:\Program Files (x86)\DingDing\main\current\DingTalk.exe"
CELLREF = re.compile(r"^[A-Z]{1,2}\d{1,4}$")
SHEET_HWND = None          # 运行时发现: 表格控件树所在窗口
DING_HWND = None           # 运行时发现: 钉钉主窗口


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------- 凭证 / 纯逻辑决策(可独立单测, 不碰窗口) ----------------
def load_last_result():
    """上一次成功登记的凭证: {date, ok, code, cell}。缺失/损坏返回 {}。"""
    try:
        with open(RESULT, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_result(code, cell, ok=True):
    d = {"date": time.strftime("%Y-%m-%d"), "ok": bool(ok),
         "code": code, "cell": cell, "source": "write_return_code.py 每日登记"}
    try:
        os.makedirs(os.path.dirname(RESULT), exist_ok=True)
        with open(RESULT, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    except Exception as ex:
        log(f"警告: 凭证写入失败 {ex}")


def decide_action(before, prev_code, new_code, last_date, today):
    """纯逻辑: 根据「表格里读到的旧码」「上次登记凭证」「当日码」决定动作。
    返回 (action, reason)，action ∈ SKIP / WRITE / ABORT。

    严谨口径:
      - 今日已用同一码登记 -> SKIP(零操作)
      - 该格已是当日码       -> SKIP(凭证可能丢失, 补写凭证即可)
      - 该格值 != 上次登记码 -> ABORT(有人改过/上次未成功/码撞车), 拒绝盲目覆盖
      - 仅当 701 主锚点确认 + 旧码对版通过, 才允许 WRITE(单格)
    """
    if last_date == today and prev_code == new_code:
        return "SKIP", "凭证显示今日已用同一码成功登记"
    if before == new_code:
        return "SKIP", "该格已是当日码(凭证可能缺失), 补写凭证"
    if prev_code and before and before != prev_code:
        return "ABORT", (f"该格当前值 {before!r} != 上次登记码 {prev_code!r} "
                         "(有人改过 / 上次未成功 / 码撞车), 拒绝盲目覆盖")
    if not before:
        return "WRITE", "该格为空, 无上次码可对版(已标注, 仅靠 701 主锚点)"
    if prev_code and before == prev_code:
        return "WRITE", f"对版通过: 该格当前值 = 上次登记码 {prev_code!r}"
    return "WRITE", "无上次码凭证, 仅靠 701 主锚点定位(已标注)"


def self_test():
    """纯逻辑自测: 不采集、不碰紫鸟、不碰钉钉、不改任何数据。"""
    T = "2026-11-07"
    Y = "2026-11-06"
    cases = [
        ("今日已登记同码 -> 零操作", (None, "BE1B261A", "BE1B261A", T, T), "SKIP"),
        ("该格已是当日码 -> 跳过", ("C0FFEE01", "BE1B261A", "C0FFEE01", Y, T), "SKIP"),
        ("正常跨日换码 -> 写入", ("BE1B261A", "BE1B261A", "C0FFEE01", Y, T), "WRITE"),
        ("表格值与凭证不符 -> 中止", ("DEADBEEF", "BE1B261A", "C0FFEE01", Y, T), "ABORT"),
        ("无凭证 -> 仅靠701写入", ("BE1B261A", None, "C0FFEE01", None, T), "WRITE"),
        ("空单元格 -> 写入并标注", ("", "BE1B261A", "C0FFEE01", Y, T), "WRITE"),
        ("当天码中途更新 -> 允许再写", ("BE1B261A", "BE1B261A", "C0FFEE01", T, T), "WRITE"),
    ]
    bad = 0
    for name, args, expect in cases:
        act, why = decide_action(*args)
        ok = act == expect
        bad += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {act} (期望 {expect}) | {why}")
    print(f"SELF_TEST {'ALL_OK' if bad == 0 else 'FAILED'} ({len(cases) - bad}/{len(cases)})")
    return 0 if bad == 0 else 1


# ---------------- 窗口发现(不涉及坐标) ----------------
def all_windows():
    out = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def cb(h, l):
        n = user32.GetWindowTextLengthW(h)
        if n > 0:
            b = ctypes.create_unicode_buffer(n + 2)
            user32.GetWindowTextW(h, b, n + 2)
            cl = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(h, cl, 256)
            out.append((h, bool(user32.IsWindowVisible(h)), cl.value, b.value))
        return True

    user32.EnumWindows(cb, 0)
    return out


def ding_main_hwnd():
    """钉钉主窗口: 可见优先; 隐藏则恢复; 托盘常驻但无窗口则自行拉起"""
    global DING_HWND

    def pick(visible_only):
        for h, vis, cl, t in all_windows():
            if visible_only and not vis:
                continue
            if cl.startswith("Qt51511QWindowIcon") and ("钉钉" in t or "DingTalk" in t):
                return h, t
        return None, None

    h, t = pick(True)
    if h:
        DING_HWND = h
        return h

    h, t = pick(False)
    if h:
        user32.ShowWindow(h, 9)
        user32.SetForegroundWindow(h)
        time.sleep(2.5)
        if user32.IsWindowVisible(h):
            DING_HWND = h
            log(f"已从隐藏状态恢复钉钉主窗口 hwnd={h}")
            return h

    if os.path.exists(DING_EXE):
        log("钉钉主窗口不在, 自行拉起钉钉客户端")
        try:
            subprocess.Popen([DING_EXE], close_fds=True)
        except Exception as ex:
            log(f"启动钉钉失败: {ex}")
        for _ in range(20):
            time.sleep(1.5)
            h, t = pick(True)
            if h:
                DING_HWND = h
                log(f"已重新拉起钉钉主窗口 hwnd={h}")
                return h
    raise RuntimeError("未找到钉钉应用窗口(请确认钉钉已登录并运行)")


def candidate_hwnds(wide=False):
    """优先扫钉钉/文档类窗口; wide 时再带上浏览器窗口(文档可能被外部打开)"""
    fast, slow = [], []
    for h, vis, cl, t in all_windows():
        if not vis:
            continue
        if (cl.startswith("Qt51511") or "钉钉" in t or "DingTalk" in t
                or "退货" in t or "文档" in t):
            fast.append(h)
        elif cl == "Chrome_WidgetWin_1":
            slow.append(h)
    return fast + (slow if wide else [])


# ---------------- 控件树 ----------------
def scan_hwnd(h):
    """返回 (names, refs, subs)。任何窗口异常都吞掉, 不影响其他候选"""
    try:
        s = auto.ElementFromHandle(h).FindAll(TreeScope_Subtree, auto.CreateTrueCondition())
    except Exception:
        return [], [], None
    names, refs = [], []
    for i in range(s.Length):
        try:
            nm = (s.GetElement(i).CurrentName or "").strip()
        except Exception:
            continue
        if nm:
            names.append((i, nm))
            if CELLREF.match(nm):
                refs.append(nm)
    return names, refs, s


def subs_of(h=None):
    h = h or SHEET_HWND or DING_HWND or ding_main_hwnd()
    return auto.ElementFromHandle(h).FindAll(TreeScope_Subtree, auto.CreateTrueCondition())


def find_sheet_window(wide=False):
    """在候选窗口里找: 同时出现「退货码更新」标签 和 单元格名称框"""
    for h in candidate_hwnds(wide):
        names, refs, _ = scan_hwnd(h)
        if refs and any("退货码更新" in nm for _, nm in names):
            return h, refs
    return None, None


def sheet_ready(wide=False):
    global SHEET_HWND
    h, refs = find_sheet_window(wide=False)
    if h is None and wide:
        h, refs = find_sheet_window(wide=True)
    if h:
        SHEET_HWND = h
        return True, refs
    return False, []


# ---------------- 激活(仅焦点, 不改窗口状态) ----------------
def focus_hwnd(h, settle=2.5):
    """只切窗口焦点(不是坐标/不改窗口状态); settle 用于等 Chromium 无障碍树构建完整。
    2026-11-07: 已在前台则零激活(仍做控件级 SetFocus), 避免整轮反复把窗口切前台、
    以及与广告费任务互抢焦点(11/07 13:42:29 两边同秒激活钉钉导致表格双双打不开)。"""
    try:
        auto.ElementFromHandle(h).SetFocus()
    except Exception:
        pass
    if user32.GetForegroundWindow() == h:
        return
    user32.keybd_event(0x12, 0, 0, 0); user32.keybd_event(0x12, 0, 2, 0)
    fg = user32.GetForegroundWindow()
    tf = user32.GetWindowThreadProcessId(fg, None)
    tm = k32.GetCurrentThreadId()
    user32.AttachThreadInput(tm, tf, True)
    user32.SetForegroundWindow(h)
    user32.SwitchToThisWindow(h, True)
    user32.AttachThreadInput(tm, tf, False)
    time.sleep(settle)


def activate_ctrl(e):
    """控件级激活: 仅 UIA Pattern / SetFocus, 完全不使用坐标或矩形"""
    for pid, cls, m in ((10018, IUIAutomationLegacyIAccessiblePattern, "DoDefaultAction"),
                        (10000, IUIAutomationInvokePattern, "Invoke"),
                        (10010, IUIAutomationSelectionItemPattern, "Select")):
        try:
            getattr(e.GetCurrentPattern(pid).QueryInterface(cls), m)()
            return m
        except Exception:
            continue
    try:
        e.SetFocus()
        return "SetFocus"
    except Exception:
        return None


def key(vk):
    user32.keybd_event(vk, 0, 0, 0); user32.keybd_event(vk, 0, 2, 0)


def type_chars(s):
    for ch in s.upper():
        user32.keybd_event(ord(ch), 0, 0, 0); user32.keybd_event(ord(ch), 0, 2, 0)
        time.sleep(0.04)


# ---------------- 打开表格 ----------------
def doc_items_in(main_hwnd):
    names, _, s = scan_hwnd(main_hwnd)
    if s is None:
        return []
    return [s.GetElement(i) for i, nm in names if nm == "退货登记码"]


def enter_doc_view(main_hwnd):
    """进入文档视图(2026-11-06 实测修复):
    只按固定入口激活一次 —— 精确名称「文档」导航项, 或 Qt 自绘
    client_ding::NavigatorScrollView 内第一个 ListItem(UIA 名称为空,
    Legacy DoDefaultAction 可用)。激活后钉钉出现「文档 · 钉钉文档」Pane、
    「最近打开」以及固定列「退货登记码」条目。
    禁止逐个试无名称 ListItem、禁止搜索框、禁止翻聊天消息、不依赖坐标。"""
    try:
        s = subs_of(main_hwnd)
    except Exception:
        return False
    # 1) 精确名称优先
    for i in range(s.Length):
        try:
            e = s.GetElement(i)
            if (e.CurrentName or "").strip() == "文档":
                activate_ctrl(e)
                time.sleep(3.0)
                if doc_items_in(main_hwnd):
                    log("已进入应用内文档视图(按名称激活「文档」)")
                    return True
                break
        except Exception:
            continue
    # 2) Qt 自绘固定列: NavigatorScrollView 之后第一个 ListItem, 只激活一次
    for i in range(s.Length):
        try:
            e = s.GetElement(i)
            if (e.CurrentClassName or "") == "client_ding::NavigatorScrollView":
                for j in range(i + 1, min(i + 4, s.Length)):
                    it = s.GetElement(j)
                    if it.CurrentControlType == 50007:
                        activate_ctrl(it)
                        time.sleep(3.0)
                        if doc_items_in(main_hwnd):
                            log("已进入应用内文档视图(固定列导航项一次激活)")
                            return True
                        break
                break
        except Exception:
            continue
    return False


def ensure_sheet_open():
    global SHEET_HWND
    ok, refs = sheet_ready()
    if ok:
        log(f"表格已打开, 窗口={SHEET_HWND}, 名称框={refs[:2]}")
        return

    main_hwnd = ding_main_hwnd()
    focus_hwnd(main_hwnd, settle=3.0)
    log(f"钉钉主窗口={main_hwnd}, 尝试应用内打开「退货登记码」")

    for attempt in range(2):
        items = doc_items_in(main_hwnd)
        if not items:
            enter_doc_view(main_hwnd)
            items = doc_items_in(main_hwnd)
        if not items:
            log(f"第{attempt+1}次: 文档列表仍无「退货登记码」, 重试")
            focus_hwnd(main_hwnd)
            continue
        how = activate_ctrl(items[0])
        log(f"第{attempt+1}次: 激活文档条目「退货登记码」({how})")
        for k in range(15):
            time.sleep(2)
            ok, refs = sheet_ready(wide=(k % 4 == 3))
            if ok:
                log(f"表格已打开, 窗口={SHEET_HWND}, 名称框={refs[:2]}")
                return
        log(f"第{attempt+1}次: 30s 内表格控件未就绪, 重新激活")
        focus_hwnd(main_hwnd)

    raise RuntimeError("文档已激活但表格控件未就绪(已扫描全部候选窗口): 请确认钉钉能看到「退货登记码」且表格能正常加载")


# ---------------- 写入 + 回读验证 ----------------
def ensure_foreground():
    """按键前确认表格窗口仍在前台, 否则焦点会跑到任务栏导致 Ctrl+F 完全失效"""
    if user32.GetForegroundWindow() != SHEET_HWND:
        focus_hwnd(SHEET_HWND, settle=2.0)


def read_ref(wait=1.0):
    """读名称框(单元格引用)。无障碍树缩水时短轮询等待"""
    for _ in range(3):
        _, refs, _ = scan_hwnd(SHEET_HWND)
        if refs:
            return refs[0]
        time.sleep(wait)
    return None


def sheet_find(text, expect=None, tries=3):
    """表内 Ctrl+F 查找 -> 名称框回读; expect 给定期望列前缀, 不满足则重试"""
    last = None
    for k in range(tries):
        ensure_foreground()
        user32.keybd_event(0x11, 0, 0, 0); key(0x46); user32.keybd_event(0x11, 0, 2, 0)
        time.sleep(1.3)
        type_chars(text)
        time.sleep(0.7)
        key(0x0D); time.sleep(1.2)
        key(0x1B); time.sleep(1.2)
        last = read_ref()
        if last and (expect is None or last.startswith(expect)):
            return last
        log(f"查找 {text} 第{k+1}次回读={last}, 重试")
    return last


def _col(ref):
    return "".join(c for c in ref if c.isalpha())


def _row(ref):
    try:
        return int("".join(c for c in ref if c.isdigit()))
    except Exception:
        return -1


def align_cell(target):
    """已确认方向键生效后, 用方向键把选中格微调到 target, 每步读名称框确认"""
    for k in range(8):
        cur = read_ref(wait=0.4)
        if not cur:
            return False
        if cur == target:
            return True
        if _col(cur) < _col(target):
            key(0x27)                       # RIGHT
        elif _col(cur) > _col(target):
            key(0x25)                       # LEFT
        elif _row(cur) < _row(target):
            key(0x28)                       # DOWN
        else:
            key(0x26)                       # UP
        time.sleep(0.6)
    return read_ref(wait=0.4) == target


def goto_cell(target):
    """Ctrl+F 后键盘焦点常停在查找面板, 方向键无效。
    逐个对表格里可聚焦的网格容器 SetFocus(), 找出真正能接收方向键的那个,
    再用方向键对齐到 target。全程以名称框为准, 未确认绝不输入。"""
    ensure_foreground()
    _, _, s = scan_hwnd(SHEET_HWND)
    if s is None:
        return False
    cands = []
    for i in range(s.Length):
        e = s.GetElement(i)
        try:
            if (e.CurrentControlType in (50026, 50030)
                    and e.CurrentIsKeyboardFocusable):
                cands.append(e)
        except Exception:
            continue
    log(f"表格内可聚焦网格容器候选 {len(cands)} 个")
    for idx, e in enumerate(cands[:25]):
        before = read_ref(wait=0.3)
        try:
            e.SetFocus()
        except Exception:
            continue
        time.sleep(0.3)
        key(0x27)                           # 试探方向键是否生效
        time.sleep(0.5)
        after = read_ref(wait=0.3)
        if not after or after == before:
            continue                        # 这个容器不接收表格方向键
        log(f"网格容器#{idx} 接收方向键: {before} -> {after}")
        if align_cell(target):
            return True
    return False


def dismiss_freeze():
    """用户已授权: 表格提示『窗口过小, 无法正常滚动冻结区域』时, 用控件名点『临时取消冻结』"""
    names, _, s = scan_hwnd(SHEET_HWND)
    if s is None:
        return False
    for i, nm in names:
        if nm != "临时取消冻结":
            continue
        e = s.GetElement(i)
        try:
            if e.CurrentControlType != 50000:      # Button
                continue
        except Exception:
            continue
        how = activate_ctrl(e)
        if how:
            log(f"已点击应用内「临时取消冻结」({how})")
            time.sleep(1.5)
            return True
    return False


def activate_sheet_tab():
    """用控件名激活工作表, 顺带把键盘焦点带进表格网格"""
    names, _, s = scan_hwnd(SHEET_HWND)
    if s is None:
        return False
    for i, nm in names:
        if nm == "退货码更新":
            try:
                s.GetElement(i).GetCurrentPattern(10000).QueryInterface(
                    IUIAutomationInvokePattern).Invoke()
                time.sleep(1.2)
                log("已激活工作表 退货码更新")
                return True
            except Exception:
                return False
    return False


FORMULA_SKIP = {"\ufeff", "=", "插入", "选择", "菜单", "溢出", "更多", "评论"}


def read_formula_value():
    """读公式栏真值: 名称框之后第一个像单元格值的 Text 即该格内容。
    纯控件读取, 不截图、不用坐标, 也不靠 Ctrl+F 是否跳转来猜成功。
    返回: 真值字符串 / "" (该格为空) / None (连名称框都读不到, 无法对版)。
    {2,20} 兼容 '701'(3位) 与 8 位授权码; 跳过零宽字符与工具栏词。"""
    names, _, _ = scan_hwnd(SHEET_HWND)
    p = None
    for k, (i, nm) in enumerate(names):
        if CELLREF.match(nm):
            p = k
            break
    if p is None:
        return None
    for k in range(p + 1, min(p + 9, len(names))):
        v = names[k][1].strip().lstrip("\ufeff").strip()
        if not v or v in FORMULA_SKIP:
            continue
        return v if re.fullmatch(r"[A-Za-z0-9]{2,20}", v) else ""
    return ""


def set_clipboard(text):
    """纯 ctypes 写系统剪贴板(受管 venv 无 tkinter 也能用)"""
    try:
        u = ctypes.windll.user32
        k = ctypes.windll.kernel32
        k.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
        k.GlobalAlloc.restype = ctypes.c_void_p
        k.GlobalLock.argtypes = [ctypes.c_void_p]
        k.GlobalLock.restype = ctypes.c_void_p
        k.GlobalUnlock.argtypes = [ctypes.c_void_p]
        u.OpenClipboard.argtypes = [ctypes.c_void_p]
        u.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
        u.SetClipboardData.restype = ctypes.c_void_p
        if not u.OpenClipboard(None):
            return False
        u.EmptyClipboard()
        data = text.encode("utf-16-le") + b"\x00\x00"
        h = k.GlobalAlloc(0x0002, len(data))
        p = k.GlobalLock(h)
        ctypes.memmove(p, data, len(data))
        k.GlobalUnlock(h)
        u.SetClipboardData(13, h)      # CF_UNICODETEXT
        u.CloseClipboard()
        return True
    except Exception:
        return False


def formula_controls():
    """名称框之后的那几个控件(公式栏/编辑区)。不含字号等工具栏控件"""
    names, _, s = scan_hwnd(SHEET_HWND)
    if s is None:
        return []
    p = None
    for k, (i, nm) in enumerate(names):
        if CELLREF.match(nm):
            p = k
            break
    if p is None:
        return []
    out = []
    for i, nm in names[p + 1:p + 8]:
        e = s.GetElement(i)
        try:
            if e.CurrentControlType in (50004, 50020, 50030):
                out.append(e)
        except Exception:
            continue
    return out


def verify_cell(target):
    """回到目标格并读公式栏真值"""
    if read_ref(wait=0.5) != target and not align_cell(target):
        if not goto_cell(target):
            return None
    return read_formula_value()


def try_once(code, target):
    """依次试: 剪贴板粘贴 -> 公式栏控件 SetValue -> 逐字符; 每步用真值判定"""
    # A. 剪贴板 Ctrl+V
    if set_clipboard(code):
        key(0x2E); time.sleep(0.35)
        user32.keybd_event(0x11, 0, 0, 0); key(0x56); user32.keybd_event(0x11, 0, 2, 0)
        time.sleep(0.6); key(0x0D); time.sleep(1.4)
        v = verify_cell(target)
        log(f"  剪贴板Ctrl+V -> 真值={v}")
        if v == code:
            return "剪贴板Ctrl+V"
    # B. 直接对公式栏控件 SetValue(绕开键盘焦点)
    for e in formula_controls():
        for pid, cls in ((10002, IUIAutomationValuePattern),
                         (10018, IUIAutomationLegacyIAccessiblePattern)):
            try:
                e.GetCurrentPattern(pid).QueryInterface(cls).SetValue(code)
                time.sleep(0.4); key(0x0D); time.sleep(1.2)
                v = verify_cell(target)
                log(f"  控件SetValue({pid}) -> 真值={v}")
                if v == code:
                    return f"控件SetValue({pid})"
            except Exception:
                continue
    # C. 逐字符回退
    key(0x2E); time.sleep(0.3)
    type_chars(code); time.sleep(0.3)
    key(0x0D); time.sleep(1.3)
    v = verify_cell(target)
    log(f"  逐字符 -> 真值={v}")
    return "逐字符" if v == code else None


def write_code(code, prev_code=None, last_date=None, today=None):
    """返回 (status, target)，status ∈ WROTE / SKIP。

    只在 701 右邻「单一单元格」替换, 绝不做表格级全局查找替换(Ctrl+H):
      - 主锚点 701 常量, 且必须读 B 列公式栏真值 == '701' 才允许继续
      - 从 C 列公式栏读「昨天填写的退货码」(直接从表格读, 不依赖本地凭证)
      - 与上次登记凭证对版: 不符 -> ABORT 中止并如实报错, 绝不盲目覆盖
      - 写后读真值验证 == 当日码, 成功才更新凭证供次日对版
    """
    today = today or time.strftime("%Y-%m-%d")
    activate_sheet_tab()
    ref = sheet_find("701", expect="B")
    if not ref or not ref.startswith("B"):
        raise RuntimeError(
            f"表内查找 701 失败/未落到B列(名称框={ref}); 请确认 701 单元格内容是否完好")

    # 主锚点对版: B 列真值必须是 701, 防止改错行
    b_truth = read_formula_value()
    log(f"{ref} 主锚点公式栏真值 = {b_truth!r}")
    if b_truth != "701":
        raise RuntimeError(f"{ref} 真值={b_truth!r} != '701', 中止保护数据")

    row = int(ref[1:])
    target = f"C{row}"
    if not goto_cell(target):
        raise RuntimeError(f"无法把选中格移到 {target}(名称框未确认), 已中止以保护 701")

    # 读取「昨天填写的退货码」= 表格里该格当前真值
    before = read_formula_value()
    if before is None:                       # 控件树偶发抖动, 有限重试
        for _ in range(2):
            time.sleep(1.0)
            before = read_formula_value()
            if before is not None:
                break
    log(f"{target} 表格中原有码(上次登记) = {before!r}; 凭证上次登记码 = {prev_code!r}")
    if before is None:
        raise RuntimeError(f"{target} 公式栏真值读不到, 无法对版, 中止(绝不盲目写入)")

    action, why = decide_action(before, prev_code, code, last_date, today)
    log(f"对版决策: {action} | {why}")
    if action == "ABORT":
        raise RuntimeError(f"状态异常, 已中止且未改动任何单元格: {why}")
    if action == "SKIP":
        save_result(code, target)
        return "SKIP", target

    log(f"仅在 {target} 单格把 {before!r} 替换为当日码 {code!r} (不做全局搜索替换)")
    for attempt in range(2):
        log(f"第{attempt+1}轮写入尝试")
        how = try_once(code, target)
        if how:
            log(f"真值验证通过: {target} = {code} (方式 {how})")
            save_result(code, target)
            return "WROTE", target
        if not goto_cell(target):
            break

    final = verify_cell(target)
    # 写入失败时, 尽量把原值恢复回去, 不留下被清空的单元格
    if before and final != before:
        log(f"写入失败, 尝试把 {target} 恢复为原值 {before}")
        try_once(before, target)
        final = verify_cell(target)
    raise RuntimeError(
        f"{target} 公式栏真值仍为 {final}, 未写成 {code}; 已尝试剪贴板/控件SetValue/逐字符, 不谎报成功")


# ---------------- 当日码来源(数据已知即停) ----------------
def code_from_args():
    """--data=XXXX 手动指定当日码(最高优先级)。"""
    for a in sys.argv[1:]:
        if a.startswith("--data="):
            c = a.split("=", 1)[1].strip().upper()
            if re.fullmatch(r"[A-Z0-9]{6,12}", c):
                return c
            raise RuntimeError(f"--data 不是合法授权码: {c!r}")
    return None


def code_from_log(today=None):
    """从 return_code.log 读『当天最后一次真实采集』的码。
    数据已知即停: 当天已采过就直接复用, 不再打开紫鸟、不重新采集。"""
    today = today or time.strftime("%Y-%m-%d")
    try:
        with open(LOG, "r", encoding="utf-8") as f:
            txt = f.read()
    except Exception:
        return None
    hits = re.findall(
        r"(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}:\d{2} 美客多刷新后当日授权码:\s*([A-Za-z0-9]{6,12})",
        txt)
    today_hits = [c.upper() for d, c in hits if d == today]
    return today_hits[-1] if today_hits else None


def collect_code():
    """兜底: 当天日志无任何采集凭证时, 才实时刷新美客多读一次码(不反复刷)。"""
    sys.path.insert(0, r"D:\workjiyineiron\2026-11-02-09-03-07")
    from daily_return_code import read_ml_code
    code = read_ml_code()
    log(f"美客多刷新后当日授权码: {code}")
    return code


def main():
    if "--self-test" in sys.argv:
        return self_test()
    force = "--force" in sys.argv
    today = time.strftime("%Y-%m-%d")

    # 一级幂等(硬规则: 数据已知即停): 当日已成功登记 -> 零操作,
    # 不采集、不碰紫鸟、不碰钉钉窗口。当天码若中途更新, 用 --force 重跑。
    last = load_last_result()
    if not force and last.get("date") == today and last.get("ok"):
        log(f"SKIP 今日已成功登记 {last.get('code')} @ {last.get('cell')}, 零操作")
        print(f"DONE(已登记) 退货码={last.get('code')} 单元格={last.get('cell')}")
        return 0

    # 取码优先级: --data > 日志当天已采集(零采集) > 兜底实时采集一次
    code = code_from_args()
    if code:
        log(f"当日退货码(--data 指定): {code} —— 不触碰紫鸟、不重新采集")
    else:
        code = code_from_log(today)
        if code:
            log(f"当日退货码(日志已有采集凭证): {code} —— 数据已知即停, 不触碰紫鸟、不重新采集")
        elif "--no-collect" in sys.argv:
            raise RuntimeError(
                "当日 return_code.log 无采集凭证, 且已禁用紫鸟采集(--no-collect); 如实报未登记")
        else:
            log("当日无采集凭证, 执行唯一一次紫鸟刷新采集(失败即如实报错, 不反复刷页面)")
            code = collect_code()

    ensure_sheet_open()            # 动态发现窗口并自行打开表格
    focus_hwnd(SHEET_HWND, settle=3.0)   # 让表格无障碍树构建完整 + 键盘焦点落在表格
    dismiss_freeze()               # 用户授权: 自己点掉「临时取消冻结」

    prev_code = last.get("code") if last.get("ok") else None
    status, pos = write_code(code, prev_code, last.get("date"), today)
    if status == "SKIP":
        log(f"SKIP 该格已是当日码 {code} @ {pos}, 未改动表格")
        print(f"DONE(已登记) 退货码={code} 单元格={pos}")
    else:
        log(f"DONE 退货码={code} 已写入钉钉 701 行右邻格 {pos}")
        print(f"DONE 退货码={code} 单元格={pos}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as ex:
        log("ERROR: " + str(ex))
        print("ERROR:", ex, flush=True)
        sys.exit(1)
