# -*- coding: utf-8 -*-
"""钉钉「广告费消耗」表格回填：按日期把营业额/广告费/点击写入 B/C/D 列。

用户口径：
  - 文档在钉钉【最左侧固定列】，跟「退货登记码」同一列，按名称直接点；
    2026-11-06 用户明令：不翻文档首页内容、不用搜索框、不逐个试无名称 ListItem、
    不汇报中间探测；数据已知/已登记就直接跳过，不打开紫鸟浏览器、不切窗口前台；
  - 表头 A=701店铺 B=营业额 C=广告费 D=点击 E=广告比例(公式, 自动算)；
  - 日期行 A2=11月1日 ... A5=11月4日；
  - 只在开头激活一次窗口（键盘粘贴必需），其余全部 UIA 控件操作，不反复跳窗口；
  - 每格写入后必须读「公式栏真值」验证，禁止靠名称框/查找跳转假装成功。

零坐标、零截图、不读消息内容。
"""
import ctypes, sys, time, re, unicodedata, json, os
sys.path.insert(0, r"D:\workjiyineiron\2026-11-02-09-03-07")
import write_return_code as w

CELLREF = re.compile(r"^[A-Z]{1,2}\d{1,4}$")
TOOLBAR = {"插入", "选择", "菜单", "默认", "溢出", "常规", "搜索", "首页",
           "我的文档", "团队文件", "知识库", "最近打开", "更多", "评论"}
LOG = r"D:\workjiyineiron\2026-11-02-09-03-07\.workbuddy\tmp\ads_cost.log"
DOC = "广告费消耗"

# (行, 营业额B, 广告费C, 点击D)
ROWS = [
    (2, 450, 40, 54),
    (3, 1159, 546, 210),
    (4, 1258, 546, 281),
    (5, 1262, 442, 186),
]

u = ctypes.windll.user32
main = None
RECEIVER = None


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def clean(t):
    return "".join(ch for ch in (t or "")
                   if unicodedata.category(ch) not in ("Cf", "Cc", "Cs")
                   and not (0xE000 <= ord(ch) <= 0xF8FF)).strip()


def scan():
    s = w.subs_of(main)
    out = []
    if s:
        for i in range(s.Length):
            e = s.GetElement(i)
            try:
                out.append((i, e.CurrentControlType, clean(e.CurrentName), e))
            except Exception:
                out.append((i, -1, "", None))
    return out


def get_ref(out=None):
    out = out if out is not None else scan()
    for i, ct, nm, e in out:
        if CELLREF.match(nm):
            return nm
    return None


def get_formula(out=None):
    """公式栏真值 = 名称框之后的 Text(50020)。写入的是数字，优先取纯数值文本，
    避免空单元格时误把后面的工具栏词（插入/选择）当成单元格内容。"""
    out = out if out is not None else scan()
    for k, (i, ct, nm, e) in enumerate(out):
        if not CELLREF.match(nm):
            continue
        cands = []
        for j in range(k + 1, min(k + 10, len(out))):
            i2, ct2, nm2, e2 = out[j]
            if (ct2 == 50020 and nm2 and nm2 not in TOOLBAR
                    and not CELLREF.match(nm2)):
                cands.append(nm2.strip())
        for c in cands:
            if re.fullmatch(r"[\d,]+(?:\.\d+)?", c):
                return c
        return cands[0] if cands else None
    return None


def split(ref):
    m = re.match(r"^([A-Z]{1,2})(\d{1,4})$", ref or "")
    return (m.group(1), int(m.group(2))) if m else ("A", 1)


def focus_grid(out):
    """把键盘焦点放回表格网格。

    不能用缓存的控件索引：写入后工具栏/格式面板会增减控件（实测 280 -> 321），
    绝对索引会漂移，导致焦点设到错误元素、Ctrl+V 落空。
    实测「名称框之前最近的可聚焦容器」这个相对关系稳定，故每次按当前控件树现算。
    只 SetFocus，不切前台窗口。
    """
    p = None
    for k, (i, ct, nm, e) in enumerate(out):
        if CELLREF.match(nm):
            p = k
            break
    if p is None:
        return False
    for k in range(p - 1, -1, -1):
        i, ct, nm, e = out[k]
        if ct in (50026, 50030) and e is not None:
            try:
                if e.CurrentIsKeyboardFocusable:
                    e.SetFocus()
                    return True
            except Exception:
                continue
    return False


def find_receiver():
    """一次性找出能接收方向键的网格容器（DOWN 试探），之后复用。"""
    global RECEIVER
    out = scan()
    for k, (i, ct, nm, e) in enumerate(out):
        if ct not in (50026, 50030) or e is None:
            continue
        try:
            if not e.CurrentIsKeyboardFocusable:
                continue
        except Exception:
            continue
        before = get_ref(scan())
        try:
            e.SetFocus()
        except Exception:
            continue
        time.sleep(0.2)
        w.key(0x28)                      # DOWN
        time.sleep(0.5)
        after = get_ref(scan())
        if after and after != before:
            RECEIVER = k
            log(f"网格容器 #{i} 接收方向键: {before} -> {after}")
            return True
    return False


def goto(target):
    """方向键对齐到目标格，每步读名称框确认；不确认绝不写入。"""
    for step in range(30):
        out = scan()
        cur = get_ref(out)
        if not cur:
            return False
        if cur == target:
            return True
        c, r = split(cur)
        tc, tr = split(target)
        if c < tc:
            vk = 0x27        # RIGHT
        elif c > tc:
            vk = 0x25        # LEFT
        elif r < tr:
            vk = 0x28        # DOWN
        else:
            vk = 0x26        # UP
        focus_grid(out)
        time.sleep(0.15)
        w.key(vk)
        time.sleep(0.45)
        new = get_ref(scan())
        if new == cur:       # 焦点丢了，重设一次再试
            focus_grid(scan())
            time.sleep(0.15)
            w.key(vk)
            time.sleep(0.5)
            new = get_ref(scan())
            if new == cur:
                log(f"  方向键失效: 停在 {cur}, 目标 {target}")
                return False
    return get_ref(scan()) == target


def paste_value(value):
    if not w.set_clipboard(str(value)):
        return False
    u.keybd_event(0x11, 0, 0, 0)
    w.key(0x56)
    u.keybd_event(0x11, 0, 2, 0)
    time.sleep(0.6)
    w.key(0x0D)              # 提交
    time.sleep(0.8)
    return True


def write_cell(target, value):
    if not goto(target):
        return False, f"无法定位 {target}"
    before = get_formula(scan())
    last = None
    for attempt in range(2):
        if not goto(target):
            return False, f"第{attempt+1}轮无法回到 {target}"
        focus_grid(scan())       # 按当前控件树现算，避免索引漂移
        time.sleep(0.2)
        if not paste_value(value):
            return False, "剪贴板写入失败"
        time.sleep(0.8)          # 等单元格内容异步落定
        if not goto(target):      # 回车会下移一格，走回来读真值
            return False, f"写入后无法回到 {target} 验证"
        v = get_formula(scan())
        last = v
        if v is not None and v.strip() == str(value):
            return True, f"真值={v!r}"
        log(f"  {target} 第{attempt+1}轮真值={v!r}，重试")
    if before and last != before:
        log(f"  {target} 失败，尝试恢复原值 {before}")
        paste_value(before)
    return False, f"真值={last!r}"


def sheet_open(out):
    """名称框 + 文档标题(控件树里的 Pane) 同时出现，才算目标表格就绪。"""
    refs = [nm for i, ct, nm, e in out if CELLREF.match(nm)]
    title = any(ct == 50030 and DOC in nm for i, ct, nm, e in out)
    return bool(refs) and title


def open_fixed_column():
    """左侧最外列「文档」入口按名称点一次，让常驻文档列展开。

    用户 2026-11-06 明令：最左侧那一列里「广告费消耗」是固定入口，
    直接按名称点。这里只按名称激活「文档」导航项（不是逐个试空白 ListItem、
    不用搜索框、不翻消息、不读文档首页内容），展开后立即按名称定位目标文档。
    """
    cands = [(i, ct, e) for i, ct, nm, e in scan() if nm == "文档" and e is not None]
    cands.sort(key=lambda x: 0 if x[1] == 50020 else 1)
    for i, ct, e in cands:
        w.activate_ctrl(e)
        time.sleep(2.0)
        if any(nm == DOC for i2, ct2, nm, e2 in scan()):
            log("左侧固定列已展开，按名称定位「广告费消耗」")
            return True
    # 2026-11-07 定点修复: 钉钉刚从隐藏恢复、停在「消息」IM 视图时控件树
    # 会缩水(~162 个), 完全没有名为「文档」的控件。按 2026-11-06 16:00 实测:
    # client_ding::NavigatorScrollView 之后第一个 Qt 自绘 ListItem(50007,
    # 名称/AutomationId 全空) 执行一次 DoDefaultAction, 约 3s 后文档视图展开、
    # 固定列出现「广告费消耗」。只激活一次, 不逐个试、不用搜索框、不翻消息。
    try:
        s = w.subs_of(main)
        for i in range(s.Length):
            try:
                e = s.GetElement(i)
                if (e.CurrentClassName or "") != "client_ding::NavigatorScrollView":
                    continue
                for j in range(i + 1, min(i + 4, s.Length)):
                    try:
                        it = s.GetElement(j)
                        if it.CurrentControlType != 50007:
                            continue
                        how = w.activate_ctrl(it)
                        log(f"文档导航项一次激活({how})，等待固定列展开")
                        # 钉钉刚从隐藏恢复时加载较慢，多轮等固定列出现，只激活一次
                        for k in range(6):
                            time.sleep(1.5)
                            if any(nm == DOC for i2, ct2, nm, e2 in scan()):
                                log("固定列已出现「广告费消耗」")
                                return True
                        break
                    except Exception as ex2:
                        log(f"文档导航项候选异常: {ex2}")
                        continue
                break
            except Exception as ex1:
                log(f"文档导航项扫描异常: {ex1}")
                continue
    except Exception as ex:
        log(f"文档导航项回退异常: {ex}")
    return False


def ensure_sheet():
    """表格已开则直接返回；否则只按左侧固定列的精确名称点开「广告费消耗」。

    不翻文档首页内容、不用搜索框、不逐个试无名称 ListItem、不汇报探测细节。
    """
    out = scan()
    if sheet_open(out):
        log("表格已打开: 广告费消耗")
        return True
    if not any(nm == DOC for i, ct, nm, e in out):
        # IM 视图下固定列控件未展开：按名称激活一次「文档」入口
        open_fixed_column()
        out = scan()
    cands = [(i, ct, e) for i, ct, nm, e in out if nm == DOC and e is not None]
    if not cands:
        log("左侧固定列里没有「广告费消耗」条目")
        return False
    # 实测(2026-11-06): 文档目录里同名控件有 Button(50000) 与 Text(50020)，
    # 只有 Text 控件的 DoDefaultAction 能真正切换文档；Button 点了标题不变。
    # 按控件类型优先 Text，并允许对同名候选逐个回退，每步都以表格真就绪为准。
    cands.sort(key=lambda x: 0 if x[1] == 50020 else 1)
    log("按名称点开左侧固定列「广告费消耗」")
    for i, ct, e in cands:
        w.activate_ctrl(e)
        for k in range(10):
            time.sleep(1.5)
            if sheet_open(scan()):
                log("表格就绪")
                return True
    return False


def main():
    global main
    main = w.ding_main_hwnd()
    # 已在前台则零激活；否则整轮只激活这一次（键盘粘贴需要），之后全程后台
    if u.GetForegroundWindow() != main:
        w.focus_hwnd(main, settle=2.0)

    if not ensure_sheet():
        raise RuntimeError("未能打开「广告费消耗」表格，已停止，未写入任何数据")

    if not find_receiver():
        raise RuntimeError("找不到能接收方向键的网格容器，已停止，未写入任何数据")

    results = {}
    for row, sales, ads, clicks in ROWS:
        for col, value in (("B", sales), ("C", ads), ("D", clicks)):
            target = f"{col}{row}"
            ok, note = write_cell(target, value)
            results[target] = {"ok": ok, "value": value, "note": note}
            log(f"  {target} <- {value}: {'OK' if ok else 'FAIL'} {note}")
            if not ok:
                # 单个格子失败不硬凑，立刻如实停止并报出已完成的部分
                log(f"ABORT: {target} 未写成，停止后续写入")
                print(json.dumps(results, ensure_ascii=False), flush=True)
                return 1

    log(f"DONE 已写入并逐格验证 {len(results)} 格 (11/1-11/4 营业额/广告费/点击)")
    print(json.dumps(results, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as ex:
        log("ERROR: " + str(ex))
        print("ERROR:", ex, flush=True)
        sys.exit(1)
