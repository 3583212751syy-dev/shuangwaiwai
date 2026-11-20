# -*- coding: utf-8 -*-
"""每日广告费消耗登记：2026-11-07 起与退货码任务彻底分开、错峰在 14:00 独立运行，
采集【前一天/昨天】的
营业额、点击量、广告费，按日期写入钉钉「广告费消耗」表对应行的 B/C/D 列。

用户口径：
  - 表头 A=701店铺 B=营业额 C=广告费 D=点击 E=广告比例(公式自动算)；
  - A 列是日期文本「11月5日」，按日期找行，不写当天日期，周日照常；
  - 广告费只在 0-800 可信区间内填写，超出视为页面卡顿，不填写并如实反馈；
  - 必须先确认目标日期/目标行，才允许写入；写入后逐格读公式栏真值验证；
  - 全程后台：必要的前台激活只在紫鸟(改地址栏)和钉钉(键盘写入)各一次，
    不反复跳窗口；其余走 UIA 控件。

用法：
  python daily_ads_cost.py                # 正常：采昨天并写入；当天已登记则零操作跳过
  python daily_ads_cost.py --reuse-last   # 复用上次已核验数据直接写，不碰紫鸟
  python daily_ads_cost.py --data 676,101,106  # 已知数据直接写，不碰紫鸟
  python daily_ads_cost.py --dry-run      # 只采集+定位行，不写入(自检用)
  python daily_ads_cost.py --date 2026-11-04   # 指定日期(补录用)

2026-11-06 用户明令（勿再违背）：
  - 钉钉最左侧固定列里的「广告费消耗」是常驻入口，按名称直接点；不翻文档首页
    内容、不用搜索框、不逐个试无名称 ListItem、不汇报中间探测过程。
  - 数据已知 / 当天已成功登记 -> 直接跳过，不打开紫鸟浏览器、不重新采集、
    不把任何窗口切到前台。
  - 全程后台：不反复激活紫鸟/钉钉；键盘写入最多开头一次必要激活。
"""
import ctypes, re, sys, time, json, argparse
from datetime import date, timedelta
import comtypes
from comtypes.client import GetModule
GetModule("UIAutomationCore.dll")
from comtypes.gen.UIAutomationClient import (
    CUIAutomation, IUIAutomation, TreeScope_Subtree,
    IUIAutomationValuePattern, IUIAutomationInvokePattern,
    IUIAutomationLegacyIAccessiblePattern, IUIAutomationExpandCollapsePattern,
    IUIAutomationSelectionItemPattern)

sys.path.insert(0, r"D:\workjiyineiron\2026-11-02-09-03-07")
import write_return_code as w
import write_ads_cost as a
import finish_ads_cost as f

ctypes.windll.ole32.CoInitializeEx(0, 2)
u = ctypes.windll.user32
auto = comtypes.client.CreateObject(CUIAutomation, interface=IUIAutomation)

MESES = {1: ("enero", "ene"), 2: ("febrero", "feb"), 3: ("marzo", "mar"),
         4: ("abril", "abr"), 5: ("mayo", "may"), 6: ("junio", "jun"),
         7: ("julio", "jul"), 8: ("agosto", "ago"), 9: ("septiembre", "sep"),
         10: ("octubre", "oct"), 11: ("noviembre", "nov"), 12: ("diciembre", "dic")}
LOG = r"D:\workjiyineiron\2026-11-02-09-03-07\.workbuddy\tmp\daily_ads_cost.log"
OUT = r"D:\workjiyineiron\2026-11-02-09-03-07\.workbuddy\tmp\daily_ads_cost_last.json"
# dry-run 不能覆盖 OUT：OUT 是「当天已成功登记」的凭证，被自检覆盖会导致重复采集
DRY_OUT = r"D:\workjiyineiron\2026-11-02-09-03-07\.workbuddy\tmp\daily_ads_cost_dry.json"
RETURN_LOG = r"D:\workjiyineiron\2026-11-02-09-03-07\.workbuddy\tmp\return_code.log"
ML_HWND = None
CELLREF = a.CELLREF


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fp:
        fp.write(line + "\n")


# ---------------- 紫鸟窗口 ----------------
def find_ml_hwnd():
    global ML_HWND
    wins = []
    CB = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def cb(h, l):
        if u.IsWindowVisible(h):
            n = u.GetWindowTextLengthW(h)
            if n:
                b = ctypes.create_unicode_buffer(n + 2)
                u.GetWindowTextW(h, b, n + 2)
                if "G-701" in b.value:
                    wins.append(h)
        return True
    u.EnumWindows(CB(cb), 0)
    ML_HWND = wins[0] if wins else None
    return ML_HWND


def ml_els():
    return auto.ElementFromHandle(ML_HWND).FindAll(
        TreeScope_Subtree, auto.CreateTrueCondition())


def ml_names():
    s = ml_els()
    out = []
    for i in range(s.Length):
        e = s.GetElement(i)
        try:
            out.append((i, e.CurrentControlType, (e.CurrentName or "").strip(), e))
        except Exception:
            continue
    return out


def ml_url():
    for i, ct, nm, e in ml_names():
        if ct == 50004:
            try:
                v = e.GetCurrentPattern(10002).QueryInterface(
                    IUIAutomationValuePattern).CurrentValue
                if v and "mercadolibre.com.mx" in v:
                    return e, v
            except Exception:
                continue
    return None, None


def ml_select_tab(kw):
    for i, ct, nm, e in ml_names():
        if ct == 50019 and (kw in nm
                            or (kw == "广告" and ("Publicidad" in nm
                                                  or "Mercado Ads" in nm))
                            or (kw == "业务指标" and "Métricas" in nm)):
            try:
                e.GetCurrentPattern(10010).QueryInterface(
                    IUIAutomationSelectionItemPattern).Select()
                time.sleep(2.5)
                return True
            except Exception:
                try:
                    e.GetCurrentPattern(10018).QueryInterface(
                        IUIAutomationLegacyIAccessiblePattern).DoDefaultAction()
                    time.sleep(2.5)
                    return True
                except Exception:
                    continue
    return False


def ml_goto(url):
    e, old = ml_url()
    if e is None:
        return False
    if not w.set_clipboard(url):
        return False
    e.SetFocus()
    time.sleep(0.4)
    u.keybd_event(0x11, 0, 0, 0); w.key(0x41); u.keybd_event(0x11, 0, 2, 0)
    time.sleep(0.3)
    u.keybd_event(0x11, 0, 0, 0); w.key(0x56); u.keybd_event(0x11, 0, 2, 0)
    time.sleep(0.7)
    w.key(0x0D)
    return True


def date_patterns(d):
    """目标日的日期按钮文本特征（中文 + 西语 + 任意月份）"""
    y, m, day = d.year, d.month, d.day
    full, abbr = MESES[m]
    return [
        re.compile(rf"^{y}年{m}月{day}日", re.I),
        re.compile(rf"(?<!\d){day} de {full} de {y}", re.I),
        re.compile(rf"(?<!\d){day} {abbr} {y}", re.I),
    ]


def day_token(d):
    """日期控件里的单日标记，如 '5 nov 2026' 或 '2026年11月5日'"""
    y, m, day = d.year, d.month, d.day
    full, abbr = MESES[m]
    return [
        f"{y}年{m}月{day}日",
        f"{day} {abbr} {y}",
        f"{day} de {full} de {y}",
    ]


# ---------------- 采集：业务指标页 ----------------
def collect_business(d):
    ml_select_tab("业务指标")
    stamp = f"{d.year}-{d.month:02d}-{d.day:02d}T06%3A00%3A00.000Z"
    url = ("https://vendedores.mercadolibre.com.mx/metricas?start_period=custom"
           f"&from_current={stamp}&to_current={stamp}")
    if not ml_goto(url):
        return None, "找不到紫鸟地址栏"
    tokens = day_token(d)
    for k in range(8):
        time.sleep(3.0)
        names = [nm for i, ct, nm, e in ml_names()]
        date_txt = next((nm for nm in names
                         if re.search(rf"{d.year}年\d+月\d+日", nm)
                         or re.search(r"\d+ (?:ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic) "
                                      + str(d.year), nm, re.I)), None)
        ok = bool(date_txt and any(t in date_txt for t in tokens))
        sales = visits = None
        for nm in names:
            if sales is None:
                m = (re.match(r"^总销售额\s+([\d.,]+)美元", nm)
                     or re.match(r"^Ventas brutas\s+\$\s*([\d.,]+)", nm))
                if m:
                    sales = m.group(1)
            if visits is None:
                m = (re.match(r"^访问\s+([\d.,]+)", nm)
                     or re.match(r"^Visitas\s+([\d.,]+)", nm))
                if m:
                    visits = m.group(1)
        log(f"  业务指标 轮{k+1}: 日期={date_txt!r} 营业额={sales} 访问={visits}")
        if ok and sales is not None and visits is not None:
            return {"营业额": sales.replace(",", ""), "点击": visits.replace(",", "")}, None
    return None, "业务指标页日期未确认或指标未加载"


# ---------------- 采集：广告费页 ----------------
CAL_BTN = re.compile(
    r"\d{4}年\d+月\d+日，|\d+ de (?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
    r"septiembre|octubre|noviembre|diciembre) de \d{4}|\d+ "
    r"(?:ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic) \d{4}", re.I)
DATE_RANGE = re.compile(
    r"\d{4}年\d+月\d+日\s*-\s*\d{4}年\d+月\d+日|"
    r"\d{1,2} (?:ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)\.? \d{4}\s*-\s*"
    r"\d{1,2} (?:ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)\.? \d{4}", re.I)


def ads_scan():
    out = []
    for i, ct, nm, e in ml_names():
        out.append((i, ct, nm, e))
    ranges = [nm for i, ct, nm, e in out if nm and DATE_RANGE.search(nm)]
    cal = sum(1 for i, ct, nm, e in out
              if ct == 50000 and nm and CAL_BTN.search(nm))
    invest = None
    for i, ct, nm, e in out:
        if ct == 50000 and nm and invest is None:
            m = re.match(r"^(?:投资|Inversión)\s+\$?\s*([\d.,]+)", nm, re.I)
            if m:
                invest = m.group(1)
    return ranges, cal, invest


def ads_expand():
    for i, ct, nm, e in ml_names():
        if ct == 50003 and nm in ("时期", "Periodo", "Período"):
            try:
                e.GetCurrentPattern(10005).QueryInterface(
                    IUIAutomationExpandCollapsePattern).Expand()
                return True
            except Exception:
                continue
    return False


def ads_pick(kind):
    for i, ct, nm, e in ml_names():
        if ct != 50007:
            continue
        low = nm.lower()
        hit = (("过去7天" in nm or "últimos 7 días" in low or "7 días" in low)
               if kind == "p7" else
               (nm in ("个性化", "自定义", "Personalizado") or "personaliz" in low))
        if hit:
            try:
                e.GetCurrentPattern(10010).QueryInterface(
                    IUIAutomationSelectionItemPattern).Select()
                return True
            except Exception:
                continue
    return False


def ads_open_calendar():
    # 2026-11-15 定点修复（已实跑验证，业务校验零改动）：
    # 旧逻辑 `if ads_scan()[1] >= 10: return True` 会把「遗留开着的日历弹层」
    # 当成日历已就绪，跳过「过去7天 -> 个性化」切换。此时日期控件仍显示
    # "Últimos 7 días"、投资卡是 7 天累计(>800)，双击目标日不生效，
    # 连续多轮报「日期未确认 []」且失败后日历仍开着，导致整天反复失败。
    # 判据：日历开着但没有任何日期区间文本 = preset 没切 -> 重载广告页复位，
    # 再强制走完整 preset 流程；日历已开且有区间则保持原行为（正常路径零影响）。
    ranges, cal, invest = ads_scan()
    if cal >= 10:
        if ranges:
            return True
        log("  检测到遗留日历弹层(preset未切)，重载广告页复位后重走 preset")
        ads_url = ("https://vendedores.mercadolibre.com.mx/"
                   "publicidad/product-ads/admin/campaigns")
        if not ml_goto(ads_url):
            return False
        for _ in range(12):
            time.sleep(2.5)
            _, url = ml_url()
            if url and ("/admin/campaigns" in url
                        or "ads.mercadolibre.com.mx/hub" in url):
                break
        time.sleep(2.0)
        if ads_scan()[1] >= 10:
            return False
    ads_expand(); time.sleep(1.5)
    ads_pick("p7"); time.sleep(4.0)
    ads_expand(); time.sleep(1.5)
    ads_pick("custom")
    for _ in range(8):
        time.sleep(1.0)
        if ads_scan()[1] >= 10:
            return True
    return False


def ads_double_click(d):
    pats = date_patterns(d)
    for i, ct, nm, e in ml_names():
        if ct != 50000 or not nm or not any(p.search(nm) for p in pats):
            continue
        ok = 0
        for _ in range(2):
            try:
                e.GetCurrentPattern(10000).QueryInterface(
                    IUIAutomationInvokePattern).Invoke()
                ok += 1
            except Exception:
                try:
                    e.GetCurrentPattern(10018).QueryInterface(
                        IUIAutomationLegacyIAccessiblePattern).DoDefaultAction()
                    ok += 1
                except Exception:
                    pass
            time.sleep(0.12)
        log(f"  双击日期按钮 {nm[:60]!r} 触发{ok}次")
        return ok == 2
    return False


def date_range_ok(ranges, d):
    tokens = day_token(d)
    for nm in ranges:
        # 单日区间：起止都要是目标日
        if any(re.search(rf"{re.escape(t)}\s*-\s*{re.escape(t)}", nm) for t in tokens):
            return nm
        short = f"{d.day} {MESES[d.month][1]} {d.year}"
        if re.search(rf"(?<!\d){re.escape(short)}\s*-\s*{re.escape(short)}", nm, re.I):
            return nm
        # 新版 Mercado Ads 日期控件: "05 nov. 2026 - 05 nov. 2026"
        # (日期带前导零、月份缩写带点)，起止都必须是目标日才算单日确认。
        abbr = MESES[d.month][1]
        if re.search(rf"0?{d.day} {abbr}\.? {d.year}\s*-\s*0?{d.day} {abbr}\.? {d.year}",
                     nm, re.I):
            return nm
    return None


def plausible(v):
    try:
        return 0 <= float(str(v).replace(",", "")) <= 800
    except Exception:
        return False


def collect_ads(d):
    ml_select_tab("广告")
    ranges, cal, invest = ads_scan()
    _, url = ml_url()
    # 2026-11-06 起 /publicidad/product-ads/admin/campaigns 会 302 到新版
    # ads.mercadolibre.com.mx/hub/summary；两种地址都算广告后台页，
    # 日期仍必须走 Período -> 过去7天 -> 个性化 -> 目标日双击，不吃 URL 参数。
    if not url or ("/admin/campaigns" not in url
                   and "ads.mercadolibre.com.mx/hub" not in url):
        # 2026-11-07 定点修复：紫鸟当天未预先打开广告后台标签页时，
        # ml_select_tab("广告") 切不过去、地址栏仍停在业务指标页。
        # 用 URL 只打开广告页（不带任何日期参数），日期筛选仍严格走下面的
        # Período 下拉流程，符合硬性规则 3。一次性必要导航，不反复翻页。
        ads_url = ("https://vendedores.mercadolibre.com.mx/"
                   "publicidad/product-ads/admin/campaigns")
        log("  广告后台标签页未打开，导航到广告页(不带日期参数)")
        if not ml_goto(ads_url):
            return None, "找不到紫鸟地址栏，无法打开广告页"
        for _ in range(12):
            time.sleep(2.5)
            _, url = ml_url()
            if url and ("/admin/campaigns" in url
                        or "ads.mercadolibre.com.mx/hub" in url):
                break
        else:
            return None, f"导航后仍不在广告后台页({(url or '')[:90]})"
        log(f"  已进入广告后台页 {(url or '')[:90]}")
    for attempt in range(2):
        if not ads_open_calendar():
            return None, "日历未打开"
        if not ads_double_click(d):
            continue
        confirmed = None
        for k in range(10):
            time.sleep(1.5)
            ranges, cal, invest = ads_scan()
            if cal >= 10 or not ranges:
                continue
            confirmed = date_range_ok(ranges, d)
            if confirmed:
                break
        if not confirmed:
            log(f"  第{attempt+1}轮日期未确认 {ranges[:1]}")
            continue
        vals = []
        for k in range(2):
            time.sleep(2.0)
            ranges, cal, invest = ads_scan()
            if not date_range_ok(ranges, d):
                break
            if invest is not None:
                vals.append(invest)
        log(f"  广告费 日期={confirmed!r} 两轮={vals}")
        if len(vals) == 2 and vals[0] == vals[1] and plausible(vals[0]):
            return {"广告费": vals[0].replace(",", "")}, None
    return None, "广告费未拿到可信值(日期未确认/超出0-800/页面卡顿)"


# ---------------- 钉钉表格写入 ----------------
def find_row_by_date(d, label):
    """按日期找行。

    实测：这张表的 Ctrl+F 不生效（三次都停在 A5、公式栏返回 '='），
    所以不能靠查找框定位。表结构是 A1 为表头、A2=当月1日、A{d+1}=当月d日
    （11/1-11/4 回填已逐格验证过这个映射）。
    定位后再读 A{row} 公式栏真值对版：读到明确日期但不匹配目标日 -> 拒绝写入；
    读到 '=' / 空（钉钉公式栏偶发残留）则记警告、仍按已验证的行映射继续。
    """
    a.main = w.ding_main_hwnd()
    row = d.day + 1
    target = f"A{row}"
    if not f.goto(target):
        log(f"  无法定位 {target}，拒绝写入")
        return None
    val = a.get_formula(a.scan())
    log(f"  日期 {label} -> {target} 公式栏真值={val!r}")
    if val and val != "=" and re.search(r"\d", val):
        # 明确日期必须对版：兼容 2026/11/4、11月4日、4 nov 2026
        nums = re.findall(r"\d+", val)
        hit = (f"{d.year}/{d.month}/{d.day}" in val
               or f"{d.month}月{d.day}日" in val
               or (len(nums) >= 2 and int(nums[-2]) == d.month and int(nums[-1]) == d.day)
               or re.search(rf"(?<!\d){d.day}\s+{MESES[d.month][1]}\s+{d.year}", val, re.I))
        if not hit:
            log(f"  {target} 真值 {val!r} 与 {label} 不符，拒绝写入")
            return None
        log(f"  {target} 真值对版通过")
    else:
        log(f"  {target} 真值不可读({val!r})，按已验证的行映射继续（写入后仍逐格真值验证）")
    return row


def write_row(row, data):
    targets = [(f"B{row}", data["营业额"]), (f"C{row}", data["广告费"]),
               (f"D{row}", data["点击"])]
    done = {}
    for target, value in targets:
        ok, note = f.write_cell(target, value)
        done[target] = {"ok": ok, "value": value, "note": note}
        log(f"  {target} <- {value}: {'OK' if ok else 'FAIL'} {note}")
        if not ok:
            return False, done
    return True, done


def wait_return_code_done(max_wait=180):
    """【默认已停用，仅 --wait 显式调用】旧设计: 两任务同为 13:30 时靠轮询日志互等。
    2026-11-07 起退货码 13:30 / 广告费 14:00 错峰独立运行，不再靠日志轮询串行，
    避免退货码失败时白等 180s、也避免两边同时激活钉钉互抢焦点。
    保留此函数只为手动排障；只等今天日志里出现 DONE，超时则如实记一笔后继续。"""
    today = date.today().strftime("%Y-%m-%d")
    for k in range(max_wait // 10):
        try:
            with open(RETURN_LOG, "r", encoding="utf-8") as fp:
                txt = fp.read()
            if any(line.startswith(today) and "DONE 退货码=" in line
                   for line in txt.splitlines()):
                log("退货码任务今日已写入完成，继续执行")
                return True
        except Exception:
            pass
        time.sleep(10)
    log(f"等待退货码任务 {max_wait}s 未见 DONE，继续执行（如冲突会在真值验证阶段暴露）")
    return False


def load_last():
    """上一次成功写入的结果，作为「数据已知 / 当天已登记」的凭证。"""
    try:
        with open(OUT, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return None


def already_done(d):
    """当天已成功登记 -> 零窗口操作跳过：不碰紫鸟、不碰钉钉、不切前台。"""
    last = load_last()
    if last and last.get("date") == d.isoformat() and last.get("ok") is True:
        return last
    return None


def known_data(args):
    """已知数据直接写入，完全不打开紫鸟浏览器、不做任何采集。"""
    if args.data:
        parts = [p.strip() for p in args.data.replace(";", ",").split(",")]
        if len(parts) != 3 or any(not re.fullmatch(r"\d+(?:\.\d+)?", p) for p in parts):
            raise ValueError("--data 需为三个纯数字：营业额,广告费,点击（不含千分位逗号）")
        sales, ads, clicks = parts
        log(f"使用命令行已知数据，跳过紫鸟采集: {sales}/{ads}/{clicks}")
    else:
        last = load_last()
        if not last or not all(k in (last.get("data") or {})
                              for k in ("营业额", "广告费", "点击")):
            raise RuntimeError("--reuse-last 需要上一次已核验的采集值，未找到")
        sales = str(last["data"]["营业额"]).replace(",", "")
        ads = str(last["data"]["广告费"]).replace(",", "")
        clicks = str(last["data"]["点击"]).replace(",", "")
        log(f"复用上次已核验采集值({last.get('date')})，跳过紫鸟采集: {sales}/{ads}/{clicks}")
    if not plausible(ads):
        raise RuntimeError(f"已知广告费 {ads} 不在 0-800 可信区间，拒绝写入")
    return {"营业额": sales, "广告费": ads, "点击": clicks}


def ensure_ding_ready():
    """钉钉已在前台则零激活；否则整轮只做一次必要激活（键盘粘贴所需）。
    绝不反复切窗口、不抢焦点。"""
    a.main = w.ding_main_hwnd()
    if u.GetForegroundWindow() == a.main:
        log("钉钉已在前台，本轮零激活")
        return
    log("钉钉不在前台，执行本轮唯一一次必要激活（键盘写入所需）")
    w.focus_hwnd(a.main, settle=2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--date", help="YYYY-MM-DD，默认昨天")
    ap.add_argument("--no-wait", action="store_true", help="兼容旧参数: 默认已不等退货码任务")
    ap.add_argument("--wait", action="store_true",
                    help="显式等待退货码任务当日 DONE(默认关闭: 两任务已错峰, 不再互等)")
    ap.add_argument("--reuse-last", action="store_true",
                    help="复用上次已核验采集值直接写入，不打开紫鸟浏览器")
    ap.add_argument("--data", help="已知数据 营业额,广告费,点击，直接写入，不打开紫鸟浏览器")
    args = ap.parse_args()
    d = (date.fromisoformat(args.date) if args.date
         else date.today() - timedelta(days=1))
    label = f"{d.month}月{d.day}日"
    log(f"=== 每日广告费登记 目标日 {d.isoformat()} (表内日期文本 {label}) dry_run={args.dry_run} ===")

    # 用户口径：需要填的数据知道了就行。当天已成功登记 -> 零窗口操作直接跳过。
    done = already_done(d)
    if done and not (args.reuse_last or args.data):
        log(f"SKIP {d.isoformat()} 已登记(第{done.get('row')}行)，不打开紫鸟/钉钉、不切前台")
        print(json.dumps({"skipped": True, "date": d.isoformat(),
                          "row": done.get("row"), "data": done.get("data")},
                         ensure_ascii=False), flush=True)
        return 0

    if args.data or args.reuse_last:
        data = known_data(args)          # 完全不碰紫鸟窗口
    else:
        if not args.dry_run and args.wait:
            wait_return_code_done()
        if find_ml_hwnd() is None:
            raise RuntimeError("未找到紫鸟 G-701 店铺窗口")
        # 只有确实需要采集时才激活紫鸟，整轮仅此一次；不反复导航、不重复翻页面
        w.focus_hwnd(ML_HWND, settle=2.0)
        biz, err = collect_business(d)
        if biz is None:
            raise RuntimeError(f"业务指标采集失败: {err}")
        ads, err = collect_ads(d)
        if ads is None:
            raise RuntimeError(f"广告费采集失败: {err}（按要求不填写，等下次或人工处理）")
        data = {**biz, **ads}
        log(f"采集结果 {d.isoformat()}: {data}")

    if args.dry_run:
        log("DRY-RUN: 不写入，采集链路自检通过")
        with open(DRY_OUT, "w", encoding="utf-8") as fp:
            json.dump({"date": d.isoformat(), "label": label, "data": data,
                       "dry_run": True}, fp, ensure_ascii=False, indent=2)
        print(json.dumps({"dry_run": True, "date": d.isoformat(), "data": data},
                         ensure_ascii=False), flush=True)
        return 0

    # 钉钉：已在前台则零激活；否则整轮唯一一次必要激活，之后全程后台
    ensure_ding_ready()
    if not a.ensure_sheet():
        raise RuntimeError("未能打开「广告费消耗」表格，未写入")
    row = find_row_by_date(d, label)
    if row is None:
        raise RuntimeError(f"表内未找到日期行 {label!r}（已读公式栏真值对版），未写入")
    log(f"日期 {label} 位于第 {row} 行 -> B{row}/C{row}/D{row}")
    ok, done = write_row(row, data)
    result = {"date": d.isoformat(), "label": label, "row": row,
              "data": data, "ok": ok, "cells": done}
    with open(OUT, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    if not ok:
        raise RuntimeError(f"第 {row} 行写入未全部通过真值验证，详见 {OUT}")
    log(f"DONE {label} 第{row}行: 营业额={data['营业额']} 广告费={data['广告费']} 点击={data['点击']}")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as ex:
        log("ERROR: " + str(ex))
        print("ERROR:", ex, flush=True)
        sys.exit(1)
