# -*- coding: utf-8 -*-
"""
ecom_report_tool.py — 电商产品图片分析报表工具 v2.0
=====================================================
核心能力：
1. 多图嵌入：产品图 / 1688采购图 / 平台售卖图 放同一 sheet 对应行，一一对照分析
2. 物流选项：sea 海运 / air 空运 / mixed 海运+空运结合（自动给出备货比例建议）
3. 平台选项：shopee-id(本土) / shopee-id-cb(跨境) / lazada-id / tiktok-id / amazon-us
4. 记忆库：SQLite 记录历史分析 -> 重复产品自动停止
5. 禁用条件：负利润 / 需资质 / 重复使用 -> 任一命中即停止出表并输出原因报告
6. 自动搜索：--search-json 注入 AI(WebSearch) 竞品数据；--auto-search 用内置平台价格库
7. 完成通知：结束打印 DONE + 输出 result.json（AI 侧负责 present_files 通知用户）

用法示例：
  python ecom_report_tool.py --product "浴室脏衣篮.jpg" --platform shopee-id --logistics mixed \
      --purchase 4.00 --benchmark 109000 --ad 5 --price-range "3.6,16.0"
  python ecom_report_tool.py --image-dir "D:\\产品图" --platform lazada-id --logistics sea --search-json "竞品数据.json"
  python ecom_report_tool.py --config config.json
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import sqlite3
import sys

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ============================================================
# 1. 平台配置（可扩展）
# ============================================================
# fx: 1 当地货币 = ? CNY
PLATFORMS = {
    "shopee-id": {
        "name": "Shopee 印尼本土店(PT)", "currency": "IDR", "fx": 0.0003782,
        "commission": 0.05, "trans": 0.02, "pay": 0.01,
        "intl_ship": 3.00, "local_ship_idr": 10000, "cert": [],  # cert=强制资质关键词
    },
    "shopee-id-cb": {
        "name": "Shopee 印尼跨境店(CB)", "currency": "IDR", "fx": 0.0003782,
        "commission": 0.08, "trans": 0.06, "pay": 0.01,
        "intl_ship": 3.00, "local_ship_idr": 8000, "cert": [],
    },
    "lazada-id": {
        "name": "Lazada 印尼(LGS)", "currency": "IDR", "fx": 0.0003782,
        "commission": 0.06, "trans": 0.02, "pay": 0.01,
        "intl_ship": 3.00, "local_ship_idr": 10000, "cert": [],
    },
    "tiktok-id": {
        "name": "TikTok Shop 印尼", "currency": "IDR", "fx": 0.0003782,
        "commission": 0.02, "trans": 0.02, "pay": 0.01,
        "intl_ship": 3.00, "local_ship_idr": 10000, "cert": [],
    },
    "amazon-us": {
        "name": "Amazon 美国站", "currency": "USD", "fx": 7.20,
        "commission": 0.15, "trans": 0.01, "pay": 0.01,
        "intl_ship": 5.00, "local_ship_usd": 4.50, "cert": [],
    },
}
# 物流方案（国际段 CNY/件；amazon-us 用 USD/件 -> 由平台 fx 折算，这里统一按 CNY 输入则跳过）
LOGISTICS = {
    "sea":   {"label": "海运 LCL（20-35天）", "intl_ship": 3.00, "days": "20-35天"},
    "air":   {"label": "空运（5-10天）", "intl_ship": 22.00, "days": "5-10天"},
    "mixed": {"label": "海运+空运结合（首轮海运+补货空运）", "intl_ship": 3.00, "air_ship": 22.00, "days": "海运20-35天/空运5-10天"},
}
# 强制资质规则：(平台通配, 关键词正则, 资质要求说明)；命中即停止
CERT_RULES = [
    ("*", r"food\s*contact|食品接触|餐盒|奶瓶|水杯|吸管", "SNI 7323:2008（食品接触塑料强制）"),
    ("*", r"cosmetic|化妆品|面膜|防晒|彩妆|口红", "BPOM 注册（化妆品强制）"),
    ("*", r"drug|药品|保健品|维生素|protein|蛋白粉", "BPOM 注册（药品/保健品强制）"),
    ("*", r"electronic|电子|充电|电源|灯|light|battery|电池|耳机|phone|手机", "SNI 强制认证（电子产品）"),
    ("*", r"baby|儿童|玩具|toy|child", "SNI 强制认证（儿童用品）"),
    ("*", r"vape|电子烟|alcohol|酒|香烟|cigarette", "平台禁售/限制类目"),
    ("shopee-id|shopee-id-cb|lazada-id|tiktok-id", r"halal|清真", "Halal 认证（食品类强制）"),
]
# 内置平台价格库（auto-search 模式，按产品关键词粗配）
AUTO_PRICE_LIB = {
    "laundry|脏衣篮|收纳篮|脏衣篓|basket|hamper": {
        "range_cny": (3.60, 16.00), "mid_cny": 6.7,
        "idr_band": (30000, 80000), "idr_main": (50000, 150000),
        "competitors_1688": [
            ("无品牌普料壁挂脏衣篮", 3.60, "塑料薄、挂钩易断"),
            ("折叠手提脏衣篓", 3.90, "尺寸偏小、用料单薄"),
            ("免打孔壁挂收纳篮", 4.20, "胶贴易脱落、承重差"),
            ("大容量衣物收纳筐", 4.50, "与本品采购价同档"),
            ("卫生间挂式脏衣篮", 5.30, "同质化严重"),
            ("可折叠脏衣篓(热销)", 5.50, "价格战最烈档"),
            ("防水牛津布脏衣篮", 7.80, "贵、起订量大"),
            ("带盖壁挂脏衣篮", 10.50, "高价小众"),
            ("加厚塑料收纳筐", 12.80, "高端慢销"),
            ("品牌定制款", 16.00, "品牌溢价"),
        ],
        "competitors_platform": [
            ("无品牌塑料脏衣篮", 45000, "红海带底部"),
            ("入门折叠布艺款", 65000, "主销带"),
            ("大众织物款", 85000, "大众段"),
            ("带盖织物款", 120000, "中高端"),
            ("设计款多格", 160000, "设计溢价"),
            ("高级品牌锚点", 2299000, "品牌锚点"),
        ],
    },
}

# ============================================================
# 2. 记忆库（SQLite）
# ============================================================
MEMORY_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ecom_report_memory.db")


def init_memory(db_path=MEMORY_DB):
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE IF NOT EXISTS reports(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_hash TEXT, product_name TEXT, platform TEXT, logistics TEXT,
        purchase REAL, benchmark REAL, result TEXT, note TEXT,
        file_path TEXT, created_at TEXT)""")
    conn.commit()
    return conn


def product_hash(name):
    return hashlib.md5(name.strip().lower().encode("utf-8")).hexdigest()[:16]


def check_duplicate(conn, product_name, platform):
    """返回已存在的记录；无则 None"""
    cur = conn.execute(
        "SELECT id, product_name, platform, result, created_at, file_path FROM reports "
        "WHERE product_hash=? AND platform=? ORDER BY id DESC LIMIT 1",
        (product_hash(product_name), platform))
    return cur.fetchone()


def save_report(conn, product_name, platform, logistics, purchase, benchmark, result, note, file_path):
    conn.execute(
        "INSERT INTO reports(product_hash, product_name, platform, logistics, purchase, benchmark, result, note, file_path, created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)",
        (product_hash(product_name), product_name, platform, logistics, purchase, benchmark, result, note,
         file_path, datetime.datetime.now().isoformat(timespec="seconds")))
    conn.commit()


def list_reports(conn, limit=10):
    cur = conn.execute("SELECT created_at, product_name, platform, result, file_path FROM reports ORDER BY id DESC LIMIT ?", (limit,))
    return cur.fetchall()

# ============================================================
# 3. 工具函数
# ============================================================
def fmt_cny(v): return f"¥{v:,.2f}"
def fmt_pct(v): return f"{v*100:.1f}%"
def fmt_money(v, cur="IDR"):
    if cur == "IDR": return f"Rp{v:,.0f}"
    if cur == "USD": return f"${v:,.2f}"
    return f"{v:,.2f}"

def to_cny(amount, cur, fx):
    if cur == "CNY": return amount
    return amount * fx

def convert_price(amount, cur, fx, to="CNY"):
    if cur == to: return amount
    if to == "CNY": return amount * fx
    return amount / fx

# ============================================================
# 4. 样式
# ============================================================
HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
HEADER_FONT = Font(name="微软雅黑", size=11, bold=True, color="FFFFFF")
TITLE_FONT = Font(name="微软雅黑", size=14, bold=True, color="1F4E79")
SUB_FONT = Font(name="微软雅黑", size=10, color="595959")
BODY_FONT = Font(name="微软雅黑", size=10)
BOLD_FONT = Font(name="微软雅黑", size=10, bold=True)
GREEN_FONT = Font(name="微软雅黑", size=10, bold=True, color="1E7B34")
RED_FONT = Font(name="微软雅黑", size=10, bold=True, color="C00000")
WARN_FILL = PatternFill("solid", fgColor="FFF2CC")
OK_FILL = PatternFill("solid", fgColor="E2EFDA")
BAD_FILL = PatternFill("solid", fgColor="FCE4EC")
THIN = Side(style="thin", color="B7B7B7")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)
RIGHT = Alignment(horizontal="right", vertical="center", wrap_text=True)

# ============================================================
# 5. 资质核查（禁用条件 1）
# ============================================================
def check_certification(product_name, platform_key):
    """返回 (是否需资质, 资质清单)。命中任一规则 -> 停止"""
    needed = []
    for plat_wild, pattern, cert_name in CERT_RULES:
        wilds = plat_wild.split("|")
        if "*" not in wilds and platform_key not in wilds:
            continue
        if re.search(pattern, product_name, re.I):
            needed.append(cert_name)
    return (len(needed) > 0, needed)

# ============================================================
# 6. 利润计算
# ============================================================
def calc_profit(price_cur, ad_rate, purchase_cny, plat, ship_intl_cny, extra_cny=0.0):
    """单件净利 CNY"""
    rev_cny = to_cny(price_cur, plat["currency"], plat["fx"])
    if plat["currency"] == "IDR":
        local = to_cny(plat.get("local_ship_idr", 10000), "IDR", plat["fx"])
    elif plat["currency"] == "USD":
        local = to_cny(plat.get("local_ship_usd", 4.50), "USD", plat["fx"])
    else:
        local = 0.0
    cost = purchase_cny + ship_intl_cny + local + extra_cny
    fee = rev_cny * (plat["commission"] + plat["trans"] + plat["pay"] + ad_rate)
    return rev_cny - fee - cost

def calc_rate(price_cur, ad_rate, purchase_cny, plat, ship_intl_cny, extra_cny=0.0):
    rev = to_cny(price_cur, plat["currency"], plat["fx"])
    if rev == 0:
        return 0.0
    return calc_profit(price_cur, ad_rate, purchase_cny, plat, ship_intl_cny, extra_cny) / rev

# ============================================================
# 7. 报表生成（5-Sheet + 图片页）
# ============================================================
def embed_images(ws, anchor, img_paths, col_span=4, img_h=110):
    """把一组图片横排嵌入 sheet。img_paths: {label: path}"""
    from openpyxl.utils.cell import coordinate_from_string, column_index_from_string
    col_str, row_str = coordinate_from_string(anchor)
    c_start = column_index_from_string(col_str)
    row = int(row_str)
    c = c_start
    for label, path in img_paths.items():
        if not path or not os.path.exists(path):
            ws.cell(row=row, column=c, value=f"[{label}:缺图]").font = SUB_FONT
        else:
            try:
                img = XLImage(path)
                img.height = img_h
                img.width = img_h  # 保持方形缩略；如需按比例可换用 PIL
                ws.add_image(img, f"{get_column_letter(c)}{row}")
            except Exception as e:
                ws.cell(row=row, column=c, value=f"[{label}:图读失败]").font = SUB_FONT
        c += 1
        if c >= c_start + col_span:
            break


def build_report(args, plat, logistics, purchase_cny, benchmark, ad_rates, discount_steps,
                 competitors_1688, competitors_platform, extra_cny, result_meta):
    """生成 5-sheet xlsx；result_meta 供负利润判定"""
    out_path = args.output
    wb = Workbook()
    ws = wb.active
    ws.title = "产品与竞品图"
    ncol = 3 + 4 * 2

    # ---------- Sheet 0: 产品与竞品图（图片对照页） ----------
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncol)
    ws["A1"] = f"产品与竞品图对照 — {args.product_name}（{plat['name']} / {logistics['label']}）"
    ws["A1"].font = TITLE_FONT
    img_headers = ["产品图", "1688采购图", "平台售卖图", "说明/备注"]
    for i, h in enumerate(img_headers, 1):
        cell = ws.cell(row=3, column=i, value=h)
        cell.font = HEADER_FONT; cell.fill = HEADER_FILL; cell.alignment = CENTER; cell.border = BORDER
    embed_images(ws, "A4", {
        "产品图": args.product_image, "1688采购图": args.purchase_image,
        "平台售卖图": args.platform_image,
    })
    ws.cell(row=4, column=4, value=(
        f"产品：{args.product_name}\n平台：{plat['name']}\n物流：{logistics['label']}\n"
        f"采购价：{fmt_cny(purchase_cny)}\n基准售价：{fmt_money(benchmark, plat['currency'])}"
    )).font = BODY_FONT
    ws.cell(row=4, column=4).alignment = LEFT
    for cc in range(1, 5):
        ws.cell(row=4, column=cc).border = BORDER
    ws.cell(row=4, column=4).border = BORDER
    ws.row_dimensions[4].height = 90
    for i, w in enumerate([16, 16, 16, 46], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A3"

    # ---------- Sheet 1: 利润明细 ----------
    ws1 = wb.create_sheet("利润明细")
    ws1.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncol)
    ws1["A1"] = f"利润明细 — {args.product_name} × {plat['name']}"
    ws1["A1"].font = TITLE_FONT
    ship_label = logistics["intl_ship"]
    air_ship = logistics.get("air_ship")
    ws1.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncol)
    ws1["A2"] = (f"基准售价 {fmt_money(benchmark, plat['currency'])} | 采购价 {fmt_cny(purchase_cny)} | "
                 f"物流 {logistics['label']}(国际段 {fmt_cny(ship_label)}{'+空运' + fmt_cny(air_ship) if air_ship else ''}/件) | "
                 f"佣金{plat['commission']*100:.0f}%+交易{plat['trans']*100:.0f}%+支付{plat['pay']*100:.0f}%")
    ws1["A2"].font = SUB_FONT
    headers = ["降价档", f"售价({plat['currency']})", "售价(CNY)"]
    for ad in ad_rates:
        headers += [f"广告{int(ad*100)}%净利(海运)", "利润率"]
    row = 3
    for c, h in enumerate(headers, 1):
        cell = ws1.cell(row=row, column=c, value=h)
        cell.font = HEADER_FONT; cell.fill = HEADER_FILL; cell.alignment = CENTER; cell.border = BORDER
    row += 1
    any_negative = False
    for d in discount_steps:
        price = int(benchmark * (1 - d))
        label = "原价" if d == 0 else f"降{d*100:.0f}%"
        ws1.cell(row=row, column=1, value=label).font = BOLD_FONT
        ws1.cell(row=row, column=2, value=fmt_money(price, plat["currency"])).font = BODY_FONT
        ws1.cell(row=row, column=3, value=f"≈{fmt_cny(to_cny(price, plat['currency'], plat['fx']))}").font = BODY_FONT
        col = 4
        for ad in ad_rates:
            p = calc_profit(price, ad, purchase_cny, plat, ship_label, extra_cny)
            r = calc_rate(price, ad, purchase_cny, plat, ship_label, extra_cny)
            c1 = ws1.cell(row=row, column=col, value=fmt_cny(p)); c1.alignment = RIGHT; col += 1
            c2 = ws1.cell(row=row, column=col, value=fmt_pct(r)); c2.alignment = RIGHT; col += 1
            for c in (c1, c2):
                c.border = BORDER
                c.font = GREEN_FONT if p > 0 else RED_FONT
            if p <= 0:
                any_negative = True
        for cc in range(1, len(headers) + 1):
            ws1.cell(row=row, column=cc).border = BORDER
        row += 1
    # 空运敏感性行（mixed/air 时展示）
    if air_ship:
        ws1.cell(row=row, column=1, value=f"空运参考({logistics['label']})").font = BOLD_FONT
        ws1.cell(row=row, column=2, value=fmt_money(int(benchmark*0.5), plat["currency"]))  # 半价档
        ws1.cell(row=row, column=3, value=f"≈{fmt_cny(to_cny(int(benchmark*0.5), plat['currency'], plat['fx']))}")
        p_air = calc_profit(int(benchmark*0.5), 0.05, purchase_cny, plat, air_ship, extra_cny)
        ws1.cell(row=row, column=6, value=fmt_cny(p_air)).font = RED_FONT if p_air <= 0 else GREEN_FONT
        for cc in range(1, len(headers)+1):
            ws1.cell(row=row, column=cc).border = BORDER
            ws1.cell(row=row, column=cc).fill = BAD_FILL if p_air <= 0 else OK_FILL
        row += 2
    else:
        row += 1
    ws1.freeze_panes = "A4"
    widths = [16, 12, 12] + [11, 9] * len(ad_rates)
    for i, w in enumerate(widths, 1):
        ws1.column_dimensions[get_column_letter(i)].width = w

    # ---------- Sheet 2: 竞品全景 ----------
    ws2 = wb.create_sheet("竞品全景")
    ws2.merge_cells("A1:G1")
    ws2["A1"] = "竞品全景 — 1688 供应链价格 vs 平台售价（来源：AI 自动搜索 / 内置价格库）"
    ws2["A1"].font = TITLE_FONT
    headers2 = ["序号", "商品/卖家", "1688价(CNY)", f"平台价({plat['currency']})", "折合CNY", "月销/热度", "对标痛点"]
    for c, h in enumerate(headers2, 1):
        cell = ws2.cell(row=2, column=c, value=h)
        cell.font = HEADER_FONT; cell.fill = HEADER_FILL; cell.alignment = CENTER; cell.border = BORDER
    r = 3
    ws2.cell(row=r, column=1, value="本品基准").font = BOLD_FONT
    ws2.cell(row=r, column=2, value=args.product_name).font = BODY_FONT
    ws2.cell(row=r, column=3, value=fmt_cny(purchase_cny)).font = BODY_FONT
    ws2.cell(row=r, column=4, value=fmt_money(benchmark, plat["currency"])).font = BODY_FONT
    ws2.cell(row=r, column=5, value=f"≈{fmt_cny(to_cny(benchmark, plat['currency'], plat['fx']))}").font = BODY_FONT
    ws2.cell(row=r, column=6, value="★ 待售").font = BODY_FONT
    ws2.cell(row=r, column=7, value="基准锚点").font = BODY_FONT
    for cc in range(1, 8):
        ws2.cell(row=r, column=cc).border = BORDER; ws2.cell(row=r, column=cc).fill = WARN_FILL
    r += 1
    for i, (name, price, pain) in enumerate(competitors_1688, 1):
        ws2.cell(row=r, column=1, value=f"1688-{i}").font = BODY_FONT
        ws2.cell(row=r, column=2, value=name).font = BODY_FONT
        ws2.cell(row=r, column=3, value=fmt_cny(price)).font = BODY_FONT
        ws2.cell(row=r, column=4, value="—").font = BODY_FONT
        ws2.cell(row=r, column=5, value="—").font = BODY_FONT
        ws2.cell(row=r, column=6, value="—").font = BODY_FONT
        ws2.cell(row=r, column=7, value=pain).font = BODY_FONT
        for cc in range(1, 8):
            ws2.cell(row=r, column=cc).border = BORDER; ws2.cell(row=r, column=cc).alignment = LEFT
        r += 1
    for i, (name, price, note) in enumerate(competitors_platform, 1):
        ws2.cell(row=r, column=1, value=f"平台-{i}").font = BODY_FONT
        ws2.cell(row=r, column=2, value=name).font = BODY_FONT
        ws2.cell(row=r, column=3, value="—").font = BODY_FONT
        ws2.cell(row=r, column=4, value=fmt_money(price, plat["currency"])).font = BODY_FONT
        ws2.cell(row=r, column=5, value=f"≈{fmt_cny(to_cny(price, plat['currency'], plat['fx']))}").font = BODY_FONT
        ws2.cell(row=r, column=6, value=note).font = BODY_FONT
        ws2.cell(row=r, column=7, value="").font = BODY_FONT
        for cc in range(1, 8):
            ws2.cell(row=r, column=cc).border = BORDER; ws2.cell(row=r, column=cc).alignment = LEFT
        r += 1
    for i, w in enumerate([10, 34, 13, 16, 12, 16, 36], 1):
        ws2.column_dimensions[get_column_letter(i)].width = w
    ws2.freeze_panes = "A3"

    # ---------- Sheet 3: 决策结论（禁用条件结果 + 定价） ----------
    ws3 = wb.create_sheet("决策结论")
    ws3.merge_cells("A1:B1")
    ws3["A1"] = "决策结论 — 记忆库/资质/利润三重检查"
    ws3["A1"].font = TITLE_FONT
    ws3.cell(row=2, column=1, value="检查项").font = HEADER_FONT
    ws3.cell(row=2, column=2, value="结果").font = HEADER_FONT
    for c in (1, 2):
        ws3.cell(row=2, column=c).fill = HEADER_FILL; ws3.cell(row=2, column=c).alignment = CENTER
    r = 3
    checks = result_meta.get("checks", [])
    for name, ok, detail in checks:
        ws3.cell(row=r, column=1, value=name).font = BOLD_FONT
        ws3.cell(row=r, column=2, value=detail).font = GREEN_FONT if ok else RED_FONT
        ws3.cell(row=r, column=1).border = BORDER; ws3.cell(row=r, column=2).border = BORDER
        ws3.cell(row=r, column=1).fill = OK_FILL if ok else BAD_FILL
        ws3.cell(row=r, column=2).fill = OK_FILL if ok else BAD_FILL
        r += 1
    r += 1
    ws3.cell(row=r, column=1, value="定价建议").font = BOLD_FONT
    r += 1
    advice = result_meta.get("advice", [])
    for a in advice:
        ws3.merge_cells(start_row=r, start_column=1, end_row=r, end_column=2)
        ws3.cell(row=r, column=1, value=a).font = BODY_FONT
        ws3.cell(row=r, column=1).alignment = LEFT
        r += 1
    ws3.column_dimensions["A"].width = 18
    ws3.column_dimensions["B"].width = 100

    # ---------- Sheet 4: 供应链物流 ----------
    ws4 = wb.create_sheet("供应链物流")
    ws4.merge_cells("A1:G1")
    ws4["A1"] = f"供应链与物流 — {logistics['label']}（物流方式：{args.logistics}）"
    ws4["A1"].font = TITLE_FONT
    headers4 = ["环节", "说明", "海运方案", "海运费用(CNY/件)", "空运方案", "空运费用(CNY/件)", "备注"]
    for c, h in enumerate(headers4, 1):
        cell = ws4.cell(row=2, column=c, value=h)
        cell.font = HEADER_FONT; cell.fill = HEADER_FILL; cell.alignment = CENTER; cell.border = BORDER
    rows4 = [
        ("① 采购", "工厂采购", "同左", purchase_cny, "同左", purchase_cny, f"采购价{fmt_cny(purchase_cny)}"),
        ("② 国内集运", "工厂→集货仓", "头程专线", 0.50, "同左", 0.50, "含国内运输"),
        ("③ 国际段", "中国→印尼/美国", "海运LCL 20-35天", 3.00, "空运5-10天", 22.00, "海运USD15-25/m³；空运¥15-25/kg"),
        ("④ 清关+仓", "清关入库", "第三方仓", 2.00, "第三方仓", 2.00, "需进口资质"),
        ("⑤ 本地配送", "仓→买家", "JNE/J&T 1-3天", 4.00, "JNE/J&T 1-3天", 4.00, "Rp8k-15k/单"),
        ("⑥ 包装", "彩盒+袋", "随货", 1.00, "随货", 1.00, "防压"),
    ]
    r = 3
    for row_ in rows4:
        for c, v in enumerate(row_, 1):
            cell = ws4.cell(row=r, column=c, value=v)
            cell.font = BODY_FONT; cell.border = BORDER; cell.alignment = LEFT
        r += 1
    r += 1
    sea_total = purchase_cny + 0.5 + 3.0 + 2.0 + 4.0 + 1.0
    air_total = purchase_cny + 0.5 + 22.0 + 2.0 + 4.0 + 1.0
    ws4.cell(row=r, column=1, value="单件物流合计").font = BOLD_FONT
    ws4.cell(row=r, column=3, value=f"海运到岸 {fmt_cny(sea_total)}").font = BOLD_FONT
    ws4.cell(row=r, column=5, value=f"空运到岸 {fmt_cny(air_total)}").font = BOLD_FONT
    if args.logistics == "mixed":
        ws4.cell(row=r, column=7, value=f"混合建议：首批 70% 海运 + 30% 空运测款补货").font = RED_FONT
    for cc in range(1, 8):
        ws4.cell(row=r, column=cc).border = BORDER; ws4.cell(row=r, column=cc).fill = WARN_FILL
    for i, w in enumerate([16, 22, 14, 16, 16, 16, 36], 1):
        ws4.column_dimensions[get_column_letter(i)].width = w
    ws4.freeze_panes = "A3"

    wb.save(out_path)
    return out_path, any_negative


# ============================================================
# 8. 主流程
# ============================================================
def resolve_images(args):
    """图片解析：--product/--purchase/--platform 显式 或 --image-dir 目录按文件名匹配"""
    if args.product_image:
        return args.product_image, args.purchase_image, args.platform_image
    if args.image_dir and os.path.isdir(args.image_dir):
        files = os.listdir(args.image_dir)
        def find(patterns, exclude=()):
            for f in sorted(files):
                low = f.lower()
                if any(p in low for p in patterns) and not any(e in low for e in exclude):
                    return os.path.join(args.image_dir, f)
            return None
        product = find(["product", "产品", "主图", "jpg", "jpeg", "png"], exclude=["1688", "shopee", "竞品", "clipboard"])
        purchase = find(["1688", "采购"])
        platform = find(["shopee", "lazada", "tiktok", "amazon", "售卖", "平台"])
        return product, purchase, platform
    return None, None, None


def pick_auto_competitors(product_name):
    """内置价格库匹配 -> (1688竞品, 平台竞品)"""
    for key, lib in AUTO_PRICE_LIB.items():
        if re.search(key, product_name, re.I):
            return lib["competitors_1688"], lib["competitors_platform"]
    return [], []


def main(argv=None):
    ap = argparse.ArgumentParser(description="电商产品图片分析报表工具 v2.0")
    ap.add_argument("--config", help="JSON 配置文件（可选，覆盖以下默认参数）")
    ap.add_argument("--product", dest="product_image", help="产品图路径")
    ap.add_argument("--purchase-image", dest="purchase_image", help="1688采购图路径")
    ap.add_argument("--platform-image", dest="platform_image", help="平台售卖图路径")
    ap.add_argument("--image-dir", dest="image_dir", help="图片目录（按文件名自动匹配产品图/采购图/售卖图）")
    ap.add_argument("--product-name", dest="product_name", default="", help="产品名称（默认取文件名）")
    ap.add_argument("--platform", default="shopee-id", choices=list(PLATFORMS.keys()), help="售卖平台")
    ap.add_argument("--logistics", default="sea", choices=list(LOGISTICS.keys()), help="物流方式：sea海运/air空运/mixed混合")
    ap.add_argument("--purchase", type=float, default=4.00, help="采购价 CNY")
    ap.add_argument("--benchmark", type=float, default=109000, help="基准售价（当地货币）")
    ap.add_argument("--ad", default="0,5,10,15", help="广告率百分比，逗号分隔")
    ap.add_argument("--discount", default="5,10,15,20,25", help="降价阶梯百分比，逗号分隔（不含原价）")
    ap.add_argument("--extra-cost", dest="extra_cost", type=float, default=0.0, help="其他固定成本 CNY/件")
    ap.add_argument("--search-json", dest="search_json", help="AI 搜索竞品数据 JSON（自动搜索注入）")
    ap.add_argument("--auto-search", action="store_true", help="用内置价格库自动匹配竞品")
    ap.add_argument("--output", default="", help="输出 xlsx 路径（默认 桌面/产品名_平台选品分析报表.xlsx）")
    ap.add_argument("--force", action="store_true", help="跳过记忆库去重（强制生成）")
    ap.add_argument("--embed-images", dest="embed_images_flag", action="store_true", default=True, help="嵌入图片到报表（默认开）")
    ap.add_argument("--no-embed-images", dest="embed_images_flag", action="store_false")
    args = ap.parse_args(argv)

    # 配置文件覆盖
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if getattr(args, k, None) in (None, "", False, []) or k in ("output", "product_name"):
                setattr(args, k, v)

    plat = PLATFORMS[args.platform]
    logistics = LOGISTICS[args.logistics]
    ad_rates = [float(x) / 100 for x in args.ad.split(",")]
    discount_steps = [float(x) / 100 for x in args.discount.split(",")]

    # 图片解析
    prod_img, pur_img, plat_img = resolve_images(args)
    if not args.product_name:
        base = os.path.basename(prod_img) if prod_img else "未命名产品"
        args.product_name = os.path.splitext(base)[0]

    # ---------- 禁用条件检查（顺序：资质 -> 重复 -> 利润） ----------
    checks = []
    # 1) 资质
    need_cert, cert_list = check_certification(args.product_name, args.platform)
    checks.append(("资质检查", not need_cert,
                   "✅ 无强制资质门槛" if not need_cert else f"❌ 需强制资质：{'、'.join(cert_list)}（停止出表）"))
    if need_cert:
        result_meta = {"checks": checks, "advice": [f"该产品触发强制资质（{'、'.join(cert_list)}），按禁用规则停止生成报表。请先完成认证或更换产品。"]}
        _emit_blocked(args, result_meta, "资质不足", cert_list)
        return 1

    # 2) 记忆库去重
    conn = init_memory()
    dup = None if args.force else check_duplicate(conn, args.product_name, args.platform)
    if dup:
        checks.append(("记忆库去重", False,
                       f"❌ 重复使用：'{dup[1]}' 曾在 {dup[2]} 生成过（{dup[3]}，{dup[4]}）→ {dup[5]}（停止出表）"))
        result_meta = {"checks": checks, "advice": [f"该产品在 {args.platform} 已分析过（见记忆库记录），按禁用规则停止。如需重新分析请加 --force。"]}
        _emit_blocked(args, result_meta, "重复使用", dup[5])
        conn.close()
        return 1
    checks.append(("记忆库去重", True, "✅ 新项目，无重复记录"))

    # 3) 竞品数据（AI search-json 优先，其次内置库）
    comp_1688, comp_plat = [], []
    if args.search_json and os.path.exists(args.search_json):
        with open(args.search_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        comp_1688 = data.get("competitors_1688", [])
        comp_plat = data.get("competitors_platform", [])
        checks.append(("自动搜索", True, f"✅ 已注入 AI 搜索结果（1688 {len(comp_1688)} 款 / 平台 {len(comp_plat)} 款）"))
    elif args.auto_search:
        comp_1688, comp_plat = pick_auto_competitors(args.product_name)
        checks.append(("自动搜索", True, f"✅ 内置价格库匹配（1688 {len(comp_1688)} 款 / 平台 {len(comp_plat)} 款）"))
    else:
        checks.append(("自动搜索", True, "ℹ️ 未启用自动搜索（可用 --auto-search 或 --search-json 注入）"))

    # 4) 利润计算 + 负利润判定
    main_ad = ad_rates[1] if len(ad_rates) > 1 else ad_rates[0]
    main_ship = logistics.get("air_ship", logistics["intl_ship"]) if args.logistics == "air" else logistics["intl_ship"]
    # 主力场景：基准售价*0.5（Flash档）+ 常规广告 + 海运
    flash_price = int(args.benchmark * 0.5) if args.benchmark else args.benchmark
    main_p = calc_profit(flash_price, main_ad, args.purchase, plat, main_ship, args.extra_cost)
    main_rate = calc_rate(flash_price, main_ad, args.purchase, plat, main_ship, args.extra_cost)
    checks.append(("利润判定", main_p > 0,
                   f"{'✅' if main_p > 0 else '❌'} 主力场景（半价档+{int(main_ad*100)}%广告+{args.logistics}）单件净利 {fmt_cny(main_p)}，利润率 {fmt_pct(main_rate)}"))
    if main_p <= 0:
        result_meta = {"checks": checks, "advice": [f"主力场景利润 {fmt_cny(main_p)} ≤ 0，按禁用规则停止生成报表。建议：谈判采购价 / 改物流 / 提高售价或放弃该产品。"]}
        _emit_blocked(args, result_meta, "负利润", fmt_cny(main_p))
        conn.close()
        return 1

    # 5) 生成报表
    advice = [
        "✅ 项目可行：主力场景正利润。建议按 Sheet1 矩阵选择定价档位与广告预算。",
        f"物流建议：{logistics['label']}；若选 mixed，首批 70% 海运 + 30% 空运测款。",
        "定价建议：半价档 Flash 只在大促/清库存使用；日常价落在 60-85% 基准价区间保利润。",
    ]
    result_meta = {"checks": checks, "advice": advice}
    out_path = args.output or os.path.join(
        os.path.expanduser("~"), "Desktop", f"{args.product_name}_{args.platform}_选品分析报表.xlsx")
    try:
        built, any_neg = build_report(args, plat, logistics, args.purchase, args.benchmark,
                                      ad_rates, discount_steps, comp_1688, comp_plat,
                                      args.extra_cost, result_meta)
    except Exception as e:
        import traceback
        traceback.print_exc()
        result_meta = {"checks": checks, "advice": [f"报表生成失败：{e}"]}
        _emit_blocked(args, result_meta, "生成失败", str(e))
        conn.close()
        return 1

    save_report(conn, args.product_name, args.platform, args.logistics, args.purchase,
                args.benchmark, "OK" if not any_neg else "OK_有负档", built, out_path)
    conn.close()

    # 6) 结果 JSON（供 AI 通知）
    result = {
        "status": "OK", "product": args.product_name, "platform": args.platform,
        "logistics": args.logistics, "output": built,
        "main_profit": main_p, "main_rate": main_rate,
        "checks": [{"name": n, "ok": b, "detail": d} for n, b, d in checks],
    }
    result_json = os.path.join(os.path.dirname(out_path), f"{args.product_name}_result.json")
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("DONE")
    print("OUTPUT:", built)
    print("RESULT_JSON:", result_json)
    print("MAIN_PROFIT:", fmt_cny(main_p), "RATE:", fmt_pct(main_rate))
    print("CHECKS:", json.dumps([n for n, b, d in checks], ensure_ascii=False))
    return 0


def _emit_blocked(args, result_meta, reason, detail):
    """禁用条件触发：输出停止报告（不生成报表）"""
    out_dir = os.path.expanduser("~") + os.sep + "Desktop"
    block_file = os.path.join(out_dir, f"{args.product_name}_停止原因报告.txt")
    with open(block_file, "w", encoding="utf-8") as f:
        f.write(f"产品：{args.product_name}\n平台：{args.platform}\n物流：{args.logistics}\n\n")
        f.write(f"【停止原因】{reason}: {detail}\n\n检查明细：\n")
        for name, ok, d in result_meta["checks"]:
            f.write(f"- {name}: {d}\n")
        f.write("\n建议：\n")
        for a in result_meta["advice"]:
            f.write(f"- {a}\n")
    print("BLOCKED:", reason, "-", detail)
    print("BLOCKED_REPORT:", block_file)
    print("NO_REPORT_GENERATED")


if __name__ == "__main__":
    sys.exit(main())
