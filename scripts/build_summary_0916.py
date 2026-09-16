# -*- coding: utf-8 -*-
"""5 产品汇总对比表 v3.1（2026-09-16 批次）：物流按「实重 vs 体积重择大」计费。
数据取自各产品 _result.json / _停止原因报告.txt，零手填。p5 结构性亏损单独标注。"""
import json, os, re

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

OUTDIR = r"E:/Desktop"
IMGDIR = r"E:/AI/WorkBuddy/2026-09-14-14-49-16/git_refs/img4"
JSON_BACKUP = r"C:/Users/lenovo/AppData/Local/Temp/ecom-report-json-backup"

# (full_name, short, key, ext, 采购¥, 基准Rp, 实重g, 包装尺寸(cm), 体积重g, 海运¥, 本地¥, 备注)
PRODUCTS = [
    ("Wadah Peniris Multifungsi 6in1 双层家用洗菜盆沥水篮篓厨房客厅水果盘糖果零食盘洗菜篮滤水篮子",
     "双层洗菜盆沥水篮 6件套", "p1", "png", 0.65, 110000, 600, "28×25×15", 1750, 6.30, 3.69,
     "基准=Tokopedia Cypruz 塑料双层沥水篮在售最高 Rp110,000。⚠️抛货最重：体积重 1750g = 实重 2.9×，海运占成本 78%。"),
    ("MAISONNE Spatula Scoop Penjepit 不锈钢煎铲夹子二合一煎鱼铲夹牛排夹烧烤夹子厨房加厚煎夹防烫夹",
     "不锈钢煎铲夹二合一", "p2", "jpg", 0.95, 123750, 95, "26×8×3", 104, 0.374, 2.46,
     "基准=Tokopedia denispujas 同款 Spatula Jepit Gorengan 在售最高 Rp123,750。重货小件，实重胜，利润最厚 71.4%。"),
    ("Kacamata Las Buka Tutup Flip Up 电焊眼镜防强光打磨切割飞溅焊工劳保氩弧焊接打眼护目镜",
     "电焊护目镜 翻盖式", "p3", "jpg", 3.00, 75000, 150, "17×10×8", 227, 0.816, 2.46,
     "基准=Tokopedia toko sempam 翻盖焊接护目镜 Rp75,000（本品 Shopee 券后仅 Rp24,975）。体积重略胜实重 1.5×。"),
    ("POP IT Mini Elektronik 迷你掌上游戏机卡通图案打地鼠电子发光解压玩具小钥匙扣",
     "迷你打地鼠游戏机 钥匙扣", "p4", "jpg", 1.95, 44975, 37, "11×8×3", 44, 0.158, 2.46,
     "基准=Tokopedia Ciptawarna2000 同规格 mini 款 Rp44,975。⚠️资质警告：SNI 儿童用品/电子产品（已豁免，仅供选品参考）。"),
    ("Taplak Kulkas Cover 家具冰箱巾顶盖布防灰尘布防尘罩滚筒洗衣机罩微波炉单开门冰箱罩",
     "家具冰箱防尘罩 单开门", "p5", "jpg", 1.30, 15833, 100, "25×18×3", 225, 0.81, 2.46,
     "基准=FastMoss TikTok 同款 PEVA 带兜通用款在售高位 Rp15,833（本品 Shopee Rp9,750）。体积重 225g = 实重 2.25×。"),
]

THIN = Side(style="thin", color="CCCCCC")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
FILL_HEADER = PatternFill("solid", fgColor="1F4E79")
FILL_OK = PatternFill("solid", fgColor="E2EFDA")
FILL_BAD = PatternFill("solid", fgColor="FCE4EC")
FILL_WARN = PatternFill("solid", fgColor="FFF2CC")
FONT_WHITE = Font(name="微软雅黑", size=10, bold=True, color="FFFFFF")
FONT = Font(name="微软雅黑", size=10)
FONT_S = Font(name="微软雅黑", size=9)
GREEN = Font(name="微软雅黑", size=10, bold=True, color="1E7B34")
RED = Font(name="微软雅黑", size=10, bold=True, color="C00000")
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)
RATE = 0.0003782
AD = 0.05


def load_state(name):
    rj = os.path.join(OUTDIR, f"{name}_result.json")
    if not os.path.exists(rj):
        rj = os.path.join(JSON_BACKUP, f"{name}_result.json")
    if os.path.exists(rj):
        d = json.load(open(rj, encoding="utf-8"))
        return {"status": "OK", "profit": d["main_profit"], "rate": d["main_rate"],
                "sea": d.get("sea_fee"), "local": d.get("local_fee"), "vol_kg": d.get("vol_kg")}
    bt = os.path.join(OUTDIR, f"{name}_停止原因报告.txt")
    if os.path.exists(bt):
        txt = open(bt, encoding="utf-8").read()
        reason = re.search(r"【停止原因】([^\n]+)", txt)
        p = re.search(r"单件净利 ¥([-\d,.]+)，利润率 ([-\d.]+%)", txt)
        return {"status": "BLOCKED", "reason": reason.group(1) if reason else "?",
                "profit": float(p.group(1).replace(",", "")) if p else None,
                "rate": p.group(2) if p else "", "txt": txt}
    return {"status": "MISSING", "profit": None, "rate": ""}


wb = Workbook()
ws = wb.active
ws.title = "5产品选品汇总对比"
ws.sheet_view.showGridLines = False
headers = ["产品图", "产品名", "1688最低价\n(CNY)", "平台基准价\n=竞品最高价(IDR)",
           "实重 / 体积重\n(包装尺寸)", "到岸成本明细\n(CNY/件)", "主力净利\n(CNY/件)",
           "利润率", "闸门结果", "体积重结论 / 基准来源"]
ncol = len(headers)
ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncol)
ws["A1"] = "5产品 Shopee 印尼本土店选品汇总（v3.1 · 物流按「实重 vs 体积重择大」计费）"
ws["A1"].font = Font(name="微软雅黑", size=14, bold=True, color="1F4E79")
ws["A1"].alignment = CENTER
ws.row_dimensions[1].height = 28
ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncol)
ws["A2"] = ("物流口径（国内发印尼真实计费）：海运拼箱 LCL DDP 双清包税 600¥/CBM，按 1CBM=1000kg 与实重「择大」计费，最低 1 CBM 起运；"
            "本地派送按 长×宽×高÷6000 与实重择大（首 1kg ¥2.46 + 续重 ¥1.23/kg）。"
            "成本 = 采购 + 0.5集运 + 海运费 + 2.0印尼仓操作 + 本地派送 + 1.0包装；费率 佣金5%+交易2%+支付1%；主力场景 = 基准价 + 5%广告。")
ws["A2"].font = Font(name="微软雅黑", size=9, color="C00000")
ws["A2"].alignment = LEFT
ws.row_dimensions[2].height = 40

for c, h in enumerate(headers, 1):
    cell = ws.cell(row=3, column=c, value=h)
    cell.font = FONT_WHITE; cell.fill = FILL_HEADER; cell.alignment = CENTER; cell.border = BORDER
ws.row_dimensions[3].height = 36

rows = []
for i, (name, short, key, ext, pc, bench, wg, dims, volkg, sea, local, note) in enumerate(PRODUCTS, 4):
    st = load_state(name)
    ws.row_dimensions[i].height = 84
    img_path = os.path.join(IMGDIR, f"{key}_product.{ext}")
    if os.path.exists(img_path):
        img = XLImage(img_path); img.width = 70; img.height = 70
        ws.add_image(img, f"A{i}")
    ws.cell(row=i, column=1).border = BORDER

    sea_v = st.get("sea") if st.get("sea") is not None else sea
    local_v = st.get("local") if st.get("local") is not None else local
    cost = pc + 0.5 + sea_v + 2.0 + local_v + 1.0
    cost_txt = f"¥{pc:.2f}+0.5+¥{sea_v:.2f}海运+2+¥{local_v:.2f}本地+1 = ¥{cost:.2f}"

    big = "体积重" if volkg > wg else "实重"
    ratio = max(volkg, wg) / min(volkg, wg)
    wt_txt = f"{wg}g / 体积重 {volkg}g\n({dims}cm，{big}胜 {ratio:.1f}×)"

    profit = st.get("profit")
    rate = st.get("rate")
    rate_txt = f"{rate*100:.1f}%" if isinstance(rate, (int, float)) else str(rate or "")

    ws.cell(row=i, column=2, value=short).font = FONT; ws.cell(row=i, column=2).alignment = LEFT
    ws.cell(row=i, column=3, value=pc).font = FONT
    ws.cell(row=i, column=4, value=f"Rp{bench:,}").font = FONT
    ws.cell(row=i, column=5, value=wt_txt).font = FONT_S
    ws.cell(row=i, column=6, value=cost_txt).font = FONT; ws.cell(row=i, column=6).alignment = LEFT
    pfont = GREEN if (profit or 0) > 0 else RED
    ws.cell(row=i, column=7, value=(None if profit is None else round(profit, 2))).font = pfont
    ws.cell(row=i, column=8, value=rate_txt).font = pfont

    if st["status"] == "OK":
        ws.cell(row=i, column=9, value="✅ 通过出表").fill = FILL_OK
        extra = ("利润健康，可优先测款。" if (profit or 0) >= 5 else "利润偏薄，建议先小批量测款。")
        ws.cell(row=i, column=10, value=extra + note).fill = FILL_OK if (profit or 0) >= 5 else FILL_WARN
    else:
        ws.cell(row=i, column=9, value=f"⛔ 停止闸：{st.get('reason','?')}").fill = FILL_BAD
        fixed = 0.5 + sea_v + 2.0 + local_v + 1.0
        cost_max = bench * RATE * (1 - 0.13) - fixed
        if cost_max > 0:
            price_per = cost_max / 3 if pc > 3 else cost_max  # 多件装时给单件口径
            msg = (f"未出表（净利 ¥{st.get('profit')}）。采购价需压到 ¥{cost_max:.2f} 以内才转正"
                   f"（现 ¥{pc:.2f}）。" + note)
        else:
            req_bench = (fixed / 0.87) / RATE  # 采购价压到0时所需的基准售价(IDR)
            msg = (f"未出表（净利 ¥{st.get('profit')}，利润率 {rate_txt}）。⚠️结构性亏损：即便采购价压到 ¥0，"
                   f"固定物流+平台费 ¥{fixed:.2f} 仍 > 到手 ¥{bench*RATE*0.87:.2f}。"
                   f"须把售价提到 ≥Rp{req_bench:,.0f}/件（现 Rp{bench:,}）或换更省物流（如本地仓直发免海运）才转正，"
                   f"采购端已无可压空间。" + note)
        ws.cell(row=i, column=10, value=msg).fill = FILL_BAD

    for c in range(1, ncol + 1):
        cell = ws.cell(row=i, column=c); cell.border = BORDER
        if c in (3, 4, 5, 7, 8, 9):
            cell.alignment = CENTER
        elif c == 10:
            cell.alignment = LEFT
        if c == 10:
            cell.font = FONT_S
        elif cell.font is None or cell.font.name is None:
            cell.font = FONT
    rows.append((name, bench, st))

for col, w in zip("ABCDEFGHIJ", [10, 30, 12, 16, 20, 36, 12, 10, 20, 66]):
    ws.column_dimensions[col].width = w
ws.freeze_panes = "A4"

out = os.path.join(OUTDIR, "5产品选品汇总对比_0916.xlsx")
wb.save(out)
print("SAVED:", out)
for name, bench, st in rows:
    print(f"  {name[:30]:32s} Rp{bench:<9,} {st['status']:8s} profit={st.get('profit')} rate={st.get('rate')}")
