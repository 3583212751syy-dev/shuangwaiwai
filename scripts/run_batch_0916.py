# -*- coding: utf-8 -*-
"""Batch run 5 products with v3.1 production script (volumetric weight, 2026-09-16)
P1 沥水篮6件套 / P2 煎铲夹二合一 / P3 电焊护目镜 / P4 打地鼠游戏机 / P5 冰箱防尘罩"""
import subprocess, os, json

PY = r"C:\Users\lenovo\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
SCRIPT = r"E:\Desktop\双接口\scripts\ecom_report_tool.py"
IMG = r"E:\AI\WorkBuddy\2026-09-14-14-49-16\git_refs\img4"
OUT = r"E:\Desktop"

PRODUCTS = [
    {
        "key": "p1", "ext": "png",
        "name": "Wadah Peniris Multifungsi 6in1 双层家用洗菜盆沥水篮篓厨房客厅水果盘糖果零食盘洗菜篮滤水篮子",
        "purchase": 0.65,          # 1688 六件套PP加厚同款最低（已售43万件）
        "benchmark": 110000,       # Tokopedia Cypruz 塑料双层沥水篮在售最高 Rp110,000
        "weight": 600, "dim": "28x25x15",   # 6层套叠+彩袋，最大篮27.5x24x9(fall-finds规格)
        "comp_1688": [
            ["塑料双层洗菜盆沥水篮子厨房多功能客厅家用水果盘", 0.55, "1688 已售68,486"],
            ["洗菜篮子沥水篮六件套PP加厚同款", 0.65, "1688精选 已售431,008"],
            ["双层洗菜篮八件套多功能塑料", 0.58, "1688 已售21"],
            ["六件套沥水篮子家用双层加厚水果蔬菜清洗盆", 1.30, "1688精选 已售118"],
            ["双多能用水果盆洗菜盆沥水篮八件套", 3.00, "1688精选 已售36,256"],
        ],
        "comp_plat": [
            ["Cypruz Mangkok Baskom Saringan Ganda Cuci Buah Sayur Drain Basket (Tokopedia)", 110000, "tokopedia.com 在售"],
            ["[Dapat 6in1] Wadah Keranjang Tirisan Multifungsi 6in1 (TikTok Shop/FastMoss)", 77215, "fastmoss.com 高位变体"],
            ["LetYeah Baskom saringan sayur 2 in 1 keranjang (Tokopedia)", 45540, "tokopedia.com"],
            ["Masterhome Baskom Cuci Transparan 2in1 Peniris (Tokopedia)", 31500, "tokopedia.com"],
            ["IDOHOUSE Set Baskom Cuci Serbaguna 2in1 (Tokopedia)", 17160, "tokopedia.com"],
        ],
    },
    {
        "key": "p2", "ext": "jpg",
        "name": "MAISONNE Spatula Scoop Penjepit 不锈钢煎铲夹子二合一煎鱼铲夹牛排夹烧烤夹子厨房加厚煎夹防烫夹",
        "purchase": 0.95,          # 1688 304不锈钢夹牛排夹煎鱼夹最低（已售28,886）
        "benchmark": 123750,       # Tokopedia denispujas 同款"Spatula Jepit Gorengan"在售最高 Rp123,750
        "weight": 95, "dim": "26x8x3",     # 304不锈钢二合一铲夹，长约25cm
        "comp_1688": [
            ["批发304不锈钢夹牛排夹煎鱼夹多功能食物夹", 0.95, "1688精选 已售28,886"],
            ["304不锈钢牛排夹烤肉夹烤鱼夹食物夹", 0.99, "1688 已售171,604"],
            ["不锈钢煎铲夹子二合一煎鱼铲夹牛排夹", 0.99, "1688 已售1,863"],
            ["304不锈钢煎鱼夹创意烘焙牛排煎夹", 1.20, "1688 已售32,196"],
            ["【网红爆款】304不锈钢牛排夹煎鱼煎蛋厨房加厚食物夹", 1.50, "1688精选 已售1,455"],
        ],
        "comp_plat": [
            ["Spatula Masak Goreng Stainless Steel Spatula Jepit Gorengan (Tokopedia denispujas)", 123750, "tokopedia.com 在售"],
            ["MAISONNE Penjepit Makanan Stainless Steel Food Tong NPJ01 (Shopee 本品)", 15000, "shopee.co.id 原价Rp15,000/闪购Rp11,503"],
            ["Spatula Masak Goreng Stainless Steel Jepit Gorengan (Tokopedia mjp store)", 15000, "tokopedia.com"],
            ["Jepitan Daging BBQ Stainless Spatula Jepit Steak Food Tong (Tokopedia)", 7999, "tokopedia.com"],
        ],
    },
    {
        "key": "p3", "ext": "jpg",
        "name": "Kacamata Las Buka Tutup Flip Up 电焊眼镜防强光打磨切割飞溅焊工劳保氩弧焊接打眼护目镜",
        "purchase": 3.00,          # 1688 翻盖式同款最低（四款镜片任意搭配，已售170）
        "benchmark": 75000,        # Tokopedia toko sempam 同为翻盖式焊接护目镜在售 Rp75,000
        "weight": 150, "dim": "17x10x8",   # 翻盖双镜片结构；NANKAI同类包装16x16x7cm/0.2kg
        "comp_1688": [
            ["添新焊友烧电焊眼镜焊工墨镜防晒紫外线护目镜", 3.00, "1688 已售170"],
            ["多功能防护眼镜翻上变透明眼镜电焊护目", 3.50, "1688精选 已售555"],
            ["添新焊友翻盖电焊眼镜护目镜强弧防强光", 4.85, "1688精选 已售16,972"],
            ["厂家直销电焊眼镜防强光打磨切割飞溅氩弧焊", 5.50, "1688 同款 已售43"],
            ["电焊眼镜批发防护眼镜焊工专用绑带", 0.59, "1688 已售6,261(非翻盖款)"],
        ],
        "comp_plat": [
            ["Kaca Mata Las Buka Tutup Iwara Welding Goggle Flip (Tokopedia toko sempam)", 75000, "tokopedia.com 在售"],
            ["Krisbow Goggle Welding Kacamata Pengaman Las (Tokopedia Wero Wero)", 69000, "tokopedia.com"],
            ["Kacamata Safety Goggle Kaca Mata Las (Tokopedia ibdahcel)", 64000, "tokopedia.com"],
            ["Baru Kacamata Kerja/Kacamata UV Safety/Kacamata Las Gerinda (Shopee 本品)", 24975, "shopee.co.id 券后价"],
            ["NANKAI Kacamata Sporty Hitam 护目镜(官方目录价)", 19950, "nankai.co.id"],
        ],
    },
    {
        "key": "p4", "ext": "jpg",
        "name": "POP IT Mini Elektronik 迷你掌上游戏机卡通图案打地鼠电子发光解压玩具小钥匙扣",
        "purchase": 1.95,          # 1688 跨境同款最低（已售13,979/24,497）
        "benchmark": 44975,        # Tokopedia Ciptawarna2000 同规格mini款在售 Rp44,975
        "weight": 37, "dim": "11x8x3",     # Tokopedia STALLION 实标 Berat ±37gram，含AG13电池
        "comp_1688": [
            ["热卖儿童打地鼠跨境迷你掌上游戏机卡通图案", 1.95, "1688 已售24,497"],
            ["热卖儿童打地鼠跨培迷你掌上游戏机发光解压", 1.95, "1688 已售13,979"],
            ["创意打地鼠游戏机儿童益智玩具", 2.10, "1688精选 已售2,243"],
            ["掌上打地鼠儿童游戏机练手速不伤眼发光解压", 2.53, "1688精选 已售1,329"],
            ["打地鼠玩具挂件儿童迷你益智小玩具解压按钮", 2.20, "1688 已售41,557"],
        ],
        "comp_plat": [
            ["Mainan Pop It Elektronik Push Bubble Fidget Game Anak DBS (Tokopedia Ciptawarna)", 44975, "tokopedia.com 在售"],
            ["Game pop it elektrik mini elektrik pop push mainan (Tokopedia joysource)", 42500, "tokopedia.com 在售"],
            ["Pop it Electric Viral Mainan Anak Fidget Game Machine (Tokopedia Bang Belie)", 39900, "tokopedia.com 1rb+terjual"],
            ["SANRIO Pop It Electric Fast Push Game Gen3 (Tokopedia fsstore)", 36900, "tokopedia.com"],
            ["Mainan Pop It Elektrik Mini Gantungan Fidget (Tokopedia Zanta.id)", 21550, "tokopedia.com"],
            ["Mainan Game Anak Pop It Elektrik Fast Push Puzzle (Tokopedia gamarsnew 异规格大号)", 80000, "tokopedia.com 仅参考"],
        ],
    },
    {
        "key": "p5", "ext": "jpg",
        "name": "Taplak Kulkas Cover 家具冰箱巾顶盖布防灰尘布防尘罩滚筒洗衣机罩微波炉单开门冰箱罩",
        "purchase": 1.30,          # 1688 PEVA冰箱盖布最低（已售75,799）
        "benchmark": 15833,        # FastMoss TikTok 同款PEVA带兜通用款在售最高 Rp15,833
        "weight": 100, "dim": "25x18x3",   # PEVA布折叠装袋；mrgrosir实标100g/130x55cm展开
        "comp_1688": [
            ["家用PEVA冰箱盖布防尘罩冰箱收纳袋单双开门", 1.30, "1688 已售75,799"],
            ["冰箱防尘罩防尘单双开门收纳简约洗衣机盖布", 1.50, "1688精选 已售11,062"],
            ["冰箱洗衣机家用防尘罩时尚印花多功能收纳袋", 1.58, "1688 已售46,269"],
            ["现代简约PEVA微波炉防尘罩家用电烤箱防油盖布", 2.23, "1688精选 已售91,392"],
            ["翻盖洗衣机罩波轮滚筒洗衣机防尘罩通用防晒", 5.23, "1688精选 已售184,889"],
        ],
        "comp_plat": [
            ["[BISA COD] Taplak Kulkas Penutup Kulkas Cover Anti Air PEVA (TikTok/FastMoss)", 15833, "fastmoss.com 在售高位"],
            ["Taplak Kulkas Mesin Cuci Penutup Sarung Cover (Tokopedia Sejati Official)", 14000, "tokopedia.com 129x54cm"],
            ["Taplak Kulkas Sarung Cover Anti Air (Distroop)", 13500, "distroop.com"],
            ["1234 OS Taplak Kulkas Cover Penutup Kulkas (Shopee 本品)", 9750, "shopee.co.id 在售"],
            ["Taplak Kulkas Waterproof Sarung Kulkas (mrgrosir)", 7600, "mrgrosir.com 130x55cm"],
            ["Taplak Cover Kulkas Besar 2 Pintu 180x60cm (Tokopedia MRR10STORE 异规格)", 33999, "tokopedia.com 仅参考"],
        ],
    },
]

for p in PRODUCTS:
    k, name = p["key"], p["name"]
    js_path = os.path.join(IMG, f"{k}_search.json")
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump({"competitors_1688": p["comp_1688"], "competitors_platform": p["comp_plat"]},
                  f, ensure_ascii=False, indent=2)
    out_file = os.path.join(OUT, f"{name}_shopee-id_选品分析报表.xlsx")
    cmd = [PY, SCRIPT,
           "--product", os.path.join(IMG, f"{k}_product.{p['ext']}"),
           "--purchase-image", os.path.join(IMG, f"{k}_purchase.png"),
           "--platform-image", os.path.join(IMG, f"{k}_platform.{ 'jpg' if k=='p1' else 'png'}"),
           "--product-name", name,
           "--platform", "shopee-id", "--logistics", "sea",
           "--purchase", str(p["purchase"]), "--benchmark", str(p["benchmark"]),
           "--weight", str(p["weight"]), "--dim", p["dim"],
           "--search-json", js_path, "--allow-cert", "--force", "--output", out_file]
    print(f"\n{'='*70}\nRunning: {name[:52]}\n  purchase={p['purchase']} benchmark={p['benchmark']} weight={p['weight']}g dim={p['dim']}")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print("  STDOUT:", (r.stdout or "")[-260:])
    if r.stderr:
        errs = [l for l in r.stderr.splitlines() if l and "Warning" not in l]
        if errs: print("  STDERR:", "\n".join(errs[-8:]))
    print("  RC:", r.returncode)

print("\n===== BATCH COMPLETE =====")
