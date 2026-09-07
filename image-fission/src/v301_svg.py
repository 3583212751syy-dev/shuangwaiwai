"""v301 配套：3 张 BACARDÍ 经典版式 SVG 字体模板（可缩放无损）
用户可在 Illustrator / Inkscape / PS 自由改字、改色、改位置。
引用真实字体路径，渲染引擎会用本机已安装的 Abril Fatface。"""
import os

OUT_DIR = r"E:\Desktop\双接口\image-fission\jobs\v301"
os.makedirs(OUT_DIR, exist_ok=True)

# BACARDÍ 经典版式锚点（与 v301 主图同坐标系，画布 1552 x 2000）
# SVG 用绝对坐标 <text> 排版 + textPath 沿弧线实现顶弧绕环
variants = {
    "up":     ("SHADOW OF THE WING",  "NOCTAVEN",  "DISTILLERY"),
    "spread": ("WINGS OF TWILIGHT",   "DUSKBAT",   "RESERVE"),
    "fold":   ("GUARDIAN OF THE DARK","MOONBAT",   "NOCTURNE"),
}

FONT_FAMILY = "Abril Fatface, 'Abril Fatface Regular', serif"

for name, (arc, brand, sub) in variants.items():
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1552 2000" width="1552" height="2000">
  <!-- 背景透明：叠加到任何底图上都自然显示 -->
  <defs>
    <!-- 顶弧路径：以 (776, 746) 为圆心、半径 430，从 -125° 到 -55° 的弧 -->
    <path id="topArc_{name}"
          d="M {776 - 430*0.8192:.2f} {746 - 430*0.5736:.2f}
             A 430 430 0 0 1 {776 + 430*0.8192:.2f} {746 - 430*0.5736:.2f}"
          fill="none"/>
  </defs>

  <!-- 顶弧绕环 -->
  <text font-family="{FONT_FAMILY}" font-size="78" fill="#000" letter-spacing="6">
    <textPath href="#topArc_{name}" startOffset="50%" text-anchor="middle">{arc}</textPath>
  </text>

  <!-- EST. 1862 行 -->
  <text x="432" y="760" font-family="{FONT_FAMILY}" font-size="58" fill="#000">EST.</text>
  <text x="1120" y="760" font-family="{FONT_FAMILY}" font-size="58" fill="#000">1862</text>

  <!-- 品牌大写（中央） -->
  <text x="776" y="1240" font-family="{FONT_FAMILY}" font-size="260" fill="#000"
        text-anchor="middle" font-weight="900" letter-spacing="2">{brand}</text>

  <!-- 副字 -->
  <text x="776" y="1390" font-family="{FONT_FAMILY}" font-size="130" fill="#000"
        text-anchor="middle" font-weight="900" letter-spacing="3">{sub}</text>

  <!-- 三角 ▼ -->
  <polygon points="748,1480 804,1480 776,1520" fill="#000"/>

  <!-- 文字层使用说明（不影响最终合成；导出去掉这行注释） -->
</svg>
'''
    p = os.path.join(OUT_DIR, f"v301_{name}_text.svg")
    with open(p, 'w', encoding='utf-8') as f:
        f.write(svg)
    print(f"  {p}")

print("\nSVG 模板就绪。\n")
print("字体 fallback 说明：")
print("  SVG 内 font-family='Abril Fatface'。如果系统没装，会 fallback 到 serif。")
print("  本机真字体：E:\\Desktop\\双接口\\image-fission\\ComfyUI\\models\\fonts\\AbrilFatface-Regular.ttf")
print("  PS / AI 里可以 Load Font 直接加载使用。")
