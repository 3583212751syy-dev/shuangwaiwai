# RULES.md — image-fission 裂变硬规则全文（🔴 不可破坏）

> 本文件是**细则正本**，随仓库版本控制。`.workbuddy/memory/MEMORY.md` 只放索引 + 跨轮偏好。
> **任何裂变任务动手前，先 Read 本文件。** 改动规则时同步更新这里，并在 commit message 里写清。

---

## 已确认的图像 ↔ 代码映射（每图一码，🔴31；改某图只改它自己那行）

| 图 ID | 内容 | 代码名 | 文件 | 方法 |
|---|---|---|---|---|
| `6978` | 紫蝠徽章 + Didone 弧字 | `bat_badge` | `styles/subject_badge_text.py` | **档 A~B（剪影标识）**：在原图剪影基础上裂变 + Playfair 弧字 |
| `b78e60` | 军牌迷彩 + BlackOpsOne | `dogtag_camo` | `styles/camo_pattern.py` | mode3 img2img + `camo_blob_morph` 保色形变 + 文字原位替换 |
| `pinterest3` | 牛仔拼布蝶 + denim 贴布字 | `denim_butterfly` | `src/v381_p3final.py`（+`styles/subject_badge_text.py` 文字） | SDXL 三蝶原生重生 + `repair_dots` + `overlay_fringe` |
| `pinterest4` | 迷彩棕榈（矢量线稿，无字） | `camo_palm_regen` | `src/v368_p4v2.py` + `styles/palm_draw.py` | **档 A~B**：程序化重绘**同族异树形**（禁搬移原笔画，🔴34）；可传参出多版候选 |
| `pinterest6` | 鹰骷髅金属 + 尖刺标题 | `eagle_skull_canny` | `src/v390_p6canny.py` + `src/v401_p6parts.py` + `src/v403_words.py` | **档 C（自由重绘）**：整体弱 cn + 分部件通道 → 文本第二步（🔴37/41） |

---

## 总纲

- **🔴1** 出图只能走**本地 ComfyUI**；禁云端 API / ImageGen / 远程 generate。
- **🔴2 真裂变 = 原图元素的「结构级重设计」**（换内容；保物种/配色/构图/线质；幅度肉眼可辨；结果须解剖/结构合理）。
  三种❌：微旋转·仿射搬移 / 叠加原图元素 / AI 自由重画成异物·畸形。
- **🔴38 手法分档**（第24轮口谕；判据 = 该元素有没有「剪影辨识 / 品牌标识」功能）：
  - **自由型**（老鹰 / 骷髅这类装饰主体）→ **档 C**：AI **保色彩 + 保构图**，元素**整只重绘**（结构随便变，只要物种/配色/画面位置对上 + 解剖合理）。
  - **标识·剪影型**（**蝙蝠徽章**、**棕榈矢量线稿**）→ **档 A~B**：**必须在原图基础上裂变** —— 形态真变，但骨架/剪影家族特征要留到"看得出跟原图有关联"；**禁脱离原形自由重绘**。
  - ⚠️ 与 🔴34 不冲突：A~B 的"在原图基础上裂变"**仍禁搬移/拉伸原笔画**，要**重绘出同族的异形**。
  - 档位未定的图**先问用户，别猜**。
- **🔴3** 逐元素覆盖**所有**元素（小佩斯利 / 链条 / 边框 / 小字），不只大字。
- **🔴4** 同位同大同版式同配色；锁原图色族（LAB Reinhard α=0.85）。
- **🔴5** 商标原位改写，禁色块遮盖 / 纯擦除 → LaMa + PIL 重画。
- **🔴6** 禁矩形色块遮字（含羽化）；正确 = inpaint + 直接贴新字。
- **🔴7** 字体映射：military→BlackOpsOne / Didone→PlayfairDisplay-Bold / denim→LeagueSpartan-Black / metal→MetalMania。
  capH 探针 @200 → `fontsize = tgt/capH_ratio*4*0.95`，4x supersample + LANCZOS，**禁 per-char**。
- **🔴8** LaMa 唯一路线：本地 Big-LaMa（`ComfyUI/models/RMBG/Lama/big-lama.pt`）+ 傅里叶卷积，只动 mask，dilate 8。
- **🔴9** 无主体（迷彩 / 底纹）→ 抹字 + 写词 + 小元素裂变；有主体 → 保主体 + 换文字。
- **🔴10** 回归集 `regression_set/set.json`：① 13c8b7 头巾 ② b78e60 狗牌 ③ Pinterest(4) 棕榈 ④ Pinterest(6) 鹰骷髅 ⑤ Pinterest(3) 蝶 ⑥ 6978 蝙蝠。验收每张 8/10+。
- **🔴11** `styles/` 每风格独立模块；反馈只改对应风格；路由 `fission_router.py` 只选 style_key。

---

## 文字层

- **🔴12** 擦字掩膜按对比度自动检测（亮底深字 lum<110 / 暗底亮字 lum>150），主体几何圆排除；
  一次性全局掩膜 + 单次 LaMa + 统一重绘，**禁羽化融合**。弧字擦除用径向带 `|d-r|<=0.63*cap`，禁角向跨度筛。
- **🔴13** 文字重画：PIL `d.text` 的 y 是行顶非 cap 顶 → 用 `getbbox` 墨迹包围盒居中；超框用 `shrink`/`squeeze`；带外附属符号用 `extra_specks`。
- **🔴14** 字标：SDXL 宽画布 ≥1664 必拼错 → **字标一律 1024²**；满宽白标题
  `ink=lum>30 & sat<62` → closing(disk12) → 触顶最大域 → fill+dilate(disk10) → 常数底填。
- **🔴15 弧字铁律**：新字在原弧半径 + cap*1.02 内，裂变只走字距(+1~2px) / 整体角偏(+4°) / 新词长度差；**缎带弧线不换**（改半径会跨线翻转）。
- **🔴36** 风格化标题**先旋转后排版**：逐字旋转 reshape → 实测宽高 → kfill 只缩不放 → 依序铺设；±6°、垂直中心 0.42·band。
- **🔴37** **主体与文本严格分两步**：① 主体裂变完成且解剖合理 → ② 再文本裂变叠回。
- **🔴41 文本裂变 = 词库机制**（用户第25轮："我要的是把文本单词都裂变"）
  沿用 `src/text_fission.py` 的词库思路 —— **词库每个词是"另一个意思"**，逐词出一版，供直接对比。
  - 载体：`src/v403_glyphs.py`（SDXL + Harrlogos_XL_v2 生成字标，`REUSE` 复用旧 n1~n6）+ `src/v403_words.py`（合成，擦字只做一次）。
  - ⚠️ **字标必须做极性自动判定**：模型有时吐**黑字浅底**、有时**白字黑底**。按**图缘中位亮度**判：底亮 ⇒ 先反相。
    反相过的用**更严阈值** `(ga-0.55)/0.35`（用 0.42/0.30 会漏出淡灰矩形鬼影），未反相用原 `0.42/0.30`。
- **🔴42 p6 标题排版几何**：横向满幅 `TGT_W=3513` + 墨迹顶对齐 `TOP_Y=88`
  （原图 MRCHOSR 白墨 `x[12,3524] y[88,1800]`）；字标 slab 宽高比 2.06 ≈ 原图 2.05 → 纯等比即可重合。

- **🔴43 字标风格对齐原图 = 骨架 + 弱 canny + 高 LoRA**（用户第26轮："字体按照原图字体风格做相同风格设计"）
  - **症状**：LoRA 自由生成 → 风格对但 ≥2:1 宽画布上 ≥9 字母**拼写必崩**（🔴14）；纯强 canny 结构透传 →
    拼写对但风格被压平成骨架字（≈MetalMania，无尖刺）。
  - **正解**：PIL 用 **MetalMania 排好 `{WORD}`** → 取 canny 骨架 → ComfyUI `ControlNetApplyAdvanced`
    **弱引导 `cn .50 / end .50`** + Harrlogos_XL_v2 **LoRA weight 1.30** + `cfg 7.5` @ **1536×512** + seed 888。
    骨架锁字母结构（拼写 100%）；弱 canny + 高 LoRA 让模型在骨架周围"长出"尖刺/刀锋（风格到位）。
    `cn_end<1.0` 是关键（=1.0 只描一遍、无尖刺）；`cn .60-.75 / end .85` 会把风格压平。
  - **验收（对齐原图 MRCHOSR）**：`aspect≈4.49 / fill≈0.40 / stroke≈52.3`，成品高度落在原图 **926px 带内**（实测新 10 词 742–1156px）。
  - **载体**：`src/v404_vorg_style.py`（6 轮探索，`R5b_c50_e50_l13_s888` 定稿）+ `src/v405_bank_style.py`（9 词批量同规格重做）；拼版 `src/v404_sheets.py` 确定性再生成。
  - ⚠️ **脏底板 bug（必查）**：`jobs/**/_base_notitle.jpg` 若曾用默认字标 `n4(MOURNGRAVE)` 跑过 → 标题已烙进底板 →
    之后**每一版词变体都会双字重影**（第25轮 10 版全中招）。修法：`make_base()` 传**不存在的字标名**
    （`P6_WORD_SRC="__no_such_glyph__.png"`）使 `make_v329` 的 `if G.exists()` 为假、跳过贴字分支 → 干净底板。

---

## 形变层

- **🔴16** 位移场几何：双层法（擦主体→nn 重建底板→主体成层做层内形变→贴回，背景零拖动）；
  旋转切向位移 ∝ 力臂 dy0，枢轴 y 全量、下方 ~70px smoothstep 归零；权重过渡带 σ16(~60px)。
  旋转符号 `d=(M(-φ)-I)(p-pivot)`；迷彩重排**降主频比降强度有效**（st.050/fq0.90 只丢 5%）。
- **🔴17** 形变许可范围：**只许"扩张型"**（新形完全盖旧形，仿射两轴>1）；
  收缩 / 旋转 / 剪切一律腾空禁用（除腾空区落纯平滑底）。层 bbox padding ≥ max 缩放倍数。
  硬剪影化 = `alpha>0.5` + 0.8σ 回抗齿。唯一安全不对称 = 单侧额外径向扩张。
- **🔴18** 程序化对象线宽按"对象尺寸比例"给，禁写死。
- **🔴25 三层法**（多部件形变定式）：细长跨部件的白线元素（闪电射线 / 细弧线）**必须单列成层做刚体变换** ——
  混进"单位分解"权重场会被权重边界**剪断成碎段**。分层严格照原图 z 序：`底 → 射线 → 主体 → 标题`。
  **单位分解（partition of unity）** 是治"多部件形变互掐 / pinch"的正解：各部件平台核权重 `w_i = 1-smoothstep(plate)` → **归一化 Σw=1** → 再合成位移场。
  **背景纯黑 ⇒ 形变无需 inpaint**（腾空区自动变黑）；背景非纯色（p4 迷彩）才需 EDT 清底。

---

## 重生层（SDXL）

- **🔴19** 重生管线：主体掩膜 → 裁 bbox+margin → ComfyUI 原生 SDXL inpaint（VAEEncode + SetLatentNoiseMask + KSampler，dpmpp_2m/karras）
  → 仅掩膜内贴回，**掩膜外原图零改动**。
- **🔴20** **禁 IPAdapter style transfer**（把原主体外观一起迁移 → 内容锁死 + 泄出原剪影幽灵）。
  靠 prompt 锁色 + 后期 LAB Reinhard（`_match_region` α=0.9）。
- **🔴21** **重生定式 = ControlNet canny 结构引导**：无结构约束 + 高 denoise → 自由发挥崩形。
  `Canny(0.35/0.75)` → `ControlNetApplyAdvanced(strength,0,end_percent)`，**cn_end<1.0 是关键**（=1.0 只描一遍、变化≈0）。
  负提示必压色漂（加 pink/magenta/neon/glow/white）。⚠️ depth 预处理器缺 ckpt 会超时，别用。
- **🔴22** **重生分辨率不得低于原图尺寸**（降采样 = 线稿碎化）。p4 判墨迹必须 `lum<30`
  （校准：<25→22.04% / <30→22.75% / <35→24.38% / <40→41.0%；`lum<60` 会把迷彩底算进墨）。
- **🔴23** 密度匹配阈值优于 Otsu：令掩膜内落墨数 == 原墨迹数 → `t=percentile(lg[sel], 100*ink0/sel.sum())` 夹 [20,200]。（Otsu 会把新剪影切厚 1.5x）
- **🔴24** 成品 = 重生接进管线：`make_v329.py` 的 `REBIRTH_FILES` + `_rebirth_path(iid)`（`SUBJECT_REBIRTH=0` 回退 PIL）。交付必须**整图**，禁主体裁片。
- **🔴26** 重生必须"换内容"，不能只跟骨架描一遍。
- **🔴27** 局部色相漂移治不了（LAB Reinhard 只对齐整体均值/方差）→ 只能靠 **prompt 锁色**：颜色写进 prompt + NEG 删与之打架的词。
- **🔴29 + 🔴30 空洞 / 细碎元素不得焊进主体掩膜**：症状 = 轨迹小圆点被重生整条吃掉（变成发白圆环 / 灰色空心圆）。
  根因：点缘间隙 ~11px < `binary_closing(disk 9)` 桥接能力(~18px) → 整条轨迹焊进主体连通域。
  **三件套（缺一不可）**：
  1. 闭运算**前**剥掉孤立小圆斑 `trail_dots`（面积 20~260 + bbox 4~20px + 圆度>0.45）；
  2. 回贴**后** `protect_full` 把受保护像素**从原图强制还原**（回贴环带是无条件贴的，会压到邻近元素）；
  3. 检不出时（已被焊进主体）用 `repair_dots`（原图厚暗斑 + bbox≤24px + 新图被提亮 >18 处还原原像素）。
  **通则：主体掩膜 = 仅该主体本体；任何"邻近但独立"的元素（轨迹点 / 编号 / 小符号）都必须进 protect 掩膜并被强制还原。**
- **🔴32** 量化前先剔 JPEG 噪点（连通块数统计前，先在墨掩膜里剔 <12px 分量）。
- **🔴33** 放松 canny 后必须用 prompt 写死部位归属：
  POS「white head above the skull / beak attached to the eagle's face / pitch-black nasal cavity」
  + NEG「yellow nose / nasal cavity / teeth, beak on the skull, second beak」。
- **🔴34** **p4 严禁"叠加 / 搬移原图元素"充裂变**：整树仿射换位、整组原笔画刚体变换（v395）**同样算叠加**
  → 必须**重绘新树形**，保矢量平涂风格 + 墨量对得上（原 22.63%）。
- **🔴35 p6 主体验收 = 解剖合理**：肉眼须见 ① 白头 + 黄喙清晰 ② 喙长在鹰脸上 ③ 黄爪抓在颅骨上 ④ 无悬空色块 ⑤ 翼羽连贯。缺一不合格。
- **🔴39 多部件通道**（v401 定式，载体 `src/v401_p6parts.py`）：
  整幅一把 canny 强度时各部件对自由度要求**相反** → 必须按部件分通道，各自从**原图**取 canny 结构 + 独立 cn/dn/seed，
  再按软 alpha 依次贴回（骷髅 → 左角 → 右角 → 鹰头，后贴者赢重叠）。
  p6 实测分工：整体 `cn.12/end.35/dn.98@2560`（翼/身真换形）→ 骷髅 `.30/.55/.78`（治"颅盖被画成波浪毛发"）→ 双角 `.34/.60/.75`（保平涂分节）→ 鹰头 `.32/.50/.88`（守解剖）。
  骷髅掩膜：被内部黑线稿切碎 → **组件规则**（area≥40000 + bbox 完整落在 x[900,2700] + y_min>2450 → 排除触边的翅膀白羽），
  并集后 close(13)+fill_holes；**不要**再取 biggest（会丢下颌 / 牙列）。

---

## 工具位置速查

- 形变 `styles/subject_morph.py`
- 重生 `src/v342_rebirth.py`（`rebirth_subject` / `mask_p6`，支持 `protect=` / `wide=`）·`v377_p6rb.py`(`snap_p6`)·`v390_p6canny.py`(CLI)·`v401_p6parts.py`(🔴39)·`v397_p6head.py`(头通道/`head_mask`)·`v384_p6beak.py`(锁色)
- p3 `src/v378_p3rb.py` / `v381_p3final.py` / `v380_p3fringe.py`
- p4 `src/v368_p4v2.py` + `styles/palm_draw.py`
- 文本 `src/text_fission.py`(词库原件) · `src/v403_glyphs.py` · `src/v403_words.py`
- 管线 `make_v329.py` · `build_v373.py` · `qc_v341*.py`
- 历史 `history/build_history.py`（🔴40）
- **EDT 精确清底**（优于 LaMa/gaussian）：`idx=distance_transform_edt(masked,return_indices=True); out[masked]=a[idx][masked]` — 原色·硬边·零模糊。

## v402 / v403 定稿参数

- p6 主体：`v390_p6canny --cn-strength 0.12 --cn-end 0.35 --denoise 0.98 --max-side 2560 --seed 888`（A 通道）
  → `v401_p6parts`：骷髅 .30/.55/.78 s4242 · 左角 .34/.60/.75 s5151 · 右角 同 s6161 · 鹰头 .32/.50/.88 s888。
- p6 文本：词库 10 词（RAVEN / VORGRAVEN / MOURNGRAVE / SKARVALD / IRONVEIL / GRAVETIDE / STORMHELM / ASHREAVER / DUSKBANE / THORNMOURN），排版按 🔴42。
- p4：`v368_p4v2.py --tag X --dilate --tilt --crown-lo/hi --size-scale --seed --rseed --solid/chevron/scribble/tuft --out-dir <dir>`。
