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
    （该模式下 `make_v329` 末尾日志会引用未定义的 `tw` → 必须先把 `tw=th=cx0=cy_place=0` 初始化。）

- **🔴44 新文本必须按"原图排版位置 + 大概形状构图"**（用户第27轮："不允许换位置生成，也不允许用色块背景遮盖原图内容"）
  - **量测口径**：原图 MRCHOSR 在**左区 x<1150**（避开鹰头）量「密集字身带」（行墨量 > 25% 峰值）：
    `y[337,1313] h≈976`，墨迹全幅 `x[12,3524]`。旧 v405 只有 `y[190,734] h=544` 且 y>800 无墨 →
    用户读成「一排整齐小字浮在上方」。
  - **落位铁律**：按**密集字身带顶 = y359** 对齐（不是"墨迹顶 y88"—— 那只管住最上面一根尖刺，字身落哪不管）。
  - **骨架必须自适应纵拉**：MetalMania 字母宽窄差异极大（I/L 窄、M/O 宽）→ 同样"排满 94% 宽"，
    含窄字母的词要用更大字号 → 排出来更高。**固定纵拉倍数会失控**（实测 IRONVEIL 骨架 1440×685
    超出画布被裁、RAVEN 成品 dh 1290、THORNMOURN 只 832）。
    正解 = 排字后按**墨迹高宽比归一到 0.358**（由 VORGRAVEN 定稿实测反推）→ 10 词成品 dh 收在 951~1023（原图 976）。
  - **背景洁净度判据（必查）**：字标必须"白字纯黑底"。判据 `中灰(0.28~0.72)占比 <12% 且 白墨(>0.75)占比 >10%`。
    实测 RAVEN/IRONVEIL 有种子吐"灰底荆棘纹理"（**白墨 0%**）却 `placed_dh` 很准（975/1001）→
    只看高度会选中废图，合成后字母发灰/消失。选 seed 时该判据必须参与打分。
  - 载体：`src/v407_probe_geo.py`(几何量测) + `src/v407_p6_title.py`(骨架+生成) + `src/v407_compose.py`(落位合成) + `src/v407_deliver.py`(整套交付)。
  - ⚠️ 擦字仍走 **LaMa**，**禁色块遮盖**（🔴5/🔴6）；合成只用白色墨迹 alpha 叠加。

- **🔴45 主体"拉开"≠压引导**（用户第27轮）⚠️ **第28轮已升级，见 🔴46**
  - 单纯把 canny 压到 `cn .15 / dn 1.0` → **解剖必崩**：鹰头糊成黑块、颅骨长黄斑（🔴35 红线）。
  - 第27轮的权宜解 = 引导力度回到已认可档 `cn .20 / end .55 / dn .98 @2048`、靠**换种子**逼变化；
    选种判据 = 鹰头清晰可辨 + 黄喙在鹰脸上 + 颅骨无黄渍（黄像素占比 ≤ 原图 2.89%）。
    实测 seed 2024 入选（黄 1.62%）。**但换种子只能改笔触/细节，改不了姿态**（用户第28轮仍说"没拉开"）。
  - 载体：`src/v407_p6subject.py`（`--pos v390` 隔离种子变量）；底板用 `P6_REBIRTH=<成品>` 环境变量切换主体，不必改 `REBIRTH_FILES`。

- **🔴46 高自由度图（🔴38 档 C）要"变化大"→ 用 v147 逐元素区域控制 + hires 两遍**（用户第28轮原话：
  「这种自由度高的图片就裂变变化大一点，参考本机 v147 的风格」）
  - **v147 是什么**（本机历史基线，别重新发明）：`src/smoke_v146_region_tile.py`（commit `8f8b78b`/`8285e76`
    的"真裂变基线"）= **每个元素一条独立提示词**（`RegionalListCombine` + `ConditioningSetAreaPercentage`），
    提示词里**显式要求换朝向/换姿态**（老鹰"正对镜头 NOT 侧俯冲"、骷髅"3/4 而非正面"、火焰改斜升、铁链改三段沿边），
    再配 Tile 0.60 + Canny 0.25 + denoise 0.80，每条区域 prompt 拼 `COHESIVE`，全局 NEG 拦
    `elements touching / fused elements / melted edges`。
  - ⚠️ **单遍在 3.9MP 上换不动姿态**（第28轮实测）：把 3312×3561 裁块降到 2048 长边 = **4× SDXL 原生分辨率**，
    Tile/Canny 的"保外观"作用被放大到压过提示词 → 剪影 IoU 0.88、鹰头仍侧脸。
  - **定式 = hires 两遍**：① 第一遍在 **~1.10MP 近原生**画布跑区域提示词（`--pass1-mp 1.10 --pass1-dn 0.92`，
    Tile 0.25 / Canny 0.15 end0.40）→ 模型才有自由度换姿态；② `LatentUpscale` 到目标分辨率后走 **denoise≈0.45**
    的细节遍（Tile 0.50 / Canny 0.12），**参考图必须来自第一遍解码结果（自引导）** ——
    拿原图当 Tile 参考会把姿态拽回去，等于白做。
  - **解剖靠"守护区"而不是靠压 denoise**：把最容易崩的小部件单列成高 strength 区域 ——
    鹰头 `strength 1.55`（写死"喙必须长在脸正面、无第二只喙、双眼可见、无熔化"）、
    鹰爪 `1.45`（"五爪分明、爪落在颅骨上、无悬空黄块"）。实测：不做守护区时去掉 Tile（E/F 档）
    IoU 能到 0.68~0.70 但**鹰喙熔成一团**；做守护区后 G7 档 IoU 0.82 且鹰头成形。
  - **实测定稿**（v408，seed 888）：IoU 0.733 / 结构差异 0.817 / 边缘密度 7.13（原图 6.45）/ 黄像素 0.58%（原图 0.95%）。
    变化是**姿态级**的：鹰头侧脸 → 正对镜头、翼展与羽层重排、颅骨下颌重画、射线重排。
  - **底色保护**：纯黑背景图必须做 `_bg_protect`（原图为近黑处，除"新画出的成片强亮白射线"外一律还原原图）
    → 杀掉"黑底留灰雾/灰块"这一 p6 历史顽疾。
  - ⚙️ **ComfyUI 细节（踩过坑）**：`RegionalListCombine` 的 `INPUT_TYPES` 只声明 `region2..region8`，
    但实现是 `run(self, global_cond, region1, **kwargs)`；ComfyUI 对**未声明输入仍按节点 id 解析链接**，
    所以 `region1` 键**确实生效**（已读 `comfy_execution/graph.py` + `execution.py:159-227` 确认），
    最多可挂 8 个区域。另：`build_wf*()` 这类构建函数务必留 `**ignored`，否则不同档位的 CLI 参数互传会 TypeError。
  - 载体：`src/v408_p6regional.py`（构建+两遍+保护）、`src/v408_eval.py`（量化+拼版）、`src/v407_deliver.py --desk ... --tag ...`（整套交付）。
    产物是**含原图标题带的整图**中间件，喂 `P6_REBIRTH` 给 `make_v329` 走 v407 文本通道（🔴37 主体/文本分两步）。

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
