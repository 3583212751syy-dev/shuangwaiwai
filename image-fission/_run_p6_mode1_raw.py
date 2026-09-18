"""_run_p6_mode1_raw.py — p6 canonical mode1 主体源（**不叠文字** + 背景压回黑底）。

为什么不用 subject_badge_textless.fission() 的现成输出：它会调 _add_text() 把
set.json 的 RAVEN 画上去，而 do_pinterest6() 之后还要擦标题+叠 VORTCRAVEN →
两套文字打架（RAVEN 残留在 VORTCRAVEN 上方，实测可见）。故这里直接跑 src/fission.py
mode1（= 同一条用户点赞路径），只保留：① LAB 色锁 ② 背景压黑。

背景压黑：mode1 把原本纯黑底生成成了"布纹灰底"。判据 = 低饱和(sat<45) + 中灰
(45<lum<215) + **与画框连通的连通域**（骷髅是内部区域，被黑墨/白射线与画框隔开 → 免疫）。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from styles import base  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

SRC = "E:/Desktop/图裂变测试图/Pinterest (6).jpg"
OUT = ROOT / "jobs" / "v386b_p6mode1raw"
OUT.mkdir(parents=True, exist_ok=True)

PROMPT = ("black metal poster, bald eagle perched on horned skull, white lightning bolts, "
          "brown white black palette, spiked gothic typography, no real band name")

args = [
    "--input", SRC, "--mode", "mode1", "--prompts", PROMPT,
    "--count", "2", "--out", str(OUT), "--seed", "8888", "--steps", "28", "--cfg", "5.0",
    "--width", "880", "--height", "1240",
    "--color-strength", "0.60", "--composition-strength", "0.55",
    "--ipadapter-noise", "0.10",
    "--controlnet-strength", "0.50", "--controlnet-end", "0.90",
]
base.run_fission_cli(args)

ref = base.Image.open(SRC).convert("RGB")
W, H = ref.size

outs = [p for p in sorted(OUT.rglob("*.jpg")) if not p.name.startswith("_")]
print("[raw] mode1 outputs:", [p.name for p in outs])


def blacken_bg(img):
    a = np.asarray(img.convert("RGB"), np.float32)
    mx = a.max(2); mn = a.min(2)
    sat = mx - mn
    lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
    cand = (sat < 45.0) & (lum > 45.0) & (lum < 215.0)
    # 只保留与画框连通的候选域（= 背景布纹），内部连不通的（骷髅/主体）自动免疫
    lab, n = ndi.label(cand, np.ones((3, 3), bool))
    if n:
        border = set(np.unique(np.concatenate([
            lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])))
        border.discard(0)
        keep = np.isin(lab, list(border))
        keep = ndi.binary_dilation(keep, ndi.generate_binary_structure(2, 2), iterations=2)
        keep = ndi.binary_closing(keep, np.ones((9, 9), bool))
        a[keep] = 0.0
        print(f"[raw] bg blackened {100*keep.mean():.1f}% of frame")
    return base.Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


for op in outs:
    gen = base.Image.open(op).convert("RGB").resize((W, H), base.Image.LANCZOS)
    locked = base.lab_color_lock(gen, ref, alpha=0.85)
    locked = blacken_bg(locked)
    dst = OUT / ("clean_" + op.name)
    locked.save(str(dst), quality=95)
    print("[raw] saved", dst)
