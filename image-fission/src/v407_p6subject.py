# -*- coding: utf-8 -*-
"""v407_p6subject.py —— p6 主体「跟原图拉开」的种子/姿态扫描（第 27 轮）

用户第 27 轮原话：
  「主体老鹰跟骷髅头的裂变需要跟原图拉开」

诊断
----
现状主体 = v402 step1（A 通道 cn .20/end .55/dn .98 seed 888 + B 通道鹰头）。
**姿态/构图与原图几乎一致**（同翼展、同头位、同颅角），用户读成"没拉开"。
本轮试过把 cn 压到 .15 / dn 1.0 → 变得多但**解剖崩**（s1234 鹰头糊成黑块、
s2024/s42 颅骨上长黄斑 = 🔴35 红线）。

做法（保解剖、换姿态）
--------------------
① 引导力度回到**用户已认可的 v393 档**（cn .20 / end .55 / dn .98）；
② 差异改由 **prompt 显式要求换姿态**（翼展更宽/更收、头转向另一侧、颅骨倾角变化）
   + **换种子** 来逼出结构级变化；
③ 扫多个种子，人工目检：鹰头清晰可辨 + 喙在鹰脸上 + 颅骨无黄渍 + 角成形。

产出：jobs/v407_subject2/p6_<tag>_snap.jpg
用法：python src/v407_p6subject.py --seeds 91,1234,2024,42
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from v342_rebirth import mask_p6, rebirth_subject          # noqa: E402
from v377_p6rb import snap_p6                              # noqa: E402
from v390_p6canny import CN, CKPT                          # noqa: E402

SRC = Path("E:/Desktop/图裂变测试图/Pinterest (6).jpg")
OUT = ROOT / "jobs" / "v407_subject2"
OUT.mkdir(parents=True, exist_ok=True)

CN_S, CN_E, DN, MS = 0.20, 0.55, 0.98, 2048

# 在 v390 已认可 prompt 基础上，**加强"换姿态"的语言**（这是本轮唯一的差异驱动）
POS = ("bald eagle with spread wings perched on top of a horned demon skull, "
       "completely new wing pose, wings raised much higher and swept back at a different angle, "
       "different wingspan clearly wider or narrower than the reference, "
       "feather layers rearranged in new overlapping rows, very dark chocolate brown wing feathers, "
       "white feathered eagle head raised high above the skull, "
       "head turned to the other side at a new angle, beak pointing a different direction, "
       "bright golden-yellow hooked beak attached to the eagle's face, fierce visible eye, "
       "redesigned skull tilted at a new angle, "
       "clean white bone skull with even fine cross-hatch shading, "
       "empty hollow pitch-black eye sockets and pitch-black nasal cavity, "
       "different curved ribbed horns, "
       "flat vector illustration, bold screen print, hard clean edges, crisp linework, "
       "solid flat colors, tan horns, pure black background, white lightning bolts, "
       "centered composition, high contrast, no text, no letters")
NEG = ("yellow patches, yellow stains, yellow spots on skull, yellow feathers on skull, "
       "yellow nose, yellow nasal cavity, yellow teeth, beak on the skull, second beak, "
       "dirty bone, dark stains, smudges, blotches, mud, grime, "
       "glowing eyes, luminous eyes, eye light, text, letters, words, watermark, "
       "blurry, low quality, photo, realistic, 3d render, airbrush, painterly, "
       "soft gradients, smooth shading, extra heads, extra skulls, extra birds, "
       "deformed, asymmetric, gray haze, fog, noise, speckle, dirty background, "
       "halo, glow, pale beak, white beak")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="91,1234,2024,42")
    ap.add_argument("--ms", type=int, default=MS)
    ap.add_argument("--pos", default="v407", choices=("v390", "v407"),
                    help="v390=用户已认可的原 prompt（隔离种子变量）；v407=加强换姿态语言")
    a = ap.parse_args()
    pos = POS
    if a.pos == "v390":
        from v390_p6canny import POS as POS390          # noqa: PLC0415
        pos = POS390
    src = Image.open(SRC).convert("RGB")
    mask = mask_p6(src)
    for s in [int(x) for x in a.seeds.split(",") if x.strip()]:
        t = time.time()
        tag = f"{a.pos}_s{s}_cn{CN_S}"
        full = rebirth_subject(src, mask, pos, NEG, ckpt=CKPT, denoise=DN,
                               ipa_weight=0.0, color_match=0.0, cn_name=CN,
                               cn_strength=CN_S, cn_pre="canny", cn_end=CN_E,
                               margin=60, grow=10, seed=s, tag=tag, max_side=a.ms)
        snap = snap_p6(src, full, mask)
        dst = OUT / f"p6_{tag}_snap.jpg"
        snap.save(str(dst), quality=93)
        print(f"[subj] pos={a.pos} seed={s} cn={CN_S}/{CN_E} dn={DN} ms={a.ms} → {dst.name} "
              f"({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
