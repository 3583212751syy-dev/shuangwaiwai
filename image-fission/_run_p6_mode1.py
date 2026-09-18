"""_run_p6_mode1.py — 跑 pinterest6 的**正规路径** subject_badge_textless.fission()（mode1）。

用途：v373 把 p6 主体源换成了本地 inpaint 重生(=原图近似复制)，用户否决
("这点裂变跟原图的区别是什么")。本脚本恢复 canonical 代码路径：
  subject_badge_textless.fission()  →  src/fission.py --mode mode1 (IPAdapter 双锁 + Canny 0.5)
产出即"用户明确点赞"的整幅主体裂变，随后由 make_v329.do_pinterest6() 叠标题。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from styles import subject_badge_textless as sbt  # noqa: E402

CFG = json.loads((ROOT / "regression_set" / "set.json").read_text(encoding="utf-8"))
OUT = ROOT / "jobs" / "v386_p6mode1"
OUT.mkdir(parents=True, exist_ok=True)

paths = sbt.fission("E:/Desktop/图裂变测试图/Pinterest (6).jpg", OUT, CFG, seed=None)
print("[p6 mode1] variants:")
for p in paths:
    print("   ", p)
