#!/usr/bin/env python
"""快速测试 LaMa + 全带 mask 在 b78e60 上的效果。"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'styles'))

from PIL import Image
from styles import base

cfg = json.load(open('regression_set/set.json'))
for it in cfg['images']:
    if it['id'] != 'b78e60':
        continue
    img = Image.open(it['path']).convert('RGB')
    plan = it.get('text_plan', [])
    out = base.replace_text_plan_v2(img, plan, use_material=True)
    out_dir = 'jobs/regression_v325_upgrade/text_only/b78e60'
    os.makedirs(out_dir, exist_ok=True)
    out.save(f'{out_dir}/01_text_only.jpg', quality=92)
    print(f"saved -> {out_dir}/01_text_only.jpg")
    break
