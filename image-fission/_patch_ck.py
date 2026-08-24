import re, os, glob
ROOT = r"E:\Desktop\双接口\image-fission\venv\Lib\site-packages\comfy_kitchen"
files = glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True)
patched = 0
NEED = ("Dict", "List", "Optional", "Tuple", "Union")

def ensure_typing(s):
    if "from typing import" in s or "import typing" in s:
        # 追加缺失名（放在已有 from typing import 行后）
        for nm in NEED:
            if re.search(r"from typing import[^\n]*\b" + nm + r"\b", s) is None:
                s = re.sub(r"(from typing import[^\n]*)",
                           lambda m: m.group(1) + ("" if nm in m.group(1) else f", {nm}"),
                           s, count=1)
        return s
    line = "from typing import Dict, List, Optional, Tuple, Union  # patched for torch2.6\n"
    # 必须放在 from __future__ import annotations 之后（若有），否则语法错
    m = re.search(r"from __future__ import annotations\s*\n", s)
    if m:
        idx = m.end()
        return s[:idx] + line + s[idx:]
    return line + s

for fp in files:
    with open(fp, encoding="utf-8") as f:
        s = f.read()
    orig = s
    s = s.replace("dict[", "Dict[").replace("tuple[", "Tuple[").replace("list[", "List[")
    s = re.sub(r"([A-Za-z_][\w.]*)\s*\|\s*([A-Za-z_][\w.]*)\s*\|\s*None",
               r"Union[\1, \2, None]", s)
    s = re.sub(r"([A-Za-z_][\w.]*)\s*\|\s*None", r"Optional[\1]", s)
    if s != orig:
        s = ensure_typing(s)
        with open(fp, "w", encoding="utf-8") as f:
            f.write(s)
        patched += 1
        print("patched:", os.path.relpath(fp, ROOT))
print(f"TOTAL patched files: {patched}")
