"""fission_router.py — 大图裂变智慧体路由层。

职责（仅此而已，不含任何风格专属逻辑）：
1. 接收图片（单张或批量）
2. 决定 style_key：
   - 若 set.json 中该图已显式指派 style_key → 直接用（回归/已知图优先）
   - 否则调各风格 classify() 取最高置信度自动分类
3. 调对应风格模块的 fission() 出图
4. 返回 {image, style_key, variants, selfcheck}

升级约定：改分类逻辑只改这里；改某风格裂变只改 styles/<style_key>.py（硬规则 #11）。
"""
from __future__ import annotations
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styles import registry

SET_JSON = ROOT / "regression_set" / "set.json"


def load_set() -> dict:
    if SET_JSON.exists():
        return json.loads(SET_JSON.read_text(encoding="utf-8"))
    return {}


def _known_style_for_image(image_path: str, cfg: dict) -> str | None:
    """回归集里若该图有显式 style_key，直接返回。"""
    name = Path(image_path).name
    for it in cfg.get("images", []):
        if it.get("filename") == name or it.get("path") == image_path:
            return it.get("style_key")
    return None


def auto_classify(image_path: str) -> str:
    """对各风格跑 classify()，取最高分。分数全中性(0.5)时回退 camo_pattern 并提示人工确认。"""
    best, best_score = None, -1.0
    for key in registry.list_styles():
        try:
            mod = registry.get_module(key)
            score = mod.classify(image_path)
        except Exception as e:
            print(f"[router] classify {key} failed: {e}")
            score = 0.0
        if score > best_score:
            best, best_score = key, score
    if best_score <= 0.5:
        print(f"[router] 置信度低({best_score:.2f})，默认 {best}，建议人工确认风格")
    return best


def route_one(image_path: str, out_dir: Path, seed: int | None = None) -> dict:
    cfg = load_set()
    style_key = _known_style_for_image(image_path, cfg) or auto_classify(image_path)
    mod = registry.get_module(style_key)
    # 取该图在 set.json 里的专属配置（prompts / text_replacements / comfyui_params）
    img_cfg = _image_cfg(image_path, cfg)
    merged = {**cfg, **img_cfg}
    variants = mod.fission(image_path, out_dir, merged, seed=seed)
    return {
        "image": image_path,
        "style_key": style_key,
        "style_desc": registry.STYLE_DESCRIPTIONS.get(style_key, ""),
        "variants": [str(v) for v in variants],
        "selfcheck": mod.selfcheck_notes(),
    }


def _image_cfg(image_path: str, cfg: dict) -> dict:
    name = Path(image_path).name
    for it in cfg.get("images", []):
        if it.get("filename") == name or it.get("path") == image_path:
            return {
                "prompts": it.get("prompts", []),
                "text_replacements": it.get("text_replacements", {}),
            }
    return {"prompts": [], "text_replacements": {}}


def route_batch(image_dir: str | list, out_root: Path, seed: int | None = None) -> list[dict]:
    paths = [image_dir] if isinstance(image_dir, str) else image_dir
    results = []
    for p in paths:
        res = route_one(p, out_root / Path(p).stem, seed=seed)
        results.append(res)
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="图裂变智慧体路由：识图→选风格→出图")
    ap.add_argument("--input", required=True, help="图片路径或目录")
    ap.add_argument("--out", default="jobs/router_out", help="输出根目录")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    out_root = Path(args.out)
    if Path(args.input).is_dir():
        imgs = [str(p) for p in sorted(Path(args.input).glob("*.*"))]
    else:
        imgs = [args.input]
    results = route_batch(imgs, out_root, seed=args.seed)
    print(json.dumps(results, ensure_ascii=False, indent=2))
