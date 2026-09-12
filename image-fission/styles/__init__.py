"""image-fission 风格化裂变模块包。

每个风格一个独立模块（pattern_no_subject / camo_pattern / subject_badge_text /
subject_badge_textless），各自持有该风格的 ComfyUI 参数、文字替换策略、prompt 模板、
专属后处理。升级只改对应模块，绝不"一改全改"（硬规则 #11）。

路由层 `fission_router.py` 只负责识图→选 style_key→调对应模块，不含任何风格逻辑。
"""

__all__ = [
    "base",
    "registry",
    "pattern_no_subject",
    "camo_pattern",
    "subject_badge_text",
    "subject_badge_textless",
]
