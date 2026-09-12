"""styles/registry.py — 风格注册表（style_key → 模块）。

路由层与风格模块之间的唯一契约。新增风格只需在此登记，不改路由逻辑。
"""
from __future__ import annotations
from importlib import import_module

STYLE_REGISTRY = {
    "pattern_no_subject": "styles.pattern_no_subject",
    "camo_pattern": "styles.camo_pattern",
    "subject_badge_text": "styles.subject_badge_text",
    "subject_badge_textless": "styles.subject_badge_textless",
}

# 风格描述（供智慧体向用户解释）
STYLE_DESCRIPTIONS = {
    "pattern_no_subject": "无主体图案（佩斯利/头巾/底纹）：保留颜色与布局，只改小元素母题",
    "camo_pattern": "迷彩类：改湖泊色块形变、小元素物种不变、文字按原排版替换",
    "subject_badge_text": "有主体+文字：主体物种不变改角度细节、文字按原排版替换",
    "subject_badge_textless": "有主体(文字缺失)：保留点赞的主体裂变效果，补文字",
}


def get_module(style_key: str):
    if style_key not in STYLE_REGISTRY:
        raise KeyError(f"未知 style_key: {style_key}，已注册：{list(STYLE_REGISTRY)}")
    return import_module(STYLE_REGISTRY[style_key])


def list_styles() -> list[str]:
    return list(STYLE_REGISTRY.keys())
