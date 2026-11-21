"""Pure validation: importing this module never reads or controls desktop windows."""
import re
from datetime import date
from decimal import Decimal, InvalidOperation

MONTHS = {name: i + 1 for i, name in enumerate(
    ('ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic'))}


def validate_code(value):
    code = str(value).strip().upper()
    if not re.fullmatch(r'[0-9A-F]{8}', code):
        raise ValueError('退货码必须是 8 位十六进制字符（0-9、A-F）')
    return code


def validate_data(data):
    if not isinstance(data, dict):
        raise ValueError('广告数据必须包含营业额、广告费和点击')
    result = {}
    for key in ('营业额', '广告费', '点击'):
        text = str(data.get(key, '')).strip()
        if not re.fullmatch(r'\d+(?:\.\d+)?', text):
            raise ValueError(f'{key} 必须是非负有限数字，不含千分位逗号')
        try:
            number = Decimal(text)
        except InvalidOperation as ex:
            raise ValueError(f'{key} 无效') from ex
        if not number.is_finite() or (key == '广告费' and number > 800):
            raise ValueError('广告费必须在 0–800 范围内')
        if key == '点击' and number != number.to_integral_value():
            raise ValueError('点击必须是整数')
        result[key] = text
    return result


def reusable_pending(pending, today, target):
    if not isinstance(pending, dict) or pending.get('date') != today or pending.get('target_date') != target:
        return False
    try:
        validate_code(pending.get('code', ''))
        validate_data(pending.get('data'))
        return True
    except ValueError:
        return False


def parse_sheet_date(value, target_year):
    """Accept explicit dates only. Month/day labels use the target year, never the row number."""
    text = str(value or '').strip()
    if text.startswith("'"):
        text = text[1:].strip()
    patterns = (
        (r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', 'ymd'),
        (r'(?:(\d{4})年)?(\d{1,2})月(\d{1,2})日', 'ymd'),
        (r'(\d{1,2})[-/](\d{1,2})', 'md'),
        (r'(\d{1,2})\s+(ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)\.?\s+(\d{4})', 'es'),
    )
    for pattern, kind in patterns:
        match = re.fullmatch(pattern, text, re.I)
        if not match:
            continue
        groups = match.groups()
        if kind == 'ymd':
            year, month, day = int(groups[0] or target_year), int(groups[1]), int(groups[2])
        elif kind == 'md':
            year, month, day = target_year, int(groups[0]), int(groups[1])
        else:
            year, month, day = int(groups[2]), MONTHS[groups[1].lower()], int(groups[0])
        try:
            return date(year, month, day)
        except ValueError:
            return None
    return None


def find_unique_date_row(target, read_value, max_row=40):
    """Read only column A; require one exact match. No unreadable-date fallback."""
    matches = [row for row in range(2, max_row + 1)
               if parse_sheet_date(read_value(row), target.year) == target]
    if len(matches) != 1:
        raise ValueError(f'目标日期 {target.isoformat()} 匹配 {len(matches)} 行，必须唯一确认后才能填写')
    return matches[0]
