"""Bounded, observable navigation policies. No Windows dependency or fixed screen coordinates."""
import re
import unicodedata


def label(value):
    return ' '.join(unicodedata.normalize('NFKC', str(value or '')).casefold().split())


def extract_return_code(records):
    """Use list positions, not sparse UIA indices; reject conflicting visible codes."""
    found = set()
    for pos, (_, _, text, _) in enumerate(records):
        if 'código de autorización para hoy' not in label(text):
            continue
        same_line = re.split(r'código de autorización para hoy\s*:?', str(text), flags=re.I)[-1]
        candidates = [same_line] + [row[2] for row in records[pos + 1:pos + 7]]
        for candidate in candidates:
            match = re.search(r'\b([0-9a-f]{8})\b', str(candidate), re.I)
            if match:
                found.add(match.group(1).upper())
                break
    if len(found) > 1:
        raise RuntimeError('页面出现多个不同授权码，不能唯一确认')
    return next(iter(found), None)


def find_return_page(snapshot, activate, wait, report):
    """Read current page, then named sales tabs, then observed sales/returns links. Never guess URLs."""
    code = extract_return_code(snapshot())
    if code:
        return code
    attempts = set()
    for kind in ('tab', 'link'):
        records = snapshot()
        for _, control, text, element in records:
            name = label(text)
            tab_match = control == 50019 and any(word in name for word in ('ventas', '销售', '退货', 'devoluciones'))
            link_match = control in (50000, 50005, 50011) and name in {
                'ventas', 'ventas y envíos', 'devoluciones', 'ventas y devoluciones',
                '销售', '销售与配送', '退货', '退货管理'}
            if not (tab_match if kind == 'tab' else link_match):
                continue
            if (kind, name) in attempts:
                continue
            attempts.add((kind, name))
            if len(attempts) > 6:
                break
            # Refresh after each action; do not reuse an element from an old UI tree.
            fresh = [row for row in snapshot() if row[1] == control and label(row[2]) == name]
            if len(fresh) != 1:
                continue
            report(f'[取码导航] 按可见控件名称打开 {text!r}')
            if not activate(fresh[0][3]):
                continue
            for _ in range(4):
                wait(1)
                code = extract_return_code(snapshot())
                if code:
                    return code
    raise RuntimeError('未找到当日退货码模块：已检查当前页、销售/退货标签及可见导航入口；未写入表格')
