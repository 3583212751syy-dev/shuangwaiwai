import ast
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from navigation import extract_return_code, find_return_page

ROOT = Path(__file__).resolve().parent


def row(index, control, name, element=None):
    return (index, control, name, element)


def pure_function(filename, function_name, globals_):
    tree = ast.parse((ROOT / filename).read_text(encoding='utf-8'))
    function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == function_name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<isolated navigation>', 'exec'), globals_)
    return globals_[function_name]


class ReturnPageTests(unittest.TestCase):
    def test_sparse_indices_and_inline_lowercase_code(self):
        self.assertEqual(extract_return_code([row(88, 50020, 'Código de autorización para hoy:'),
                                              row(104, 50020, 'c0ffee01')]), 'C0FFEE01')
        self.assertEqual(extract_return_code([row(3, 50020, 'Código de autorización para hoy: c0ffee01')]), 'C0FFEE01')

    def test_unrelated_hex_and_conflicts(self):
        self.assertIsNone(extract_return_code([row(1, 50020, 'C0FFEE01')]))
        with self.assertRaises(RuntimeError):
            extract_return_code([row(1, 50020, 'Código de autorización para hoy: C0FFEE01'),
                                 row(2, 50020, 'Código de autorización para hoy: DEADBEEF')])

    def test_named_sales_tab_is_actually_activated(self):
        current = [row(5, 50019, 'Ventas | Mercado Libre', 'sales')]
        invoked = []
        def activate(element):
            invoked.append(element)
            current[:] = [row(100, 50020, 'Código de autorización para hoy: C0FFEE01')]
            return True
        self.assertEqual(find_return_page(lambda: list(current), activate, lambda _: None, lambda _: None), 'C0FFEE01')
        self.assertEqual(invoked, ['sales'])

    def test_existing_page_has_no_navigation(self):
        def forbidden(element):
            self.fail('unnecessary navigation')
        code = find_return_page(lambda: [row(2, 50020, 'Código de autorización para hoy: C0FFEE01')],
                                forbidden, lambda _: None, lambda _: None)
        self.assertEqual(code, 'C0FFEE01')

    def test_no_product_search_or_unlabelled_clicks(self):
        invoked = []
        with self.assertRaises(RuntimeError):
            find_return_page(lambda: [row(2, 50004, '搜索商品', 'search'), row(3, 50000, '', 'unknown')],
                             lambda e: invoked.append(e), lambda _: None, lambda _: None)
        self.assertEqual(invoked, [])


class NameBoxTests(unittest.TestCase):
    def environment(self, elements):
        controls = SimpleNamespace(GetElement=lambda index: elements[index][1])
        names = [(i, name) for i, (name, _) in enumerate(elements)]
        state = {'ref': 'B2'}
        calls = []
        globals_ = dict(CELLREF=re.compile(r'^[A-Z]{1,2}\d{1,4}$'), SHEET_HWND=1,
                        scan_hwnd=lambda h: (names, [], controls),
                        IUIAutomationValuePattern=object,
                        ensure_foreground=lambda: None, key=lambda vk: calls.append(vk),
                        time=SimpleNamespace(sleep=lambda _: None),
                        read_ref=lambda **kwargs: state['ref'], log=lambda _: None)
        return globals_, state, calls

    def box(self, state, readonly=False):
        pattern = SimpleNamespace(CurrentValue='B2', CurrentIsReadOnly=readonly,
                                  SetValue=lambda value: state.update(ref=value))
        pattern.QueryInterface = lambda cls: pattern
        return SimpleNamespace(CurrentControlType=50004, GetCurrentPattern=lambda _: pattern, SetFocus=lambda: None)

    def test_name_box_navigates_without_typing_cell_values(self):
        globals_, state, calls = self.environment([])
        element = self.box(state)
        globals_['scan_hwnd'] = lambda h: ([(0, 'B2')], [], SimpleNamespace(GetElement=lambda _: element))
        goto = pure_function('write_return_code.py', 'goto_by_name_box', globals_)
        self.assertTrue(goto('C25'))
        self.assertEqual(state['ref'], 'C25')
        self.assertEqual(calls, [0x0D])

    def test_unidentified_edit_is_not_used_as_name_box(self):
        globals_, state, calls = self.environment([])
        element = self.box(state)
        globals_['scan_hwnd'] = lambda h: ([(0, '搜索消息')], [], SimpleNamespace(GetElement=lambda _: element))
        self.assertFalse(pure_function('write_return_code.py', 'goto_by_name_box', globals_)('C25'))
        self.assertEqual(calls, [])

    def test_document_search_does_not_use_chat_search(self):
        state = {'ref': 'B2'}
        element = self.box(state)
        globals_ = dict(ding_cef_windows=lambda h: [],
                        w=SimpleNamespace(scan_hwnd=lambda h: ([(0, '搜索消息')], [],
                            SimpleNamespace(GetElement=lambda _: element))),
                        _plain=lambda text: text, log=lambda _: None)
        self.assertFalse(pure_function('daily_701_oneflow.py', 'search_doc_by_name', globals_)(1, '退货登记码'))

    def test_unverified_grid_cannot_send_filling_keys(self):
        calls = []
        globals_ = dict(goto_cell=lambda target: True, focus_grid_for_cell=lambda target: False,
                        key=lambda vk: calls.append(vk), set_clipboard=lambda text: calls.append(text))
        write = pure_function('write_return_code.py', 'try_once', globals_)
        with self.assertRaises(RuntimeError):
            write('C0FFEE01', 'C25')
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
