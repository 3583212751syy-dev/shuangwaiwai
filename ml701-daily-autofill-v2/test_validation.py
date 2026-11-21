import ast
import argparse
import contextlib
import io
import time
import os
import traceback
from types import SimpleNamespace
from datetime import timedelta
from unittest.mock import patch
import unittest
from datetime import date
from pathlib import Path
from validation import validate_code, validate_data, reusable_pending, parse_sheet_date, find_unique_date_row

ROOT = Path(__file__).resolve().parent


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.data = {'营业额': '100.50', '广告费': '0', '点击': '12'}
        self.pending = {'date': '2026-11-21', 'target_date': '2026-11-20',
                        'code': 'C0FFEE01', 'data': self.data}

    def test_same_day_cache_for_different_target_is_rejected(self):
        self.assertTrue(reusable_pending(self.pending, '2026-11-21', '2026-11-20'))
        self.assertFalse(reusable_pending(self.pending, '2026-11-21', '2026-11-19'))
        self.assertFalse(reusable_pending(self.pending, '2026-11-22', '2026-11-20'))

    def test_zero_cost_cache_and_missing_fields(self):
        self.assertEqual(validate_data(self.data)['广告费'], '0')
        for key in self.data:
            bad = dict(self.data)
            del bad[key]
            self.assertFalse(reusable_pending({**self.pending, 'data': bad}, '2026-11-21', '2026-11-20'))

    def test_numbers_and_integer_clicks(self):
        for value in ('NaN', 'Infinity', '-1', '801', '1,000', ''):
            with self.assertRaises(ValueError):
                validate_data({**self.data, '广告费': value})
        with self.assertRaises(ValueError):
            validate_data({**self.data, '点击': '1.5'})

    def test_codes(self):
        self.assertEqual(validate_code(' c0ffee01 '), 'C0FFEE01')
        for value in ('G0FFEE01', 'ABC123', '', None):
            with self.assertRaises(ValueError):
                validate_code(value)

    def test_explicit_date_formats(self):
        for value in ('2026/11/20', '2026-11-20', '2026年11月20日', '11月20日', '11/20', '20 nov. 2026'):
            self.assertEqual(parse_sheet_date(value, 2026), date(2026, 11, 20))
        for value in ('=', '', '11月31日', '2026/11/200', 'note 11/20'):
            self.assertIsNone(parse_sheet_date(value, 2026))

    def test_shifted_rows_and_duplicate_rows(self):
        values = {9: '11月20日', 21: '11月19日'}
        read = lambda row: values.get(row, '')
        self.assertEqual(find_unique_date_row(date(2026, 11, 20), read), 9)
        values[22] = '11月20日'
        with self.assertRaises(ValueError):
            find_unique_date_row(date(2026, 11, 20), read)

    def test_missing_or_wrong_year_does_not_use_row_guess(self):
        for value in ('=', '', '2025/11/20', '11月19日'):
            with self.assertRaises(ValueError):
                find_unique_date_row(date(2026, 11, 20), lambda row: value)

    def test_scan_failure_propagates(self):
        def broken(row):
            raise RuntimeError('cannot navigate')
        with self.assertRaises(RuntimeError):
            find_unique_date_row(date(2026, 11, 20), broken)

    def test_actual_cell_mismatch_not_hidden_by_receipt(self):
        tree = ast.parse((ROOT / 'write_return_code.py').read_text(encoding='utf-8'))
        function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'decide_action')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<pure decision>', 'exec'), namespace)
        decide = namespace['decide_action']
        self.assertEqual(decide('DEADBEEF', 'C0FFEE01', 'C0FFEE01', '2026-11-21', '2026-11-21')[0], 'ABORT')
        self.assertEqual(decide('C0FFEE01', 'C0FFEE01', 'C0FFEE01', '2026-11-21', '2026-11-21')[0], 'SKIP')

    def test_dry_run_collection_failure_returns_failure_without_window_writes(self):
        tree = ast.parse((ROOT / 'daily_701_oneflow.py').read_text(encoding='utf-8'))
        function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'main')
        saved = {}

        def failed_collection(*args):
            raise RuntimeError('collection failed')

        def forbidden(*args):
            self.fail('dry-run must not reach desktop write stages')

        namespace = dict(argparse=argparse, date=date, timedelta=timedelta, time=time,
                         os=os, traceback=traceback, json=__import__('json'),
                         validate_code=validate_code, validate_data=validate_data,
                         reusable_pending=reusable_pending, STATE={},
                         u=SimpleNamespace(GetForegroundWindow=lambda: 1),
                         log=lambda *args: None, return_done=lambda *args: False,
                         ads_done=lambda *args: False, install_ads_stability=lambda: None,
                         load_json=lambda *args: None, PENDING='pending', OUT='out',
                         log_code_today=lambda *args: 'C0FFEE01',
                         collect_ads_data=failed_collection,
                         save_json=lambda path, value: saved.update({path: value}),
                         restore_foreground=lambda: None, discover_ding=forbidden)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<orchestration>', 'exec'), namespace)
        with patch('sys.argv', ['oneflow', '--dry-run']), contextlib.redirect_stdout(io.StringIO()):
            result = namespace['main']()
        self.assertEqual(result, 1)
        self.assertFalse(saved['out']['overall_ok'])
        self.assertIn('collection failed', saved['out']['collected']['ads_error'])


if __name__ == '__main__':
    unittest.main()
