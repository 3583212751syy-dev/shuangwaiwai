// 从金山文档「选品定价表.xlsx」的「产品排重」A列与「通用品」G列（表头均为「同行SKU」）
// 生成扩展内置 SKU 集合。输入文件由 WorkBuddy 通过金山文档 API 导出（见 README）。
import fs from 'node:fs';
import path from 'node:path';

const here = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'));
const args = process.argv.slice(2);
const dedupFile = args[0];    // 产品排重 A 列 rangeData JSON
const generalFile = args[1];  // 通用品 G 列 rangeData JSON
const notesFile = args[2];    // 备注列 MLM ID JSON（可选）
const outFile = args[3] || path.join(here, 'sku-set.json');

function readCells(file) {
  if (!file || !fs.existsSync(file)) return [];
  const data = JSON.parse(fs.readFileSync(file, 'utf8'));
  return data?.data?.detail?.rangeData || data?.data?.range_data || [];
}

function norm(raw) {
  return String(raw || '').trim().toUpperCase().replace(/[^A-Z0-9]/g, '');
}

function collect(cells, sheet) {
  const out = new Map();
  for (const c of cells) {
    const text = String(c.cellText || c.cell_text || '').trim();
    if (!text || text === '同行SKU') continue;
    if (!/^MLM-?\d/i.test(text) && !/^B0[A-Z0-9]{8}$/i.test(text)) continue;
    const sku = norm(text);
    if (!sku) continue;
    const row = (c.rowFrom ?? c.row_from ?? 0) + 1;
    if (!out.has(sku)) out.set(sku, { sku, type: sku.startsWith('MLM') ? 'mlm' : 'other', sources: [] });
    out.get(sku).sources.push({ sheet, row });
  }
  return out;
}

const items = new Map();
for (const [file, sheet] of [[dedupFile, '产品排重'], [generalFile, '通用品']]) {
  for (const [sku, v] of collect(readCells(file), sheet)) {
    if (!items.has(sku)) items.set(sku, v);
    else items.get(sku).sources.push(...v.sources);
  }
}
if (notesFile && fs.existsSync(notesFile)) {
  const mlm = JSON.parse(fs.readFileSync(notesFile, 'utf8'));
  for (const [id, v] of Object.entries(mlm)) {
    const sku = norm(id);
    if (!sku || items.has(sku)) continue;
    items.set(sku, { sku, type: 'mlm', sources: [{ sheet: '主管审核备注', row: v.row ?? null }] });
  }
}

const set = {
  version: 2,
  generatedAt: new Date().toISOString(),
  source: '金山文档 选品定价表.xlsx（与我共享）',
  sourceUrl: 'https://www.kdocs.cn/l/ch0h8m4dZ6ho',
  note: '自有 SKU = 产品排重 A 列 + 通用品 G 列（表头均为「同行SKU」，值为 MLM 平台 ID）；备注列补充。',
  count: items.size,
  items: [...items.values()].sort((a, b) => a.sku.localeCompare(b.sku)),
};
fs.writeFileSync(outFile, JSON.stringify(set, null, 2));
console.log('wrote', outFile, 'skus', set.count);
