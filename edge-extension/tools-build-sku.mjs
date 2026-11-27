// 从金山文档「选品定价表.xlsx」导出的 SKU 生成扩展内置 SKU 集合。
// 输入文件由 tools/fetch-skuset.mjs 抓取（见 README）。
import fs from 'node:fs';
import path from 'node:path';

const args = process.argv.slice(2);
const dedupFile = args[0];
const generalFile = args[1];
const mlmFile = args[2];
const outFile = args[3] || path.join(process.cwd(), 'sku-set.json');

function readCells(file) {
  const raw = fs.readFileSync(file, 'utf8');
  const data = JSON.parse(raw);
  return data?.data?.detail?.rangeData || data?.data?.range_data || [];
}

function skuFromCells(cells) {
  const out = new Map();
  for (const c of cells) {
    const text = String(c.cellText || c.cell_text || '').trim();
    if (!text || text === '0' || text === 'sku' || text === 'SKU编码') continue;
    if (!/[A-Za-z]/.test(text)) continue;
    if (!/^[A-Z]{2,5}-[A-Z0-9-]+$/i.test(text)) continue;
    const row = (c.rowFrom ?? c.row_from ?? 0) + 1;
    if (!out.has(text)) out.set(text, { sku: text, sources: [] });
    out.get(text).sources.push({ sheet: '选品定价表', row });
  }
  return out;
}

const items = new Map();
for (const [file, sheet] of [[dedupFile, '产品排重'], [generalFile, '通用品']]) {
  if (!file || !fs.existsSync(file)) throw new Error('缺少输入文件: ' + file);
  const cells = readCells(file);
  for (const [sku, v] of skuFromCells(cells)) {
    const sources = v.sources.map(s => ({ sheet, row: s.row }));
    if (!items.has(sku)) items.set(sku, { sku, type: 'internal', sources });
    else items.get(sku).sources.push(...sources);
  }
}
if (mlmFile && fs.existsSync(mlmFile)) {
  const mlm = JSON.parse(fs.readFileSync(mlmFile, 'utf8'));
  for (const [id, v] of Object.entries(mlm)) {
    items.set(id, { sku: id, type: 'mlm', sources: [{ sheet: '选品定价表/产品排重·主管审核备注', row: v.row ?? null }] });
  }
}

const set = {
  version: 1,
  generatedAt: new Date().toISOString(),
  source: '金山文档 选品定价表.xlsx（与我共享）',
  sourceUrl: 'https://www.kdocs.cn/l/ch0h8m4dZ6ho',
  note: '内部 SKU 来自「产品排重」与「通用品」G 列；MLM 平台 ID 来自两张表的主管审核备注列。',
  count: items.size,
  items: [...items.values()].sort((a, b) => a.sku.localeCompare(b.sku)),
};
fs.writeFileSync(outFile, JSON.stringify(set, null, 2));
console.log('wrote', outFile, 'skus', set.count, 'internal', [...items.values()].filter(x => x.type === 'internal').length, 'mlm', [...items.values()].filter(x => x.type === 'mlm').length);
