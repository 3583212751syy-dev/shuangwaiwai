/* SKU 标准化与重复判定，浏览器与 Node 共用 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SKUMatch = api;
})(typeof self !== 'undefined' ? self : globalThis, function () {
  const MLM_RE = /\bMLM-?(\d{7,10})\b/gi;
  const trim = s => String(s == null ? '' : s).trim();

  function normalizeSKU(raw) {
    let s = trim(raw).toUpperCase().replace(/\s+/g, '');
    if (!s) return '';
    const m = s.match(/^MLM-?(\d+)$/);
    if (m) return 'MLM' + m[1].replace(/^0+(?=\d)/, '');
    s = s.replace(/^SKU[:：]?/, '').replace(/^[（(]/, '').replace(/[）)]$/, '');
    return s.replace(/[-_]/g, '-');
  }

  function extractSKUs(text) {
    const found = new Map();
    const src = trim(text);
    for (const m of src.matchAll(MLM_RE)) {
      const id = 'MLM' + m[1];
      if (!found.has(id)) found.set(id, { sku: id, evidence: m[0] });
    }
    // 内部 SKU：前缀-数字-可选字母，如 ZLP-A1836A / PFY-53-X2
    const internal = /\b([A-Z]{2,5}-[A-Z0-9]{2,}[A-Z]?)\b/g;
    for (const m of src.matchAll(internal)) {
      const id = m[1].toUpperCase();
      if (/^\d+$/.test(id)) continue;
      if (!found.has(id)) found.set(id, { sku: id, evidence: m[0] });
    }
    return [...found.values()];
  }

  function buildIndex(set) {
    const map = new Map();
    for (const item of set?.items || []) {
      const k = normalizeSKU(item.sku);
      if (!k) continue;
      if (!map.has(k)) map.set(k, item);
    }
    return map;
  }

  function matchSKUs(candidates, index) {
    const hits = [];
    for (const c of candidates) {
      const k = normalizeSKU(c.sku || c);
      if (k && index.has(k)) hits.push({ ...(typeof c === 'object' ? c : { sku: c }), sku: k, item: index.get(k) });
    }
    return hits;
  }

  // 返回 {duplicate, hits, candidates, verdict}
  function verdict(candidates, index, opts = {}) {
    const list = Array.isArray(candidates) ? candidates : extractSKUs(candidates);
    const hits = matchSKUs(list, index);
    const duplicate = hits.length > 0;
    return {
      duplicate,
      hits,
      candidates: list,
      verdict: duplicate ? '产品重复（SKU 命中自有集合）' : (list.length ? 'SKU 未命中，不判为重复' : '未识别到 SKU，无法判定')
    };
  }

  return { normalizeSKU, extractSKUs, buildIndex, matchSKUs, verdict, MLM_RE };
});
