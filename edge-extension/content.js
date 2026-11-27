/* 网页一键扫描：提取商品主图、名称、规格描述、SKU */
(function () {
  if (window.__wbScannerLoaded) return;
  window.__wbScannerLoaded = true;

  function text(el) { return (el?.textContent || '').replace(/\s+/g, ' ').trim(); }
  const clean = s => (s || '').replace(/\s+/g, ' ').trim();

  function meta(name) {
    const q = s => document.querySelector(s);
    return q(`meta[property="${name}"]`)?.content || q(`meta[name="${name}"]`)?.content || '';
  }

  function jsonLD() {
    const out = [];
    for (const s of document.querySelectorAll('script[type="application/ld+json"]')) {
      try {
        const v = JSON.parse(s.textContent);
        for (const x of (Array.isArray(v) ? v : [v])) {
          if (!x) continue;
          if (x['@graph']) out.push(...x['@graph']);
          else out.push(x);
        }
      } catch { /* 忽略非法 JSON-LD */ }
    }
    return out;
  }

  function findProduct() {
    const graphs = jsonLD();
    return graphs.find(x => /product/i.test(x['@type'] || '')) || graphs[0] || {};
  }

  function getTitle() {
    const p = findProduct();
    return clean(p.name) || meta('og:title') || text(document.querySelector('h1')) || document.title || '';
  }

  function getImages() {
    const urls = [];
    const add = u => {
      if (!u) return;
      try {
        const abs = new URL(u, location.href).href;
        if (/^https?:$/.test(new URL(abs).protocol) && !urls.includes(abs)) urls.push(abs);
      } catch { /* 忽略无效 URL */ }
    };
    const p = findProduct();
    if (Array.isArray(p.image)) p.image.forEach(x => add(typeof x === 'string' ? x : x?.url));
    else if (p.image) add(typeof p.image === 'string' ? p.image : p.image.url);
    add(meta('og:image'));
    add(meta('twitter:image'));
    for (const el of document.querySelectorAll('img[itemprop="image"],img[data-testid*="image"],img[class*="gallery"],img[class*="product-image"]')) add(el.src || el.dataset.src || el.getAttribute('data-original'));
    return urls.filter(u => !/\.svg(\?|$)/i.test(u)).slice(0, 12);
  }

  function getSpecs() {
    const lines = [];
    const seen = new Set();
    const push = s => {
      const v = clean(s);
      if (v.length >= 4 && v.length <= 240 && !seen.has(v)) { seen.add(v); lines.push(v); }
    };
    // 常见规格区：Mercado Libre / Shopee / Amazon / 1688
    const selectors = [
      '[data-testid="product-detail-specs"] li',
      '[class*="specification"] li',
      '[class*="specification"] tr',
      '[class*="product-info"] li',
      '[class*="item-prop"] li',
      '#productDetails li',
      '#productDetails tr',
      '#detail-bullets li',
      'ul.a-unordered-list li',
      '[class*="attribute"] li',
      '[itemprop="description"] li',
      'section li', 'article li'
    ];
    for (const sel of selectors) {
      for (const el of document.querySelectorAll(sel)) push(text(el));
      if (lines.length >= 40) break;
    }
    const p = findProduct();
    if (p.description) push(p.description.replace(/<[^>]+>/g, ' '));
    if (!lines.length) {
      for (const li of document.querySelectorAll('li')) push(text(li));
    }
    return lines.slice(0, 40);
  }

  function getPrice() {
    const p = findProduct();
    const offers = Array.isArray(p.offers) ? p.offers[0] : p.offers;
    return clean(offers?.price || offers?.lowPrice) || meta('og:price:amount') || meta('product:price:amount') || '';
  }

  function getSKUCandidates() {
    const out = new Map();
    const add = (s, where) => {
      const v = clean(s);
      if (!v) return;
      for (const m of v.matchAll(/\bMLM-?(\d{7,10})\b/gi)) {
        const id = 'MLM' + m[1];
        if (!out.has(id)) out.set(id, { sku: id, evidence: m[0], where });
      }
      for (const m of v.matchAll(/\b(?:SKU|C[oó]digo|S[Kk]U代码)[:：]?\s*([A-Z]{2,5}-[A-Z0-9-]{2,})/gi)) {
        const id = m[1].toUpperCase();
        if (!out.has(id)) out.set(id, { sku: id, evidence: m[0], where });
      }
    };
    const p = findProduct();
    add(p.sku, 'JSON-LD sku');
    add(p.mpn, 'JSON-LD mpn');
    add(p.productID, 'JSON-LD productID');
    for (const name of ['product:retailer_item_id', 'itemprop:sku', 'sku', 'product_id']) add(meta(name), 'meta ' + name);
    add(location.href, 'URL');
    // 页面文本只取"明确标注"的编号，避免把相关推荐商品的 MLM ID 误当主商品 SKU。
    const body = document.body?.innerText || '';
    const labeled = body.match(/(?:商品ID|商品编号|商品编码|SKU|C[oó]digo(?:\s+de\s+producto)?|Item\s*ID|Product\s*ID|ID)[\s:：]*\(?(MLM-?\d{7,10})\)?/gi) || [];
    for (const s of labeled) add(s, '页面标注 ID');
    return [...out.values()].slice(0, 30);
  }

  function scan() {
    return {
      url: location.href,
      title: document.title,
      name: getTitle(),
      price: getPrice(),
      images: getImages(),
      specs: getSpecs(),
      skuCandidates: getSKUCandidates(),
      scannedAt: new Date().toISOString()
    };
  }

  chrome.runtime.onMessage.addListener((msg, sender, reply) => {
    if (msg?.type === 'WB_SCAN_PAGE') {
      try { reply({ ok: true, data: scan() }); }
      catch (e) { reply({ ok: false, error: String(e?.message || e) }); }
      return true;
    }
  });
})();
