import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
const S = SKUMatch; // 全局注入
let pass = 0;
const ok = (name, fn) => { fn(); console.log('PASS', name); pass++; };

// 加载 SKU 集合
const set = JSON.parse(fs.readFileSync(path.join(process.cwd(), 'sku-set.json'), 'utf8'));
const idx = S.buildIndex(set);
ok('SKU 集合数量与来源', () => {
  assert.ok(set.count >= 696);
  assert.ok(set.items.some(x => x.sku === 'ZLP-A1836A'));
  assert.ok(set.items.some(x => x.sku === 'MLM3137701327'));
});

ok('SKU 标准化', () => {
  assert.equal(S.normalizeSKU(' MLM-2850659521 '), 'MLM2850659521');
  assert.equal(S.normalizeSKU('zlp-a1836a'), 'ZLP-A1836A');
});

ok('文本提取 MLM 与内部 SKU', () => {
  const got = S.extractSKUs('商品ID:MLM2850659521(MLM66631409) 同款 ZLP-A1836A');
  const ids = got.map(x => x.sku);
  assert.ok(ids.includes('MLM2850659521'));
  assert.ok(ids.includes('MLM66631409'));
  assert.ok(ids.includes('ZLP-A1836A'));
});

ok('重复判定：命中平台 ID', () => {
  const r = S.verdict(['MLM3137701327'], idx);
  assert.equal(r.duplicate, true);
  assert.equal(r.hits[0].sku, 'MLM3137701327');
});

ok('重复判定：命中内部 SKU', () => {
  const r = S.verdict(['ZLP-A1836A'], idx);
  assert.equal(r.duplicate, true);
});

ok('重复判定：未命中', () => {
  const r = S.verdict(['MLM9999999999'], idx);
  assert.equal(r.duplicate, false);
  assert.equal(r.verdict, 'SKU 未命中，不判为重复');
});

ok('重复判定：无 SKU', () => {
  const r = S.verdict([], idx);
  assert.equal(r.duplicate, false);
  assert.match(r.verdict, /未识别到 SKU/);
});

// content.js 在模拟 DOM 下的扫描
ok('content.js 提取商品页字段', () => {
  const html = `<html><head>
    <meta property="og:title" content="Interruptor Inalámbrico 433mhz + Receptor Relé 1 Canal 12vcc Negro">
    <meta property="og:image" content="https://http2.mlstatic.com/D_NQ_NP_1.jpg">
    <script type="application/ld+json">{"@type":"Product","name":"Interruptor Inalámbrico 433mhz","sku":"MLM3137701327","image":["https://http2.mlstatic.com/D_NQ_NP_2.jpg"],"offers":{"price":"179.00"}}</script>
  </head><body><h1>Interruptor Inalámbrico</h1>
    <ul><li>Cantidad de controles remotos: 1</li><li>Potencia máxima de carga de 110W</li></ul>
  </body></html>`;
  const listeners = {};
  const document = {
    title: 'Producto | Mercado Libre',
    querySelector: sel => {
      if (sel.includes('og:title')) return { content: 'Interruptor Inalámbrico 433mhz + Receptor Relé 1 Canal 12vcc Negro' };
      if (sel.includes('og:image')) return { content: 'https://http2.mlstatic.com/D_NQ_NP_1.jpg' };
      if (sel === 'h1') return { textContent: 'Interruptor Inalámbrico' };
      return null;
    },
    querySelectorAll: sel => {
      if (sel.includes('ld+json')) return [{ textContent: '{"@type":"Product","name":"Interruptor Inalámbrico 433mhz","sku":"MLM3137701327","image":["https://http2.mlstatic.com/D_NQ_NP_2.jpg"],"offers":{"price":"179.00"}}' }];
      if (sel === 'li') return [{ textContent: 'Cantidad de controles remotos: 1' }, { textContent: 'Potencia máxima de carga de 110W' }];
      return [];
    },
    body: { innerText: '商品ID:MLM2850659521(MLM66631409)' }
  };
  const context = {
    window: { __wbScannerLoaded: false },
    document,
    location: { href: 'https://www.mercadolibre.com.mx/p/MLM2850659521' },
    URL: globalThis.URL,
    Date: globalThis.Date,
    chrome: { runtime: { onMessage: { addListener: fn => { listeners.scan = fn; } } } }
  };
  vm.runInNewContext(fs.readFileSync(path.join(process.cwd(), 'content.js'), 'utf8'), context);
  let out;
  listeners.scan({ type: 'WB_SCAN_PAGE' }, {}, r => { out = r; });
  assert.equal(out.ok, true);
  assert.match(out.data.name, /Interruptor Inalámbrico/);
  assert.equal(out.data.price, '179.00');
  assert.ok(out.data.images.includes('https://http2.mlstatic.com/D_NQ_NP_2.jpg'));
  assert.ok(out.data.specs.some(x => x.includes('110W')));
  assert.ok(out.data.skuCandidates.some(x => x.sku === 'MLM3137701327'));
  assert.ok(out.data.skuCandidates.some(x => x.sku === 'MLM2850659521'));
});

console.log('ALL PASS', pass);
