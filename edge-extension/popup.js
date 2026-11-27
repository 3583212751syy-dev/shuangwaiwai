const $ = id => document.getElementById(id);
let current = null;
let index = new Map();
let setInfo = null;

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

async function loadSet() {
  const built = await fetch(chrome.runtime.getURL('sku-set.json')).then(r => r.json()).catch(() => null);
  const cfg = await chrome.storage.local.get({ customSKUs: [], siteUrl: '' });
  const combined = { items: [...(built?.items || []), ...(cfg.customSKUs || []).map(s => ({ sku: s, type: 'custom', sources: [{ sheet: '手动添加' }] }))] };
  setInfo = { built, custom: cfg.customSKUs || [], siteUrl: cfg.siteUrl };
  index = SKUMatch.buildIndex(combined);
  $('footer').textContent = `自有 SKU 集合：${setInfo.built?.count || 0} 条内置 + ${setInfo.custom.length} 条手动 · 数据来自 ${setInfo.built?.source || '—'}`;
}

function render(data) {
  current = data;
  const r = SKUMatch.verdict(data.skuCandidates || [], index);
  const v = $('verdict');
  v.className = r.duplicate ? 'dup' : (r.candidates.length ? 'safe' : 'unknown');
  v.innerHTML = esc(r.verdict) + '<small>' + (r.hits.length
    ? '命中：' + r.hits.map(h => esc(h.sku)).join('、')
    : (r.candidates.length ? '扫描到：' + r.candidates.map(c => esc(c.sku)).join('、') : '请在商品详情页使用')) + '</small>';

  const specs = (data.specs || []).map(s => '<li>' + esc(s) + '</li>').join('');
  const imgs = (data.images || []).slice(0, 4).map(u => '<img class="thumb" src="' + esc(u) + '" alt="">').join('');
  $('product').innerHTML =
    '<h3>商品信息</h3>' +
    '<div class="name">' + esc(data.name || '未识别到名称') + '</div>' +
    '<div class="meta">' + esc(data.url) + (data.price ? ' · 价格 ' + esc(data.price) : '') + '</div>' +
    (imgs ? '<div style="margin-top:8px">' + imgs + '</div>' : '') +
    (specs ? '<h3>规格描述</h3><div class="spec"><ul>' + specs + '</ul></div>' : '<div class="meta" style="margin-top:8px">未识别到规格描述</div>') +
    (r.hits.length ? '<h3>重复依据</h3>' + r.hits.map(h => {
      const it = h.item || {};
      const src = (it.sources || []).map(x => x.sheet + (x.row ? ' 第' + x.row + '行' : '')).join('、');
      return '<div class="hit"><b>' + esc(h.sku) + '</b> · ' + esc(it.type === 'mlm' ? '平台商品 ID' : '内部 SKU') + (src ? '<br><span class="meta">' + esc(src) + '</span>' : '') + '</div>';
    }).join('') : '');
}

async function scan() {
  $('scan').disabled = true;
  $('status').textContent = '正在扫描页面…';
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id) throw new Error('没有可用标签页');
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ['content.js'] });
    const out = await chrome.tabs.sendMessage(tab.id, { type: 'WB_SCAN_PAGE' });
    if (!out?.ok) throw new Error(out?.error || '页面未返回结果');
    render(out.data);
    $('status').textContent = '扫描完成 · ' + new Date(out.data.scannedAt).toLocaleString('zh-CN');
  } catch (e) {
    $('status').textContent = '扫描失败：' + (e?.message || e) + '（系统页面或商店页面可能不允许注入）';
  } finally {
    $('scan').disabled = false;
  }
}

$('scan').onclick = scan;
$('compare').onclick = () => chrome.tabs.create({ url: setInfo?.siteUrl || 'https://product-image-match.app.workbuddy.host/?sharecode=MTCP4KDYCY' });
$('copy').onclick = async () => {
  if (!current) return;
  const text = [
    '商品名称：' + current.name,
    '商品链接：' + current.url,
    'SKU：' + (current.skuCandidates || []).map(x => x.sku).join('、'),
    '规格：', ...(current.specs || []),
    '主图：', ...(current.images || [])
  ].join('\n');
  try { await navigator.clipboard.writeText(text); $('status').textContent = '已复制扫描结果'; }
  catch { $('status').textContent = '复制失败，请手动选择文本'; }
};

loadSet().catch(e => $('status').textContent = 'SKU 集合加载失败：' + e.message);
