const $ = id => document.getElementById(id);
let current = null;
let index = new Map();
let setInfo = null;
let galleryTabId = null;

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function zoom(src) {
  const lb = document.getElementById('lightbox');
  document.getElementById('lightbox-img').src = src;
  lb.style.display = 'flex';
}
function zoomAttr() {
  return "onclick=\"event.preventDefault();event.stopPropagation();zoom(this.src)\"";
}
function resetMatches(msg) {
  $('match-panel').classList.remove('hidden');
  $('match-grid').innerHTML = '';
  $('match-status').textContent = msg || '';
}
function renderMatches(items) {
  if (!items.length) { resetMatches('暂未找到图库匹配图'); return; }
  $('match-panel').classList.remove('hidden');
  $('match-status').textContent = '共 ' + items.length + ' 组';
  $('match-grid').innerHTML = items.map(x => {
    const q = x.query || current?.images?.[0] || '';
    return '<div class="match-box"><h4>网页商品图</h4>' +
      (q ? '<img src="' + esc(q) + '" alt="网页图" ' + zoomAttr() + '>' : '<div class="meta">无图</div>') +
      '</div><div class="match-box"><h4>' + esc(x.title || '图库候选') + '</h4>' +
      (x.src ? '<img src="' + esc(x.src) + '" alt="图库匹配图" ' + zoomAttr() + '>' : '<div class="meta">无图</div>') +
      '<div class="meta">' + esc(x.sub || '') + '</div></div>';
  }).join('');
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
  const imgs = (data.images || []).slice(0, 6).map(u =>
    '<a class="img-link" href="' + esc(u) + '" target="_blank" title="' + esc(u) + '">' +
    '<img class="thumb" src="' + esc(u) + '" alt="商品图" loading="lazy" ' +
    'onerror="this.style.display=\'none\';this.parentNode.textContent=\'图加载失败·点开\'" ' +
    zoomAttr() +
    '</a>').join('');
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
    const out = await chrome.tabs.sendMessage(tab.id, { type: 'WB_SCAN_PAGE_V2' });
    if (!out?.ok) throw new Error(out?.error || '页面未返回结果');
    render(out.data);
    $('status').textContent = '扫描完成 · ' + new Date(out.data.scannedAt).toLocaleString('zh-CN');
    // 一键扫描里直接带上图库比对，结果回填到本面板，不抢前台焦点。
    openGalleryCompare();
  } catch (e) {
    $('status').textContent = '扫描失败：' + (e?.message || e) + '（系统页面或商店页面可能不允许注入）';
  } finally {
    $('scan').disabled = false;
  }
}

$('scan').onclick = scan;
$('lightbox').onclick = () => { $('lightbox').style.display = 'none'; $('lightbox-img').src = ''; };
function galleryUrl() {
  const base = setInfo?.siteUrl || 'https://product-image-match.app.workbuddy.host/?sharecode=MTCP4KDYCY';
  const u = new URL(base);
  for (const img of (current?.images || []).slice(0, 8)) u.searchParams.append('img', img);
  if (u.searchParams.has('img')) u.searchParams.set('autorun', '1');
  return u.toString();
}
async function pollGallery(tabId, attempt = 0) {
  try {
    const tab = await chrome.tabs.get(tabId);
    if (!tab) return;
    const out = await chrome.scripting.executeScript({
      target: { tabId },
      func: () => {
        const cards = [...document.querySelectorAll('#results article.card')];
        const items = cards.slice(0, 8).map(card => {
          const im = card.querySelector('img');
          return {
            title: card.querySelector('h3')?.textContent?.trim() || '图库候选',
            sub: card.querySelector('.body .sub')?.textContent?.trim().slice(0, 90) || '',
            src: im ? new URL(im.currentSrc || im.src, location.href).href : ''
          };
        });
        return { done: /查找完成|导入失败|失败/.test(document.querySelector('#query-status')?.textContent || ''), items };
      }
    });
    const r = out?.[0]?.result;
    if (r?.items?.length) { renderMatches(r.items.map((x, i) => ({ ...x, query: current?.images?.[i] || current?.images?.[0] }))); return; }
    if (r?.done) { resetMatches('图库对比完成，未找到匹配图'); return; }
  } catch { /* 页面尚未加载完，继续等 */ }
  if (attempt < 60) setTimeout(() => pollGallery(tabId, attempt + 1), 2500);
  else resetMatches('图库对比超时，可点“图片对比工具”重试');
}
async function openGalleryCompare() {
  if (!current?.images?.length) { resetMatches('未扫描到商品图'); return; }
  const url = galleryUrl();
  resetMatches('正在后台与图库对比…');
  try {
    if (galleryTabId) await chrome.tabs.update(galleryTabId, { url, active: false });
    else {
      const tab = await chrome.tabs.create({ url, active: false });
      galleryTabId = tab.id;
    }
    setTimeout(() => pollGallery(galleryTabId), 4000);
  } catch (e) { resetMatches('打开图库失败：' + e.message); }
}
$('compare').onclick = openGalleryCompare;
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
