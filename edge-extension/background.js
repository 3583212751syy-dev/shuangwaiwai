// 后台：默认打开检查用网站；保留扩展安装信息。
chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.get({ customSKUs: [], siteUrl: 'https://product-image-match.app.workbuddy.host/?sharecode=MTCP4KDYCY' }, cfg => {
    chrome.storage.local.set(cfg);
  });
});
