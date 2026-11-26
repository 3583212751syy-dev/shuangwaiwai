// 生成线上覆盖表：761 组人工标签（按来源行号）+ 2430 条 DINOv2 向量（int8 量化）。
// 线上底库存在云端桶里、类目和向量都是旧的，用这张表在前端覆盖，避免去重写 2430 条云端元数据。
import {readFileSync, writeFileSync} from 'node:fs';
const repo = 'D:/GPT/2026-11-21/new-chat-2/work/git-handoff/projects/product-image-compare';
const groups = JSON.parse(readFileSync(repo + '/work/groups.json', 'utf8'));
const labels = JSON.parse(readFileSync(repo + '/work/labels-manual.json', 'utf8')).labels;
const dino = JSON.parse(readFileSync(repo + '/work/embeddings-dinov2.json', 'utf8'));

// 行号 -> [名称, 类目, 确认状态]；名字里带「待复核」的组保留后缀和「待确认」状态
const byRow = {};
let labelled = 0, confirmed = 0;
groups.forEach((g, i) => {
  const l = labels[String(i)];
  if (!l) return;
  const pending = /待复核/.test(l.name || '');
  byRow[String(g.row)] = [l.name, l.category, pending ? '待确认' : '已确认'];
  if (!pending) confirmed++;
  labelled++;
});

// DINOv2：按向量 L2 归一后量化为 int8，保留方向即可（检索只用余弦）
const files = [], chunks = [];
let totalBytes = 0;
for (const f of Object.keys(dino).sort()) {
  const v = dino[f];
  if (!Array.isArray(v) || v.length !== 384) continue;
  const n = Math.hypot(...v) || 1;
  const q = Buffer.alloc(384);
  for (let k = 0; k < 384; k++) {
    const s = Math.max(-127, Math.min(127, Math.round((v[k] / n) * 127)));
    q[k] = s < 0 ? s + 256 : s;
  }
  files.push(f);
  chunks.push(q);
  totalBytes += 384;
}
const b64 = Buffer.concat(chunks).toString('base64');

const out = {
  v: 1,
  generatedAt: new Date().toISOString(),
  note: 'labels 按来源行号覆盖线上底库的品名/类目；dino 为 int8 量化向量，前端除以 127 还原',
  labels: byRow,
  dino: {dim: 384, files, b64},
};
const p = repo + '/site-source/overrides.json';
writeFileSync(p, JSON.stringify(out));
console.log('写出', p);
console.log('标签组数', labelled, '（已确认', confirmed, '/ 待复核', labelled - confirmed, '） DINOv2 向量', files.length, ' 原始字节', totalBytes, ' 文件大小', (JSON.stringify(out).length / 1048576).toFixed(2) + 'MB');
