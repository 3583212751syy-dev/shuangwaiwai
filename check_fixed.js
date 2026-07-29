const fs = require('fs');
const path = require('path');
const { PNG } = require('pngjs');
const jpeg = require('jpeg-js');
const v8 = require('v8');

const DIR = path.join(__dirname, '..');
const HASH_SIZE = 8;
const SMALL_SIZE = 32;
const HIGH_SIM_THRESHOLD = 5;
const MED_SIM_THRESHOLD = 10;
const BATCH_SIZE = 500;  // 分批处理，避免内存溢出

function hammingDistance(a, b) {
  let d = 0;
  for (let i = 0; i < a.length; i++) {
    let x = a[i] ^ b[i];
    while (x) { d += x & 1; x >>>= 1; }
  }
  return d;
}

function hashToString(hash) {
  let s = '';
  for (const n of hash) s += n.toString(16).padStart(2, '0');
  return s;
}

function packBits(bits) {
  const out = [];
  for (let i = 0; i < bits.length; i += 8) {
    let byte = 0;
    for (let j = 0; j < 8; j++) byte = (byte << 1) | (bits[i + j] ? 1 : 0);
    out.push(byte & 0xff);
  }
  return out;
}

function nearestNeighborResize(pixels, srcW, srcH, dstW, dstH) {
  const out = new Uint8ClampedArray(dstW * dstH * 4);
  const xRatio = srcW / dstW;
  const yRatio = srcH / dstH;
  for (let y = 0; y < dstH; y++) {
    for (let x = 0; x < dstW; x++) {
      const sx = Math.floor(x * xRatio);
      const sy = Math.floor(y * yRatio);
      const si = (sy * srcW + sx) * 4;
      const di = (y * dstW + x) * 4;
      out[di] = pixels[si];
      out[di + 1] = pixels[si + 1];
      out[di + 2] = pixels[si + 2];
      out[di + 3] = pixels[si + 3];
    }
  }
  return out;
}

function toGrayscale(pixels) {
  const gray = new Float64Array(pixels.length / 4);
  for (let i = 0, j = 0; i < pixels.length; i += 4, j++) {
    gray[j] = 0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2];
  }
  return gray;
}

function computeDCT(gray, N) {
  const out = new Float64Array(N * N);
  const PI = Math.PI;
  for (let u = 0; u < N; u++) {
    for (let v = 0; v < N; v++) {
      let sum = 0;
      for (let x = 0; x < N; x++) {
        for (let y = 0; y < N; y++) {
          sum += gray[x * N + y] * Math.cos((2 * x + 1) * u * PI / (2 * N)) * Math.cos((2 * y + 1) * v * PI / (2 * N));
        }
      }
      const cu = u === 0 ? 1 / Math.SQRT2 : 1;
      const cv = v === 0 ? 1 / Math.SQRT2 : 1;
      out[u * N + v] = 0.25 * cu * cv * sum;
    }
  }
  return out;
}

function pHash(graySmall, N, small) {
  const dct = computeDCT(graySmall, small);
  const low = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      low[i * N + j] = dct[i * small + j];
    }
  }
  let sum = 0;
  for (let i = 1; i < low.length; i++) sum += low[i];
  const mean = sum / (low.length - 1);
  const bits = [];
  for (let i = 0; i < low.length; i++) bits.push(low[i] > mean);
  return packBits(bits);
}

function dHash(graySmall, small) {
  const bits = [];
  for (let y = 0; y < small; y++) {
    for (let x = 0; x < small - 1; x++) {
      bits.push(graySmall[y * small + x] > graySmall[y * small + (x + 1)]);
    }
  }
  return packBits(bits);
}

function loadImage(filePath) {
  const buf = fs.readFileSync(filePath);
  const header = buf.slice(0, 3);
  let img;
  if (header[0] === 0xff && header[1] === 0xd8) {
    img = jpeg.decode(buf, { maxMemoryUsageInMB: 1024 });
  } else if (header[0] === 0x89 && header.toString('ascii', 1, 4) === 'PNG') {
    img = PNG.sync.read(buf);
  } else {
    throw new Error('Unsupported format: ' + path.basename(filePath));
  }
  const result = { data: img.data, width: img.width, height: img.height };
  // 释放 buffer 引用帮助 GC
  if (typeof img.data !== 'undefined') img = null;
  return result;
}

function processImage(filePath, fileName) {
  const { data, width, height } = loadImage(filePath);
  const small = nearestNeighborResize(data, width, height, SMALL_SIZE, SMALL_SIZE);
  const gray = toGrayscale(small);
  const ph = pHash(gray, HASH_SIZE, SMALL_SIZE);
  const dh = dHash(gray, SMALL_SIZE);
  return {
    file: fileName,
    width,
    height,
    sizeKB: (fs.statSync(filePath).size / 1024).toFixed(1),
    pHash: ph,
    dHash: dh
  };
}

function fileSizeKB(p) {
  return (fs.statSync(p).size / 1024).toFixed(1);
}

// ====== 主流程 ======
const allFiles = fs.readdirSync(DIR).filter(f => /\.(jpg|jpeg|png)$/i.test(f));
console.log('共发现 ' + allFiles.length + ' 张图片');
console.log('分批处理中（每批 ' + BATCH_SIZE + ' 张）...\n');

const entries = [];
const errors = [];
const totalBatches = Math.ceil(allFiles.length / BATCH_SIZE);

for (let batch = 0; batch < totalBatches; batch++) {
  const start = batch * BATCH_SIZE;
  const end = Math.min(start + BATCH_SIZE, allFiles.length);
  const batchFiles = allFiles.slice(start, end);
  
  for (const f of batchFiles) {
    const p = path.join(DIR, f);
    try {
      const entry = processImage(p, f);
      entries.push(entry);
    } catch (e) {
      errors.push({ file: f, error: e.message });
    }
  }
  
  // 每批完成后释放内存
  const mem = process.memoryUsage();
  console.log(`[批次 ${batch + 1}/${totalBatches}] 已完成 ${entries.length}/${allFiles.length} 张 ` +
    `(失败 ${errors.length}) | 内存: ${(mem.heapUsed/1024/1024).toFixed(0)}MB`);
  
  if (global.gc) global.gc();
}

console.log('\n成功分析 ' + entries.length + ' 张，失败 ' + errors.length + ' 张\n');

// 输出前 30 张图片信息
console.log('=== 图片信息一览（前30张）===');
for (let i = 0; i < Math.min(30, entries.length); i++) {
  const e = entries[i];
  console.log(`- ${e.file}   ${e.width}x${e.height}   ${e.sizeKB} KB   pHash=${hashToString(e.pHash)}`);
}
if (entries.length > 30) console.log(`... 还有 ${entries.length - 30} 张\n`);

console.log('=== 重复/高度相似图片（汉明距离 <= ' + HIGH_SIM_THRESHOLD + '，视为同一图片）===');
const highPairs = [];
const medPairs = [];
for (let i = 0; i < entries.length; i++) {
  for (let j = i + 1; j < entries.length; j++) {
    const d1 = hammingDistance(entries[i].pHash, entries[j].pHash);
    if (d1 <= HIGH_SIM_THRESHOLD) highPairs.push([entries[i], entries[j], d1]);
    else if (d1 <= MED_SIM_THRESHOLD) medPairs.push([entries[i], entries[j], d1]);
  }
}

if (highPairs.length === 0) {
  console.log('（未发现完全一致/几乎一致的图片）');
} else {
  for (const [a, b, d] of highPairs) {
    console.log('* ' + a.file + '  <->  ' + b.file + '   (pHash 汉明距离 = ' + d + ')   高度相似 / 重复');
  }
}
console.log();

console.log('=== 疑似来源相同/轻微修改的图片（汉明距离 ' + (HIGH_SIM_THRESHOLD + 1) + '-' + MED_SIM_THRESHOLD + '）===');
if (medPairs.length === 0) {
  console.log('（未发现中度相似图片）');
} else {
  for (const [a, b, d] of medPairs) {
    console.log('? ' + a.file + '  <->  ' + b.file + '   (pHash 汉明距离 = ' + d + ')   中度相似');
  }
}
console.log();

console.log('=== 分析摘要 ===');
console.log('总图片数：' + entries.length);
console.log('完全/高度相似组（距离<=' + HIGH_SIM_THRESHOLD + '）：' + highPairs.length + ' 对');
console.log('中度相似组（距离<=' + MED_SIM_THRESHOLD + '）：' + (highPairs.length + medPairs.length) + ' 对');
console.log();
if (highPairs.length > 0) {
  console.log('⚠ 结论：检测到 ' + highPairs.length + ' 对图片存在高度相似/重复，存在重复使用或侵权风险，请人工复核。');
} else if (medPairs.length > 0) {
  console.log('⚠ 结论：检测到 ' + medPairs.length + ' 对图片存在中度相似，建议进行人工复核。');
} else {
  console.log('✓ 结论：在当前目录图片之间未检测到明显的重复或高度相似。');
}

if (errors.length > 0) {
  console.log('\n=== 处理失败的文件（共 ' + errors.length + ' 个）===');
  for (const e of errors) console.log('x ' + e.file + ': ' + e.error);
}

// 保存结果到 JSON 文件，方便后续处理
const resultJson = {
  total: entries.length,
  errors: errors.length,
  highSimPairs: highPairs.map(([a, b, d]) => ({ file1: a.file, file2: b.file, distance: d })),
  medSimPairs: medPairs.map(([a, b, d]) => ({ file1: a.file, file2: b.file, distance: d })),
  highSimCount: highPairs.length,
  medSimCount: medPairs.length,
  generatedAt: new Date().toISOString()
};
fs.writeFileSync(path.join(__dirname, 'check_result.json'), JSON.stringify(resultJson, null, 2));
console.log('\n结果已保存到 check_result.json');
