import {
  Canvas,
  createCanvas,
  createImageData,
  Image as CanvasImage,
  ImageData,
} from 'canvas';

type RGB = [red: number, green: number, blue: number];
type Hue = number;

export type HEX = `#${string}`;
export type HSL = [hue: number, saturation: number, lightness: number];
export type VariantName = number | string;
export type Image = CanvasImage | HTMLImageElement;

const hexToUint32LE = (hex: HEX) => {
  const n = Number.parseInt(hex.slice(1), 16) & 0x00ffffff;
  const [r, g, b] = int32ToRgb(n);
  return (b << 16) | (g << 8) | r;
};

const int32ToRgb = (color: number): RGB => {
  const r = (color >> 16) & 0xff;
  const g = (color >> 8) & 0xff;
  const b = color & 0xff;
  return [r, g, b];
};

const hex = (value: number) => Math.round(value).toString(16).padStart(2, '0');
const rgbToHex = ([r, g, b]: RGB): HEX => `#${hex(r)}${hex(g)}${hex(b)}`;
const int32ToHex = (color: number) => {
  const [r, g, b] = int32ToRgb(color);
  return rgbToHex([b, g, r]);
};

const applyHue = (color: number, hue: number) => {
  const h = hue / 360;
  const r = (color & 0xff) / 255;
  const g = ((color >> 8) & 0xff) / 255;
  const b = ((color >> 16) & 0xff) / 255;
  const min = Math.min(r, g, b);
  const max = Math.max(r, g, b);
  const delta = max - min;
  let s;
  const l = (min + max) / 2;
  if (max === min) {
    s = 0;
  } else if (l <= 0.5) {
    s = delta / (max + min);
  } else {
    s = delta / (2 - max - min);
  }

  let t2;
  let t3;
  let val;

  if (s === 0) {
    val = l * 255;
    return (val << 16) | (val << 8) | val;
  }

  if (l < 0.5) {
    t2 = l * (1 + s);
  } else {
    t2 = l + s - l * s;
  }

  const t1 = 2 * l - t2;
  let result0 = 0;
  let result1 = 0;
  let result2 = 0;
  for (let i = 0; i < 3; i++) {
    t3 = h + (1 / 3) * -(i - 1);
    if (t3 < 0) {
      t3++;
    }

    if (t3 > 1) {
      t3--;
    }

    if (6 * t3 < 1) {
      val = t1 + (t2 - t1) * 6 * t3;
    } else if (2 * t3 < 1) {
      val = t2;
    } else if (3 * t3 < 2) {
      val = t1 + (t2 - t1) * (2 / 3 - t3) * 6;
    } else {
      val = t1;
    }

    if (i === 0) {
      result0 = val * 255;
    } else if (i === 1) {
      result1 = val * 255;
    } else {
      result2 = val * 255;
    }
  }

  return (result2 << 16) | (result1 << 8) | result0;
};

const createCanvasFromImage = (image: Image) => {
  const canvas = createCanvas(image.width, image.height);
  const context = canvas.getContext('2d');
  context.drawImage(image as unknown as CanvasImage, 0, 0);
  return canvas;
};

const equals = (a: ImageData, b: ImageData) => {
  if (a.width !== b.width || a.height !== b.height) {
    return false;
  }
  for (let i = 0; i < a.data.length; i++) {
    if (a.data[i] !== b.data[i]) {
      return false;
    }
  }
  return true;
};

const colorCacheSlot = (keys: Uint32Array, key: number, mask: number) => {
  let slot = Math.imul(key, 0x9e3779b1) & mask;
  let cachedKey = keys[slot];
  while (cachedKey !== 0 && cachedKey !== key) {
    slot = (slot + 1) & mask;
    cachedKey = keys[slot];
  }
  return slot;
};

// Palette art has few colors, but photo-like inputs can have one color per pixel.
// Keep the per-color cache bounded and compute overflow colors without retaining them.
const maxCachedColors = 4096;

export default function paletteSwap(
  image: Image,
  inputVariants: ReadonlyMap<VariantName, ReadonlyMap<HEX, HEX> | Hue>,
  staticColors?: Set<HEX> | null,
  images?: ReadonlyMap<VariantName, Image> | null,
  options?: {
    imageName?: string;
    ignoreMissing?: boolean;
  }
): ReadonlyMap<VariantName, Canvas> {
  const canvas = createCanvasFromImage(image);
  const context = canvas.getContext('2d');
  const { height, width } = canvas;
  const size = width * height;
  const imageData = new Uint32Array(
    context.getImageData(0, 0, width, height).data.buffer
  );
  const results = new Map();

  const variantNames: Array<VariantName> = [];
  const variantImageData: Array<ImageData> = [];
  const variantBuffers: Array<Uint32Array> = [];
  const variantHues: Array<Hue | null> = [];
  let i = 0;
  const palettes = new Map<number, Array<number>>();
  for (const [variant, palette] of inputVariants) {
    const newImageData = createImageData(
      new Uint8ClampedArray(size * 4),
      width
    );
    if (typeof palette !== 'number') {
      for (const [key, value] of palette) {
        const k = hexToUint32LE(key);
        const v = palettes.get(k) || [];
        v[i] = hexToUint32LE(value);
        palettes.set(k, v);
      }
    }
    variantNames.push(variant);
    variantImageData.push(newImageData);
    variantBuffers.push(new Uint32Array(newImageData.data.buffer));
    variantHues.push(typeof palette === 'number' ? palette : null);

    i++;
  }
  const variantCount = variantNames.length;
  const staticColorNumbers = staticColors
    ? new Set([...staticColors].map(hexToUint32LE))
    : null;

  const trackMissing = !options?.ignoreMissing;
  const missing = new Set<[VariantName, number]>();
  const colorCacheKeys = new Uint32Array(maxCachedColors * 2);
  const colorCacheValues = new Uint32Array(
    colorCacheKeys.length * variantCount
  );
  const colorCacheMask = colorCacheKeys.length - 1;
  let colorCacheSize = 0;
  const uncachedColors = new Uint32Array(variantCount);
  const buffer0 = variantBuffers[0];
  const buffer1 = variantBuffers[1];
  const buffer2 = variantBuffers[2];
  const buffer3 = variantBuffers[3];
  const buffer4 = variantBuffers[4];
  const buffer5 = variantBuffers[5];
  const buffer6 = variantBuffers[6];
  const buffer7 = variantBuffers[7];
  for (let index = 0; index < size; index++) {
    const originalColor = imageData[index];
    const a = (originalColor >> 24) & 0xff;
    if (a === 0) {
      continue;
    }

    const key = originalColor >>> 0;
    const slot = colorCacheSlot(colorCacheKeys, key, colorCacheMask);
    const cachedKey = colorCacheKeys[slot];
    const canCache = cachedKey === key || colorCacheSize < maxCachedColors;
    const colors = canCache ? colorCacheValues : uncachedColors;
    const colorOffset = canCache ? slot * variantCount : 0;
    if (cachedKey !== key) {
      if (canCache) {
        colorCacheKeys[slot] = key;
        colorCacheSize++;
      }

      const color = originalColor & 0x00ffffff;
      const isStatic = staticColorNumbers?.has(color);
      const paletteColors = !isStatic ? palettes.get(color) : null;

      for (let i = 0; i < variantCount; i++) {
        if (isStatic) {
          colors[colorOffset + i] = originalColor;
          continue;
        }

        const hue = variantHues[i];
        if (hue) {
          colors[colorOffset + i] = (a << 24) | applyHue(color, hue);
        } else if (paletteColors?.[i]) {
          colors[colorOffset + i] = (a << 24) | paletteColors[i];
        } else if (!staticColorNumbers) {
          colors[colorOffset + i] = originalColor;
        } else {
          colors[colorOffset + i] = 0;
          if (trackMissing) {
            missing.add([variantNames[i], color]);
          }
        }
      }
    }

    if (variantCount === 8) {
      const color0 = colors[colorOffset];
      const color1 = colors[colorOffset + 1];
      const color2 = colors[colorOffset + 2];
      const color3 = colors[colorOffset + 3];
      const color4 = colors[colorOffset + 4];
      const color5 = colors[colorOffset + 5];
      const color6 = colors[colorOffset + 6];
      const color7 = colors[colorOffset + 7];
      if (color0) {
        buffer0[index] = color0;
      }
      if (color1) {
        buffer1[index] = color1;
      }
      if (color2) {
        buffer2[index] = color2;
      }
      if (color3) {
        buffer3[index] = color3;
      }
      if (color4) {
        buffer4[index] = color4;
      }
      if (color5) {
        buffer5[index] = color5;
      }
      if (color6) {
        buffer6[index] = color6;
      }
      if (color7) {
        buffer7[index] = color7;
      }
    } else if (variantCount === 2) {
      const color0 = colors[colorOffset];
      const color1 = colors[colorOffset + 1];
      if (color0) {
        buffer0[index] = color0;
      }
      if (color1) {
        buffer1[index] = color1;
      }
    } else {
      for (let i = 0; i < variantCount; i++) {
        const color = colors[colorOffset + i];
        if (color) {
          variantBuffers[i][index] = color;
        }
      }
    }
  }

  if (trackMissing && missing.size) {
    throw new Error(
      `${
        options?.imageName ? `${options.imageName}` : 'Palette Swap'
      }: Missing ${Array.from(
        new Set(
          [...missing].map(
            ([variant, color]) => `'${int32ToHex(color)}' in '${variant}'`
          )
        )
      ).join(', ')}.`
    );
  }

  for (let i = 0; i < variantCount; i++) {
    const name = variantNames[i];
    const existingImage = images?.get(name);
    const existingImageData = existingImage
      ? createCanvasFromImage(existingImage)
          .getContext('2d')
          .getImageData(0, 0, width, height)
      : null;

    const newImageData = variantImageData[i];
    if (!existingImageData || !equals(newImageData, existingImageData)) {
      const newCanvas = createCanvas(width, height);
      newCanvas.getContext('2d').putImageData(newImageData, 0, 0);
      results.set(name, newCanvas);
    }
  }

  return results;
}
