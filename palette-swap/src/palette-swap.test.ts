import { createHash } from "node:crypto";
import { join } from "node:path";
import { createCanvas, loadImage, type Canvas } from "canvas";
import { expect, test } from "vitest";
import paletteSwap, { type HEX, type VariantName } from "./palette-swap";

type Variants = ReadonlyMap<VariantName, ReadonlyMap<HEX, HEX> | number>;

const hashCanvas = (canvas: Canvas) => {
  const context = canvas.getContext("2d");
  const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
  return createHash("sha256").update(imageData.data).digest("hex");
};

const getResult = (
  results: ReadonlyMap<VariantName, Canvas>,
  variant: VariantName,
) => {
  const canvas = results.get(variant);
  expect(canvas).toBeDefined();
  return canvas as Canvas;
};

const yoshiVariants: Variants = new Map<VariantName, ReadonlyMap<HEX, HEX> | number>([
  [
    2,
    new Map<HEX, HEX>([
      ["#006000", "#570061"],
      ["#00a800", "#8c00a8"],
      ["#00f800", "#de00f8"],
    ]),
  ],
  [3, 220],
]);

const yoshiStaticColors = new Set<HEX>([
  "#000000",
  "#903020",
  "#f84020",
  "#f89000",
  "#ff0000",
  "#f8c0a8",
  "#f8e0d0",
  "#f8f8f8",
]);

test("produces stable raw pixels for the README example", async () => {
  const image = await loadImage(join(process.cwd(), "example/Yoshi.png"));
  const results = paletteSwap(image, yoshiVariants, yoshiStaticColors);

  expect([...results.keys()]).toEqual([2, 3]);
  expect(hashCanvas(getResult(results, 2))).toBe(
    "9795432e22e485c3de5cf75cdfc324f22ba0c0775569ca8652ee37d9d19cf687",
  );
  expect(hashCanvas(getResult(results, 3))).toBe(
    "2b22a59d898ca04087e1e4a52530228a438ab1c8010e157b4fb75626089f490b",
  );
});

test("omits variants whose existing image already matches", async () => {
  const image = await loadImage(join(process.cwd(), "example/Yoshi.png"));
  const firstResults = paletteSwap(image, yoshiVariants, yoshiStaticColors);
  const secondResults = paletteSwap(image, yoshiVariants, yoshiStaticColors, firstResults);

  expect(secondResults.size).toBe(0);
});

test("keeps ignored missing colors transparent", () => {
  const canvas = createCanvas(1, 1);
  const context = canvas.getContext("2d");
  const imageData = context.createImageData(1, 1);
  imageData.data.set([255, 0, 0, 255]);
  context.putImageData(imageData, 0, 0);

  const results = paletteSwap(
    canvas,
    new Map([[1, new Map<HEX, HEX>([["#00ff00", "#0000ff"]])]]),
    new Set<HEX>(),
    null,
    { ignoreMissing: true },
  );

  expect(hashCanvas(getResult(results, 1))).toBe(hashCanvas(createCanvas(1, 1)));
});
