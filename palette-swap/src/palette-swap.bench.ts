import { createHash } from "node:crypto";
import { existsSync, readdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { type Canvas, type Image as CanvasImage, loadImage } from "canvas";
import { bench, describe } from "vitest";
import paletteSwap, { type HEX, type VariantName } from "./palette-swap";

type Variants = ReadonlyMap<VariantName, ReadonlyMap<HEX, HEX> | number>;
type BenchmarkImage = readonly [path: string, image: CanvasImage];
type BenchmarkCase = Readonly<{
  images: ReadonlyArray<BenchmarkImage>;
  name: string;
  options: NonNullable<Parameters<typeof paletteSwap>[4]>;
  staticColors: Set<HEX>;
  variants: Variants;
}>;

const root = process.cwd();
const suite = process.env.BENCHMARK_SUITE === "all" ? "all" : "quick";

const hexHash = (buffer: NodeJS.ArrayBufferView) =>
  createHash("sha256").update(buffer).digest("hex").slice(0, 12);

const canvasHash = (canvas: Canvas) => {
  const context = canvas.getContext("2d");
  const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
  return hexHash(imageData.data);
};

const hashResults = (results: ReadonlyMap<VariantName, Canvas>) =>
  hexHash(
    Buffer.from(
      [...results]
        .map(([name, canvas]) => `${name}:${canvas.width}x${canvas.height}:${canvasHash(canvas)}`)
        .join("|"),
    ),
  );

const loadImages = async (paths: ReadonlyArray<string>) =>
  Promise.all(paths.map(async (path) => [path, await loadImage(path)] as const));

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

const variant0 = new Map<HEX, HEX>([
  ["#3d002d", "#433428"],
  ["#690027", "#423327"],
  ["#7e114f", "#6e5541"],
  ["#c32621", "#917a5c"],
  ["#ff3f35", "#b3a07c"],
  ["#ff4f3c", "#ccc19b"],
  ["#ff8578", "#ccc19b"],
  ["#ffbbb4", "#e6dbbb"],
]);

const variant5 = new Map<HEX, HEX>([
  ["#3d002d", "#063326"],
  ["#690027", "#366b00"],
  ["#7e114f", "#477e11"],
  ["#c32621", "#5ea318"],
  ["#ff3f35", "#bbd742"],
  ["#ff4f3c", "#b4d742"],
  ["#ff8578", "#bfd672"],
  ["#ffbbb4", "#daffb5"],
]);

const playerVariants: Variants = new Map<VariantName, ReadonlyMap<HEX, HEX> | number>([
  [0, variant0],
  [1, 325],
  [2, 30],
  [3, 210],
  [4, 270],
  [5, variant5],
  [6, 5],
  [7, 165],
]);

const playerStaticColors = new Set<HEX>([
  "#000000",
  "#202e37",
  "#2a2a2a",
  "#312c29",
  "#394a50",
  "#3c4b73",
  "#4336f7",
  "#4d4744",
  "#584e49",
  "#6b5a63",
  "#76655d",
  "#79888f",
  "#887870",
  "#8f7860",
  "#908880",
  "#ad7757",
  "#afc3cc",
  "#b8c8b0",
  "#bf9273",
  "#c09473",
  "#c8b8a8",
  "#cea783",
  "#cebdad",
  "#d7b594",
  "#e09455",
  "#f8f8f8",
  "#fbc774",
  "#ffffff",
]);

const athenaVariantsPath = () => {
  const candidates = [
    process.env.ATHENA_CRISIS_ART && join(process.env.ATHENA_CRISIS_ART, "variants"),
    resolve(root, "../athena-crisis/art/variants"),
    resolve(root, "../../athena-crisis/art/variants"),
  ].filter((path): path is string => !!path);
  return candidates.find((path) => existsSync(path));
};

const selectedAthenaImages = () => {
  const variantsPath = athenaVariantsPath();
  if (!variantsPath) {
    return [];
  }

  const quick = ["Units-Sniper.png", "Units-Dragon.png", "Decorators.png", "NavalExplosion.png"];
  const all = readdirSync(variantsPath)
    .filter((file) => file.endsWith(".png"))
    .sort();

  return (suite === "all" ? all : quick)
    .filter((file) => existsSync(join(variantsPath, file)))
    .map((file) => join(variantsPath, file));
};

const createCases = async (): Promise<Array<BenchmarkCase>> => {
  const cases: Array<BenchmarkCase> = [
    {
      images: await loadImages([join(root, "example/Yoshi.png")]),
      name: "readme-yoshi",
      options: {},
      staticColors: yoshiStaticColors,
      variants: yoshiVariants,
    },
  ];

  const athenaImages = selectedAthenaImages();
  if (athenaImages.length) {
    cases.push({
      images: await loadImages(athenaImages),
      name: `athena-${suite}`,
      options: { ignoreMissing: true },
      staticColors: playerStaticColors,
      variants: playerVariants,
    });
  }

  return cases;
};

const runCase = (
  { images, options, staticColors, variants }: BenchmarkCase,
  includeHash: boolean,
) => {
  const hash = createHash("sha256");
  let pixelCount = 0;
  let resultCount = 0;

  for (const [path, image] of images) {
    const results = paletteSwap(image, variants, staticColors, null, {
      imageName: path,
      ...options,
    });
    pixelCount += image.width * image.height;
    resultCount += results.size;
    if (includeHash) {
      hash.update(hashResults(results));
    }
  }

  return {
    hash: includeHash ? hash.digest("hex").slice(0, 12) : null,
    pixelCount,
    resultCount,
  };
};

const cases = await createCases();

describe(`paletteSwap (${suite})`, () => {
  for (const item of cases) {
    const verification = runCase(item, true);
    const megapixels = (verification.pixelCount / 1_000_000).toFixed(2);
    console.log(
      [
        item.name,
        `${item.images.length} image(s)`,
        `${verification.resultCount} result(s)`,
        `${megapixels} MP`,
        `hash ${verification.hash}`,
      ].join(" | "),
    );

    bench(item.name, () => {
      runCase(item, false);
    });
  }
});
