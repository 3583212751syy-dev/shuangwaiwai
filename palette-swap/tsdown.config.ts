import { defineConfig } from 'tsdown';

export default defineConfig({
  clean: true,
  dts: true,
  entry: 'src/palette-swap.ts',
  fixedExtension: false,
  format: 'esm',
  outDir: 'lib',
  platform: 'node',
  target: 'node22',
});
