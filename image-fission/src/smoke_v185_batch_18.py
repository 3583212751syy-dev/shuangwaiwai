"""v185 批量裂变 — 桌面 图裂变测试图/ 下 18 张（已排除 6 张同主题 + 1 张违规大麻）

- 每张 ref 用 v164 5区域提示模板（保留主体+可改角度/数量/小元素）
- 共用 v147 技术管线：ProteusV0.4 / IPA 0.18 style / add-detail-xl 1.0 /
  Canny 0.25 / Tile 0.60 / KSampler 24+20 / 4x_NMKD-Siax upscaler
- 3 张侵权文字图（FIREBALL/BACARDI/ARMED FORCES）跑完后做 PIL 烧字 1:1 复刻换词

用法：python smoke_v185_batch_18.py  (按 REFS 跑全部 18 张)
"""
import time, requests, sys, os, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI = "http://127.0.0.1:8188"

SEED = 700501
CKPT = "ProteusV0.4.safetensors"
DENOISE = 0.80
IPA_WEIGHT = 0.18
LORA_DETAIL = 1.0
MEGA_PIXELS = 1.2
CN_CANNY = "controlnet-canny-sdxl-1.0.fp16.safetensors"
CN_TILE = "controlnet-tile-sdxl-1.0.safetensors"
CANNY_STRENGTH = 0.25
TILE_STRENGTH = 0.60
REGION_STRENGTH_SCALE = 0.55

NEG_BASE = (
    "frame, border, white border, edge border, box outline, padding, margin, empty corners, letterbox, "
    "text, letters, words, writing, typography, signature, caption, label, paragraph, alphabet, "
    "banner, banner inscription, engraved lettering, runic text, readable text, glyphs, calligraphy, "
    "3d, photographic, painterly, illustration by child, beginner drawing, "
    "blur, soft focus, smooth shading, smudge, soft airbrush, watercolor, "
    "bleeding borders, fused elements, melted edges, soft halo, gradient transition, "
    "out of focus, dreamy, ethereal, foggy, hazy, low contrast, pastel, "
    "small subject, distant view, zoomed out, far away, miniature, tiny, "
    "noise, grain, pixelated, jagged edges, aliasing, duplicate image, exact copy, watermark, "
    "mutated, malformed, deformed anatomy, broken bones, extra limbs, missing claws, "
    "melted, fused, smudged, bleeding, water damaged, anatomically incorrect, "
    "extra wings, asymmetric error, garbled forms, nonsense, AI artifact, tiling, repeating pattern, "
    "clipping through other objects, intersecting geometry, overlapping errors, "
    "elements touching each other, elements touching neighboring elements, "
    "adjacent objects merged, adjacent objects blending into each other, "
    "crowded center, cluttered middle area, "
    "low contrast between adjacent elements, no clear black separating outline, "
    "new colors, different color palette, extra colors, color shift"
)

COHESIVE = (
    "cohesive with the rest of the design, anatomically connected, "
    "no floating disconnected parts, no clipping through other elements, "
    "natural overlap hierarchy, fits the overall composition"
)

# ========== 18 张 REFS（已排除 6 张同主题 + 1 张违规大麻） ==========
REFS = [

    # === 1. 嘻哈风卡通宝宝 ===
    {
        "id": "hiphop_baby", "ref_img": "test_0c4719b1bdd76d452559fc4586a6a3cd.jpg",
        "global_pos": ("bold hip-hop cartoon chibi baby illustration, pure black background, "
                       "oversized head, oversized eyes, vivid neon splashes, "
                       "black leather jacket, silver chains, oversized gold ring, "
                       "tiny diamond stud earrings, skateboard, "
                       "neon color palette: hot pink, electric blue, lime green, sunshine yellow, purple, "
                       "no text, no letters, no words, no banner anywhere, "
                       "cohesive composition, all elements connected and spatially consistent"),
        "regions": [
            {"x": 0.20, "y": 0.05, "w": 0.60, "h": 0.65, "strength": 1.30,
             "prompt": ("ONE large chibi baby character centered, "
                        "oversized bald head, two HUGE round eyes, tiny nose, puckered lips, "
                        "single raised eyebrow, "
                        "wearing oversized black leather biker jacket with silver studs, "
                        "multiple gold chains around neck, oversized gold dollar-sign ring, "
                        "tiny diamond stud in one ear, "
                        "skin tone warm peach, jacket glossy black, "
                        "background burst of NEON paint splashes radiating from baby. " + COHESIVE)},
            {"x": 0.05, "y": 0.40, "w": 0.50, "h": 0.55, "strength": 1.20,
             "prompt": ("the baby's right hand holding a pink skateboard tilted 30 degrees, "
                        "skateboard deck with neon blue wheels, "
                        "wearing baggy light-blue denim shorts and chunky white-and-pink sneakers, "
                        "sneaker laces tied, white socks visible. " + COHESIVE)},
            {"x": 0.50, "y": 0.35, "w": 0.45, "h": 0.60, "strength": 1.15,
             "prompt": ("the baby's left hand extended outward, fingers splayed, "
                        "showing a chunky silver wristwatch, "
                        "background splash of cyan and lime green paint droplets exploding outward. " + COHESIVE)},
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.10,
             "prompt": ("explosion of vivid neon paint splashes radiating outward behind the baby, "
                        "neon hot pink, electric blue, lime green, sunshine yellow, purple, orange droplets, "
                        "drips and splatters of paint in all directions, "
                        "abstract chaotic energy, "
                        "PURE BLACK background between splashes, no text anywhere. " + COHESIVE)},
        ],
    },

    # === 2. 橙色佩斯利腰果花纹 4 格 ===
    {
        "id": "orange_paisley", "ref_img": "test_13c8b7bf8dae757e6c2d4b3d6a860f9d.jpg",
        "global_pos": ("ornate orange paisley bandana print pattern, "
                       "warm orange and white and dark brown and deep red color blocks, "
                       "traditional persian paisley teardrop motifs, "
                       "fabric bandana quality, repeatable seamless pattern feel, "
                       "no text, no letters, no words anywhere, "
                       "sharp clean edges, intricate detail"),
        "regions": [
            {"x": 0.00, "y": 0.00, "w": 0.50, "h": 0.50, "strength": 1.30,
             "prompt": ("a top-left quadrant panel with ornate paisley motifs in warm orange tones, "
                        "a single large central PAISLEY teardrop motif outlined in dark brown with cream inside, "
                        "surrounded by smaller teardrops and floral curls, "
                        "deep red accents, warm orange field. " + COHESIVE)},
            {"x": 0.50, "y": 0.00, "w": 0.50, "h": 0.50, "strength": 1.30,
             "prompt": ("a top-right quadrant panel, deep coral red field with cream paisley teardrops scattered, "
                        "two large central teardrop motifs, smaller teardrops forming diamond clusters, "
                        "warm orange accents. " + COHESIVE)},
            {"x": 0.00, "y": 0.50, "w": 0.50, "h": 0.50, "strength": 1.30,
             "prompt": ("a bottom-left quadrant panel, persian rug medallion with central floral rosette, "
                        "warm orange and dark brown, "
                        "scattered paisley teardrops radiating outward symmetrically. " + COHESIVE)},
            {"x": 0.50, "y": 0.50, "w": 0.50, "h": 0.50, "strength": 1.30,
             "prompt": ("a bottom-right quadrant panel, warm orange field, "
                        "central pair of mirrored paisley teardrops like wings, "
                        "cream and dark brown outlining, "
                        "small teardrops and dots scattered, deep red accents. " + COHESIVE)},
            {"x": 0.00, "y": 0.45, "w": 1.0, "h": 0.10, "strength": 1.05,
             "prompt": ("a thin decorative band running horizontally between top and bottom quadrants, "
                        "repeating paisley motifs in cream and dark brown on warm orange, "
                        "intricate ornamental edge. " + COHESIVE)},
        ],
    },

    # === 3. RACING 赛车 T恤 ===
    {
        "id": "racing", "ref_img": "test_184432b34a4787fbed628b3b986b37a2.jpg",
        "global_pos": ("bold motorsport racing t-shirt graphic design, "
                       "red and white and black and silver color blocks, "
                       "horizontal red bands top and bottom with white center band, "
                       "motorsport typography in silver and black, "
                       "checkered flag motif, stopwatch motif, "
                       "no text, no letters, no words, no banner anywhere, "
                       "cohesive composition, all elements connected and spatially consistent"),
        "regions": [
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 0.30, "strength": 1.25,
             "prompt": ("TOP HORIZONTAL RED BAND spanning full width, "
                        "small black-and-white checkered flag motif in upper-left corner, "
                        "small circular stopwatch motif in upper-right corner, "
                        "matte crimson red field. " + COHESIVE)},
            {"x": 0.10, "y": 0.28, "w": 0.80, "h": 0.15, "strength": 1.30,
             "prompt": ("a CENTRAL BOLD ITALIC FUTURISTIC RACING TYPEFORM shape, "
                        "thick chrome-silver beveled letterforms with sharp angled edges, "
                        "black outline, italic forward-slant aggressive racing posture, "
                        "centered horizontally, occupying middle of design. " + COHESIVE)},
            {"x": 0.10, "y": 0.42, "w": 0.80, "h": 0.12, "strength": 1.25,
             "prompt": ("a thin double-line black underline motif below the central racing typeform, "
                        "small angled rectangular shapes flanking it like racing flags. " + COHESIVE)},
            {"x": 0.20, "y": 0.55, "w": 0.60, "h": 0.10, "strength": 1.20,
             "prompt": ("a SECONDARY bolder letterform below the flag motif, "
                        "thick black serif-like block letters with chrome silver highlight, "
                        "centered. " + COHESIVE)},
            {"x": 0.00, "y": 0.65, "w": 1.0, "h": 0.12, "strength": 1.20,
             "prompt": ("a thin TRACK shape running horizontally, "
                        "two stacked line motifs like a racing track straight, "
                        "black on white center band. " + COHESIVE)},
            {"x": 0.00, "y": 0.78, "w": 1.0, "h": 0.22, "strength": 1.25,
             "prompt": ("BOTTOM HORIZONTAL RED BAND spanning full width, "
                        "small white double-line italicized typeform shape centered, "
                        "matte crimson red field. " + COHESIVE)},
        ],
    },

    # === 4. 黑暗骑士跪地 AMEN ===
    {
        "id": "dark_knight", "ref_img": "test_3a300c32794aeea08f8abb2517f3afe1.jpg",
        "global_pos": ("bold gothic dark knight illustration, "
                       "monochrome black and gray and silver, "
                       "knight in full plate armor kneeling with head bowed, "
                       "vertical sword planted point-down in front, "
                       "dramatic side lighting from upper-left, "
                       "stencil-style weathered look, "
                       "no text, no letters, no words, no banner anywhere, "
                       "cohesive composition, all elements connected and spatially consistent"),
        "regions": [
            {"x": 0.10, "y": 0.10, "w": 0.45, "h": 0.55, "strength": 1.30,
             "prompt": ("a KNIGHT in full gothic plate armor kneeling on the LEFT side, "
                        "head bowed down with helmet visor closed, "
                        "both gauntleted hands gripping the hilt of a vertical sword "
                        "planted point-down into the ground in front of him, "
                        "shoulder pauldrons spiked, "
                        "torn cape draped behind, armor weathered gray-silver, "
                        "dramatic shadow under helm. " + COHESIVE)},
            {"x": 0.20, "y": 0.05, "w": 0.30, "h": 0.70, "strength": 1.25,
             "prompt": ("a TALL VERTICAL SWORD planted point-down in front of the knight, "
                        "long double-edged blade catching silver light, "
                        "ornate cross-guard with quillon tips, "
                        "round pommel wrapped in dark leather, "
                        "blade length spans from upper-mid down to ground. " + COHESIVE)},
            {"x": 0.55, "y": 0.15, "w": 0.40, "h": 0.55, "strength": 1.20,
             "prompt": ("a LARGE SHIELD or banner motif standing vertically on the right side, "
                        "weathered cracked gray-silver shield with dark central boss, "
                        "tattered edges, "
                        "subtle ornamental rivets along the rim. " + COHESIVE)},
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.10,
             "prompt": ("dramatic spotlight from upper-left casting deep shadows, "
                        "weathered cracked concrete texture in the background, "
                        "gothic cathedral atmosphere, "
                        "PURE BLACK shadows and corners, "
                        "monochrome black and gray and silver ONLY. " + COHESIVE)},
            {"x": 0.10, "y": 0.75, "w": 0.80, "h": 0.20, "strength": 1.15,
             "prompt": ("the bottom area with bold stencil-style angular typeform shapes, "
                        "thick block letterforms in chrome-silver, "
                        "centered, dominant on the lower portion. " + COHESIVE)},
        ],
    },

    # === 5. FIREBALL 火焰骷髅（侵权→PIL 烧字换词）===
    {
        "id": "fireball_skull", "ref_img": "test_581f43423ef2d71d4447c0f634411138.jpg",
        "global_pos": ("bold death-metal flaming skull illustration, "
                       "black and warm gold and orange and yellow color blocks, "
                       "stylized golden skull with flame-like tribal engravings, "
                       "horizontally split: top half warm amber, bottom half pure black, "
                       "no text, no letters, no words, no banner anywhere, "
                       "cohesive composition, all elements connected and spatially consistent"),
        "regions": [
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 0.20, "strength": 1.20,
             "prompt": ("TOP HORIZONTAL BAND in warm amber-orange, "
                        "PURE solid amber color field, no motifs, "
                        "split horizon line. " + COHESIVE)},
            {"x": 0.05, "y": 0.05, "w": 0.55, "h": 0.95, "strength": 1.35,
             "prompt": ("a LARGE STYLIZED GOLD SKULL on the LEFT side, "
                        "viewed from front-side 3/4 angle, "
                        "skull surface engraved with flame-like tribal filigree patterns, "
                        "rich metallic warm-gold color, "
                        "deep eye sockets, "
                        "upper jaw teeth visible, "
                        "lower jaw hidden in the black section, "
                        "facing right. " + COHESIVE)},
            {"x": 0.55, "y": 0.20, "w": 0.40, "h": 0.50, "strength": 1.30,
             "prompt": ("a SMALL CIRCULAR EMBLEM on the upper-right, "
                        "circular badge with a tiny stylized creature silhouette inside, "
                        "warm-gold metallic, "
                        "below it a small horizontal thin rectangular bar. " + COHESIVE)},
            {"x": 0.62, "y": 0.32, "w": 0.10, "h": 0.65, "strength": 1.30,
             "prompt": ("a TALL VERTICAL WORDMARK on the right side, "
                        "thick bold warm-gold metallic block letterforms, "
                        "stacked vertically reading downward, "
                        "occupying most of the right edge. " + COHESIVE)},
            {"x": 0.00, "y": 0.20, "w": 1.0, "h": 0.80, "strength": 1.10,
             "prompt": ("PURE BLACK background field below the amber horizon, "
                        "high contrast against the gold skull, "
                        "torn jagged horizon edge between amber and black, "
                        "no other motifs. " + COHESIVE)},
        ],
    },

    # === 6. BACARDI 蝙蝠 logo（侵权→PIL 烧字换词）===
    {
        "id": "bat_logo", "ref_img": "test_6978fabda2cc99629fa9e81f802762d3.jpg",
        "global_pos": ("bold vintage bat emblem logo illustration, "
                       "warm lavender purple and deep purple and black color blocks, "
                       "circular emblem with central bat silhouette, "
                       "ornamental ribbon banner above, "
                       "weathered cracked texture, "
                       "no text, no letters, no words, no banner inscription anywhere, "
                       "cohesive composition, all elements connected and spatially consistent"),
        "regions": [
            {"x": 0.30, "y": 0.05, "w": 0.40, "h": 0.18, "strength": 1.20,
             "prompt": ("an ORNAMENTAL CURVED RIBBON BANNER at the top, "
                        "arching over the central emblem, "
                        "warm lavender purple background with black border, "
                        "decorative scroll ends, "
                        "no inscription inside, "
                        "weathered texture. " + COHESIVE)},
            {"x": 0.30, "y": 0.20, "w": 0.40, "h": 0.45, "strength": 1.35,
             "prompt": ("a CENTRAL CIRCULAR EMBLEM, "
                        "deep purple circular medallion, "
                        "a black bat silhouette centered inside, "
                        "bat wings fully spread showing membranous structure, "
                        "small pointed ears, "
                        "wings touching the inner edge of the circle, "
                        "vintage 1860s emblem style. " + COHESIVE)},
            {"x": 0.30, "y": 0.32, "w": 0.18, "h": 0.10, "strength": 1.10,
             "prompt": ("a SMALL DATE-LIKE MARKER on the LEFT of the circle, "
                        "four characters stacked in a row, "
                        "thin black serif-like letterforms on warm lavender, "
                        "vintage emblem stamp style. " + COHESIVE)},
            {"x": 0.52, "y": 0.32, "w": 0.18, "h": 0.10, "strength": 1.10,
             "prompt": ("a SMALL DATE-LIKE MARKER on the RIGHT of the circle, "
                        "four characters stacked in a row, "
                        "thin black serif-like letterforms on warm lavender. " + COHESIVE)},
            {"x": 0.10, "y": 0.55, "w": 0.80, "h": 0.15, "strength": 1.30,
             "prompt": ("a LARGE BOLD WORDMARK below the emblem, "
                        "thick black serif block letterforms, "
                        "centered, "
                        "occupying the middle horizontal band. " + COHESIVE)},
            {"x": 0.20, "y": 0.68, "w": 0.60, "h": 0.10, "strength": 1.25,
             "prompt": ("a SMALL SECONDARY WORDMARK below the large wordmark, "
                        "slimmer italic black letterforms with a tiny circle-mark after, "
                        "centered. " + COHESIVE)},
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.10,
             "prompt": ("weathered cracked lavender purple paper background, "
                        "subtle scratched texture across the surface, "
                        "uniform light lavender field, "
                        "PURE lavender only. " + COHESIVE)},
        ],
    },

    # === 7. 暴龙+熔岩月 ===
    {
        "id": "lava_dino", "ref_img": "test_754bd0928c6b51ccdb66f161aa411f2f.jpg",
        "global_pos": ("bold prehistoric dinosaur battle illustration, "
                       "molten lava orange and pure black and teal and crimson color blocks, "
                       "two dinosaurs facing each other, "
                       "giant molten lava moon with dinosaur skull silhouette, "
                       "dramatic volcanic atmosphere, "
                       "no text, no letters, no words, no banner anywhere, "
                       "cohesive composition, all elements connected and spatially consistent"),
        "regions": [
            {"x": 0.20, "y": 0.05, "w": 0.60, "h": 0.55, "strength": 1.30,
             "prompt": ("a GIANT CIRCULAR MOLTEN LAVA MOON in the upper-center, "
                        "bright molten orange-yellow surface with cracks, "
                        "darker crust lines, "
                        "inside the moon a darker dinosaur-skull silhouette, "
                        "rays of heat radiating outward. " + COHESIVE)},
            {"x": 0.10, "y": 0.45, "w": 0.40, "h": 0.50, "strength": 1.35,
             "prompt": ("a LARGE TYRANNOSAURUS REX dinosaur facing right, "
                        "muscular hind legs, tiny forelimbs, "
                        "open mouth roaring showing sharp teeth, "
                        "dark olive-gray scaly skin, "
                        "viewed from front-side angle. " + COHESIVE)},
            {"x": 0.50, "y": 0.50, "w": 0.40, "h": 0.45, "strength": 1.30,
             "prompt": ("a SECOND TYRANNOSAURUS REX facing left, "
                        "mirror pose to the first T-rex, "
                        "open mouth also roaring, "
                        "two dinosaurs facing each other in combat. " + COHESIVE)},
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.10,
             "prompt": ("PURE BLACK and dark teal night sky background, "
                        "scattered tiny orange lava embers floating in the air, "
                        "crimson glow on the horizon behind the dinosaurs, "
                        "dramatic volcanic eruption lighting. " + COHESIVE)},
            {"x": 0.00, "y": 0.85, "w": 1.0, "h": 0.15, "strength": 1.10,
             "prompt": ("a FOREGROUND ROCKY CLIFF with lava cracks, "
                        "dark gray basalt with bright orange molten veins, "
                        "silhouetted jagged cliff edge, "
                        "where the dinosaurs stand. " + COHESIVE)},
        ],
    },

    # === 8. 黑绿泼墨 LIMITLESS ===
    {
        "id": "limitless_splash", "ref_img": "test_793cf2eb0c1dd7603cd043b163ab4935.jpg",
        "global_pos": ("bold athletic splash ink design, "
                       "pure black and forest green and emerald green color blocks, "
                       "diagonal paint splash brushstroke from lower-left to upper-right, "
                       "small neon green typographic mark in the upper area, "
                       "abstract high-contrast athletic energy, "
                       "no text, no letters, no words, no banner anywhere, "
                       "cohesive composition, all elements connected and spatially consistent"),
        "regions": [
            {"x": 0.00, "y": 0.10, "w": 1.0, "h": 0.90, "strength": 1.30,
             "prompt": ("a BOLD DIAGONAL PAINT-SPLASH BRUSHSTROKE running from lower-left to upper-right, "
                        "thick forest green paint with rough textured edges, "
                        "individual bristle marks visible, "
                        "splatters and droplets breaking off the main stroke, "
                        "covering roughly 60 percent of the canvas. " + COHESIVE)},
            {"x": 0.30, "y": 0.30, "w": 0.40, "h": 0.10, "strength": 1.25,
             "prompt": ("a SMALL NEON GREEN TYPOGRAPHIC MARK in the upper-center area, "
                        "thin neon green letterforms on black background, "
                        "futuristic monospace style, "
                        "small and minimal, "
                        "occupying only a small portion of upper-center. " + COHESIVE)},
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.10,
             "prompt": ("PURE BLACK background covering all non-splash regions, "
                        "high contrast against the green splash, "
                        "no other motifs, no gradients. " + COHESIVE)},
            {"x": 0.00, "y": 0.05, "w": 0.30, "h": 0.10, "strength": 1.05,
             "prompt": ("a SECONDARY TINY NEON GREEN DETAIL MARK in the upper-left corner, "
                        "single tiny dot or micro-mark, "
                        "matching the green color of the splash. " + COHESIVE)},
        ],
    },

    # === 9. 橙绿条+松树轮廓 ===
    {
        "id": "orange_pines", "ref_img": "test_7b445250f19d58a07b612500bb43be1d.jpg",
        "global_pos": ("bold geometric autumn mountain landscape illustration, "
                       "warm orange and bright lime green and pure black color blocks, "
                       "two vertical lime green stripes top, one horizontal band middle, "
                       "black pine tree forest silhouette at the bottom, "
                       "flat vector style, sharp clean edges, "
                       "no text, no letters, no words, no banner anywhere, "
                       "cohesive composition, all elements connected and spatially consistent"),
        "regions": [
            {"x": 0.10, "y": 0.05, "w": 0.05, "h": 0.45, "strength": 1.20,
             "prompt": ("a THIN VERTICAL BRIGHT LIME GREEN STRIPE on the upper-left, "
                        "with thin dark gray outline on both sides, "
                        "spanning from top down to the horizontal band. " + COHESIVE)},
            {"x": 0.45, "y": 0.05, "w": 0.05, "h": 0.45, "strength": 1.20,
             "prompt": ("a SECOND THIN VERTICAL BRIGHT LIME GREEN STRIPE in the upper-right, "
                        "matching the first stripe, "
                        "with thin dark gray outline, "
                        "spanning top to horizontal band. " + COHESIVE)},
            {"x": 0.00, "y": 0.45, "w": 1.0, "h": 0.06, "strength": 1.25,
             "prompt": ("a HORIZONTAL BRIGHT LIME GREEN BAND running full width, "
                        "with thin dark gray outline above and below, "
                        "separating the upper orange field from the lower orange field. " + COHESIVE)},
            {"x": 0.00, "y": 0.05, "w": 1.0, "h": 0.40, "strength": 1.20,
             "prompt": ("the UPPER ORANGE FIELD, "
                        "warm matte orange, "
                        "uniform flat color, "
                        "no motifs within the field. " + COHESIVE)},
            {"x": 0.00, "y": 0.52, "w": 1.0, "h": 0.30, "strength": 1.20,
             "prompt": ("the LOWER ORANGE FIELD, "
                        "warm matte orange, "
                        "matching the upper field color, "
                        "uniform flat color. " + COHESIVE)},
            {"x": 0.00, "y": 0.80, "w": 1.0, "h": 0.20, "strength": 1.30,
             "prompt": ("a PINE TREE FOREST SILHOUETTE across the bottom, "
                        "multiple pine tree triangular shapes in pure black, "
                        "varying heights, "
                        "trees touching each other forming a continuous forest edge, "
                        "occasional small mountain peak rising up. " + COHESIVE)},
        ],
    },

    # === 10. 暗黑紧身胸衣/束身衣 ===
    {
        "id": "corset_goth", "ref_img": "test_7c79f3bb333c1680e6399d04347ade6c.jpg",
        "global_pos": ("gothic dark green silk corset illustration, "
                       "deep forest green and pure black and silver color blocks, "
                       "lacing detail with silver eyelets, "
                       "silk satin sheen on dark green fabric, "
                       "no text, no letters, no words, no banner anywhere, "
                       "cohesive composition, all elements connected and spatially consistent"),
        "regions": [
            {"x": 0.35, "y": 0.05, "w": 0.30, "h": 0.55, "strength": 1.30,
             "prompt": ("a TALL VERTICAL LACE-UP BODICE / CORSET front panel, "
                        "dark forest green silk satin fabric, "
                        "high sheen reflecting light, "
                        "CENTER LACE-UP detail with black crisscross ribbon threading through "
                        "FIVE PAIRS of small silver metal eyelets, "
                        "tight parallel vertical seams along the bodice, "
                        "fabric folds and creases visible at top and bottom. " + COHESIVE)},
            {"x": 0.00, "y": 0.55, "w": 1.0, "h": 0.45, "strength": 1.20,
             "prompt": ("FLOWING DARK GREEN SILK FABRIC draping downward from the corset, "
                        "loose folds and shadows, "
                        "fabric bunching at the lower section, "
                        "occasional sharp black strap detail emerging from the folds. " + COHESIVE)},
            {"x": 0.65, "y": 0.50, "w": 0.35, "h": 0.50, "strength": 1.10,
             "prompt": ("on the RIGHT side, BLACK STRAP / RIBBON detail, "
                        "dark straps extending into the silk, "
                        "sharp dark accent against the green silk. " + COHESIVE)},
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.05,
             "prompt": ("overall DEEP FOREST GREEN and BLACK atmospheric background, "
                        "subtle smoke-like softness, "
                        "deep dark mood, "
                        "no motifs other than the corset and silk. " + COHESIVE)},
        ],
    },

    # === 11. 黑色战术背心 ===
    {
        "id": "tactical_vest", "ref_img": "test_805913629e34491860a892101da398fe.jpg",
        "global_pos": ("bold black tactical vest illustration, "
                       "pure black and dark gray and silver highlight color blocks, "
                       "front view of tactical body armor with multiple pouches and straps, "
                       "highly detailed product rendering style, "
                       "pure white background, sharp clean edges, "
                       "no text, no letters, no words, no banner anywhere, "
                       "cohesive composition, all elements connected and spatially consistent"),
        "regions": [
            {"x": 0.10, "y": 0.00, "w": 0.80, "h": 0.35, "strength": 1.25,
             "prompt": ("UPPER SHOULDER STRAP SECTION with TWO tall shoulder straps, "
                        "each strap featuring a silver quick-release buckle, "
                        "matte black heavy-duty fabric, "
                        "stitched edges. " + COHESIVE)},
            {"x": 0.05, "y": 0.05, "w": 0.90, "h": 0.40, "strength": 1.30,
             "prompt": ("the UPPER CHEST PLATE of the vest, "
                        "segmented matte black armor panels, "
                        "raised contoured sections for fit, "
                        "small silver rivets along edges, "
                        "central horizontal seam. " + COHESIVE)},
            {"x": 0.10, "y": 0.40, "w": 0.80, "h": 0.20, "strength": 1.25,
             "prompt": ("the MID-SECTION MOLLE WEBBING band, "
                        "horizontal rows of black webbing straps, "
                        "two side-release buckles in center. " + COHESIVE)},
            {"x": 0.05, "y": 0.55, "w": 0.90, "h": 0.20, "strength": 1.25,
             "prompt": ("the LOWER ABDOMEN PLATE area, "
                        "two large contoured front pouches, "
                        "each with velcro closure flap and silver snap button, "
                        "matte black tactical fabric. " + COHESIVE)},
            {"x": 0.30, "y": 0.72, "w": 0.40, "h": 0.20, "strength": 1.25,
             "prompt": ("the LOWER GROIN / DROPSEAT PLATE, "
                        "rounded bottom protective plate, "
                        "matte black contoured shape, "
                        "stitched seams along the curve. " + COHESIVE)},
            {"x": 0.00, "y": 0.00, "w": 0.15, "h": 1.0, "strength": 1.10,
             "prompt": ("the LEFT SIDE PANEL with a single large drop pouch, "
                        "matte black tactical fabric, "
                        "stitched edges. " + COHESIVE)},
            {"x": 0.85, "y": 0.00, "w": 0.15, "h": 1.0, "strength": 1.10,
             "prompt": ("the RIGHT SIDE PANEL with a single large drop pouch, "
                        "mirroring the left side pouch. " + COHESIVE)},
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.05,
             "prompt": ("PURE WHITE uniform background, "
                        "no shadows on background, "
                        "product shot style. " + COHESIVE)},
        ],
    },

    # === 12. 几何 X + 金色狮头 ===
    {
        "id": "x_lion", "ref_img": "test_820294597fc383943ee5758f66539081.jpg",
        "global_pos": ("bold geometric racing X-mark emblem illustration, "
                       "pure black and pure white and forest green and warm gold color blocks, "
                       "large X shape spanning the canvas, "
                       "small golden crowned lion head emblem in the upper area, "
                       "flat vector style, sharp clean edges, "
                       "no text, no letters, no words, no banner anywhere, "
                       "cohesive composition, all elements connected and spatially consistent"),
        "regions": [
            {"x": 0.20, "y": 0.05, "w": 0.60, "h": 0.50, "strength": 1.30,
             "prompt": ("a BOLD LARGE WHITE X shape across the upper-center, "
                        "thick white X with thin black outlines on both sides of each arm, "
                        "arms of the X converging at the center, "
                        "crisp geometric shape. " + COHESIVE)},
            {"x": 0.05, "y": 0.20, "w": 0.30, "h": 0.50, "strength": 1.25,
             "prompt": ("a DIAGONAL FOREST GREEN STRIPE on the LEFT side, "
                        "thick green diagonal band running parallel to the X, "
                        "matte forest green with thin black outlines, "
                        "between two white stripes. " + COHESIVE)},
            {"x": 0.45, "y": 0.20, "w": 0.10, "h": 0.50, "strength": 1.20,
             "prompt": ("a DIAGONAL FOREST GREEN STRIPE on the RIGHT side, "
                        "matching the left stripe, "
                        "thick green band with black outlines. " + COHESIVE)},
            {"x": 0.35, "y": 0.35, "w": 0.30, "h": 0.25, "strength": 1.30,
             "prompt": ("a SMALL GOLDEN CROWNED LION HEAD EMBLEM in the upper-center, "
                        "warm gold metallic, "
                        "lion facing forward, "
                        "small ornate crown on top, "
                        "detailed mane around the face, "
                        "occupying only a small portion of upper-center. " + COHESIVE)},
            {"x": 0.20, "y": 0.55, "w": 0.60, "h": 0.45, "strength": 1.30,
             "prompt": ("the LOWER BOLD WHITE X shape continuing downward, "
                        "thick white X arms converging at lower-center, "
                        "with FOREST GREEN diagonal stripes alongside. " + COHESIVE)},
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.10,
             "prompt": ("PURE BLACK background filling the negative spaces between X arms and stripes, "
                        "uniform flat black field, no gradients, no other motifs. " + COHESIVE)},
        ],
    },

    # === 13. 工业齿轮 DAVIO ===
    {
        "id": "mech_gear", "ref_img": "test_85f5d2f428cf1afb805288932f9a6ac1.jpg",
        "global_pos": ("bold industrial mechanic gear illustration, "
                       "pure black and pure white color blocks ONLY, "
                       "industrial complex gear and pipe factory scene, "
                       "stencil silhouette style with white line-work on black, "
                       "no text, no letters, no words, no banner anywhere, "
                       "cohesive composition, all elements connected and spatially consistent"),
        "regions": [
            {"x": 0.05, "y": 0.10, "w": 0.30, "h": 0.55, "strength": 1.30,
             "prompt": ("a TALL VERTICAL INDUSTRIAL TYPOGRAPHIC MARK on the LEFT side, "
                        "stacked vertical letterforms reading downward, "
                        "distressed grunge stencil style, "
                        "PURE WHITE on pure black. " + COHESIVE)},
            {"x": 0.55, "y": 0.10, "w": 0.20, "h": 0.20, "strength": 1.25,
             "prompt": ("a SMALL EMBLEM in the upper-right, "
                        "crossed wrench-and-screwdriver motif inside a circle, "
                        "PURE WHITE stencil on black, "
                        "below it a thin rectangular bar. " + COHESIVE)},
            {"x": 0.55, "y": 0.30, "w": 0.20, "h": 0.06, "strength": 1.20,
             "prompt": ("a SMALL RECTANGULAR BADGE below the emblem, "
                        "PURE WHITE stencil text on black, "
                        "small horizontal rectangular shape. " + COHESIVE)},
            {"x": 0.00, "y": 0.40, "w": 1.0, "h": 0.55, "strength": 1.25,
             "prompt": ("LOWER HALF COMPLEX INDUSTRIAL SCENE, "
                        "massive interlocking gears, "
                        "spider-like mecha figure standing tall in the center, "
                        "factory pipes and tanks, "
                        "PURE WHITE line-work on pure black background, "
                        "stencil street-art style. " + COHESIVE)},
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.05,
             "prompt": ("PURE BLACK background everywhere except the white line-work motifs, "
                        "high contrast monochrome, "
                        "no other colors anywhere. " + COHESIVE)},
        ],
    },

    # === 14. 黑白棋盘格漩涡 ===
    {
        "id": "checker_vortex", "ref_img": "test_99b27e6a189276f6ccbc6cd3bbd7028b.jpg",
        "global_pos": ("bold black and white checkerboard vortex optical illusion, "
                       "pure black and pure white color blocks ONLY, "
                       "warped checkered grid forming a swirling vortex tunnel, "
                       "op-art 3D illusion effect, sharp edges, "
                       "no text, no letters, no words, no banner anywhere, "
                       "cohesive composition, all elements connected and spatially consistent"),
        "regions": [
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.40,
             "prompt": ("a BLACK AND WHITE CHECKERBOARD PATTERN warped into a swirling vortex tunnel, "
                        "individual checkered squares follow the curvature of the vortex, "
                        "squares near the vortex center are smaller and darker, "
                        "squares at the outer edges are larger and brighter, "
                        "the vortex opening is a deep dark elliptical hole in the center, "
                        "PURE pure black and pure pure white ONLY, "
                        "no gray, no gradient, no other colors, "
                        "op-art perspective tunnel. " + COHESIVE)},
        ],
    },

    # === 15. 灰迷彩+军牌 ARMED FORCES（侵权→PIL 烧字换词）===
    {
        "id": "camo_armed", "ref_img": "test_b78e60de8dfdf44acda99395326a7298.jpg",
        "global_pos": ("bold grayscale urban camouflage with military dog-tag illustration, "
                       "gray and light gray and white and pure black color blocks, "
                       "broken geometric camouflage pattern, "
                       "single military dog-tag hanging from ball chain, "
                       "stencil high-contrast style, "
                       "no text, no letters, no words, no banner inscription anywhere, "
                       "cohesive composition, all elements connected and spatially consistent"),
        "regions": [
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.30,
             "prompt": ("a BROKEN GEOMETRIC URBAN CAMOUFLAGE PATTERN, "
                        "irregular fragmented shapes in varying grays, "
                        "alternating light gray, mid-gray, dark gray, "
                        "sharp edges between fragments, "
                        "flat stencil style, "
                        "no soft gradient. " + COHESIVE)},
            {"x": 0.40, "y": 0.55, "w": 0.20, "h": 0.25, "strength": 1.30,
             "prompt": ("a SINGLE MILITARY DOG-TAG hanging from a ball chain in the center, "
                        "oval metal dog-tag with rounded corners, "
                        "light gray steel, "
                        "small ball chain rising upward out of frame, "
                        "reflective sheen. " + COHESIVE)},
            {"x": 0.20, "y": 0.20, "w": 0.60, "h": 0.18, "strength": 1.30,
             "prompt": ("a SMALL BOLD TYPOGRAPHIC BLOCK in the upper-center area, "
                        "thick bold stencil black letterforms reading horizontally, "
                        "centered above the dog-tag, "
                        "occupying the upper-middle band. " + COHESIVE)},
            {"x": 0.20, "y": 0.40, "w": 0.60, "h": 0.10, "strength": 1.30,
             "prompt": ("a SECONDARY BOLD TYPOGRAPHIC BLOCK in the mid-center area, "
                        "thicker stencil black letterforms reading horizontally, "
                        "centered below the first block. " + COHESIVE)},
        ],
    },

    # === 16. 黑白扑克牌 A ===
    {
        "id": "ace_card", "ref_img": "test_d056ed4ab763fff030d1e4403362e32e.jpg",
        "global_pos": ("bold ornate ace of spades playing card illustration, "
                       "pure black and pure white color blocks ONLY, "
                       "vertically split card design with decorative spade center, "
                       "monochrome high contrast, sharp clean edges, "
                       "no text, no letters, no words, no banner inscription anywhere, "
                       "cohesive composition, all elements connected and spatially consistent"),
        "regions": [
            {"x": 0.00, "y": 0.00, "w": 0.50, "h": 1.0, "strength": 1.25,
             "prompt": ("the LEFT HALF in pure white, "
                        "with a small black capital A letterform and a small spade symbol below it "
                        "in the upper-left corner, "
                        "mirrored by a small spade and tiny V-letterform in the lower-left corner. " + COHESIVE)},
            {"x": 0.50, "y": 0.00, "w": 0.50, "h": 1.0, "strength": 1.25,
             "prompt": ("the RIGHT HALF in pure black, "
                        "with a small white capital A letterform and a small spade outline below it "
                        "in the upper-right corner. " + COHESIVE)},
            {"x": 0.40, "y": 0.30, "w": 0.20, "h": 0.40, "strength": 1.40,
             "prompt": ("a MASSIVE ORNATE BLACK SPADE SHAPE centered on the card, "
                        "decorative baroque filigree inside the spade body, "
                        "pure black with pure white interior scrollwork detail, "
                        "stem of the spade pointing down, "
                        "card vertically split: spade straddling the white-left / black-right divide. " + COHESIVE)},
            {"x": 0.45, "y": 0.05, "w": 0.10, "h": 0.95, "strength": 1.10,
             "prompt": ("a THIN VERTICAL LINE bisecting the card, "
                        "alternating black and white line segments, "
                        "subtle decorative border between the two halves. " + COHESIVE)},
            {"x": 0.55, "y": 0.60, "w": 0.30, "h": 0.20, "strength": 1.20,
             "prompt": ("a BOLD TYPOGRAPHIC BLOCK in the lower-right area, "
                        "PURE WHITE stencil letterforms on black, "
                        "two stacked lines of bold text. " + COHESIVE)},
        ],
    },

    # === 17. PARIS 三色条 ===
    {
        "id": "paris_stripes", "ref_img": "test_eddf45c7da4ea2615035c8d8f8cddf03.jpg",
        "global_pos": ("bold minimalist horizontal tri-color stripe composition, "
                       "pure black and warm cream and bright orange-red color blocks, "
                       "three horizontal bands, "
                       "small typographic mark with a red bar in the center, "
                       "flat vector graphic poster style, "
                       "no text, no letters, no words, no banner inscription anywhere, "
                       "cohesive composition, all elements connected and spatially consistent"),
        "regions": [
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 0.30, "strength": 1.20,
             "prompt": ("the TOP HORIZONTAL BAND in pure black, "
                        "uniform flat black field, "
                        "no motifs within. " + COHESIVE)},
            {"x": 0.00, "y": 0.30, "w": 1.0, "h": 0.32, "strength": 1.20,
             "prompt": ("the MIDDLE HORIZONTAL BAND in warm cream beige, "
                        "uniform flat field, "
                        "soft warm cream tone. " + COHESIVE)},
            {"x": 0.00, "y": 0.62, "w": 1.0, "h": 0.38, "strength": 1.20,
             "prompt": ("the BOTTOM HORIZONTAL BAND in bright orange-red, "
                        "warm vermilion red, "
                        "uniform flat field. " + COHESIVE)},
            {"x": 0.30, "y": 0.32, "w": 0.40, "h": 0.10, "strength": 1.30,
             "prompt": ("a SMALL CENTERED TYPOGRAPHIC MARK on the cream band, "
                        "thin sans-serif capital letterforms, "
                        "centered between two stacked marks, "
                        "followed by a small bright red rectangular bar to the right, "
                        "PURE minimalist mark. " + COHESIVE)},
        ],
    },

    # === 18. 鹰+骷髅+火焰+铁链（Pinterest 2）===
    {
        "id": "eagle_skull_chain", "ref_img": "test_Pinterest_2.jpg",
        "global_pos": ("bold death-metal eagle skull and chains illustration, "
                       "pure black and pure white and crimson orange color blocks, "
                       "central eagle perched on top of a pile of skulls, "
                       "iron chains wrapping around the skulls, "
                       "flames and embers radiating outward, "
                       "no text, no letters, no words, no banner inscription anywhere, "
                       "cohesive composition, all elements connected and spatially consistent"),
        "regions": [
            {"x": 0.30, "y": 0.00, "w": 0.40, "h": 0.20, "strength": 1.30,
             "prompt": ("a SMALL FLYING EAGLE in the upper-center, "
                        "wings spread, diving downward, "
                        "small clawed feet extended, "
                        "dark gray and white feathers. " + COHESIVE)},
            {"x": 0.10, "y": 0.15, "w": 0.80, "h": 0.35, "strength": 1.35,
             "prompt": ("a LARGE CENTRAL EAGLE perched on the central skull, "
                        "wings fully spread upward and outward, "
                        "large detailed white and gray feathers, "
                        "yellow beak open, "
                        "sharp talons gripping the skull, "
                        "front-facing pose. " + COHESIVE)},
            {"x": 0.15, "y": 0.45, "w": 0.70, "h": 0.40, "strength": 1.30,
             "prompt": ("a PILE OF THREE SKULLS at the center-bottom, "
                        "central large skull with two smaller skulls on either side, "
                        "white skulls with deep black eye sockets and nasal cavity, "
                        "cracked teeth visible, "
                        "gritty weathered surface. " + COHESIVE)},
            {"x": 0.30, "y": 0.30, "w": 0.40, "h": 0.55, "strength": 1.20,
             "prompt": ("a CRIMSON ORANGE FLAME BURST rising from behind the central skull, "
                        "large flowing flame tongues spreading outward, "
                        "orange and red-orange fire, "
                        "occasional small embers floating in the air, "
                        "radiating outward behind the eagle. " + COHESIVE)},
            {"x": 0.00, "y": 0.50, "w": 1.0, "h": 0.50, "strength": 1.15,
             "prompt": ("IRON CHAINS wrapping around the skulls, "
                        "heavy black chain links draping across the skulls, "
                        "small chain segments dangling at the bottom edge, "
                        "crisp metallic black links. " + COHESIVE)},
            {"x": 0.00, "y": 0.00, "w": 1.0, "h": 1.0, "strength": 1.05,
             "prompt": ("PURE BLACK background behind all elements, "
                        "with sparse small crimson orange ember splatters scattered in the air, "
                        "high contrast death-metal atmosphere. " + COHESIVE)},
        ],
    },

]


def scaled_region_strengths(ref):
    return [{**r, "strength": r["strength"] * REGION_STRENGTH_SCALE} for r in ref["regions"]]


def build(ref, seed):
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": ref["ref_img"]}}
    g["3"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["2", 0], "upscale_method": "lanczos", "megapixels": MEGA_PIXELS, "resolution_steps": 64}}
    g["4"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}}

    g["5"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {"model": ["1", 0], "preset": "PLUS (high strength)"}}
    g["6"] = {"class_type": "IPAdapterAdvanced", "inputs": {
        "model": ["1", 0], "ipadapter": ["5", 1], "image": ["3", 0],
        "weight": IPA_WEIGHT, "weight_type": "style transfer",
        "combine_embeds": "average", "start_at": 0.0, "end_at": 0.85,
        "noise": 0.05, "embeds_scaling": "V only"}}
    g["7"] = {"class_type": "LoraLoader", "inputs": {
        "model": ["6", 0], "clip": ["1", 1],
        "lora_name": "add-detail-xl.safetensors",
        "strength_model": LORA_DETAIL, "strength_clip": LORA_DETAIL}}

    g["20"] = {"class_type": "CannyEdgePreprocessor", "inputs": {
        "image": ["3", 0], "low_threshold": 0.10, "high_threshold": 0.25, "resolution": 1024}}
    g["21"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_CANNY}}
    g["22"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["pg", 0], "control_net": ["21", 0],
        "image": ["20", 0], "strength": CANNY_STRENGTH}}
    g["23"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CN_TILE}}
    g["24"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["22", 0], "control_net": ["23", 0],
        "image": ["3", 0], "strength": TILE_STRENGTH}}

    g["pg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": ref["global_pos"]}}
    g["ng"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": NEG_BASE}}

    region_nodes = []
    for i, r in enumerate(scaled_region_strengths(ref)):
        rk = f"rp{i}"
        g[rk] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": r["prompt"]}}
        sk = f"sa{i}"
        g[sk] = {"class_type": "ConditioningSetAreaPercentage", "inputs": {
            "conditioning": [rk, 0], "width": r["w"], "height": r["h"],
            "x": r["x"], "y": r["y"], "strength": r["strength"]}}
        region_nodes.append(sk)

    comb_in = {"global_cond": ["pg", 0]}
    for i, sk in enumerate(region_nodes):
        comb_in[f"region{i+1}"] = [sk, 0]
    g["comb"] = {"class_type": "RegionalListCombine", "inputs": comb_in}

    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["4", 0], "seed": seed, "steps": 24, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": DENOISE}}
    g["11"] = {"class_type": "KSampler", "inputs": {
        "model": ["7", 0], "positive": ["comb", 0], "negative": ["ng", 0],
        "latent_image": ["10", 0], "seed": seed + 1, "steps": 20, "cfg": 6.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 0.20}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 2]}}
    g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x_NMKD-Siax_200k.pth"}}
    g["14"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["13", 0], "image": ["12", 0]}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": f"v185_{ref['id']}"}}
    return g


def gen(ref, seed, out_base):
    tag = ref["id"]
    out = out_base / f"v185_{tag}.jpg"
    if out.exists() and out.stat().st_size > 100000:
        print(f"  [{tag}] 已存在 {out.stat().st_size/1024/1024:.1f}MB，跳过", flush=True); return True
    if not ref.get("regions"):
        print(f"  [{tag}] 无 regions 配置，跳过", flush=True); return True

    g = build(ref, seed)
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": g, "client_id": f"v185_{int(time.time())}_{tag}"}, timeout=30)
    try:
        j = r.json()
    except Exception:
        j = {}
    if r.status_code != 200 or "error" in j:
        print(f"[ERR] {tag}: {r.status_code} {json.dumps(j)[:1500]}", flush=True); return False
    pid = j.get("prompt_id")
    if not pid:
        print(f"[ERR] {tag}: 无 prompt_id {str(j)[:400]}", flush=True); return False
    print(f"  [{tag}] pid={pid} running...", flush=True)
    for i in range(72):
        time.sleep(5)
        try:
            h = requests.get(f"{COMFYUI}/history/{pid}", timeout=10).json()
            if pid in h:
                rec = h[pid]
                if rec.get("status", {}).get("completed"):
                    imgs = rec.get("outputs", {}).get("15", {}).get("images", [])
                    if imgs:
                        url = f"{COMFYUI}/view?filename={imgs[0]['filename']}&type=output&subfolder={imgs[0].get('subfolder','')}"
                        try:
                            data = requests.get(url, timeout=60).content
                        except Exception as e:
                            print(f"  [{tag}] 取图失败 {e}", flush=True); return False
                        out.write_bytes(data)
                        try:
                            from PIL import Image, ImageFilter
                            im = Image.open(out).convert('RGB')
                            sharp = im.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
                            sharp.save(out, 'JPEG', quality=95, optimize=True)
                            print(f"  [{tag}] USM锐化 {out.stat().st_size/1024/1024:.1f}MB", flush=True)
                        except Exception as e:
                            print(f"  [{tag}] USM失败 原图保留 {e}", flush=True)
                        print(f"  [{tag}] OK {out.stat().st_size/1024/1024:.1f}MB", flush=True); return True
                elif rec.get("status", {}).get("error"):
                    err = rec["status"].get("error")
                    print(f"  [{tag}] COMFY错误 {str(err)[:600]}", flush=True); return False
        except Exception as e:
            print(f"  [{tag}] 轮询异常 {e}", flush=True)
        if i % 6 == 0: print(f"    [{tag}] {i*5}s...", flush=True)
    print(f"  [{tag}] TIMEOUT 跳过重试", flush=True); return False


def main():
    wants = sys.argv[1:] if len(sys.argv) > 1 else [r["id"] for r in REFS]
    out = PROJECT_ROOT / "jobs" / "smoke_v185"
    out.mkdir(parents=True, exist_ok=True)
    ok = 0; fail = 0; skipped = 0
    for want in wants:
        ref = next((r for r in REFS if r["id"] == want), None)
        if not ref:
            print(f"未知 ref_id={want}，可选: {[r['id'] for r in REFS]}"); continue
        print(f"\n=== {want} ===", flush=True)
        ret = gen(ref, SEED, out)
        if ret is True: ok += 1
        else: fail += 1
    print(f"\n=== batch done: ok={ok} fail={fail} skipped={skipped} ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())