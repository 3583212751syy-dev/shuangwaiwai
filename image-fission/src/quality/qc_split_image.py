"""
qc_split_image.py —— 裂变前后拼图对照 自动量化质检
不依赖 numpy。纯 PIL 跑以下指标:
  1. 全图清晰度  Laplacian 方差（越大越清晰，越小越糊）
  2. 顶部文字区域清晰度（BACARDI / CURSE 行）
  3. 背景颜色稳定性 四角 patch 均色方差 + 主色 vs 裂变 5 色
  4. 背景异常色 检测右半出现但左半没有的异常色相
  5. 整体边缘密度（突出主轮廓是否"糊在一起"）

用法:
    python qc_split_image.py <图像路径>
"""
from __future__ import annotations
import sys, json, math
from collections import Counter
from PIL import Image, ImageFilter, ImageStat


LAP_KERNEL = [-1, -1, -1,
              -1,  8, -1,
              -1, -1, -1]  # 8-neighbor Laplacian (sum=0, scale=1)


def laplacian_variance(im: Image.Image) -> tuple[float, float]:
    """Return (stddev, variance) of Laplacian response. Higher = sharper."""
    g = im.convert("L")
    edges = g.filter(ImageFilter.Kernel((3, 3), LAP_KERNEL, scale=1))
    stat = ImageStat.Stat(edges)
    sd = stat.stddev[0]
    return sd, sd * sd


def mean_rgb(im: Image.Image) -> tuple[int, int, int]:
    stat = ImageStat.Stat(im.convert("RGB"))
    return tuple(int(round(c)) for c in stat.mean)


def quantize_topcolors(im: Image.Image, n: int = 5) -> list[tuple[int, tuple[int, int, int]]]:
    """Quantize to 16 colors, return top-n (count, rgb)."""
    small = im.convert("RGB").resize((160, int(160 * im.size[1] / im.size[0])))
    q = small.quantize(colors=16, method=Image.Quantize.FASTOCTREE).convert("RGB")
    pixels = list(q.getdata())
    cnt = Counter(pixels).most_common(n)
    return [(c, rgb) for rgb, c in cnt]


def corner_patches(im: Image.Image, margin_ratio: float = 0.06, patch_ratio: float = 0.18):
    """Return 4 RGB mean colors from 4 corners, avoiding center badge."""
    w, h = im.size
    # Use a corner band of size patch_ratio * width
    ps = max(20, int(min(w, h) * patch_ratio))
    m = max(0, int(min(w, h) * margin_ratio))
    # take from (m, m) top-left band, etc., but skip center → use corners only
    tl = im.crop((m, m, m + ps, m + ps))
    tr = im.crop((w - m - ps, m, w - m, m + ps))
    bl = im.crop((m, h - m - ps, m + ps, h - m))
    br = im.crop((w - m - ps, h - m - ps, w - m, h - m))
    return [mean_rgb(p) for p in (tl, tr, bl, br)]


def corner_color_spread(corners: list[tuple[int, int, int]]) -> float:
    """RGB stddev across corner means (consistency of background)."""
    rs = [c[0] for c in corners]
    gs = [c[1] for c in corners]
    bs = [c[2] for c in corners]
    def s(xs):
        m = sum(xs) / len(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
    return (s(rs) + s(gs) + s(bs)) / 3.0


def hist_diff(im_left: Image.Image, im_right: Image.Image, bins: int = 32) -> float:
    """Chi-square distance between normalized luminance histograms (0~1, lower=more similar)."""
    def hist(im):
        g = im.convert("L").resize((200, 200))
        h = [0] * bins
        for v in g.getdata():
            h[min(bins - 1, v * bins // 256)] += 1
        s = sum(h) or 1
        return [x / s for x in h]
    a = hist(im_left)
    b = hist(im_right)
    return 0.5 * sum((ai - bi) ** 2 / (ai + bi + 1e-9) for ai, bi in zip(a, b))


def edge_density(im: Image.Image) -> float:
    """Fraction of pixels deemed 'edge' by Sobel-like response."""
    g = im.convert("L")
    sx = g.filter(ImageFilter.Kernel((3, 3), [-1, 0, 1, -2, 0, 2, -1, 0, 1], scale=1))
    sy = g.filter(ImageFilter.Kernel((3, 3), [-1, -2, -1, 0, 0, 0, 1, 2, 1], scale=1))
    px = list(sx.getdata())
    py = list(sy.getdata())
    mag = [abs(a) + abs(b) for a, b in zip(px, py)]
    thr = 60
    return sum(1 for v in mag if v > thr) / len(mag)


def cluster_colors(rgbs: list[tuple[int, int, int]], thr: float = 25.0) -> list[tuple[int, int, int]]:
    """Cluster similar colors (Euclidean RGB distance < thr)."""
    clusters: list[list[tuple[int, int, int]]] = []
    for rgb in rgbs:
        for cl in clusters:
            if math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb, cl[0]))) < thr:
                cl.append(rgb)
                break
        else:
            clusters.append([rgb])
    return [cl[0] for cl in clusters]


def diagnose(d: dict) -> list[str]:
    """Generate red-flag human-readable conclusions."""
    flags = []
    left = d["left"]
    right = d["right"]

    # 清晰度差异
    if right["laplacian_variance"] < left["laplacian_variance"] * 0.75:
        flags.append(
            f"【糊】右半（裂变）整体清晰度方差 {right['laplacian_variance']:.1f}，"
            f"仅左半（原图）{left['laplacian_variance']:.1f} 的 "
            f"{right['laplacian_variance']/max(1e-6,left['laplacian_variance'])*100:.0f}%，"
            f"明显糊于原图"
        )
    elif right["laplacian_variance"] < 100:
        flags.append(
            f"【糊】右半 Laplacian 方差 {right['laplacian_variance']:.1f} 绝对值过低（<100），"
            f"整体偏糊"
        )

    # 文字区域
    rt = right["text_top_laplacian_var"]
    lt = left["text_top_laplacian_var"]
    if rt < lt * 0.7:
        flags.append(
            f"【字体糊在一起】右半顶部文字区域 Laplacian {rt:.1f}，"
            f"左半 {lt:.1f}，裂变后字体清晰度仅原图 "
            f"{rt/max(1e-6,lt)*100:.0f}%，疑似字母粘连/伪影"
        )
    elif rt < 120:
        flags.append(
            f"【字体糊】右半顶部文字 Laplacian {rt:.1f} 偏低，字母笔画偏糊"
        )

    # 背景颜色稳定性
    if right["corner_color_spread"] > left["corner_color_spread"] * 1.6 + 5:
        flags.append(
            f"【背景不一致/乱色】右半四角颜色差异 {right['corner_color_spread']:.1f}，"
            f"左半 {left['corner_color_spread']:.1f}，裂变后背景四角配色漂移，"
            f"出现原图没有的色斑"
        )

    # 主色差异 (容差聚类 + 跨集合匹配，避免 PIL quantize 微抖动造成假阳性)
    left_rgb = cluster_colors([tuple(c["rgb"]) for c in left["top_colors"]])
    right_rgb = cluster_colors([tuple(c["rgb"]) for c in right["top_colors"]])

    def _match(keep: list, ref: list, thr: float = 25.0) -> list:
        """Drop any in `keep` whose nearest in `ref` is within thr."""
        out = []
        for rgb in keep:
            best = min((math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb, r))) for r in ref), default=1e9)
            if best >= thr:
                out.append(rgb)
        return out

    new_added = _match(right_rgb, left_rgb)
    only_left = _match(left_rgb, right_rgb)
    if new_added:
        flags.append(
            f"【背景乱色】右半出现 {len(new_added)} 种左半没有的主色"
            f"（样例：{list(new_added)[:3]}），违背'不出现原图没有的颜色'铁律"
        )

    # 主体局部对比度（检测'糊在一起'：中心徽章区像素值 stddev，越低越糊）
    lcen = left["center_local_contrast"]
    rcen = right["center_local_contrast"]
    if rcen < lcen * 0.7 + 5:
        flags.append(
            f"【主体糊在一起】右半中心徽章区局部对比度 {rcen:.1f} 灰度 stddev，"
            f"左半 {lcen:.1f}，仅 {rcen/max(1e-6,lcen)*100:.0f}%，"
            f"主体与背景/装饰边界不分明"
        )
    elif rcen < 30:
        flags.append(
            f"【主体糊】右半中心徽章区局部灰度 stddev {rcen:.1f} 偏低（<30），"
            f"主体细节扁平化"
        )

    # 直方图差异
    if d["hist_chi2"] > 0.3:
        flags.append(
            f"【整体漂色】左右两半亮度直方图 χ² = {d['hist_chi2']:.3f}（>0.3），"
            f"裂变后整体色调显著偏离原图"
        )

    if not flags:
        flags.append("【通过】量化指标未触发任何红旗")
    return flags


def _build_compare_report(orig_img: Image.Image, gen_img: Image.Image, orig_label: str = "ORIG", gen_label: str = "FISSION") -> dict:
    """Core analysis: build the same JSON dict as the split-image mode,
    but given two separate images (orig and generated), normalized to same size."""
    if orig_img.size != gen_img.size:
        gen_img = gen_img.resize(orig_img.size, Image.LANCZOS)

    left = orig_img
    right = gen_img

    def per_half(half: Image.Image) -> dict:
        sd, var = laplacian_variance(half)
        h2 = half.size[1]
        top = half.crop((0, 0, half.size[0], int(h2 * 0.28)))
        sd2, var2 = laplacian_variance(top)
        corners = corner_patches(half)
        cw, ch = half.size
        center = half.crop((int(cw * 0.25), int(ch * 0.30),
                            int(cw * 0.75), int(ch * 0.70)))
        center_l = center.convert("L")
        center_contrast = ImageStat.Stat(center_l).stddev[0]
        return {
            "laplacian_stddev": round(sd, 2),
            "laplacian_variance": round(var, 2),
            "text_top_laplacian_var": round(var2, 2),
            "edge_density": round(edge_density(half), 4),
            "corner_color_spread": round(corner_color_spread(corners), 2),
            "center_local_contrast": round(center_contrast, 2),
            "mean_rgb": mean_rgb(half),
            "top_colors": [
                {"count": c, "rgb": list(rgb)} for c, rgb in quantize_topcolors(half, 5)
            ],
        }

    report = {
        "mode": "compare",
        "orig_label": orig_label,
        "gen_label": gen_label,
        "orig_size": list(orig_img.size),
        "gen_size": list(gen_img.size),
        "left": per_half(left),
        "right": per_half(right),
        "hist_chi2": round(hist_diff(left, right), 4),
    }
    report["flags"] = diagnose(report)
    report["passed"] = _gate(report)
    return report


def _gate(d: dict) -> bool:
    """Hard gate: must pass ALL of these to be PASS."""
    left = d["left"]; right = d["right"]
    if right["laplacian_variance"] < left["laplacian_variance"] * 0.75:
        return False
    if right["text_top_laplacian_var"] < left["text_top_laplacian_var"] * 0.75:
        return False
    if right["edge_density"] > left["edge_density"] * 1.3:
        return False
    if right["corner_color_spread"] > left["corner_color_spread"] * 1.6 + 5:
        return False
    if right["center_local_contrast"] < left["center_local_contrast"] * 0.7 + 5:
        return False
    if d["hist_chi2"] > 0.3:
        return False
    return True


def _build_side_by_side(orig_path: str, gen_path: str, out_path: str) -> None:
    """Stack orig (left) + gen (right) side-by-side, save to out_path."""
    orig = Image.open(orig_path).convert("RGB")
    gen = Image.open(gen_path).convert("RGB")
    if orig.size != gen.size:
        gen = gen.resize(orig.size, Image.LANCZOS)
    gap = 16
    w, h = orig.size
    out = Image.new("RGB", (w * 2 + gap, h), (235, 235, 238))
    out.paste(orig, (0, 0))
    out.paste(gen, (w + gap, 0))
    if out_path:
        out.save(out_path, quality=92)


def main():
    # CLI:
    #   - 1 arg: split-image mode (existing)
    #   - 2 args: compare mode (orig_path, gen_path)
    #   - 3 args: compare mode + save side-by-side preview
    if len(sys.argv) < 2:
        print("usage:")
        print("  python qc_split_image.py <split_compare_image>")
        print("  python qc_split_image.py <orig_path> <gen_path> [out_compare.jpg]")
        sys.exit(1)

    if len(sys.argv) >= 3:
        orig_path, gen_path = sys.argv[1], sys.argv[2]
        out_path = sys.argv[3] if len(sys.argv) >= 4 else None
        orig = Image.open(orig_path).convert("RGB")
        gen = Image.open(gen_path).convert("RGB")
        report = _build_compare_report(orig, gen)
        report["orig_path"] = orig_path
        report["gen_path"] = gen_path
        if out_path:
            _build_side_by_side(orig_path, gen_path, out_path)
            report["compare_image"] = out_path
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["passed"]:
            print("\n[GATE] ❌ FAIL — 至少一项硬门槛不达标，不应发图。")
            sys.exit(2)
        else:
            print("\n[GATE] ✅ PASS — 通过所有客观硬门槛。")
            sys.exit(0)

    # legacy 1-arg split-image mode
    im = Image.open(sys.argv[1]).convert("RGB")
    w, h = im.size
    mid = w // 2
    left = im.crop((0, 0, mid, h))
    right = im.crop((mid, 0, w, h))

    def per_half(half: Image.Image, tag: str) -> dict:
        sd, var = laplacian_variance(half)
        # top text strip ~ top 28% (BACARDI line)
        h2 = half.size[1]
        top = half.crop((0, 0, half.size[0], int(h2 * 0.28)))
        sd2, var2 = laplacian_variance(top)
        corners = corner_patches(half)
        # center crop: 30%~70% h, 25%~75% w  (徽章主体区域)
        cw, ch = half.size
        center = half.crop((int(cw * 0.25), int(ch * 0.30),
                            int(cw * 0.75), int(ch * 0.70)))
        center_l = center.convert("L")
        center_contrast = ImageStat.Stat(center_l).stddev[0]
        return {
            "laplacian_stddev": round(sd, 2),
            "laplacian_variance": round(var, 2),
            "text_top_laplacian_var": round(var2, 2),
            "edge_density": round(edge_density(half), 4),
            "corner_color_spread": round(corner_color_spread(corners), 2),
            "center_local_contrast": round(center_contrast, 2),
            "mean_rgb": mean_rgb(half),
            "top_colors": [
                {"count": c, "rgb": list(rgb)} for c, rgb in quantize_topcolors(half, 5)
            ],
        }

    report = {
        "image": sys.argv[1],
        "mode": "split_image",
        "size": [w, h],
        "left": per_half(left, "ORIG"),
        "right": per_half(right, "FISSION"),
        "hist_chi2": round(hist_diff(left, right), 4),
    }
    report["flags"] = diagnose(report)
    report["passed"] = _gate(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        print("\n[GATE] ❌ FAIL")
        sys.exit(2)


if __name__ == "__main__":
    main()
