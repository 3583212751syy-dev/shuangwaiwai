#!/usr/bin/env python
"""发布前打包：把 deploy-product-library 的完整底库同步成可上传的发布目录。

为什么需要它：这个 app 是 http-service 形态，数据必须随目录上传（不是云端桶），
而 2430 张原图 393MB 上传不现实。检索只用向量、图片只用于展示，所以线上用 760px
缩略图（约 100MB）。

用法：
  python tools/prepare-publish.py            # 生成缩略图 + 同步元数据/模型
  python tools/prepare-publish.py --keep-hires  # 保留已存在的原图（不重新缩图）
"""
import json
import os
import shutil
import sys
from multiprocessing import Pool

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))          # .../product-image-compare/tools
PROJ = os.path.dirname(HERE)                              # .../product-image-compare
WORK = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))  # .../new-chat-2/work
SRC = os.path.join(WORK, "deploy-product-library")
DST = os.path.join(WORK, "deploy-image-match")
MAXEDGE = 760
QUALITY = 82

SKIP_IMG = {"--keep-hires"}


def thumb(job):
    src_dir, out_dir, fn = job
    s = os.path.join(src_dir, fn)
    d = os.path.join(out_dir, fn)
    if not os.path.exists(s):
        return ("miss", fn)
    try:
        im = Image.open(s).convert("RGB")
        w, h = im.size
        k = min(1.0, MAXEDGE / max(w, h))
        if k < 1.0:
            im = im.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)
        im.save(d, "JPEG", quality=QUALITY, optimize=True, progressive=True)
        return ("ok", fn)
    except Exception as e:  # noqa: BLE001
        return ("err", fn + " :: " + str(e)[:120])


def main():
    keep = bool(SKIP_IMG & set(sys.argv[1:]))
    src_img = os.path.join(SRC, "images")
    dst_img = os.path.join(DST, "images")
    os.makedirs(dst_img, exist_ok=True)

    meta_path = os.path.join(DST, "metadata.json")
    for f in ("metadata.json", "analyses.json", "overrides.json", "app-worker.mjs"):
        shutil.copy2(os.path.join(SRC, f), os.path.join(DST, f))
    shutil.copy2(
        os.path.join(WORK, "git-handoff/projects/product-image-compare/local-runner/server.mjs"),
        os.path.join(DST, "server.mjs"),
    )
    # 读图规格 OCR：PP-OCRv6 tiny 模型 + onnxruntime-web wasm（浏览器端运行，服务端不跑 wasm）
    ocr_src = os.path.join(PROJ, "site-source", "models", "ocr")
    if os.path.isdir(ocr_src):
        for target in (DST, SRC):
            shutil.copytree(ocr_src, os.path.join(target, "models", "ocr"), dirs_exist_ok=True)

    # DINOv2 模型（pack-site 会刷新 Xenova CLIP，DINO 目录需单独带）
    dino_src = os.path.join(SRC, "model-cache", "onnx-community")
    if os.path.isdir(dino_src):
        shutil.copytree(dino_src, os.path.join(DST, "model-cache", "onnx-community"), dirs_exist_ok=True)

    names = [m["filename"] for m in json.load(open(meta_path, encoding="utf-8"))]
    if keep:
        have = set(os.listdir(dst_img))
        todo = [n for n in names if n not in have]
    else:
        todo = names
    print("缩略图待处理", len(todo), "张")
    if todo:
        with Pool(8) as p:
            res = p.map(thumb, [(src_img, dst_img, n) for n in todo], chunksize=40)
        print("  成功", sum(1 for t, _ in res if t == "ok"),
              " 缺源图", sum(1 for t, _ in res if t == "miss"),
              " 失败", sum(1 for t, _ in res if t == "err"))

    # 移出不属于底库的残留文件，保持体积可控
    keep_set = set(names)
    extra = [f for f in os.listdir(dst_img) if f not in keep_set]
    if extra:
        bak = os.path.join(WORK, "_removed_image_match_files")
        os.makedirs(bak, exist_ok=True)
        for f in extra:
            shutil.move(os.path.join(dst_img, f), os.path.join(bak, f))
        print("移出多余文件", len(extra))

    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(DST)
        for f in fs
    )
    cats = {}
    for m in json.load(open(meta_path, encoding="utf-8")):
        cats[m.get("category", "")] = cats.get(m.get("category", ""), 0) + 1
    print("底图", len(names), "张 | 类目", len(cats), "种 | 发布目录", round(total / 1048576, 1), "MB")


if __name__ == "__main__":
    main()
