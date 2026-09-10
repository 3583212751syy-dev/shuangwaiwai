"""
phase_b_anytext2.py — Phase B 集成：ComfyUI + AnyText2 v2.0 文字生成

用户明确指令（2026-09-10）："即接 ComfyUI 装 AnyText2 v2.0 进 Phase B"
本脚本把 v322 的 small_elements / text_bands 检测结果，喂给本地 ComfyUI 的
UL_AnyText* 节点，用 AnyText2 v2.0 后端在任意位置生成/改写文字。

合规：全部走本地 ComfyUI workflow（ComfyUI/models/checkpoints/anytext_v2.0.ckpt），
绝不调用任何云端生图 API（见 image-fission 永久硬规则）。

用法：
  python phase_b_anytext2.py --prompt_text "NOCTAVEN" --out "jobs/phaseB/out.png"
  python phase_b_anytext2.py --serve          # 仅验证 v2.0 后端能否连通并生一张测试图

依赖：requests（ComfyUI 自带环境一般有）；无则 pip install requests
"""
import sys, os, io, json, time, base64, argparse
import numpy as np
from PIL import Image

COMFY_URL = "http://127.0.0.1:8188"
CKPT = "anytext_v2.0.ckpt"   # 位于 ComfyUI/models/checkpoints/，Loader 自动切 AnyText2 后端
CLIENT_ID = "phaseB_anytext2_%d" % int(time.time())


def _post(api, payload, files=None):
    import requests
    if files:
        r = requests.post(COMFY_URL + api, files=files, data=payload, timeout=60)
    else:
        r = requests.post(COMFY_URL + api, json=payload, timeout=60)
    return r


def upload_mask(mask_np: np.ndarray, name="pos_mask.png") -> str:
    """把 [H,W] 0/255 遮罩上传到 ComfyUI input 目录，返回可用文件名。

    存为 RGBA：文字区域 alpha=255、其余 alpha=0，这样 LoadImage 的 MASK 输出
    (=alpha 通道) 正好是文字位置，供 UL_AnyTextFontImg/UL_AnyTextEncoder 使用。
    """
    import requests
    m = mask_np.astype(np.uint8)
    rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 3] = m  # alpha = 文字区域
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    r = requests.post(COMFY_URL + "/upload/image",
                     files={"image": (name, buf, "image/png")}, timeout=60)
    j = r.json()
    return j["name"]


def build_prompt(prompt_text, mask_name, width=512, height=512, seed=88888888,
                 mode_gen=True, base_image_name=None, a_prompt=None):
    """
    构造完整 AnyText2 v2.0 workflow。
    mode_gen=True  → 文字生成（在遮罩位置生成 prompt_text）
    mode_gen=False → 文字编辑（需 base_image_name，替换图中文字）
    """
    full_prompt = 'a clean product poster, white background, with text: "%s".' % prompt_text
    ap = a_prompt or ("best quality, extremely detailed, 4k, HD, super legible text, "
                      "clear text edges, clear strokes, neat writing, no watermarks")
    np_ = ("low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, "
           "low quality, watermark, unreadable text, messy words, distorted text")
    wf = {
        "loader": {
            "class_type": "UL_AnyTextLoader",
            "inputs": {
                "ckpt_name": CKPT,
                "control_net_name": "None",
                "miaobi_clip": "None",
                "weight_dtype": "fp16",
                "init_device": "auto",
            },
        },
        "loadmask": {
            "class_type": "LoadImage",
            "inputs": {"image": mask_name, "upload": "image"},
        },
        "formatter": {
            "class_type": "UL_AnyTextFormatter",
            "inputs": {"prompt": full_prompt},
        },
        # v322 修复：必须提供 fonts 对象（否则 AnyText2_Infer 访问 fonts.glyline_font_path 崩溃）
        "fonts": {
            "class_type": "UL_AnyText2Fonts",
            "inputs": {
                "font_hollow": False,
                "font_name": "Arial_Unicode.ttf",
                "font_color": "white",
                "font_name1": "None", "font_color1": "None",
                "font_name2": "None", "font_color2": "None",
                "font_name3": "None", "font_color3": "None",
                "font_name4": "None", "font_color4": "None",
                "font_name5": "None", "font_color5": "None",
                "font_name6": "None", "font_color6": "None",
                "font_name7": "None", "font_color7": "None",
            },
        },
        "emptylatent": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "encoder": {
            "class_type": "UL_AnyTextEncoder",
            "inputs": {
                "model": ["loader", 0],
                "mask": ["loadmask", 1],
                "prompt": ["formatter", 0],
                "texts": ["formatter", 1],
                "latent": ["emptylatent", 0],
                "font_name": "Arial_Unicode.ttf",
                "mode": mode_gen,
                "sort_radio": True,
                "a_prompt": ap,
                "n_prompt": np_,
                "random_mask": False,
                "revise_pos": False,
                "image": [base_image_name] if base_image_name else None,
                "fonts": ["fonts", 0],
                "font_apply": True,
                "show_glyph": False,
            },
        },
        "sampler": {
            "class_type": "UL_AnyTextSampler",
            "inputs": {
                "model": ["loader", 0],
                "positive": ["encoder", 0],
                "negative": ["encoder", 1],
                "seed": seed,
                "steps": 20,
                "cfg": 9.0,
                "strength": 1.0,
                "attnx_scale": 1.0,
                "eta": 0.0,
                "keep_load": True,
                "keep_device": True,
            },
        },
        "vae_decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["sampler", 0], "vae": ["loader", 1]},
        },
        "save": {
            "class_type": "SaveImage",
            "inputs": {"images": ["vae_decode", 0], "filename_prefix": "phaseB_anytext2"},
        },
    }
    return wf


def queue_and_wait(wf, timeout=600):
    import requests
    # 提交
    r = requests.post(COMFY_URL + "/prompt", json={"prompt": wf, "client_id": CLIENT_ID}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError("提交 workflow 失败: %s %s" % (r.status_code, r.text))
    pid = r.json()["prompt_id"]
    # 轮询历史
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = requests.get(COMFY_URL + "/history/" + pid, timeout=30).json()
        except Exception:
            time.sleep(3); continue
        if pid in h:
            item = h[pid]
            if "outputs" in item:
                return item["outputs"]
            if "status" in item and item["status"].get("status_str") == "error":
                st = item["status"]
                exc = st.get("exception")
                node_errs = st.get("node_errors", {})
                detail = json.dumps({"exception": exc, "node_errors": node_errs}, ensure_ascii=False)
                raise RuntimeError("ComfyUI 执行错误: " + detail)
        time.sleep(3)
    raise TimeoutError("等待 ComfyUI 超时 (%ds)" % timeout)


def fetch_first_image(outputs):
    """从 history outputs 取第一张图，返回 (bytes, filename)。"""
    import requests
    for node_id, out in outputs.items():
        if "images" in out:
            im = out["images"][0]
            url = "%s/view?filename=%s&subfolder=%s&type=%s" % (
                COMFY_URL, im["filename"], im.get("subfolder", ""), im.get("type", "output"))
            data = requests.get(url, timeout=60).content
            return data, im["filename"]
    return None, None


def run(prompt_text, out_path, width=512, height=512, mode_gen=True, base_image_path=None,
        mask_region=None, seed=88888888):
    # 1) 生成位置遮罩（默认居中矩形区域）
    if mask_region is None:
        mask_region = (int(width*0.15), int(height*0.40), int(width*0.85), int(height*0.62))
    mask = np.zeros((height, width), dtype=np.uint8)
    x0, y0, x1, y1 = mask_region
    mask[y0:y1, x0:x1] = 255
    mask_name = upload_mask(mask, name="pos_mask_%d.png" % seed)

    base_name = None
    if base_image_path and not mode_gen:
        # 上传底图（编辑模式需要）
        from PIL import Image as _Im
        buf = io.BytesIO(); _Im.open(base_image_path).convert("RGB").save(buf, format="PNG"); buf.seek(0)
        import requests
        rr = requests.post(COMFY_URL + "/upload/image", files={"image": ("base.png", buf, "image/png")}, timeout=60)
        base_name = rr.json()["name"]

    wf = build_prompt(prompt_text, mask_name, width, height, seed, mode_gen, base_name)
    outputs = queue_and_wait(wf)
    data, fname = fetch_first_image(outputs)
    if data is None:
        raise RuntimeError("未取到生成图，outputs=%s" % json.dumps(outputs, ensure_ascii=False))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path, fname


def main():
    ap = argparse.ArgumentParser(description="Phase B: ComfyUI + AnyText2 v2.0")
    ap.add_argument("--prompt_text", default="NOCTAVEN")
    ap.add_argument("--out", default="E:/Desktop/双接口/image-fission/jobs/phaseB/out.png")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--seed", type=int, default=88888888)
    ap.add_argument("--mode", choices=["gen", "edit"], default="gen")
    ap.add_argument("--base", default=None, help="编辑模式底图路径")
    ap.add_argument("--serve", action="store_true", help="仅做连通+生图冒烟测试")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    mode_gen = (args.mode == "gen")
    out, fname = run(args.prompt_text, args.out, args.width, args.height,
                     mode_gen=mode_gen, base_image_path=args.base, seed=args.seed)
    print("[PhaseB] 生成成功 -> %s (ComfyUI 文件名 %s)" % (out, fname))


if __name__ == "__main__":
    main()
