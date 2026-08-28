"""v132 Phase B: anytext_v2.0 文字替换 (专攻文字可靠渲染).

策略:
  - Phase A 已生成无字主体内容裂变图
  - 本脚本读取原图的文字区域 mask (用户或 AI 自动标) + 文字替换词
  - 用 anytext_v2.0 模型 (专门做文字渲染的 SDXL) inpaint 替换文字
  - 输出: 替换了文字的最终图

AnyText 模型 (anytext_v2.0.ckpt) 是阿里达摩院 2024 出品, 专门解决 SDXL 文字乱码
问题. 加载方式与普通 SDXL 一致.

用法:
  python smoke_v132_phaseB_text.py --base camo_4 --word "REMADE" --mask_top 0.1 --mask_bottom 0.3
  # --base: 对应 Phase A 输出 id
  # --word: 要替换的文本 (1-3 个英文短词, 中英文都可)
  # --mask_top / --mask_bottom: 文字区域 y 比例 (0-1), 仅对无 mask 文件适用
"""
import json, time, sys, argparse
from pathlib import Path
import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = Path("E:/Desktop/图裂变测试图")
COMFYUI = "http://127.0.0.1:8188"

ANYTEXT_CKPT = "anytext_v2.0.ckpt"
SEED = 132200


def run_inpaint(image_filename, word, mask_top, mask_bottom, out_dir, tag):
    """调用 anytext v2 做文字 inpaint. mask 是简单的上下水平条."""
    g = {}
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ANYTEXT_CKPT}}
    g["2"] = {"class_type": "LoadImage", "inputs": {"image": image_filename}}
    # 用空白 mask (实际生产应该用精确的 text-region mask)
    # Phase B 简易版: 占位 — 真实部署需要 ComfyUI mask editor 或自动检测
    g["17"] = {"class_type": "SaveImage",
               "inputs": {"images": ["14", 0], "filename_prefix": f"v132B_{tag}"}}
    return g


# Phase B 占位: 本脚本后续会扩展, 当前先出接口设计, 真实跑时再补 inpaint 节点
def main():
    print("v132 Phase B: anytext 文字 inpaint (placeholder, 等待 Phase A 全跑完接入)")


if __name__ == "__main__":
    main()