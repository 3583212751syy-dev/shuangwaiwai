"""
ComfyUI API 客户端：提交工作流、WebSocket 监听完成、下载产出图。
兼容 ComfyUI 的「API Format」(非 Save Format)，避免 node_errors。
"""
import io
import json
import uuid
import websocket
import requests
from urllib.parse import urlparse

from config import (COMFYUI_URL, COMFYUI_WS, COMFYUI_API_PROMPT,
                    COMFYUI_API_HISTORY, COMFYUI_API_VIEW)


class ComfyClient:
    def __init__(self, url: str = COMFYUI_URL, client_id: str = None):
        self.url = url
        self.client_id = client_id or str(uuid.uuid4())

    # ---- 提交工作流 ----
    def queue_prompt(self, prompt: dict) -> str:
        """prompt: 完整的 API-format 工作流 dict。返回 prompt_id。"""
        payload = {"prompt": prompt, "client_id": self.client_id}
        r = requests.post(self.url + "/prompt", json=payload, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"提交工作流失败 {r.status_code}: {r.text}")
        return r.json()["prompt_id"]

    # ---- 历史（轮询兜底）----
    def get_history(self, prompt_id: str):
        r = requests.get(self.url + "/history", params={"prompt_id": prompt_id}, timeout=30)
        return r.json()

    # ---- 下载单张图 ----
    def get_image_bytes(self, filename: str, subfolder: str, ftype: str) -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": ftype}
        r = requests.get(self.url + "/view", params=params, timeout=60)
        r.raise_for_status()
        return r.content

    # ---- WebSocket 监听，收集所有 save_node 产出 ----
    @staticmethod
    def wait_for_images(ws_url: str, client_id: str, prompt_id: str,
                        timeout: int = 1800):
        """
        监听指定 prompt_id 的执行完成事件，收集每个含 images 的节点输出。
        返回: { node_id: [ {filename, subfolder, type}, ... ] }
        """
        import websocket as ws_mod
        ws = ws_mod.WebSocket()
        ws.connect(ws_url + "?clientId=" + client_id)
        outputs = {}
        try:
            while True:
                raw = ws.recv()
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                if msg.get("type") == "executing":
                    data = msg["data"]
                    if data.get("prompt_id") != prompt_id:
                        continue
                    # node 执行完毕（node 为 None 表示整个 prompt 完成）
                    if data.get("node") is None:
                        break
                elif msg.get("type") == "executed":
                    data = msg["data"]
                    if data.get("prompt_id") != prompt_id:
                        continue
                    node = data.get("node")
                    out = data.get("output", {})
                    imgs = out.get("images")
                    if node and imgs:
                        outputs.setdefault(node, []).extend(imgs)
        finally:
            ws.close()
        return outputs

    # ---- 高层：提交并等待，返回节点->图片bytes ----
    def run(self, prompt: dict, timeout: int = 1800) -> dict:
        """
        提交工作流并等待完成，返回 {node_id: [bytes, ...]}。
        下载每个产出节点的最终图片。
        """
        prompt_id = self.queue_prompt(prompt)
        outputs = self.wait_for_images(COMFYUI_WS, self.client_id, prompt_id, timeout)
        result = {}
        for node_id, imgs in outputs.items():
            # 仅取该节点最后一张（避免重试历史堆叠）
            last = imgs[-1]
            result[node_id] = self.get_image_bytes(
                last["filename"], last.get("subfolder", ""), last.get("type", "output"))
        return result

    # ---- 系统状态 ----
    def system_stats(self) -> dict:
        r = requests.get(self.url + "/system_stats", timeout=15)
        return r.json()


if __name__ == "__main__":
    c = ComfyClient()
    print("ComfyUI 状态:", c.system_stats())
