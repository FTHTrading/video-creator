"""
ComfyUI API client — handles HTTP queue submission and WebSocket result polling.

ComfyUI exposes two surfaces:
  - HTTP  /prompt          — queue a workflow
  - HTTP  /history/{id}   — fetch completed job output
  - HTTP  /upload/image    — upload an input image
  - HTTP  /view            — download an output image
  - WS    /ws?clientId=… — real-time progress events
"""

import asyncio
import base64
import json
import os
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import aiofiles
import aiohttp


COMFY_HOST = os.getenv("COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.getenv("COMFY_PORT", "8188"))
COMFY_BASE_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"
COMFY_WS_URL = f"ws://{COMFY_HOST}:{COMFY_PORT}/ws"

# How long to wait for a queued job before giving up (seconds).
JOB_TIMEOUT = int(os.getenv("COMFY_JOB_TIMEOUT", "300"))


class ComfyError(RuntimeError):
    """Raised when ComfyUI returns an error or a job times out."""


class ComfyClient:
    """Thin async wrapper around the ComfyUI HTTP + WebSocket API."""

    def __init__(self, client_id: str | None = None) -> None:
        self.client_id = client_id or str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post_json(self, path: str, payload: dict) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{COMFY_BASE_URL}{path}", json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def _get_json(self, path: str) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{COMFY_BASE_URL}{path}") as resp:
                resp.raise_for_status()
                return await resp.json()

    async def _get_bytes(self, path: str) -> bytes:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{COMFY_BASE_URL}{path}") as resp:
                resp.raise_for_status()
                return await resp.read()

    async def _wait_for_job(self, prompt_id: str) -> dict:
        """
        Open a WebSocket connection and block until the job with *prompt_id*
        finishes, then return its history entry.
        """
        ws_url = f"{COMFY_WS_URL}?clientId={self.client_id}"
        deadline = asyncio.get_event_loop().time() + JOB_TIMEOUT

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                async for msg in ws:
                    if asyncio.get_event_loop().time() > deadline:
                        raise ComfyError(
                            f"Job {prompt_id} timed out after {JOB_TIMEOUT}s"
                        )
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if (
                            data.get("type") == "executing"
                            and data.get("data", {}).get("node") is None
                            and data.get("data", {}).get("prompt_id") == prompt_id
                        ):
                            # Execution finished — fetch history.
                            break
                    elif msg.type in (
                        aiohttp.WSMsgType.ERROR,
                        aiohttp.WSMsgType.CLOSED,
                    ):
                        raise ComfyError("WebSocket closed unexpectedly")

        history = await self._get_json(f"/history/{prompt_id}")
        return history.get(prompt_id, {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def queue_workflow(self, workflow: dict) -> str:
        """Submit *workflow* to the ComfyUI queue and return the prompt_id."""
        payload = {"prompt": workflow, "client_id": self.client_id}
        result = await self._post_json("/prompt", payload)
        if "error" in result:
            raise ComfyError(f"ComfyUI rejected workflow: {result['error']}")
        return result["prompt_id"]

    async def run_workflow(self, workflow: dict) -> dict:
        """Queue *workflow*, wait for it to finish, and return the history entry."""
        prompt_id = await self.queue_workflow(workflow)
        return await self._wait_for_job(prompt_id)

    async def get_output_images(self, history_entry: dict) -> list[dict]:
        """
        Extract output images from a history entry.

        Returns a list of dicts:
          { "filename": str, "subfolder": str, "type": str, "data_url": str }
        """
        images = []
        for node_output in history_entry.get("outputs", {}).values():
            for img_info in node_output.get("images", []):
                params = urlencode(
                    {
                        "filename": img_info["filename"],
                        "subfolder": img_info.get("subfolder", ""),
                        "type": img_info.get("type", "output"),
                    }
                )
                raw = await self._get_bytes(f"/view?{params}")
                b64 = base64.b64encode(raw).decode()
                images.append(
                    {
                        "filename": img_info["filename"],
                        "subfolder": img_info.get("subfolder", ""),
                        "type": img_info.get("type", "output"),
                        "data_url": f"data:image/png;base64,{b64}",
                    }
                )
        return images

    async def upload_image(self, image_path: str, overwrite: bool = True) -> dict:
        """
        Upload a local image to ComfyUI's input folder.

        Returns the server's response (contains filename and subfolder).
        """
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        async with aiofiles.open(p, "rb") as f:
            data = await f.read()

        form = aiohttp.FormData()
        form.add_field(
            "image",
            data,
            filename=p.name,
            content_type="image/png",
        )
        form.add_field("overwrite", "true" if overwrite else "false")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{COMFY_BASE_URL}/upload/image", data=form
            ) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def list_models(self) -> dict[str, list[str]]:
        """Return the available model lists from ComfyUI's object_info endpoint."""
        info = await self._get_json("/object_info")
        # Extract checkpoint, lora, and vae names from the node definitions.
        checkpoints: list[str] = []
        loras: list[str] = []
        vaes: list[str] = []

        loader = info.get("CheckpointLoaderSimple", {})
        inputs = loader.get("input", {}).get("required", {})
        checkpoints = inputs.get("ckpt_name", [None])[0] or []

        lora_loader = info.get("LoraLoader", {})
        lora_inputs = lora_loader.get("input", {}).get("required", {})
        loras = lora_inputs.get("lora_name", [None])[0] or []

        vae_loader = info.get("VAELoader", {})
        vae_inputs = vae_loader.get("input", {}).get("required", {})
        vaes = vae_inputs.get("vae_name", [None])[0] or []

        return {"checkpoints": checkpoints, "loras": loras, "vaes": vaes}

    async def get_system_stats(self) -> dict:
        """Return ComfyUI system stats (VRAM, RAM, etc.)."""
        return await self._get_json("/system_stats")

    async def get_queue_status(self) -> dict:
        """Return the current queue status."""
        return await self._get_json("/queue")

    async def interrupt(self) -> None:
        """Interrupt the currently running job."""
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{COMFY_BASE_URL}/interrupt") as resp:
                resp.raise_for_status()
