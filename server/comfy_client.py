"""
ComfyUI client — template loading, parameter injection, and API communication.

Flow for every job:
  1. load_workflow(name)        load JSON from workflows/<name>.json
  2. inject_params(wf, params)  write values into the right node fields
  3. queue(workflow)            POST /prompt  → prompt_id
  4. wait(prompt_id)            WS /ws        → block until done
  5. get_outputs(prompt_id)     GET /history/{prompt_id} → image paths
  6. download_image(info)       GET /view     → bytes

Upload flow (img2img / upscale):
  upload_image(local_path)      POST /upload/image → server filename
"""

from __future__ import annotations

import base64
import copy
import json
import os
import uuid
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import aiohttp

# ── Configuration ─────────────────────────────────────────────────────────────

COMFY_HOST    = os.getenv("COMFY_HOST", "127.0.0.1")
COMFY_PORT    = os.getenv("COMFY_PORT", "8188")
BASE_URL      = f"http://{COMFY_HOST}:{COMFY_PORT}"
WS_URL        = f"ws://{COMFY_HOST}:{COMFY_PORT}/ws"
JOB_TIMEOUT   = int(os.getenv("COMFY_JOB_TIMEOUT", "300"))

# Resolved relative to this file so the server can live anywhere.
WORKFLOWS_DIR = Path(os.getenv(
    "COMFY_WORKFLOWS_DIR",
    Path(__file__).parent.parent / "workflows",
))


# ── Exceptions ────────────────────────────────────────────────────────────────

class ComfyError(RuntimeError):
    pass


# ── Template handling ─────────────────────────────────────────────────────────

def list_workflows() -> list[str]:
    """Return the names of all workflow templates on disk (without .json)."""
    return sorted(p.stem for p in WORKFLOWS_DIR.glob("*.json"))


def load_workflow(name: str) -> dict:
    """
    Load a workflow template from disk.

    Raises FileNotFoundError if the template does not exist.
    """
    path = WORKFLOWS_DIR / f"{name}.json"
    if not path.exists():
        available = list_workflows()
        raise FileNotFoundError(
            f"Workflow '{name}' not found. Available: {available}"
        )
    return json.loads(path.read_text())


def inject_params(workflow: dict, params: dict[str, Any]) -> dict:
    """
    Deep-copy *workflow*, apply *params* using the ``__params__`` map,
    and return the modified copy with ``__params__`` stripped out
    (ComfyUI rejects unknown top-level keys).

    ``__params__`` maps a parameter name to {"node": "<id>", "field": "<key>"}.
    Only params that are present in the map AND whose value is not None are injected.
    """
    wf = copy.deepcopy(workflow)
    param_map: dict[str, dict] = wf.pop("__params__", {})

    for param_name, value in params.items():
        if value is None:
            continue
        spec = param_map.get(param_name)
        if spec is None:
            continue
        node_id = spec["node"]
        field   = spec["field"]
        if node_id not in wf:
            raise ComfyError(
                f"Param '{param_name}' targets node '{node_id}' "
                f"which doesn't exist in the workflow."
            )
        wf[node_id]["inputs"][field] = value

    return wf


# ── HTTP helpers ──────────────────────────────────────────────────────────────

async def _post(path: str, payload: dict) -> dict:
    async with aiohttp.ClientSession() as s:
        async with s.post(f"{BASE_URL}{path}", json=payload) as r:
            r.raise_for_status()
            return await r.json()


async def _get_json(path: str) -> dict:
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{BASE_URL}{path}") as r:
            r.raise_for_status()
            return await r.json()


async def _get_bytes(path: str) -> bytes:
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{BASE_URL}{path}") as r:
            r.raise_for_status()
            return await r.read()


# ── Core API calls ────────────────────────────────────────────────────────────

async def queue(workflow: dict, client_id: str | None = None) -> str:
    """POST /prompt — returns the prompt_id string."""
    cid = client_id or str(uuid.uuid4())
    result = await _post("/prompt", {"prompt": workflow, "client_id": cid})
    if "error" in result:
        detail = result.get("error", {})
        raise ComfyError(f"ComfyUI rejected the workflow: {detail}")
    return result["prompt_id"], cid


async def wait(prompt_id: str, client_id: str) -> None:
    """
    Open a WebSocket connection and block until *prompt_id* finishes executing.

    ComfyUI sends {"type": "executing", "data": {"node": null, "prompt_id": ...}}
    when a job is fully done.
    """
    deadline = time.monotonic() + JOB_TIMEOUT
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(f"{WS_URL}?clientId={client_id}") as ws:
            async for msg in ws:
                if time.monotonic() > deadline:
                    raise ComfyError(
                        f"Timed out waiting for job {prompt_id} "
                        f"after {JOB_TIMEOUT}s"
                    )
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if (
                        data.get("type") == "executing"
                        and data.get("data", {}).get("node") is None
                        and data.get("data", {}).get("prompt_id") == prompt_id
                    ):
                        return
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    raise ComfyError("WebSocket closed before job finished")


async def get_history(prompt_id: str) -> dict:
    """GET /history/{prompt_id} — returns the history entry for this job."""
    history = await _get_json(f"/history/{prompt_id}")
    return history.get(prompt_id, {})


async def get_outputs(prompt_id: str) -> list[dict]:
    """
    Return a list of output image descriptors from the history entry.

    Each dict contains:
      filename, subfolder, type, data_url (base64 PNG)
    """
    entry = await get_history(prompt_id)
    images: list[dict] = []
    for node_output in entry.get("outputs", {}).values():
        for img in node_output.get("images", []):
            params = urlencode({
                "filename":  img["filename"],
                "subfolder": img.get("subfolder", ""),
                "type":      img.get("type", "output"),
            })
            raw = await _get_bytes(f"/view?{params}")
            images.append({
                "filename":  img["filename"],
                "subfolder": img.get("subfolder", ""),
                "type":      img.get("type", "output"),
                "data_url":  "data:image/png;base64," + base64.b64encode(raw).decode(),
            })
    return images


async def get_job_status(prompt_id: str) -> dict:
    """
    Return the current status of a queued job without blocking.

    Checks /queue (running + pending lists) and /history.
    Returns a dict with keys: status, outputs.
    """
    queue_info = await _get_json("/queue")
    running_ids = [item[1] for item in queue_info.get("queue_running", [])]
    pending_ids = [item[1] for item in queue_info.get("queue_pending", [])]

    if prompt_id in running_ids:
        return {"status": "running", "outputs": []}
    if prompt_id in pending_ids:
        return {"status": "pending", "outputs": []}

    # Not in queue — check history.
    entry = await get_history(prompt_id)
    if not entry:
        return {"status": "unknown", "outputs": []}

    outputs = await get_outputs(prompt_id)
    return {"status": "completed", "outputs": outputs}


async def upload_image(local_path: str) -> str:
    """
    Upload a local image to ComfyUI's input folder.

    Returns the server-side filename to use in LoadImage nodes.
    """
    p = Path(local_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {local_path}")

    data = aiohttp.FormData()
    data.add_field(
        "image",
        open(p, "rb"),
        filename=p.name,
        content_type="image/png",
    )
    data.add_field("overwrite", "true")

    async with aiohttp.ClientSession() as s:
        async with s.post(f"{BASE_URL}/upload/image", data=data) as r:
            r.raise_for_status()
            result = await r.json()
    return result.get("name", p.name)


# ── High-level run helper ─────────────────────────────────────────────────────

async def run_workflow(workflow: dict) -> list[dict]:
    """
    Queue a ready-to-submit workflow, wait for it to finish,
    and return output image descriptors.
    """
    prompt_id, client_id = await queue(workflow)
    await wait(prompt_id, client_id)
    return await get_outputs(prompt_id)
