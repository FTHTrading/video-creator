"""
ComfyUI MCP Server

Exposes ComfyUI image-generation capabilities as MCP tools so that Cursor
(or any other MCP client) can call them from natural language.

Run with:
    python -m mcp_server.server

Or, for development with auto-reload:
    mcp dev mcp_server/server.py

Tool surface:
    comfy_generate_image      — text-to-image
    comfy_edit_image          — image-to-image editing
    comfy_upscale_image       — ESRGAN-based upscaling
    comfy_list_workflows      — list named workflows saved on disk
    comfy_run_named_workflow  — run a saved workflow JSON with injected inputs
    comfy_list_models         — list available checkpoints / LoRAs / VAEs
    comfy_queue_status        — show current ComfyUI queue depth
    comfy_interrupt           — cancel the running job
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .comfy_client import ComfyClient, ComfyError
from .workflows import (
    build_edit_image_workflow,
    build_generate_image_workflow,
    build_upscale_workflow,
)

# ── Server setup ──────────────────────────────────────────────────────────────

mcp = FastMCP(
    "ComfyUI",
    instructions=(
        "Tools for generating and editing images using a local ComfyUI instance. "
        "Always prefer comfy_generate_image for new images. "
        "Use comfy_edit_image to modify an existing image. "
        "Use comfy_upscale_image to increase resolution. "
        "Check comfy_queue_status before submitting a long job."
    ),
)

# Directory where named workflow JSON files are stored.
WORKFLOWS_DIR = Path(os.getenv("COMFY_WORKFLOWS_DIR", Path(__file__).parent / "saved_workflows"))
WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)

_client = ComfyClient()


# ── Helper ────────────────────────────────────────────────────────────────────

def _format_image_results(images: list[dict]) -> str:
    """Format a list of image dicts into a human-readable result string."""
    if not images:
        return "Job completed but no output images were found."
    lines = [f"Generated {len(images)} image(s):"]
    for img in images:
        lines.append(
            f"  • {img['filename']} — data URL length: {len(img['data_url'])} chars"
        )
    # Return the first image's data URL for inline display in Cursor.
    lines.append("")
    lines.append("First image (base64 data URL):")
    lines.append(images[0]["data_url"])
    return "\n".join(lines)


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def comfy_generate_image(
    prompt: str,
    style: str = "default",
    width: int = 512,
    height: int = 512,
    checkpoint: str = "",
    seed: int = -1,
    negative_prompt: str = "",
    steps: int = 0,
    cfg: float = 0.0,
    sampler: str = "",
    scheduler: str = "",
) -> str:
    """
    Generate an image from a text prompt using ComfyUI.

    Args:
        prompt:          Positive text prompt describing what to generate.
        style:           Style preset — one of: default, photorealistic, anime,
                         painting, cinematic.
        width:           Image width in pixels (default 512).
        height:          Image height in pixels (default 512).
        checkpoint:      Checkpoint model filename. Leave blank to use the
                         server default (COMFY_DEFAULT_CHECKPOINT env var).
        seed:            RNG seed for reproducibility (-1 = random).
        negative_prompt: What to avoid in the image. Leave blank to use the
                         style preset's default negative prompt.
        steps:           Sampling steps (0 = use style preset default).
        cfg:             CFG guidance scale (0 = use style preset default).
        sampler:         Sampler name (blank = use style preset default).
        scheduler:       Scheduler name (blank = use style preset default).

    Returns:
        Result summary including a base64 data URL for the first output image.
    """
    try:
        workflow = build_generate_image_workflow(
            prompt=prompt,
            style=style,
            width=width,
            height=height,
            checkpoint=checkpoint or None,
            seed=seed,
            negative_prompt=negative_prompt or None,
            steps=steps or None,
            cfg=cfg or None,
            sampler=sampler or None,
            scheduler=scheduler or None,
        )
        history = await _client.run_workflow(workflow)
        images = await _client.get_output_images(history)
        return _format_image_results(images)
    except ComfyError as exc:
        return f"ComfyUI error: {exc}"
    except Exception as exc:
        return f"Unexpected error: {exc}"


@mcp.tool()
async def comfy_edit_image(
    image_path: str,
    prompt: str,
    strength: float = 0.75,
    checkpoint: str = "",
    negative_prompt: str = "ugly, blurry, bad quality, watermark, text",
    steps: int = 30,
    cfg: float = 7.0,
    sampler: str = "euler",
    scheduler: str = "normal",
    seed: int = -1,
) -> str:
    """
    Edit an existing image using a text prompt (image-to-image).

    Args:
        image_path:      Local file path to the image to edit.
        prompt:          Text prompt describing the desired changes.
        strength:        How strongly to apply the edit (0.0–1.0).
                         0.0 = no change, 1.0 = completely redraw.
                         Typical values: 0.5–0.8.
        checkpoint:      Checkpoint model filename (blank = server default).
        negative_prompt: What to avoid in the output.
        steps:           Sampling steps.
        cfg:             CFG guidance scale.
        sampler:         KSampler sampler name.
        scheduler:       KSampler scheduler name.
        seed:            RNG seed (-1 = random).

    Returns:
        Result summary including a base64 data URL for the edited image.
    """
    try:
        upload_result = await _client.upload_image(image_path)
        server_filename = upload_result.get("name", Path(image_path).name)
        subfolder = upload_result.get("subfolder", "")

        workflow = build_edit_image_workflow(
            image_filename=server_filename,
            prompt=prompt,
            strength=strength,
            checkpoint=checkpoint or None,
            negative_prompt=negative_prompt,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            seed=seed,
            image_subfolder=subfolder,
        )
        history = await _client.run_workflow(workflow)
        images = await _client.get_output_images(history)
        return _format_image_results(images)
    except FileNotFoundError as exc:
        return f"File not found: {exc}"
    except ComfyError as exc:
        return f"ComfyUI error: {exc}"
    except Exception as exc:
        return f"Unexpected error: {exc}"


@mcp.tool()
async def comfy_upscale_image(
    image_path: str,
    scale: float = 4.0,
    upscale_model: str = "",
) -> str:
    """
    Upscale an image using an ESRGAN-based upscale model in ComfyUI.

    Args:
        image_path:     Local file path to the image to upscale.
        scale:          Target scale factor (e.g. 2.0, 4.0).
        upscale_model:  Upscale model filename in ComfyUI's models/upscale_models/
                        folder. Leave blank to use server default
                        (COMFY_UPSCALE_MODEL env var, default RealESRGAN_x4plus.pth).

    Returns:
        Result summary including a base64 data URL for the upscaled image.
    """
    try:
        upload_result = await _client.upload_image(image_path)
        server_filename = upload_result.get("name", Path(image_path).name)
        subfolder = upload_result.get("subfolder", "")

        workflow = build_upscale_workflow(
            image_filename=server_filename,
            scale=scale,
            upscale_model=upscale_model or None,
            image_subfolder=subfolder,
        )
        history = await _client.run_workflow(workflow)
        images = await _client.get_output_images(history)
        return _format_image_results(images)
    except FileNotFoundError as exc:
        return f"File not found: {exc}"
    except ComfyError as exc:
        return f"ComfyUI error: {exc}"
    except Exception as exc:
        return f"Unexpected error: {exc}"


@mcp.tool()
async def comfy_list_workflows() -> str:
    """
    List the named workflow JSON files saved on this server.

    These are reusable ComfyUI workflow presets that can be run with
    comfy_run_named_workflow. Returns a JSON array of workflow names.
    """
    files = sorted(WORKFLOWS_DIR.glob("*.json"))
    if not files:
        return (
            "No saved workflows found. "
            f"Add workflow JSON files to: {WORKFLOWS_DIR}"
        )
    names = [f.stem for f in files]
    return json.dumps(names, indent=2)


@mcp.tool()
async def comfy_run_named_workflow(
    name: str,
    inputs: str = "{}",
) -> str:
    """
    Run a named workflow saved on the server, optionally overriding node inputs.

    Args:
        name:    Workflow name (as returned by comfy_list_workflows, without .json).
        inputs:  JSON object mapping node IDs to input field overrides.
                 Example: {"6": {"text": "a red sunset"}}
                 This is merged into the workflow before submission.

    Returns:
        Result summary including a base64 data URL for the first output image.
    """
    workflow_file = WORKFLOWS_DIR / f"{name}.json"
    if not workflow_file.exists():
        available = [f.stem for f in WORKFLOWS_DIR.glob("*.json")]
        return (
            f"Workflow '{name}' not found. "
            f"Available: {available or ['(none)']}"
        )

    try:
        workflow = json.loads(workflow_file.read_text())
    except json.JSONDecodeError as exc:
        return f"Failed to parse workflow JSON: {exc}"

    try:
        overrides: dict[str, Any] = json.loads(inputs) if inputs.strip() else {}
    except json.JSONDecodeError as exc:
        return f"Invalid inputs JSON: {exc}"

    # Deep-merge overrides into the workflow.
    for node_id, node_inputs in overrides.items():
        if node_id in workflow and "inputs" in workflow[node_id]:
            workflow[node_id]["inputs"].update(node_inputs)
        else:
            return (
                f"Override targets node '{node_id}' which doesn't exist "
                f"in workflow '{name}'."
            )

    try:
        history = await _client.run_workflow(workflow)
        images = await _client.get_output_images(history)
        return _format_image_results(images)
    except ComfyError as exc:
        return f"ComfyUI error: {exc}"
    except Exception as exc:
        return f"Unexpected error: {exc}"


@mcp.tool()
async def comfy_list_models() -> str:
    """
    List the models available in the connected ComfyUI instance.

    Returns a JSON object with keys: checkpoints, loras, vaes.
    """
    try:
        models = await _client.list_models()
        return json.dumps(models, indent=2)
    except Exception as exc:
        return f"Failed to fetch model list: {exc}"


@mcp.tool()
async def comfy_queue_status() -> str:
    """
    Show the current ComfyUI queue status.

    Returns how many jobs are running and pending, plus system stats
    (VRAM usage, RAM usage).
    """
    try:
        queue = await _client.get_queue_status()
        stats = await _client.get_system_stats()
        running = len(queue.get("queue_running", []))
        pending = len(queue.get("queue_pending", []))
        devices = stats.get("devices", [{}])
        vram = devices[0].get("vram_free", "unknown") if devices else "unknown"
        vram_total = devices[0].get("vram_total", "unknown") if devices else "unknown"
        return (
            f"Queue: {running} running, {pending} pending\n"
            f"VRAM: {vram}/{vram_total} bytes free"
        )
    except Exception as exc:
        return f"Failed to fetch queue status: {exc}"


@mcp.tool()
async def comfy_interrupt() -> str:
    """
    Cancel / interrupt the currently executing ComfyUI job.

    Returns:
        Confirmation message.
    """
    try:
        await _client.interrupt()
        return "Interrupt signal sent to ComfyUI."
    except Exception as exc:
        return f"Failed to send interrupt: {exc}"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
