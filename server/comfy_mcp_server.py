"""
ComfyUI MCP server — register this file in Cursor's MCP settings.

Run directly:
    python server/comfy_mcp_server.py

Or point Cursor at it via .cursor/mcp.json:
    { "command": "python", "args": ["server/comfy_mcp_server.py"] }

Tools:
    list_workflows          — names of available workflow templates
    run_workflow            — text-to-image via a named template
    edit_image              — img2img editing of a local file
    upscale_image           — ESRGAN upscaling of a local file
    get_job_status          — poll an async job by prompt_id
"""

from __future__ import annotations

import json
import os
import sys
import random

# Allow running as a script with `python server/comfy_mcp_server.py` from any cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
import server.comfy_client as comfy

mcp = FastMCP(
    "comfy-tools",
    instructions=(
        "Tools for generating and editing images with a local ComfyUI instance. "
        "Start with list_workflows to see what templates are available. "
        "Use run_workflow for text-to-image. "
        "Use edit_image to modify an existing image file. "
        "Use upscale_image to increase image resolution. "
        "Use get_job_status to check the progress of a previously queued job."
    ),
)


# ── Tool: list_workflows ───────────────────────────────────────────────────────

@mcp.tool()
def list_workflows() -> str:
    """
    List the workflow templates available on this server.

    Returns a JSON array of workflow names that can be passed to run_workflow.
    """
    names = comfy.list_workflows()
    if not names:
        return f"No workflow templates found in {comfy.WORKFLOWS_DIR}"
    return json.dumps(names)


# ── Tool: run_workflow ─────────────────────────────────────────────────────────

@mcp.tool()
async def run_workflow(
    name: str,
    prompt: str,
    negative_prompt: str = "ugly, blurry, low quality, watermark, text",
    width: int = 512,
    height: int = 512,
    seed: int = -1,
    steps: int = 25,
    cfg: float = 7.0,
    checkpoint: str = "",
) -> str:
    """
    Run a named workflow template and return the generated image.

    Args:
        name:            Workflow template name (from list_workflows).
                         Use "txt2img" for text-to-image generation.
        prompt:          What to generate.
        negative_prompt: What to avoid in the output.
        width:           Output width in pixels (default 512).
        height:          Output height in pixels (default 512).
        seed:            RNG seed for reproducibility (-1 = random).
        steps:           Sampling steps (default 25).
        cfg:             CFG guidance scale (default 7.0).
        checkpoint:      Checkpoint filename — leave blank for server default.

    Returns:
        JSON with prompt_id and base64 data URL of the first output image.
    """
    try:
        workflow = comfy.load_workflow(name)
    except FileNotFoundError as exc:
        return str(exc)

    actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)

    params: dict = {
        "prompt":          prompt,
        "negative_prompt": negative_prompt,
        "width":           width,
        "height":          height,
        "seed":            actual_seed,
        "steps":           steps,
        "cfg":             cfg,
    }
    if checkpoint:
        params["checkpoint"] = checkpoint

    try:
        ready = comfy.inject_params(workflow, params)
        images = await comfy.run_workflow(ready)
    except comfy.ComfyError as exc:
        return f"ComfyUI error: {exc}"
    except Exception as exc:
        return f"Error: {exc}"

    if not images:
        return "Job completed but no output images were returned."

    return json.dumps({
        "image_count": len(images),
        "images": [
            {
                "filename": img["filename"],
                "data_url": img["data_url"],
            }
            for img in images
        ],
    })


# ── Tool: edit_image ───────────────────────────────────────────────────────────

@mcp.tool()
async def edit_image(
    path: str,
    prompt: str,
    denoise: float = 0.75,
    negative_prompt: str = "ugly, blurry, low quality, watermark, text",
    steps: int = 30,
    cfg: float = 7.0,
    seed: int = -1,
    checkpoint: str = "",
) -> str:
    """
    Edit an existing local image using a text prompt (image-to-image).

    Args:
        path:            Absolute or relative path to the image to edit.
        prompt:          Describe the changes to apply.
        denoise:         Edit strength: 0.0 = no change, 1.0 = full redraw.
                         Typical range: 0.4 – 0.85.
        negative_prompt: What to avoid in the output.
        steps:           Sampling steps.
        cfg:             CFG guidance scale.
        seed:            RNG seed (-1 = random).
        checkpoint:      Checkpoint filename — leave blank for server default.

    Returns:
        JSON with base64 data URL of the edited image.
    """
    try:
        server_filename = await comfy.upload_image(path)
    except FileNotFoundError as exc:
        return str(exc)
    except Exception as exc:
        return f"Upload failed: {exc}"

    try:
        workflow = comfy.load_workflow("img2img")
    except FileNotFoundError as exc:
        return str(exc)

    actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)

    params: dict = {
        "image":           server_filename,
        "prompt":          prompt,
        "negative_prompt": negative_prompt,
        "denoise":         max(0.0, min(1.0, denoise)),
        "seed":            actual_seed,
        "steps":           steps,
        "cfg":             cfg,
    }
    if checkpoint:
        params["checkpoint"] = checkpoint

    try:
        ready = comfy.inject_params(workflow, params)
        images = await comfy.run_workflow(ready)
    except comfy.ComfyError as exc:
        return f"ComfyUI error: {exc}"
    except Exception as exc:
        return f"Error: {exc}"

    if not images:
        return "Job completed but no output images were returned."

    return json.dumps({
        "image_count": len(images),
        "images": [
            {"filename": img["filename"], "data_url": img["data_url"]}
            for img in images
        ],
    })


# ── Tool: upscale_image ────────────────────────────────────────────────────────

@mcp.tool()
async def upscale_image(
    path: str,
    scale: float = 4.0,
    upscale_model: str = "",
) -> str:
    """
    Upscale a local image using an ESRGAN model in ComfyUI.

    Args:
        path:          Absolute or relative path to the image to upscale.
        scale:         Target scale factor relative to the source image.
                       The ESRGAN model runs at 4×; other values are
                       achieved by resizing after the model.
        upscale_model: Model filename in ComfyUI's models/upscale_models/
                       folder. Leave blank for server default
                       (COMFY_UPSCALE_MODEL env var, or RealESRGAN_x4plus.pth).

    Returns:
        JSON with base64 data URL of the upscaled image.
    """
    try:
        server_filename = await comfy.upload_image(path)
    except FileNotFoundError as exc:
        return str(exc)
    except Exception as exc:
        return f"Upload failed: {exc}"

    try:
        workflow = comfy.load_workflow("upscale")
    except FileNotFoundError as exc:
        return str(exc)

    model = upscale_model or os.getenv("COMFY_UPSCALE_MODEL", "RealESRGAN_x4plus.pth")

    # ESRGAN runs at 4×. scale_by here is relative to the post-ESRGAN image.
    scale_by = scale / 4.0

    params: dict = {
        "image":         server_filename,
        "upscale_model": model,
        "scale_by":      scale_by,
    }

    try:
        ready = comfy.inject_params(workflow, params)
        images = await comfy.run_workflow(ready)
    except comfy.ComfyError as exc:
        return f"ComfyUI error: {exc}"
    except Exception as exc:
        return f"Error: {exc}"

    if not images:
        return "Job completed but no output images were returned."

    return json.dumps({
        "image_count": len(images),
        "images": [
            {"filename": img["filename"], "data_url": img["data_url"]}
            for img in images
        ],
    })


# ── Tool: get_job_status ───────────────────────────────────────────────────────

@mcp.tool()
async def get_job_status(prompt_id: str) -> str:
    """
    Check the current status of a ComfyUI job by its prompt_id.

    Use this to poll an async job that was queued earlier.
    Returns one of: pending, running, completed, unknown.
    When completed, also returns the output image data URLs.

    Args:
        prompt_id: The prompt_id string returned by a previous workflow run.
    """
    try:
        status = await comfy.get_job_status(prompt_id)
    except Exception as exc:
        return f"Error fetching status: {exc}"

    result: dict = {"status": status["status"]}
    if status["status"] == "completed" and status["outputs"]:
        result["images"] = [
            {"filename": img["filename"], "data_url": img["data_url"]}
            for img in status["outputs"]
        ]
    return json.dumps(result)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
