#!/usr/bin/env python3
"""
Standalone ComfyUI connection test.

Run this BEFORE wiring Cursor. It validates every step of the chain:
  1. Reachability check         — GET /system_stats
  2. Model availability         — GET /object_info (lists checkpoints)
  3. Workflow template loading  — parses all three JSON files
  4. Parameter injection        — dry-run without submitting
  5. Live txt2img run           — submits a real job and waits for output
     (only runs when --live flag is passed, to avoid accidental GPU usage)

Usage:
    python test_comfy.py                   # connection + template checks only
    python test_comfy.py --live            # also submits a real txt2img job
    python test_comfy.py --live --prompt "a red apple on a white table"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlencode

import aiohttp

# Resolve imports regardless of cwd.
sys.path.insert(0, str(Path(__file__).parent))
import server.comfy_client as comfy

BASE_URL = comfy.BASE_URL
PASS     = "\033[92m✓\033[0m"
FAIL     = "\033[91m✗\033[0m"
INFO     = "\033[94m·\033[0m"


def header(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def ok(msg: str) -> None:
    print(f"  {PASS}  {msg}")


def fail(msg: str) -> None:
    print(f"  {FAIL}  {msg}")
    sys.exit(1)


def info(msg: str) -> None:
    print(f"  {INFO}  {msg}")


# ── Step 1: reachability ───────────────────────────────────────────────────────

async def check_reachability() -> None:
    header("1. ComfyUI reachability")
    info(f"Connecting to {BASE_URL}")
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{BASE_URL}/system_stats", timeout=aiohttp.ClientTimeout(total=5)) as r:
                r.raise_for_status()
                stats = await r.json()
        devices = stats.get("devices", [])
        if devices:
            d = devices[0]
            vram_free  = d.get("vram_free", "?")
            vram_total = d.get("vram_total", "?")
            ok(f"Connected  —  VRAM {vram_free}/{vram_total} bytes free")
        else:
            ok("Connected (no GPU device info returned)")
    except aiohttp.ClientConnectorError:
        fail(
            f"Cannot reach ComfyUI at {BASE_URL}. "
            "Make sure ComfyUI is running and COMFY_HOST/COMFY_PORT env vars are correct."
        )
    except Exception as exc:
        fail(f"Unexpected error: {exc}")


# ── Step 2: model availability ─────────────────────────────────────────────────

async def check_models() -> None:
    header("2. Available models")
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{BASE_URL}/object_info") as r:
                r.raise_for_status()
                info_data = await r.json()

        loader = info_data.get("CheckpointLoaderSimple", {})
        checkpoints = (
            loader.get("input", {})
                  .get("required", {})
                  .get("ckpt_name", [[]])[0]
        )
        if checkpoints:
            ok(f"Found {len(checkpoints)} checkpoint(s):")
            for ckpt in checkpoints[:5]:
                info(f"  {ckpt}")
            if len(checkpoints) > 5:
                info(f"  … and {len(checkpoints) - 5} more")
        else:
            fail(
                "No checkpoints found in ComfyUI. "
                "Add a model to ComfyUI/models/checkpoints/ before proceeding."
            )

        upscale_loader = info_data.get("UpscaleModelLoader", {})
        upscale_models = (
            upscale_loader.get("input", {})
                          .get("required", {})
                          .get("model_name", [[]])[0]
        )
        if upscale_models:
            ok(f"Found {len(upscale_models)} upscale model(s): {upscale_models[:3]}")
        else:
            info(
                "No upscale models found. "
                "Add RealESRGAN_x4plus.pth to ComfyUI/models/upscale_models/ "
                "to use upscale_image."
            )

    except Exception as exc:
        fail(f"Failed to fetch object_info: {exc}")


# ── Step 3 & 4: template loading and param injection ──────────────────────────

def check_templates() -> None:
    header("3. Workflow templates")
    names = comfy.list_workflows()
    if not names:
        fail(f"No workflow JSON files found in {comfy.WORKFLOWS_DIR}")
    ok(f"Found templates: {names}")

    header("4. Parameter injection dry-run")
    for name in names:
        try:
            wf = comfy.load_workflow(name)
            params = {
                "prompt":          "test prompt",
                "negative_prompt": "test negative",
                "width":           512,
                "height":          512,
                "seed":            1234,
                "image":           "dummy.png",
                "denoise":         0.75,
                "scale_by":        1.0,
                "upscale_model":   "RealESRGAN_x4plus.pth",
            }
            ready = comfy.inject_params(wf, params)
            assert "__params__" not in ready, "__params__ was not stripped"
            ok(f"  {name}.json — injection OK, {len(ready)} nodes")
        except Exception as exc:
            fail(f"  {name}.json — {exc}")


# ── Step 5: live txt2img run ───────────────────────────────────────────────────

async def run_live(prompt: str) -> None:
    header("5. Live txt2img job")
    info(f"Prompt: {prompt!r}")
    info("Submitting to ComfyUI … (this may take 30–120s)")

    try:
        wf = comfy.load_workflow("txt2img")
        ready = comfy.inject_params(wf, {
            "prompt":          prompt,
            "negative_prompt": "ugly, blurry",
            "width":           512,
            "height":          512,
            "seed":            42,
            "steps":           20,
            "cfg":             7.0,
        })
        images = await comfy.run_workflow(ready)
    except comfy.ComfyError as exc:
        fail(f"ComfyUI error: {exc}")
    except Exception as exc:
        fail(f"Unexpected error: {exc}")

    if not images:
        fail("Job finished but returned no images.")

    ok(f"Generated {len(images)} image(s)")
    for img in images:
        info(f"  filename: {img['filename']}  "
             f"  data_url length: {len(img['data_url'])} chars")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main(live: bool, prompt: str) -> None:
    print(f"\033[1mComfyUI connection test\033[0m  →  {BASE_URL}")
    await check_reachability()
    await check_models()
    check_templates()

    if live:
        await run_live(prompt)
    else:
        print(
            "\n\033[93mSkipped live job (pass --live to submit a real workflow).\033[0m"
        )

    print("\n\033[1mAll checks passed — ready to connect Cursor.\033[0m\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ComfyUI connection test")
    parser.add_argument("--live",   action="store_true",
                        help="Submit a real txt2img job")
    parser.add_argument("--prompt", default="a red apple on a white table",
                        help="Prompt for the live test")
    args = parser.parse_args()
    asyncio.run(main(args.live, args.prompt))
