"""
Text-to-image workflow template.

Uses the standard KSampler + VAE decode pipeline with a configurable
checkpoint, prompt, resolution, steps, CFG, and sampler.

Style presets map to (negative_prompt_addon, steps, cfg, sampler) defaults
so callers only need to pass a style name.
"""

import os
from typing import Any

DEFAULT_CHECKPOINT = os.getenv("COMFY_DEFAULT_CHECKPOINT", "v1-5-pruned-emaonly.ckpt")

STYLE_PRESETS: dict[str, dict[str, Any]] = {
    "photorealistic": {
        "negative": "cartoon, painting, illustration, (worst quality, low quality, normal quality:2)",
        "steps": 30,
        "cfg": 7.0,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
    },
    "anime": {
        "negative": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        "steps": 28,
        "cfg": 7.5,
        "sampler": "euler_ancestral",
        "scheduler": "normal",
    },
    "painting": {
        "negative": "photo, photograph, 3d render, ugly, blurry, bad quality",
        "steps": 35,
        "cfg": 8.0,
        "sampler": "dpm_2_ancestral",
        "scheduler": "normal",
    },
    "cinematic": {
        "negative": "cartoon, anime, sketch, (worst quality, low quality:2)",
        "steps": 30,
        "cfg": 6.5,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
    },
    "default": {
        "negative": "ugly, blurry, bad quality, watermark, text",
        "steps": 25,
        "cfg": 7.0,
        "sampler": "euler",
        "scheduler": "normal",
    },
}


def build_generate_image_workflow(
    prompt: str,
    style: str = "default",
    width: int = 512,
    height: int = 512,
    checkpoint: str | None = None,
    seed: int = -1,
    negative_prompt: str | None = None,
    steps: int | None = None,
    cfg: float | None = None,
    sampler: str | None = None,
    scheduler: str | None = None,
) -> dict:
    """
    Build a text-to-image ComfyUI workflow.

    Args:
        prompt:          Positive text prompt.
        style:           One of the STYLE_PRESETS keys (default "default").
        width:           Output image width in pixels.
        height:          Output image height in pixels.
        checkpoint:      Checkpoint model filename. Falls back to env var or preset.
        seed:            RNG seed (-1 = random).
        negative_prompt: Override the style's negative prompt.
        steps:           Override sampling steps.
        cfg:             Override CFG scale.
        sampler:         Override sampler name.
        scheduler:       Override scheduler name.

    Returns:
        ComfyUI API-format workflow dict.
    """
    preset = STYLE_PRESETS.get(style, STYLE_PRESETS["default"])
    ckpt = checkpoint or DEFAULT_CHECKPOINT
    neg = negative_prompt or preset["negative"]
    n_steps = steps or preset["steps"]
    cfg_scale = cfg if cfg is not None else preset["cfg"]
    sampler_name = sampler or preset["sampler"]
    scheduler_name = scheduler or preset["scheduler"]
    rng_seed = seed if seed >= 0 else __import__("random").randint(0, 2**32 - 1)

    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["1", 1],
            },
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": neg,
                "clip": ["1", 1],
            },
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": rng_seed,
                "steps": n_steps,
                "cfg": cfg_scale,
                "sampler_name": sampler_name,
                "scheduler": scheduler_name,
                "denoise": 1.0,
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["5", 0],
                "vae": ["1", 2],
            },
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["6", 0],
                "filename_prefix": "mcp_generate",
            },
        },
    }
