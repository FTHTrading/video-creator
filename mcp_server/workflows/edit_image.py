"""
Image-to-image editing workflow template (img2img via KSampler with denoising).

The caller must first upload the input image using ComfyClient.upload_image()
and pass the returned server filename here.
"""

import os
from typing import Any

DEFAULT_CHECKPOINT = os.getenv("COMFY_DEFAULT_CHECKPOINT", "v1-5-pruned-emaonly.ckpt")


def build_edit_image_workflow(
    image_filename: str,
    prompt: str,
    strength: float = 0.75,
    checkpoint: str | None = None,
    negative_prompt: str = "ugly, blurry, bad quality, watermark, text",
    steps: int = 30,
    cfg: float = 7.0,
    sampler: str = "euler",
    scheduler: str = "normal",
    seed: int = -1,
    image_subfolder: str = "",
) -> dict:
    """
    Build an image-to-image ComfyUI workflow.

    Args:
        image_filename:  Server-side filename returned by upload_image().
        prompt:          Positive text prompt describing the desired edit.
        strength:        Denoising strength (0 = no change, 1 = full redraw).
        checkpoint:      Checkpoint model filename.
        negative_prompt: What to avoid in the output.
        steps:           Sampling steps.
        cfg:             CFG guidance scale.
        sampler:         KSampler sampler name.
        scheduler:       KSampler scheduler name.
        seed:            RNG seed (-1 = random).
        image_subfolder: Subfolder on the server where the image lives.

    Returns:
        ComfyUI API-format workflow dict.
    """
    ckpt = checkpoint or DEFAULT_CHECKPOINT
    rng_seed = seed if seed >= 0 else __import__("random").randint(0, 2**32 - 1)
    denoise = max(0.0, min(1.0, strength))

    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt},
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {
                "image": image_filename if not image_subfolder else f"{image_subfolder}/{image_filename}",
            },
        },
        "3": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["2", 0],
                "vae": ["1", 2],
            },
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["1", 1],
            },
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["1", 1],
            },
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["3", 0],
                "seed": rng_seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": denoise,
            },
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["6", 0],
                "vae": ["1", 2],
            },
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["7", 0],
                "filename_prefix": "mcp_edit",
            },
        },
    }
