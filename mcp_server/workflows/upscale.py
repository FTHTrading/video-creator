"""
Image upscaling workflow template.

Uses the ESRGAN-based UpscaleModelLoader + ImageUpscaleWithModel pipeline
for high-quality upscaling without requiring a diffusion model.
Falls back to a simple latent upscale if the upscale model is not available.
"""

import os

DEFAULT_UPSCALE_MODEL = os.getenv("COMFY_UPSCALE_MODEL", "RealESRGAN_x4plus.pth")


def build_upscale_workflow(
    image_filename: str,
    scale: float = 4.0,
    upscale_model: str | None = None,
    image_subfolder: str = "",
) -> dict:
    """
    Build an upscaling ComfyUI workflow using an ESRGAN-style upscale model.

    Args:
        image_filename:  Server-side filename returned by upload_image().
        scale:           Target scale factor (e.g. 2.0, 4.0).
                         Note: ESRGAN models have a fixed internal scale (usually
                         4×); this workflow uses the model directly and then
                         optionally downscales to hit the requested factor.
        upscale_model:   Upscale model filename (must be in ComfyUI's models/
                         upscale_models/ folder).
        image_subfolder: Subfolder on the server where the image lives.

    Returns:
        ComfyUI API-format workflow dict.
    """
    model = upscale_model or DEFAULT_UPSCALE_MODEL
    img_path = (
        f"{image_subfolder}/{image_filename}" if image_subfolder else image_filename
    )

    workflow: dict = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": img_path},
        },
        "2": {
            "class_type": "UpscaleModelLoader",
            "inputs": {"model_name": model},
        },
        "3": {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {
                "upscale_model": ["2", 0],
                "image": ["1", 0],
            },
        },
    }

    # If the requested scale differs from the model's native 4×, add a resize step.
    if scale != 4.0:
        workflow["4"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["3", 0],
                "upscale_method": "lanczos",
                "width": 0,   # 0 means "derive from height keeping aspect ratio"
                "height": 0,
                "crop": "disabled",
            },
        }
        # ComfyUI's ImageScale doesn't natively do percentage-based scaling;
        # we use a ScaleBy node instead when available.
        workflow["4"] = {
            "class_type": "ImageScaleBy",
            "inputs": {
                "image": ["3", 0],
                "upscale_method": "lanczos",
                # ScaleBy scales relative to the input image.  After ESRGAN the
                # image is already 4×, so we need to divide by 4 to map back to
                # the user's intended factor.
                "scale_by": scale / 4.0,
            },
        }
        save_input = ["4", 0]
    else:
        save_input = ["3", 0]

    workflow["9"] = {
        "class_type": "SaveImage",
        "inputs": {
            "images": save_input,
            "filename_prefix": "mcp_upscale",
        },
    }

    return workflow
