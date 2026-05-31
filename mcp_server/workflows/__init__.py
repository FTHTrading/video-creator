"""
Workflow templates for common ComfyUI operations.

Each function returns a fully-formed ComfyUI API-format workflow dict that can
be passed directly to ComfyClient.queue_workflow().
"""

from .generate_image import build_generate_image_workflow
from .edit_image import build_edit_image_workflow
from .upscale import build_upscale_workflow

__all__ = [
    "build_generate_image_workflow",
    "build_edit_image_workflow",
    "build_upscale_workflow",
]
