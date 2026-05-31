# video-creator

A ComfyUI MCP server that exposes image generation, editing, and upscaling as
tools callable from Cursor's agent (Composer).

```
Cursor agent
    │  MCP tool call
    ▼
mcp_server/server.py   ← this repo
    │  HTTP + WebSocket
    ▼
ComfyUI  (running locally on port 8188)
```

---

## Quick start

### 1. Prerequisites

- Python 3.11+
- A running [ComfyUI](https://github.com/comfyanonymous/ComfyUI) instance
  (defaults to `http://127.0.0.1:8188`)
- At least one checkpoint model in ComfyUI's `models/checkpoints/` folder
- *(Optional)* `RealESRGAN_x4plus.pth` in `models/upscale_models/` for upscaling

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

The server is configured via environment variables (all optional — defaults work
for a standard local ComfyUI install):

| Variable | Default | Description |
|---|---|---|
| `COMFY_HOST` | `127.0.0.1` | ComfyUI host |
| `COMFY_PORT` | `8188` | ComfyUI port |
| `COMFY_JOB_TIMEOUT` | `300` | Seconds before a queued job times out |
| `COMFY_DEFAULT_CHECKPOINT` | `v1-5-pruned-emaonly.ckpt` | Default checkpoint model |
| `COMFY_UPSCALE_MODEL` | `RealESRGAN_x4plus.pth` | Default ESRGAN upscale model |
| `COMFY_WORKFLOWS_DIR` | `mcp_server/saved_workflows/` | Folder for named workflow JSON files |

### 4. Add to Cursor

The `.cursor/mcp.json` file in this repo is pre-configured. Cursor will pick it
up automatically when you open this folder as a workspace.

Alternatively, add the server manually via **Cursor Settings → MCP**:

```json
{
  "mcpServers": {
    "comfyui": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/video-creator",
      "env": {
        "COMFY_HOST": "127.0.0.1",
        "COMFY_PORT": "8188"
      }
    }
  }
}
```

### 5. Use from Cursor agent

Once the MCP server is registered, the Cursor Composer/agent can call your
ComfyUI tools with natural language:

> *"Generate a photorealistic sunset over a mountain lake, 768×512"*

> *"Edit ~/images/portrait.png — make the background blurred bokeh, strength 0.6"*

> *"Upscale ~/images/draft.png by 4×"*

> *"List available workflows"*

---

## Available tools

| Tool | Description |
|---|---|
| `comfy_generate_image` | Text-to-image with style presets |
| `comfy_edit_image` | Image-to-image editing (img2img) |
| `comfy_upscale_image` | ESRGAN-based resolution upscaling |
| `comfy_list_workflows` | List named workflow JSON files |
| `comfy_run_named_workflow` | Run a saved workflow with optional input overrides |
| `comfy_list_models` | List checkpoints, LoRAs, and VAEs available in ComfyUI |
| `comfy_queue_status` | Show queue depth and VRAM usage |
| `comfy_interrupt` | Cancel the currently running job |

### Style presets for `comfy_generate_image`

| Preset | Best for |
|---|---|
| `default` | General purpose |
| `photorealistic` | Photographs, realistic scenes |
| `anime` | Anime / manga style |
| `painting` | Oil paintings, concept art |
| `cinematic` | Cinematic stills, dramatic lighting |

---

## Saving named workflows

1. Design a workflow in the ComfyUI web UI.
2. Export it as **API format JSON** (enable Dev Mode in ComfyUI settings first).
3. Save it to `mcp_server/saved_workflows/<name>.json`.
4. Call `comfy_run_named_workflow` with `name="<name>"` from Cursor.

You can inject dynamic values at runtime using the `inputs` parameter:

```
comfy_run_named_workflow(
    name="my_portrait_workflow",
    inputs='{"6": {"text": "a smiling astronaut"}}'
)
```

---

## Development

Run the MCP server directly for testing:

```bash
python -m mcp_server.server
```

Inspect tools interactively with the MCP dev UI:

```bash
mcp dev mcp_server/server.py
```
