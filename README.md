# video-creator

ComfyUI MCP server for Cursor. Exposes image-generation and editing as
structured tools that the Cursor agent can call from natural language.

```
Cursor Agent
    │  MCP tool call
    ▼
server/comfy_mcp_server.py
    │  HTTP + WebSocket
    ▼
ComfyUI  (running locally on :8188)
```

---

## Repository layout

```
workflows/
  txt2img.json        ComfyUI API-format template  — text-to-image
  img2img.json        ComfyUI API-format template  — image editing
  upscale.json        ComfyUI API-format template  — ESRGAN upscaling

server/
  comfy_client.py     Template loader, param injector, HTTP + WebSocket client
  comfy_mcp_server.py FastMCP server — run this file to start the MCP server

test_comfy.py         Standalone connection test (run before connecting Cursor)

.cursor/mcp.json      Cursor MCP config — auto-loaded when workspace is opened
requirements.txt      Python dependencies
```

---

## Quick start

### 1. Prerequisites

- Python 3.11+
- ComfyUI running locally on port 8188
- At least one checkpoint model in `ComfyUI/models/checkpoints/`
- `RealESRGAN_x4plus.pth` in `ComfyUI/models/upscale_models/` (for upscaling)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Test the connection (do this first)

Before touching Cursor, verify the full chain works:

```bash
# Connection + template checks only (no GPU usage)
python test_comfy.py

# Also submit a real txt2img job
python test_comfy.py --live

# Custom prompt for the live test
python test_comfy.py --live --prompt "a red apple on a white table"
```

Fix any errors the test reports before proceeding to step 4.

### 4. Connect Cursor

`.cursor/mcp.json` is already present. Cursor picks it up automatically
when this folder is opened as a workspace. No manual settings step needed.

To point it at a non-default ComfyUI host, edit the `env` block:

```json
{
  "mcpServers": {
    "comfy-local": {
      "command": "python",
      "args": ["server/comfy_mcp_server.py"],
      "cwd": "/absolute/path/to/video-creator",
      "env": {
        "COMFY_HOST": "127.0.0.1",
        "COMFY_PORT": "8188"
      }
    }
  }
}
```

### 5. Use from Cursor Composer

Once connected, the Cursor agent can call your ComfyUI tools:

> *"List the available workflows"*

> *"Generate a photorealistic portrait of an astronaut, 768×512"*

> *"Edit ~/images/draft.png — make the sky a dramatic sunset, strength 0.6"*

> *"Upscale ~/images/draft.png by 4×"*

---

## Available tools

| Tool | Description |
|---|---|
| `list_workflows` | Names of JSON templates in `workflows/` |
| `run_workflow` | Text-to-image via a named template |
| `edit_image` | Img2img — upload a local file, apply a prompt |
| `upscale_image` | ESRGAN upscale — upload a local file, scale by N× |
| `get_job_status` | Poll an async job by `prompt_id` |

---

## Adding your own workflows

1. Build and test a workflow in the ComfyUI web UI.
2. Enable **Dev Mode** in ComfyUI settings.
3. Export as **API format JSON**.
4. Add a `__params__` block at the top that maps parameter names to node/field injection points:

```json
{
  "__params__": {
    "prompt":   { "node": "6", "field": "text" },
    "width":    { "node": "5", "field": "width" },
    "height":   { "node": "5", "field": "height" }
  },
  "5": { "class_type": "EmptyLatentImage", "inputs": { ... } },
  "6": { "class_type": "CLIPTextEncode",   "inputs": { ... } }
}
```

5. Save as `workflows/<name>.json`.
6. Call `run_workflow(name="<name>", prompt="...")` from Cursor.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `COMFY_HOST` | `127.0.0.1` | ComfyUI host |
| `COMFY_PORT` | `8188` | ComfyUI port |
| `COMFY_JOB_TIMEOUT` | `300` | Seconds before a queued job times out |
| `COMFY_UPSCALE_MODEL` | `RealESRGAN_x4plus.pth` | Default ESRGAN model filename |
| `COMFY_WORKFLOWS_DIR` | `workflows/` (repo root) | Override workflow template location |

---

## Development

Run the MCP server directly:

```bash
python server/comfy_mcp_server.py
```

Inspect tools interactively with the MCP inspector:

```bash
mcp dev server/comfy_mcp_server.py
```
