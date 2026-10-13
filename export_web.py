"""
Export trained U-Net weights for the browser lab (web/index.html).

    python export_web.py

Writes web/<name>.bin (float32, already transposed to TensorFlow.js layout)
plus web/<name>.json (name -> [offset, shape]). Re-run after retraining.
"""

import base64
import json
import os

import numpy as np
import torch

from diffusion import CONDITIONAL_MODEL_PATH, MODEL_PATH, PROJECT_DIR

WEB_DIR = os.path.join(PROJECT_DIR, "web")


def export(checkpoint, name):
    state = torch.load(checkpoint, map_location="cpu")
    manifest, chunks, offset = {}, [], 0
    for key, tensor in state.items():
        a = tensor.float().numpy()
        if a.ndim == 4:      # conv: [out, in, kh, kw] -> [kh, kw, in, out]
            a = a.transpose(2, 3, 1, 0)
        elif a.ndim == 2 and not key.startswith("class_embedding"):
            a = a.T          # linear: [out, in] -> [in, out]   (embedding tables stay as-is)
        a = np.ascontiguousarray(a, dtype=np.float32)
        manifest[key] = [offset, list(a.shape)]
        chunks.append(a.tobytes())
        offset += a.size
    os.makedirs(WEB_DIR, exist_ok=True)
    data = b"".join(chunks)
    with open(os.path.join(WEB_DIR, name + ".bin"), "wb") as f:
        f.write(data)
    # The page loads <name>.bin.txt (base64), since artifact hosting only serves text/media types.
    with open(os.path.join(WEB_DIR, name + ".bin.txt"), "w") as f:
        f.write(base64.b64encode(data).decode())
    with open(os.path.join(WEB_DIR, name + ".json"), "w") as f:
        json.dump(manifest, f)
    print(f"{name}: {offset} floats")


if __name__ == "__main__":
    export(MODEL_PATH, "unet_uncond")
    if os.path.exists(CONDITIONAL_MODEL_PATH):
        export(CONDITIONAL_MODEL_PATH, "unet_cond")
