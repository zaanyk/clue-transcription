# Compatibility alias for RunPod templates that look for rp_handler.py
from handler import handler

import runpod

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
