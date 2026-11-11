"""Run the HTTP server with: just serve"""

import subprocess

import modal

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PORT = 8000
STARTUP_TIMEOUT = 600

app = modal.App("sglang-playground")
cache = modal.Volume.from_name("sglang-playground-hf-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.10.post1-cu130-runtime")
    .entrypoint([])
    .env({"HF_HOME": "/cache/huggingface"})
)


def start_server():
    """Start SGLang inside the GPU container; logs go to Modal output."""
    return subprocess.Popen(
        [
            "python3", "-m", "sglang.launch_server",
            "--model-path", MODEL,
            "--served-model-name", MODEL,
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "--context-length", "4096",
            "--mem-fraction-static", "0.7",
            "--log-requests",
            "--log-requests-level", "2",
            "--decode-log-interval", "1",
            "--enable-request-time-stats-logging",
        ]
    )



@app.function(
    image=image, gpu=["L40S", "L4", "A10"], volumes={"/cache": cache}, timeout=900,
    scaledown_window=60, max_containers=1,
)
@modal.concurrent(max_inputs=10)
@modal.web_server(PORT, startup_timeout=STARTUP_TIMEOUT, requires_proxy_auth=True)
def serve():
    """OpenAI-compatible endpoint, protected by Modal proxy auth."""
    start_server()

