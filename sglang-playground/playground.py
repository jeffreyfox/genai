"""Local Apple Silicon SGLang server. Run with just serve."""

import argparse
import os
import runpy
import sys

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.environ["SGLANG_USE_MLX"] = "1"
    sys.argv = [
        "sglang.launch_server",
        "--model-path", MODEL,
        "--host", "127.0.0.1", "--port", str(args.port),
        "--context-length", "4096", "--disable-cuda-graph",
        "--log-requests", "--log-requests-level", "2",
        "--decode-log-interval", "1",
        "--enable-request-time-stats-logging",
    ]
    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
