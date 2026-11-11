"""Send a local OpenAI-compatible chat request using the Python standard library."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import threading
import time
import urllib.error
import urllib.request

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def send_request(prompt, port, max_tokens):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode()}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Cannot reach local server: {error.reason}. Start just serve first.") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Explain the KV cache in simple terms.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrent", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.concurrent < 1:
        parser.error("--concurrent must be at least 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    if args.concurrent == 1:
        try:
            result = send_request(args.prompt, args.port, args.max_tokens)
        except (RuntimeError, TimeoutError) as error:
            raise SystemExit(str(error)) from error
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    ready = threading.Event()

    def run_one(index):
        ready.wait()
        started = time.perf_counter()
        try:
            result = send_request(args.prompt, args.port, args.max_tokens)
            return {"request": index, "seconds": round(time.perf_counter() - started, 3),
                    "response": result}
        except Exception as error:
            return {"request": index, "seconds": round(time.perf_counter() - started, 3),
                    "error": str(error)}

    failed = 0
    with ThreadPoolExecutor(max_workers=args.concurrent) as pool:
        futures = [pool.submit(run_one, i + 1) for i in range(args.concurrent)]
        started = time.perf_counter()
        print(f"Sending {args.concurrent} concurrent requests to localhost:{args.port}", flush=True)
        ready.set()
        for future in as_completed(futures):
            result = future.result()
            failed += "error" in result
            print(json.dumps(result, ensure_ascii=False), flush=True)
    print(f"Completed {args.concurrent - failed}/{args.concurrent} successfully "
          f"in {time.perf_counter() - started:.3f}s", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
