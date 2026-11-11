# Local SGLang playground

The root files run SGLang on an Apple Silicon Mac using MLX. Remote GPU serving
is self-contained in [modal/](modal/README.md).

## Start locally

Requires Apple Silicon, macOS 14 or newer, git, uv, and just.

```bash
just setup  # one-time source/dependency installation
just serve  # keep open for startup and request logs
```

In another terminal:

```bash
just health
just chat "Explain the KV cache in simple terms."
```

The server binds to localhost:8000. Override the port consistently:

```bash
just serve 8001
just chat "Hello" 8001
```

`playground.py` configures local serving. `chat.py` sends a request and prints
formatted JSON with the answer and token usage. Neither requires Modal.
`just check` checks syntax. Earlier `*-local` recipe names remain aliases.

## Environment and logs

`just setup` installs SGLang from its upstream default branch into `.local/sglang`
and creates a separate Python 3.12 environment at `.venv-local`. It reuses an
existing source checkout without automatically updating it. Keep the checkout:
the package is installed in editable mode. Setup prints the installed revision.
Optional Rust extensions and custom Metal kernels are not built; this text-only
setup does not require Cargo. See the
[Apple Metal guide](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/apple_metal.mdx).

First startup downloads Qwen2.5-0.5B-Instruct to your local Hugging Face cache.
The server logs partial request inputs/outputs, each decode iteration, and
request timing stats. CUDA graphs are disabled on Metal. Press Ctrl-C to stop;
restart after editing the local launcher. There is no automatic idle shutdown.

## Modal

```bash
cd modal
just setup
just serve
```

See [modal/README.md](modal/README.md) for authentication, chat, deployment, and
logs. Modal dependencies and lockfile live in that directory; local and remote
SGLang versions can differ. An older root `.venv` from the original Modal setup
is unused; the new Modal setup creates `modal/.venv`.

## Concurrent requests

With `just serve` running, send N simultaneous non-streaming requests:

```bash
just chat-concurrent 4
just chat-concurrent 8 "Explain prefix caching in detail." 8000 512
```

Arguments are request count, prompt, port, and maximum output tokens. Every
request uses the same prompt. Results print as JSON in completion order with
request numbers, elapsed seconds, responses, and any errors. The command exits
with an error if any request fails. Timings include server queueing and response
generation; they are not time-to-first-token measurements. Watch `#running-req`
and `#queue-req` in the server logs to see batching. Short answers may finish
before all requests overlap.
