# SGLang on Modal

With [just](https://just.systems/) installed, run `just` to list recipes:

```bash
just setup
just auth  # only if needed
just serve
```

Other recipes: `just check`, `just deploy`, `just logs`, `just stop`, and
`just chat https://YOUR-SERVE-URL.modal.run` (requires proxy token environment variables).

Serve **Qwen2.5-0.5B-Instruct** with SGLang on a single GPU, preferring **L40S** with **L4** and **A10** fallbacks.
Only the Modal SDK runs locally; CUDA, SGLang, and model weights stay on Modal.
The ungated model needs no Hugging Face token. The container image is pinned;
model weights follow the model repository's default revision.

## Get started

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync
uv run modal setup  # only if you haven't already authenticated
uv run modal serve playground.py
```

The first request starts SGLang on a remote GPU and downloads model weights, so
allow several minutes. Later runs reuse the image and persistent Hugging Face
cache volume. GPU execution uses your Modal credits/billing.

## Keep an HTTP endpoint available

For a temporary development endpoint, leave this running in a terminal:

```bash
uv run modal serve playground.py
```

For a deployed endpoint:

```bash
uv run modal deploy playground.py
```

Copy the `serve` URL printed by Modal. The endpoint uses Modal proxy
authentication. Create a proxy auth token in your Modal workspace and export
its key and secret as `MODAL_KEY` and `MODAL_SECRET` (these are distinct from
your Modal CLI credentials).

```bash
export SGLANG_URL="https://YOUR-SERVE-URL.modal.run"
curl --fail-with-body "$SGLANG_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64}'
```

The endpoint scales to zero after 60 idle seconds and allows at most one GPU
container. Requests after scale-down incur a cold start.
Stop a deployed app with:

```bash
uv run modal app stop sglang-playground
```

## Explore

- Change `MODEL` in `playground.py` to try another supported small chat model.
- Increase `--context-length` for longer prompts, subject to GPU memory.
- Inspect server logs in the Modal dashboard when startup fails.
- A missing-token error means you need `uv run modal setup`.
- Local check: `uv run python -m py_compile playground.py`.

References: [Modal web servers](https://modal.com/docs/guide/webhooks),
[Modal proxy authentication](https://modal.com/docs/guide/webhook-proxy-auth),
[Modal SGLang example](https://modal.com/docs/examples/sglang_vlm),
[SGLang](https://github.com/sgl-project/sglang).
