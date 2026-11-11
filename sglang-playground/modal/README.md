# SGLang on Modal

Run these commands from this directory (`cd modal`):

```bash
just setup
just auth  # if not already authenticated
just serve
```

Keep the server terminal open for logs. Copy its serve URL and, in another
terminal in this directory, export Modal proxy credentials and send a request:

```bash
export MODAL_KEY="wk-..."
export MODAL_SECRET="ws-..."
just chat https://YOUR-SERVE-URL.modal.run
```

Use `just deploy` for a persistent deployment, `just logs` to follow app logs,
and `just stop` to stop the deployment. From the project root, use
`just --justfile modal/justfile <recipe>`; just runs these recipes in `modal/`.

The model is Qwen2.5-0.5B-Instruct; GPU preference is L40S, then L4, then A10.
The Hugging Face cache uses a Modal Volume. The endpoint requires proxy auth,
allows at most one container, and scales down after 60 idle seconds. GPU runs
use Modal credits/billing. Model downloads and kernel initialization add cold
start latency. The pinned CUDA image uses `python3 -m sglang.launch_server`;
its entrypoint recommendation warning can be ignored.

Request logging includes partial prompt/output content and every decode iteration.
See [Modal GPU docs](https://modal.com/docs/guide/gpu) and
[proxy tokens](https://modal.com/docs/guide/webhook-proxy-auth).
