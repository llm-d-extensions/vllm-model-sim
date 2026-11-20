# Qwen3-32B — H100 SXM5 Prefill/Decode Disaggregation

Prefill/decode disaggregation at TP=1 per role (two GPUs in total). A prefill
instance processes prompt tokens and transfers KV caches to a decode instance
via `NixlConnector`. The llm-d sidecar coordinates the two phases.

The Kubernetes stack under `k8s/` runs the calibrated **physics** simulator as
CPU pods with dummy weights, so you can exercise disaggregated serving and
vLLM's KV-transfer path without any H100s. `k8s/` is rendered from
[`stack.yaml`](stack.yaml), the catalog, the `default` profile and
[`calibrations/qwen3-32b/h100-sxm5.yaml`](../../calibrations/qwen3-32b/h100-sxm5.yaml);
see [deploy/README.md](../README.md) for the layout.

> [!NOTE]
> The simulated P/D stack (CPU + `SimulatedNixlConnector` + dummy weights) has
> not yet been validated against a live cluster; treat it as a starting point.

## Deployment

### Dependencies image

Both sim pods copy the plugin from an init container,
`ghcr.io/llm-d-extensions/vllm-sim-deps` (version pinned in
[`profiles/default.yaml`](../../profiles/default.yaml)). The NIXL connector
itself comes from the base image. See
[docker/vllm-sim-deps/VERSIONS.md](../../docker/vllm-sim-deps/VERSIONS.md) for
version compatibility.

### Kubernetes (simulated)

```bash
export VLLM_SIM_NAMESPACE=default  # or your target namespace
K=deploy/qwen3-32b-h100-sxm5-pd/k8s
kubectl apply -n $VLLM_SIM_NAMESPACE \
  -f $K/configmap.yaml \
  -f $K/prefill-deployment.yaml -f $K/prefill-service.yaml \
  -f $K/decode-deployment.yaml  -f $K/decode-service.yaml
```

Skip `sim-config.json`: it is the plain-JSON copy of the ConfigMap's
`config.json`, not a Kubernetes resource.

This deploys:

| Resource | Role | Notes |
|----------|------|-------|
| `vllm-qwen3-32b-pd-<hash>-prefill` | `kv_role: kv_producer` | vLLM on port 8000; NIXL side channel on port 5600 |
| `vllm-qwen3-32b-pd-<hash>-decode` | `kv_role: kv_consumer` + sidecar | **sidecar on port 8000** (client-facing); vLLM on port 8200 (internal); NIXL on port 5601 |

`<hash>` is a 6-character SHA-256 of the sim config. If the prefill and decode
roles resolve to different calibrations, each gets its own ConfigMap.

The decode pod runs the [llm-d](https://github.com/llm-d/llm-d-router)
disaggregation sidecar (`ghcr.io/llm-d/llm-d-router-disagg-sidecar`) with
`--kv-connector=nixlv2`. Send all client traffic to the decode Service on port
8000. The sidecar forwards the prefill phase to the prefill Service, coordinates
the KV transfer, then hands decode to the local vLLM worker on port 8200.

**HF token:** both backends read `HF_TOKEN` from a Secret named `hf-token`
(key: `token`). It is needed only if the tokenizer (`Qwen/Qwen3-32B`) is gated;
the sim uses dummy weights, so no model download occurs.

```bash
kubectl create secret generic hf-token -n $VLLM_SIM_NAMESPACE --from-literal=token=hf_...
```

> [!TIP]
> The secret reference is `optional: true`, so pods start without it if the
> tokenizer is already cached or publicly accessible.

**Readiness:** the sim starts quickly, with no ~64 GB weight download. Each vLLM
container's `startupProbe` allows up to ~150 s (plugin install + CPU engine
init). The sidecar's `/health` on port 8000 is checked independently.

**Node placement:** the sim pods request 4 CPU / 8 GiB each (per TP rank) and
no GPU, so they schedule on ordinary CPU nodes.

#### Verify deployment

```bash
DECODE=$(awk '/^  name:/ {print $2; exit}' $K/decode-service.yaml)
PREFILL=$(awk '/^  name:/ {print $2; exit}' $K/prefill-service.yaml)
kubectl port-forward -n $VLLM_SIM_NAMESPACE svc/$DECODE 8000:8000

# In another terminal:
curl http://localhost:8000/health
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-prefiller-host-port: $PREFILL:8000" \
  -d '{
    "model": "qwen3-32b",
    "messages": [
      {"role": "user", "content": "Tell me a short story about a robot."}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

> [!NOTE]
> The `x-prefiller-host-port` header explicitly triggers P/D disaggregation.

### Local (CPU, simulated)

> [!IMPORTANT]
> **vLLM must be built from source**: the PyPI package does not support
> CPU-only or macOS environments. See the [vLLM CPU build guide](https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html).

`SimulatedNixlConnector` does **not** need NIXL, UCX or RDMA hardware. It
models KV transfer latency with a bandwidth model (100 Gbps by default) and
moves no data, so this runs on any platform, including macOS.

Install this plugin from the repo root (`uv pip install -e .`) and prepare a
model directory from the rendered sim config:

```bash
mkdir -p "$TMPDIR/qwen3-32b-sim"
cp deploy/qwen3-32b-h100-sxm5-pd/k8s/sim-config.json "$TMPDIR/qwen3-32b-sim/config.json"
ENGINE="--max-model-len 8192 --max-num-batched-tokens 8192 --max-num-seqs 1024"
```

**Terminal 1 — prefill:**
```bash
VLLM_NIXL_SIDE_CHANNEL_HOST=localhost VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve "$TMPDIR/qwen3-32b-sim" --tokenizer Qwen/Qwen3-32B \
  --served-model-name qwen3-32b --load-format dummy $ENGINE --port 8000 \
  --kv-transfer-config '{"kv_connector":"SimulatedNixlConnector","kv_role":"kv_producer","kv_connector_extra_config":{"bandwidth_gbps":100,"handshake_ms":2}}'
```

**Terminal 2 — decode:**
```bash
VLLM_NIXL_SIDE_CHANNEL_HOST=localhost VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve "$TMPDIR/qwen3-32b-sim" --tokenizer Qwen/Qwen3-32B \
  --served-model-name qwen3-32b --load-format dummy $ENGINE --port 8200 \
  --kv-transfer-config '{"kv_connector":"SimulatedNixlConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"bandwidth_gbps":100,"handshake_ms":2}}'
```

**Terminal 3 — coordination** (after both servers are ready), using vLLM's
proxy script from the vLLM repo root:

```bash
python3 examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py \
  --model qwen3-32b --prefill localhost:8000 --decode localhost:8200 --port 8080
```

Send requests to `http://localhost:8080`. The Kubernetes deployment uses the
llm-d sidecar instead; to use it locally, build the `pd-sidecar` binary from
[llm-d-router](https://github.com/llm-d/llm-d-router).

## Latency variants

| Variant | Latency model |
|---------|---------------|
| `physics` | Roofline physics model with this stack's calibration (`python -m simstack resolve deploy/qwen3-32b-h100-sxm5-pd`) |
| `physics-beta-1.0` | Roofline physics model, unit betas β = [1.0, 1.0, 0.0]: the uncalibrated baseline |
| `flat` | Linear model (`base_ms` + per-token/per-seq terms), a debugging aid |

## Eval (real vs sim comparison)

```bash
bash evaluation/eval.sh deploy/qwen3-32b-h100-sxm5-pd            # physics
bash evaluation/eval.sh tune deploy/qwen3-32b-h100-sxm5-pd       # re-calibrate
bash evaluation/eval.sh promote deploy/qwen3-32b-h100-sxm5-pd    # persist the tuned beta
```

Reports land under `evaluation/results/qwen3-32b-h100-sxm5-pd/<variant>/`. The
sweep matrix is [`sweep.yaml`](sweep.yaml). In P/D mode the tuner drives the
decode pod's `/sim/config`. See [evaluation/README.md](../../evaluation/README.md).
