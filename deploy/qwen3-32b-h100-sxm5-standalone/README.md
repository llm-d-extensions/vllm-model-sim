# Qwen3-32B — H100 SXM5 Standalone

Single-instance deployment: one H100 SXM5 serving the full model (TP=1).

The Kubernetes stack under `k8s/` runs the calibrated **physics** simulator on
CPU nodes, with no GPUs required. `k8s/` is rendered from
[`stack.yaml`](stack.yaml), the catalog, the `default` profile and
[`calibrations/qwen3-32b/h100-sxm5.yaml`](../../calibrations/qwen3-32b/h100-sxm5.yaml);
see [deploy/README.md](../README.md) for the layout.

> [!WARNING]
> The shipped calibration, β = [0.152, 0.0, 126.0], is suspect: it gives about
> +441% ITL at concurrency 1 according to the llm-d-autoscaling-research
> findings. Re-tune with `eval.sh tune` before relying on its latencies.

## Deployment

### Kubernetes (simulated)

```bash
export VLLM_SIM_NAMESPACE=default  # or your target namespace
K=deploy/qwen3-32b-h100-sxm5-standalone/k8s

# Shared prerequisite (once per cluster/namespace)
kubectl apply -n $VLLM_SIM_NAMESPACE -f $K/pvc.yaml

# Simulator
kubectl apply -n $VLLM_SIM_NAMESPACE \
  -f $K/configmap.yaml -f $K/deployment.yaml -f $K/service.yaml
```

Skip `sim-config.json`: it is the plain-JSON copy of the ConfigMap's
`config.json`, not a Kubernetes resource.

Resource names follow `vllm-qwen3-32b-standalone-<hash>`, where `<hash>` is a
6-character SHA-256 of the sim config. Different calibrations therefore never
collide in one namespace.

#### Verify deployment

```bash
SVC=$(awk '/^  name:/ {print $2; exit}' $K/service.yaml)
kubectl port-forward -n $VLLM_SIM_NAMESPACE svc/$SVC 8000:8000

# In another terminal:
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-32b",
    "messages": [
      {"role": "user", "content": "Tell me a short story about a robot."}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Local (CPU, simulated)

> [!IMPORTANT]
> **vLLM must be built from source**: the PyPI package does not support
> CPU-only or macOS environments. See the [vLLM CPU build guide](https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html).

Install this plugin from the repo root (`uv pip install -e .`). Then serve a
directory whose `config.json` is the sim config:

```bash
mkdir -p "$TMPDIR/qwen3-32b-sim"
cp deploy/qwen3-32b-h100-sxm5-standalone/k8s/sim-config.json "$TMPDIR/qwen3-32b-sim/config.json"
vllm serve "$TMPDIR/qwen3-32b-sim" \
  --tokenizer Qwen/Qwen3-32B \
  --served-model-name qwen3-32b \
  --load-format dummy \
  --max-model-len 8192 --max-num-batched-tokens 8192 --max-num-seqs 1024 \
  --port 8000
```

To try another latency variant, render it and copy its `sim-config.json`
instead:

```bash
python -m simstack render deploy/qwen3-32b-h100-sxm5-standalone \
  --eval physics-beta-1.0 --out "$TMPDIR/variant"
```

## Latency variants

| Variant | Latency model |
|---------|---------------|
| `physics` | Roofline physics model with this stack's calibration (`python -m simstack resolve deploy/qwen3-32b-h100-sxm5-standalone`) |
| `physics-beta-1.0` | Roofline physics model, unit betas β = [1.0, 1.0, 0.0]: the uncalibrated baseline |
| `flat` | Linear model (`base_ms` + per-token/per-seq terms), a debugging aid |

## Eval (real vs sim comparison)

`eval.sh` renders the real and sim stacks for a variant into `.build/`, then
runs setup → sweep → teardown. The report lands under
`evaluation/results/qwen3-32b-h100-sxm5-standalone/<variant>/`; commit it.

```bash
bash evaluation/eval.sh deploy/qwen3-32b-h100-sxm5-standalone            # physics
bash evaluation/eval.sh deploy/qwen3-32b-h100-sxm5-standalone flat
```

The sweep matrix is [`sweep.yaml`](sweep.yaml). To re-calibrate, run
`eval.sh tune deploy/qwen3-32b-h100-sxm5-standalone`, then `eval.sh promote …`
to write the tuned beta into the calibration record and re-render `k8s/`. See
[evaluation/README.md](../../evaluation/README.md).
