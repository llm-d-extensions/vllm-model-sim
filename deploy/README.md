# Stacks and the configuration layout

A simulated deployment is assembled from four kinds of data, each stored
once. The `simstack` tool renders Kubernetes manifests from them.

```
catalog/
  hardware/<gpu>.yaml              # peak TFLOPs, HBM bandwidth, node selector,
                                   #   vLLM's scheduler defaults for that GPU
  models/<model>/config.json       # HF config.json, unmodified
  models/<model>/model.yaml        # HF id, served model name, weight dtype
calibrations/<model>/<gpu>.yaml    # fitted latency records, with provenance
profiles/<name>.yaml               # engine settings shared by real and sim pods,
                                   #   images and pod sizing
deploy/<stack>/
  stack.yaml                       # model, hardware, topology, tp, profile
  sweep.yaml                       # eval benchmark matrix
  README.md
  k8s/                             # rendered sim stack (committed, do not edit)
evaluation/results/<stack>/<variant>/   # eval reports
.build/<stack>/<variant>/          # rendered real+sim eval stack (git-ignored)
```

See [docs/2026-11-20-models-dir-structure-alternatives.md](../docs/2026-11-20-models-dir-structure-alternatives.md)
for why the data is split this way.

## Stacks

| Stack | Model | Hardware | Topology | TP | Calibration |
|-------|-------|----------|----------|----|-------------|
| [qwen3-32b-h100-sxm5-standalone](qwen3-32b-h100-sxm5-standalone/) | Qwen3-32B | H100 SXM5 | standalone | 1 | fitted (suspect) |
| [qwen3-32b-h100-sxm5-pd](qwen3-32b-h100-sxm5-pd/) | Qwen3-32B | H100 SXM5 | P/D | 1 | fitted |
| [qwen3-32b-h100-sxm5-pd-tp2](qwen3-32b-h100-sxm5-pd-tp2/) | Qwen3-32B | H100 SXM5 | P/D | 2 | copied from TP=1 |

## What goes where

| Setting | Lives in | Why |
|---------|----------|-----|
| Model architecture | `catalog/models/<model>/config.json` | The physics model derives FLOPs and bytes from it |
| GPU specs | `catalog/hardware/<gpu>.yaml` | The physics model divides by them |
| `max-num-batched-tokens`, `max-num-seqs`, `max-model-len` | the hardware's `vllm_defaults`, overridable in the profile | The sim runs vLLM's real scheduler, so these shape the batches it costs. Real and sim pods get the same values. |
| Fitted betas | `calibrations/<model>/<gpu>.yaml` | The only per-combination data |
| Images, CPU/memory sizing, storage class | `profiles/<name>.yaml` | Cluster plumbing |

> [!IMPORTANT]
> vLLM picks different scheduler defaults per platform: 8192/1024 on H100,
> 2048/256 on A100/L40S, 2048/128 on CPU. The renderer pins the target GPU's
> values on both pods. Otherwise the CPU sim would silently schedule like a
> CPU, not like the GPU it simulates.

## Calibration records

Each record in `calibrations/<model>/<gpu>.yaml` has a `match`, a `latency`
block and its provenance:

```yaml
- match: {topology: pd, tp: 1}           # keys: topology, tp, role
  latency: {type: physics, beta: [3.119229, 1.182904, 0.146612]}
  method: fitted                         # fitted | copied | assumed
  source: evaluation/results/.../tuned-sim-config.json
```

For each sim pod (`role` is `prefill` or `decode` in P/D stacks), the renderer
picks the record of the variant's latency type whose `match` keys all equal
the pod's values:

- **The most specific match wins.** A record that sets more `match` keys beats
  one that sets fewer.
- **A tie is an error.** Two equally specific records that both match are
  refused, never resolved silently.
- **No physics match falls back to unit betas** `[1.0, 1.0, 0.0]`, the
  uncalibrated roofline.
- **A record with no `match` applies everywhere.** It is how the `flat`
  variant's parameters are shared.

So a new combination needs no calibration file to render. Add a record only
when you have measured one. The rendered ConfigMap carries the chosen record's
`method` and `match` as annotations, and `resolve` shows the full record:

```bash
python -m simstack resolve deploy/qwen3-32b-h100-sxm5-pd
```

## Commands

```bash
python -m simstack render --all          # re-render every deploy/*/k8s/
python -m simstack check                 # fail if any k8s/ is stale (also a test)
python -m simstack render deploy/<stack> --eval physics --out <dir>
python -m simstack resolve deploy/<stack> [--variant flat]
python -m simstack promote deploy/<stack> --tuned <tuned-sim-config.json>
```

`evaluation/eval.sh <stack-dir> [variant]` calls `render --eval` itself;
`eval.sh promote` calls `promote`.

## Adding things

**A stack** for an existing model and GPU: create `deploy/<stack>/stack.yaml`
and `sweep.yaml`, run `python -m simstack render deploy/<stack>`, write its
README and add it to the table above.

```yaml
model: qwen3-32b
hardware: h100-sxm5
topology: standalone   # or pd
tp: 1
profile: default
```

**A GPU:** add `catalog/hardware/<gpu>.yaml`, copying an existing file. Use the
dense BF16 peak, not the sparsity figure. Take `vllm_defaults` from
`EngineArgs.get_batch_defaults` in vLLM's
`vllm/engine/arg_utils.py`: it depends on GPU memory and on whether the GPU is
an A100.

**A model:** add `catalog/models/<slug>/config.json` (from HuggingFace:
`huggingface-cli download <HF-id> config.json --local-dir catalog/models/<slug>/`)
and a `model.yaml`:

```yaml
hf_id: Qwen/Qwen3-32B
served_model_name: qwen3-32b
weight_dtype: bfloat16     # float8 for FP8 checkpoints
```

**A calibration:** run `evaluation/eval.sh tune deploy/<stack>`, then
`evaluation/eval.sh promote deploy/<stack>`. Promote writes an exact
`{topology, tp}` record, replacing an existing one for that match, and
re-renders `k8s/`. Commit the calibration file, the re-rendered `k8s/` and the
results directory together.
