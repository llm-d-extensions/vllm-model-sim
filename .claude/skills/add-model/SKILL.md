---
name: add-model
description: Add a new model to vllm-model-sim under models/<slug>/ with default, untuned physics latency values (beta = [1.0, 1.0, 0.0]) and a deployable standalone Kubernetes stack, plus the READMEs and inventory tables. Use whenever someone wants to simulate, support, onboard or "add" a HuggingFace model (Llama, Qwen, Mistral, Mixtral, Gemma, gpt-oss, an FP8 checkpoint, ...) or a new GPU target for an existing model, even if they only give an HF id and say "can we sim this". Not for tuning or calibrating betas, or for running real-vs-sim evaluations; those go through evaluation/eval.sh.
---

# Add a model with default latency values

The result is a new `models/<slug>/` tree whose standalone `k8s/` stack runs
the **physics roofline model with unit betas** `[1.0, 1.0, 0.0]`, with no
tuning. That is the repo's documented uncalibrated baseline (the
`physics-beta-1.0` variant for Qwen3-32B): prefill and decode times are the
pure compute/bandwidth roofline for the model's shapes on the target GPU, with
no fixed per-step overhead. It is a reasonable first approximation but sits
below real latency, most visibly for small or sparse models at low batch.
Calibration happens later with `eval.sh tune` and is out of scope here.

All paths below are relative to the repo root.

## 1. Gather inputs

| Input | Default | Notes |
|-------|---------|-------|
| HF model id | — (required) | e.g. `Qwen/Qwen3-8B`. Also used as `--tokenizer`. |
| slug | lower-case HF name, `.` → `-` | Directory name and Kubernetes resource-name stem, so only `[a-z0-9-]` (the script rejects anything else). `Llama-3.1-8B-Instruct` → `llama-3-1-8b-instruct`. Keep the quantization in it (`-fp8`) if the checkpoint is quantized. |
| served model name | slug | `--served-model-name`, the `"model"` clients send. Pass `--served-model-name` to keep dots (`llama-3.1-8b-instruct`). |
| hardware slug | `h100-sxm5` | See `references/hardware.md` for specs and slugs. |
| TP | 1 | Pick the smallest TP the real model fits in (weights + KV cache under the GPU's memory). |

If the user gave only a model id, use the defaults and say so. Don't stop to
ask unless the hardware choice matters to them and is truly ambiguous.

If `models/<slug>/` already exists, the request is probably a new hardware
target for an existing model. Skip `config.json` (the script refuses to
overwrite without `--force`; use a scratch `--root` and copy only the
`deployments/<hw>/` subtree).

## 2. Scaffold the mechanical files

```bash
python3 .claude/skills/add-model/scripts/scaffold_model.py \
  --model <HF-id> --slug <slug> --hardware <hw> --tp <n>
```

`--model` also accepts a local `config.json` path (add `--hf-id <HF-id>`).
Use a local path when the direct download fails, e.g. on gated repos or
behind a TLS-intercepting proxy:

```bash
curl -fsSL https://huggingface.co/<HF-id>/resolve/main/config.json -o "$TMPDIR/config.json"
# gated: add -H "Authorization: Bearer $HF_TOKEN", or use `huggingface-cli download`
```

If no download works (gated repo, no token), write the config by hand from
the model card or the published `config.json`, and tell the user you did.
The physics model and vLLM need: `architectures`, `model_type`,
`hidden_size`, `num_hidden_layers`, `num_attention_heads`,
`num_key_value_heads`, `head_dim` (if not `hidden_size / heads`),
`intermediate_size`, `vocab_size`, `max_position_embeddings`, `eos_token_id`
and `torch_dtype`, plus for MoE `num_experts`, `num_experts_per_tok`,
`moe_intermediate_size` and, if present, `shared_expert_intermediate_size`.
Keep `rope_scaling`/`rope_theta` too, since vLLM validates them against
`max_position_embeddings`.

The script writes:

```
models/<slug>/config.json                                  real config, cleaned
models/<slug>/deployments/<hw>/standalone/k8s/
  sim-config.json   real config + architectures=[SimulatedForCausalLM] + latency block
  configmap.yaml    ConfigMap embedding sim-config.json
  deployment.yaml   CPU sim Deployment (vllm-openai-cpu + plugin init container)
  service.yaml      ClusterIP Service
  pvc.yaml          shared hf-cache PVC
```

It also prints a JSON summary. Read it, because the rest of the job depends
on it:

- `resource_name`: `vllm-<slug>-standalone-<hash>`. The hash is the first 6
  hex chars of `sha256(json.dumps(sim_config, sort_keys=True))`, the same
  scheme the existing manifests use. If you hand-edit `sim-config.json`
  afterwards, the hash and every name containing it go stale, so regenerate
  with `--force` instead.
- `warnings`: architecture features the physics model does not represent
  (MLA, sliding-window or hybrid/linear attention, multimodal towers, MoE
  sparse-step quirks). Don't hide them. Put each one in the deployment
  README's Limitations section so readers know how far to trust the numbers.

What the script does to the real config, and why:
- Hoists decoder fields out of `text_config` for multimodal wrappers, because
  the physics model reads attributes from the top level.
- Maps Mixtral-style `num_local_experts` to `num_experts`, the name the
  physics model reads.
- Drops `quantization_config`, `auto_map` and `compression_config`. With
  dummy weights the sim can't run quantized kernels or remote code. FP8 and
  INT8 checkpoints get `weight_dtype: float8` or `int8` in the latency
  block instead, so decode still reads the right number of weight bytes.
- Sets `--max-model-len` to 16384, as the Qwen3-32B Deployment does, or to
  `max_position_embeddings` if that is smaller. The CPU pod reserves only
  4 GiB of KV cache (`VLLM_CPU_KVCACHE_SPACE`). If the user needs longer
  contexts, pass `--max-model-len` and raise that variable with it.
- Never writes `tp` into the latency block. The plugin injects it from
  `--tensor-parallel-size` at load time; the script adds that flag to the
  Deployment when `--tp > 1` and scales CPU/memory requests by TP.

Pass `--beta PF DC BASE_MS` only if the user explicitly supplies betas. The
point of this skill is the untuned default.

## 3. Sanity-check the default latencies

```bash
python3 .claude/skills/add-model/scripts/estimate_latency.py \
  models/<slug>/deployments/<hw>/standalone/k8s/sim-config.json --tp <n>
```

This prints prefill times and decode step times (ITL) from the plugin's own
`PhysicsLatencyModel`. Check the batch-1 ITL against the weight-streaming
bound, `active_param_bytes / tp / hbm_bandwidth`. A dense 8B BF16 model on
H100 gives about 16 GB / 3350 GB/s ≈ 5 ms. The script reports 4.2 ms for
Qwen3-8B, a bit lower because the physics model leaves out the embedding
and LM-head weights. An order-of-magnitude miss means a config field was
misread, usually `head_dim`, the MoE fields or `text_config`. Put the batch-1 ITL and the 1k-token prefill time in the
deployment README.

## 4. Write the READMEs and update inventories

Use the Qwen3-32B tree as the model. Match its structure and tone, not its
content:

| New file | Model on |
|----------|----------|
| `models/<slug>/README.md` | `models/qwen3-32b/README.md` (HF id, architecture, params, max context, dtype, deployments table) |
| `models/<slug>/deployments/README.md` | `models/qwen3-32b/deployments/README.md` |
| `models/<slug>/deployments/<hw>/README.md` | `models/qwen3-32b/deployments/h100-sxm5/README.md` (hardware table, configurations table) |
| `models/<slug>/deployments/<hw>/standalone/README.md` | `models/qwen3-32b/deployments/h100-sxm5/standalone/README.md` |

The standalone README differs from Qwen3-32B's in these ways, so don't copy
those parts:
- Say that `k8s/` runs the **untuned** physics model, β = [1.0, 1.0, 0.0],
  and give the step-3 estimates. Qwen3-32B's `k8s/` is calibrated; the new
  model's is not yet.
- Use the real `resource_name` from the script summary in the
  `kubectl port-forward` command, the served model name as the `"model"` in the
  `/v1/chat/completions` curl example, and `<HF-id>` as the local run's
  `--tokenizer`.
- For the local run, use the model-directory form from step 5, not the
  `VLLM_SIMULATED_PLUGIN_CONFIG=... vllm serve <HF-id>` form in the Qwen3-32B
  README. The plugin never reads that variable, so serving the HF id loads
  the real architecture with dummy weights instead of the simulator.
- If the HF repo is gated, the pod downloads the tokenizer at startup and
  fails without a token, because the Deployment marks the `hf-token` Secret
  optional. Add a step:
  `kubectl create secret generic hf-token --from-literal=token=$HF_TOKEN`.
- Include a Limitations section only if the script printed warnings or the
  config was hand-written.
- There is no `evaluation/` tree yet, so leave out
  the Latency Models, Eval Results and Eval sections. Add a short "Next
  steps" note instead: an evaluation and `eval.sh tune` will produce
  calibrated betas.

Then update the inventories:
- `models/README.md`: add a row to the **Models** table, and to **Hardware
  Slugs** if the GPU is new.
- `README.md` (repo root): add a row to **Supported Models** for the
  standalone deployment, with the mode written as `Standalone (untuned)`.
  The Quick Start calls the listed deployments "production-calibrated", and
  the label keeps readers from assuming this one is.

## 5. Verify

```bash
python3 -c 'import json,sys; [json.load(open(p)) for p in sys.argv[1:]]' \
  models/<slug>/config.json models/<slug>/deployments/<hw>/standalone/k8s/sim-config.json
grep -rn "qwen3-32b\|Qwen3-32B\|eae748" models/<slug>/   # should print nothing
python3 -c 'import yaml' 2>/dev/null && python3 -c 'import sys,yaml; [list(yaml.safe_load_all(open(p))) for p in sys.argv[1:]]' \
  models/<slug>/deployments/<hw>/standalone/k8s/*.yaml    # skip if PyYAML is missing
source .venv/bin/activate && pytest tests/              # if a venv exists
```

If a venv with vLLM exists, a local smoke run proves vLLM accepts the config.
vLLM picks the architecture from the served directory's `config.json`, so
serve a directory whose `config.json` is the sim config. That is what the
Kubernetes Deployment does by mounting the ConfigMap at `/model`:

```bash
mkdir -p "$TMPDIR/<slug>-sim"
cp models/<slug>/deployments/<hw>/standalone/k8s/sim-config.json "$TMPDIR/<slug>-sim/config.json"
vllm serve "$TMPDIR/<slug>-sim" --tokenizer <HF-id> --served-model-name <served-name> \
  --load-format dummy --max-model-len 16384 --port 8000
```

followed by a `/v1/chat/completions` request. Skip it when there's no vLLM
install and say it wasn't run. Don't claim it passed.

## Out of scope

- `evaluation/` variants (`flat`, `physics`, `physics-beta-1.0`), real-GPU
  manifests, `sweep.yaml`, and tuning: see `evaluation/README.md`.
- P/D (`pd/`) stacks. Once standalone exists, copy
  `models/qwen3-32b/deployments/h100-sxm5/pd/k8s/` and substitute the names,
  but only if the user asks.
