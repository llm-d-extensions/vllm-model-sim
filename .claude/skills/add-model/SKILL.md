---
name: add-model
description: Add a new model to vllm-model-sim (catalog/models/<slug>/ plus a deploy/<stack>/ standalone stack) with default, untuned physics latency values (beta = [1.0, 1.0, 0.0]) and a rendered Kubernetes stack, plus the READMEs and inventory tables. Use whenever someone wants to simulate, support, onboard or "add" a HuggingFace model (Llama, Qwen, Mistral, Mixtral, Gemma, gpt-oss, an FP8 checkpoint, ...) or a new GPU target for an existing model, even if they only give an HF id and say "can we sim this". Not for tuning or calibrating betas, or for running real-vs-sim evaluations; those go through evaluation/eval.sh.
---

# Add a model with default latency values

The result is a catalog entry, `catalog/models/<slug>/`, and a standalone
stack, `deploy/<slug>-<hw>-standalone[-tp<n>]/`, whose rendered `k8s/` runs
the **physics roofline model with unit betas** `[1.0, 1.0, 0.0]`, with no
tuning. Unit betas are what the renderer falls back to when
`calibrations/<slug>/<hw>.yaml` has no matching record, so nothing needs
writing there. Prefill and decode times are the pure compute/bandwidth
roofline for the model's shapes on the target GPU, with no fixed per-step
overhead. It is a reasonable first approximation but sits below real latency,
most visibly for small or sparse models at low batch. Calibration happens
later with `eval.sh tune` and is out of scope here.

Read [deploy/README.md](../../../deploy/README.md) for the layout. All paths
below are relative to the repo root.

## 1. Gather inputs

| Input | Default | Notes |
|-------|---------|-------|
| HF model id | — (required) | e.g. `Qwen/Qwen3-8B`. Also used as `--tokenizer`. |
| slug | lower-case HF name, `.` → `-` | Catalog name and Kubernetes resource-name stem, so only `[a-z0-9-]` (the script rejects anything else). `Llama-3.1-8B-Instruct` → `llama-3-1-8b-instruct`. Keep the quantization in it (`-fp8`) if the checkpoint is quantized. |
| served model name | slug | `--served-model-name`, the `"model"` clients send. Pass `--served-model-name` to keep dots (`llama-3.1-8b-instruct`). |
| hardware slug | `h100-sxm5` | A file under `catalog/hardware/`. For a GPU not there, see `references/hardware.md`. |
| TP | 1 | Pick the smallest TP the real model fits in (weights + KV cache under the GPU's memory). |

If the user gave only a model id, use the defaults and say so. Don't stop to
ask unless the hardware choice matters to them and is truly ambiguous.

If `catalog/models/<slug>/` already exists, the request is probably a new
hardware target or topology for an existing model. Then you only need a new
`deploy/<stack>/stack.yaml` and `sweep.yaml`, followed by
`python -m simstack render deploy/<stack>`; skip step 2.

## 2. Scaffold the mechanical files

```bash
.venv/bin/python .claude/skills/add-model/scripts/scaffold_model.py \
  --model <HF-id> --slug <slug> --hardware <hw> --tp <n>
```

The script needs PyYAML (it renders with `simstack`). Any Python with
`pyyaml` installed works.

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

For a GPU without a `catalog/hardware/<hw>.yaml`, also pass `--peak-tflops`,
`--hbm-gbps`, `--memory-gib` and `--gpu-product` (the `nvidia.com/gpu.product`
node label). The script then writes the hardware file, deriving vLLM's
scheduler defaults from the memory size.

The script writes:

```
catalog/models/<slug>/config.json   real config, cleaned
catalog/models/<slug>/model.yaml    hf_id, served_model_name, weight_dtype
deploy/<stack>/stack.yaml           model, hardware, topology: standalone, tp, profile
deploy/<stack>/sweep.yaml           copy of evaluation/sweep.yaml
deploy/<stack>/k8s/                 rendered: configmap.yaml, sim-config.json,
                                    deployment.yaml, service.yaml, pvc.yaml
```

It also prints a JSON summary. Read it, because the rest of the job depends
on it:

- `resource_name`: `vllm-<slug>-standalone[-tp<n>]-<hash>`. The hash is the
  first 6 hex chars of `sha256(json.dumps(sim_config, sort_keys=True))`.
  Never hand-edit `k8s/`. Change the catalog, profile or calibrations and
  re-render with `python -m simstack render deploy/<stack>`;
  `tests/test_simstack.py` fails on stale renders.
- `engine`: the `--max-model-len`, `--max-num-batched-tokens` and
  `--max-num-seqs` both the sim and the real pods get. They come from the
  GPU's `vllm_defaults` and the profile, and `max-model-len` is capped at
  `max_position_embeddings`.
- `warnings`: architecture features the physics model does not represent
  (MLA, sliding-window or hybrid/linear attention, multimodal towers, MoE
  sparse-step quirks). Don't hide them. Put each one in the stack README's
  Limitations section so readers know how far to trust the numbers.

What the script does to the real config, and why:
- Hoists decoder fields out of `text_config` for multimodal wrappers, because
  the physics model reads attributes from the top level.
- Maps Mixtral-style `num_local_experts` to `num_experts`, the name the
  physics model reads.
- Drops `quantization_config`, `auto_map` and `compression_config`. With
  dummy weights the sim can't run quantized kernels or remote code. FP8 and
  INT8 checkpoints get `weight_dtype: float8` or `int8` in `model.yaml`
  instead, so decode still reads the right number of weight bytes.

The CPU pod reserves only 4 GiB of KV cache (`sim.kv_cache_space_gib` in
`profiles/default.yaml`). If the user needs longer contexts, add a profile
with a larger `engine.max_model_len` and KV space and name it in `stack.yaml`.

Pass `--beta PF DC BASE_MS` only if the user explicitly supplies betas; it
appends a `method: assumed` record to `calibrations/<slug>/<hw>.yaml`. The
point of this skill is the untuned default.

## 3. Sanity-check the default latencies

```bash
python3 .claude/skills/add-model/scripts/estimate_latency.py \
  deploy/<stack>/k8s/sim-config.json --tp <n>
```

This prints prefill times and decode step times (ITL) from the plugin's own
`PhysicsLatencyModel`. Check the batch-1 ITL against the weight-streaming
bound, `active_param_bytes / tp / hbm_bandwidth`. A dense 8B BF16 model on
H100 gives about 16 GB / 3350 GB/s ≈ 5 ms. The script reports 4.2 ms for
Qwen3-8B, a bit lower because the physics model leaves out the embedding
and LM-head weights. An order-of-magnitude miss means a config field was
misread, usually `head_dim`, the MoE fields or `text_config`. Put the batch-1
ITL and the 1k-token prefill time in the stack README.

## 4. Write the READMEs and update inventories

Use the Qwen3-32B files as the model. Match their structure and tone, not
their content:

| New file | Model on |
|----------|----------|
| `catalog/models/<slug>/README.md` | `catalog/models/qwen3-32b/README.md` (HF id, architecture, params, max context, dtype, stacks table) |
| `deploy/<stack>/README.md` | `deploy/qwen3-32b-h100-sxm5-standalone/README.md` |

The stack README differs from Qwen3-32B's in these ways, so don't copy those
parts:
- Say that `k8s/` runs the **untuned** physics model, β = [1.0, 1.0, 0.0],
  and give the step-3 estimates.
- Use the served model name as the `"model"` in the `/v1/chat/completions`
  curl example, `<HF-id>` as the local run's `--tokenizer`, and the summary's
  `engine` values as the local run's engine flags.
- If the HF repo is gated, the pod downloads the tokenizer at startup and
  fails without a token, because the Deployment marks the `hf-token` Secret
  optional. Add a step:
  `kubectl create secret generic hf-token --from-literal=token=$HF_TOKEN`.
- Include a Limitations section only if the script printed warnings or the
  config was hand-written.
- Keep the Eval section short: `eval.sh deploy/<stack>` and `eval.sh tune`
  will produce calibrated betas, but no report exists yet.

Then update the inventories:
- `deploy/README.md`: add a row to the **Stacks** table, with calibration
  `none (unit betas)`.
- `README.md` (repo root): add a row to **Supported Models**, with the mode
  written as `Standalone (untuned)`. The Quick Start calls the listed
  deployments "calibrated", and the label keeps readers from assuming this one
  is.

## 5. Verify

```bash
.venv/bin/python -m simstack check                     # renders are fresh
.venv/bin/python -m simstack resolve deploy/<stack>    # unit betas, method: assumed
grep -rn "qwen3-32b\|Qwen3-32B" catalog/models/<slug>/ deploy/<stack>/   # should print nothing
source .venv/bin/activate && pytest tests/test_simstack.py   # if a venv exists
```

If a venv with vLLM exists, a local smoke run proves vLLM accepts the config.
vLLM picks the architecture from the served directory's `config.json`, so
serve a directory whose `config.json` is the sim config. That is what the
Kubernetes Deployment does by mounting the ConfigMap at `/model`:

```bash
mkdir -p "$TMPDIR/<slug>-sim"
cp deploy/<stack>/k8s/sim-config.json "$TMPDIR/<slug>-sim/config.json"
vllm serve "$TMPDIR/<slug>-sim" --tokenizer <HF-id> --served-model-name <served-name> \
  --load-format dummy --max-model-len <engine.max_model_len> --port 8000
```

followed by a `/v1/chat/completions` request. Skip it when there's no vLLM
install and say it wasn't run. Don't claim it passed.

## Out of scope

- Real-vs-sim evaluation and tuning: see `evaluation/README.md`. The real GPU
  manifests are rendered on the fly by `eval.sh`, so there's nothing to write
  for them here.
- P/D stacks. Once standalone exists, add `deploy/<slug>-<hw>-pd/stack.yaml`
  with `topology: pd` and render it, but only if the user asks.
