# Alternatives to the `models/` directory structure

Status: option B implemented (2026-11-20). See [deploy/README.md](../deploy/README.md)
for the resulting layout and the `simstack` tool.

## Problem

Today a deployable simulator lives at
`models/<model>/deployments/<hardware>/<topology>/{k8s,evaluation/<variant>}/`.
Each leaf holds a full copy of the Kubernetes manifests, the ConfigMap and a
plain-JSON copy of the same config.

One model (Qwen3-32B) on one GPU (H100 SXM5) with three topologies (`standalone`,
`pd`, `pd-tp2`) and three latency variants (`flat`, `physics`, `physics-beta-1.0`)
is already **312 files**. Latency depends on many more dimensions than the tree
encodes (TP, parallel layout, P/D role, CUDA-graph regime, batch budget, vLLM
version, quantization), so adding them as path segments multiplies the tree again.

The copies have already drifted:

- `pd/k8s/sim-config.json` ships `beta = [3.12, 1.18, 0.15]`, while
  `pd/evaluation/physics/sim-config.json` has `[0.152, 0.0, 126.0]`.
- `pd-tp2/` carries the same betas as standalone TP=1, so they were copied, not
  calibrated for TP=2.
- The shipped standalone betas `[0.152, 0.0, 126.0]` give about +441% ITL at
  concurrency 1. 0.152 is exactly BLIS iter29's global β1a. The tuning report's
  `[1.22, 1.18, 4.09]` was never copied into the ConfigMap
  (llm-d-autoscaling-research `findings/derived/vllm-model-sim/vllm-model-sim.md`).
- The real and sim manifests do not share engine settings. The standalone sim
  serves with `--max-model-len=16384` while the real deployment uses `8192`. The sim
  leaves `max-num-batched-tokens`/`max-num-seqs` unset, so it gets vLLM's CPU
  defaults (2048/128) instead of H100's (8192/1024). Its KV cache is 4 GiB
  (`VLLM_CPU_KVCACHE_SPACE`), not the real pool.

The tree mixes facts of different kinds (architecture, hardware specs, fitted
numbers, engine settings, k8s plumbing, evaluation output) and copies each one
into every leaf. Every alternative below starts by separating them.

## What actually drives latency

Each dimension falls into one of these classes, and the class decides how the
dimension is stored.

| Class | Examples | How the sim handles it | Storage |
|-------|----------|------------------------|---------|
| Architecture | layers, hidden, heads, KV heads, FFN, MoE experts | Analytic: roofline FLOPs and bytes from HF `config.json` | One `config.json` per model |
| Hardware specs | peak TFLOPs, HBM GB/s, memory, interconnect | Analytic: divide FLOPs and bytes | One file per GPU (about 10 files total) |
| Scheduler knobs | `max-num-batched-tokens`, `max-num-seqs`, chunked prefill, prefix caching, `max-model-len`, KV blocks | **Emulated for free**: the sim runs the real vLLM scheduler, so these shape the batches the latency model costs | Engine profile shared by the real and sim manifests. **Not** a calibration key |
| Kernel regime | CUDA-graph capture size, attention backend, KV-cache dtype, weight quantization, async scheduling | Not modeled. Changes per-step efficiency (MoE above 512 tokens costs at least 43 ms without CUDA graphs) | Calibration key, or a future analytic term |
| Parallel layout and role | TP, DP+EP, prefill vs decode pod | TP divides work analytically; communication cost and EP are not modeled | Calibration key |
| Software version | vLLM image | Not modeled | Calibration provenance; a key only when it changes results |
| Sim host overhead | vLLM CPU step cost (0.5–19 ms, state-dependent) | Absorbed into `beta_base` today | A separate host term, not part of the GPU calibration |

The task brief says "hardware only impacts latency values". The research
findings agree and add one subtlety: vLLM picks **different scheduler defaults
per GPU** (H100 8192/1024, A100/L40S 2048/256, CPU 2048/128). The sim must pin the
target GPU's values explicitly, otherwise a "hardware" change silently becomes a
scheduler change. The engine profile below exists to handle this.

The research findings (BLIS refit, D3 hardware portability) also show that:

- **Fitted coefficients do not transfer across models.** BLIS's single global
  set has 39% RMS error on Qwen3-8B; a per-model refit gives 6%.
- **Fitted coefficients transfer poorly across GPUs.** With one GPU it is not
  identifiable which terms scale with FLOPs, which with bandwidth, and which
  are fixed. Two refits that fit H100 equally well diverge by up to 30% on
  L40S/B200.

So calibration is inherently per model × GPU × layout. The design must make
"no calibration yet" cheap (fall back to analytic defaults) instead of
requiring a directory per combination.

## How BLIS handled the same explosion

BLIS (`inference-sim/inference-sim`) went through three stages:

1. **Exact-key table.** `defaults.yaml` listed about 120 `alpha/beta` entries,
   each keyed on `(model, GPU, TP, vllm_version)`. A miss aborted the run, and
   almost every entry was for vLLM v0.8.4. This did not scale.
2. **Global coefficients** (`crossmodel`, `trained-roofline`). One small set was
   shared by every model and GPU, and per-config variation came from analytic
   features. Easy to scale but inaccurate, as shown above.
3. **Scoped registry (current, `blis-substrate`).** The data is split into three
   stores:
   - `catalog/hardware/<chip>.yaml`;
   - `catalog/models/<name>/{config.json,model.yaml}`;
   - `registry/coefficients/*.yaml`, where each entry carries `value`, `method`
     (measured / vendor_spec / assumed), `sources` and an optional `scope`
     (`hardware`, `model`, `tp`, ...). Resolution picks the best method rank,
     then the most specific scope, and refuses ties.

   A scenario YAML names the engine version, model, hardware and engine knobs
   (`max_num_batched_tokens`, `cudagraph_mode`, `tp/dp`). The knobs feed the
   model; they are **not** lookup keys.

## Alternatives

### A. Keep the tree, but stop copying (minimal)

Keep `models/<model>/deployments/<hw>/<topology>/`, but:

- Delete the `sim-config.json` sidecars and make `tune.py` read and write the
  ConfigMap directly.
- Make `k8s/` the only copy of the tuned config. Evaluation variants reference
  it instead of duplicating it.
- Move `results/` out of the config tree.

**Pros:** small change; no new tooling.
**Cons:** the path still encodes only hardware and topology, so TP, regime and
version end up as more directory names (`pd-tp2`, `pd-tp2-cg0`, ...). Growth is
still multiplicative. Real/sim engine parity remains manual.

### B. Normalized catalogs + calibration records + rendered manifests

```
catalog/
  hardware/h100-sxm5.yaml          # peak_tflops, hbm_gbps, memory, vLLM default
                                   #   max_num_batched_tokens/max_num_seqs
  models/qwen3-32b/config.json     # HF config, untouched
calibrations/
  qwen3-32b/h100-sxm5/tp1-standalone.yaml
  qwen3-32b/h100-sxm5/tp1-decode.yaml
  qwen3-32b/h100-sxm5/tp2-decode.yaml
templates/                         # one copy of each k8s shape
  standalone/  pd/
profiles/                          # engine settings shared by real and sim
  default.yaml  long-context.yaml
evaluation/
  sweeps/*.yaml
  results/<calibration-id>/<date>/ # or a release artifact, not git
```

A calibration record is small and self-describing:

```yaml
model: qwen3-32b
hardware: h100-sxm5
layout: {tp: 1, role: standalone}
engine: {vllm: v0.31.0, cudagraph_capture_size: 512, kv_cache_dtype: auto}
latency: {type: physics, beta: [1.22, 1.18, 4.09]}
fitted_on: {sweep: sweeps/c1.yaml, report: results/qwen3-32b-h100-tp1/2026-11-02}
error: {itl_mape: 0.06, ttft_mape: 0.09}   # illustrative values
```

A small CLI renders the deployable stack from the four inputs:

```bash
vllm-sim render --model qwen3-32b --hardware h100-sxm5 \
  --topology pd --tp 2 --profile default > stack.yaml
```

It emits the real and the sim manifests from the **same** profile, so scheduler
settings match by construction. Renders can be committed under `deploy/` for a
few curated "golden" stacks if a git-native `kubectl apply -f` path is wanted.

**Pros:**
- File count grows additively: about 10 hardware files, plus one `config.json`
  per model, plus one record per actual calibration.
- Uncalibrated combinations still render, using unit betas.
- Each fitted number has provenance and a measured error.

**Cons:** needs a render tool and tests for it. Users cannot browse a ready-made
YAML for every combination.

### C. Kustomize base + components (git-native B)

Same separation as B, expressed with Kustomize instead of a custom renderer:

```
k8s/base/{standalone,pd}/
k8s/components/hardware/h100-sxm5/   # patches hardware block into ConfigMap
k8s/components/model/qwen3-32b/      # config.json, tokenizer, served name
k8s/components/tp/2/                 # --tensor-parallel-size, GPU count
k8s/components/calibration/qwen3-32b-h100-tp1/   # beta patch
k8s/overlays/qwen3-32b-h100-pd-tp2/kustomization.yaml   # 5 lines: picks components
```

**Pros:** no new tool; `kubectl apply -k` works; overlays are tiny.
**Cons:**
- JSON-inside-ConfigMap patching is awkward, so it needs a `configMapGenerator`
  with merged files.
- Calibration lookup with fallback cannot be expressed; each overlay has to name
  its calibration explicitly.
- Real/sim parity needs a shared component that both bases include.

### D. Scoped coefficient registry, resolved by the sim at startup

Like BLIS's current design. The plugin itself ships the catalog and a flat
registry of coefficient entries with optional scopes:

```yaml
# registry/physics.yaml
- {beta: [1.0, 1.0, 0.0], method: assumed}                        # universal default
- {scope: {hardware: h100-sxm5}, beta_base: 2.3, method: fitted}  # per-GPU step overhead
- {scope: {model: qwen3-32b, hardware: h100-sxm5, tp: 1},
   beta: [1.22, 1.18, 4.09], method: fitted, source: results/...}
- {scope: {model_family: moe, cudagraph: false}, beta_dc: 1.9, method: fitted}
```

The sim resolves `(model, hardware, tp, role, regime)` at load time. It already
knows the model (HF config) and TP (`parallel_config`); it reads role from the
kv-transfer config and hardware from a single `--hf-overrides` /
`VLLM_SIM_HARDWARE=h100-sxm5`. The most specific match wins per coefficient, so
a per-GPU overhead can combine with a per-model efficiency term. Deployments then
need **no latency block at all**: the ConfigMap shrinks to the HF config plus a
hardware name.

**Pros:**
- Handles the explosion best: one entry per measured fact, partial-scope
  sharing (e.g. one per-GPU overhead for all models), and graceful fallback.
- Matches where BLIS converged.
- Separate per-coefficient scopes fit the D3 finding: fixed overhead scales
  with hardware, while efficiency multipliers are per model.

**Cons:**
- Most engineering work: resolution rules, tie handling, and a way to see which
  entries were used (the sim should log the resolved betas and their sources at
  startup).
- Changing a coefficient means releasing a new plugin image, unless the registry
  can also be mounted.
- Mixing scopes per coefficient makes a fit harder to reproduce than one
  self-contained record.

## Recommendation

Do **B now, with D's resolution rule inside the renderer**, and move the
resolution into the plugin (full D) only if runtime lookup proves useful:

1. Split the data into `catalog/hardware`, `catalog/models` and `calibrations/`
   records with provenance and measured error. Calibration records are keyed on
   `model × hardware × layout(tp, role)`, plus `engine` fields (vLLM version,
   CUDA-graph regime) **only when** a measurement shows they change the fit.
2. Make scheduler knobs a shared **engine profile** that renders into both the
   real and sim manifests. Pin the target GPU's `max-num-batched-tokens`,
   `max-num-seqs`, `max-model-len` and KV-block count so the CPU sim matches. This
   also fixes the parity drift listed above.
3. Resolve calibrations most-specific-first with fallback to unit betas, and
   print which record was used. Uncalibrated combinations then need no files.
4. Keep evaluation results out of the config tree: `evaluation/results/` keyed
   by calibration id and date, or published as CI artifacts.
5. Keep a handful of rendered "golden" stacks under `deploy/` for users who
   want a plain `kubectl apply`, regenerated and diffed in CI.

This turns growth from multiplicative (directories per combination) into additive
(one record per measurement). It keeps fitted numbers reproducible, one record
each. And it removes the copies that let the current betas drift.

## Independent of the layout

- Fix the shipped Qwen3-32B betas (`[0.152, 0.0, 126.0]`). The `pd-tp2` copy is not
  a TP=2 calibration.
- Split a sim-host CPU-overhead term out of `beta_base`, so GPU calibrations
  stop absorbing the CPU's own step cost and can be reused on a different sim
  host.
- Probe a second GPU before claiming hardware portability. D3 proposes two
  decode probes (B=1 at L=128, B=256 at L=1k) to tell the fixed overhead apart
  from the bandwidth term.

## Open questions

- Should rendered manifests be committed (reviewable, browsable) or generated only?
- Is the vLLM version a calibration key or just provenance? So far there is
  data for one version only (v0.31.0), so this is undecided.
- Where does the `flat` latency model fit? It is a debugging aid, not a
  calibrated model; it could become a profile rather than a variant directory.
- Migration of the `add-model` skill and `eval.sh`, which currently take a
  variant directory as their only argument.
