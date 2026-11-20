# Qwen3-32B — H100 SXM5 Prefill/Decode Disaggregation (TP=2)

Prefill/decode disaggregation with tensor-parallel size 2: each role runs
across two H100 GPUs (four in total). Everything else (ports, sidecar, KV
transfer, local runs) is as in the [TP=1 P/D stack](../qwen3-32b-h100-sxm5-pd/README.md);
this page lists only the differences.

> [!WARNING]
> This stack's calibration is a **copy** of the standalone TP=1 betas
> (`method: copied` in
> [`calibrations/qwen3-32b/h100-sxm5.yaml`](../../calibrations/qwen3-32b/h100-sxm5.yaml)).
> It has never been fitted against a real TP=2 deployment. Run
> `eval.sh tune` and `eval.sh promote` on this stack before using its latencies.

## Differences from TP=1

- The sim pods start with `--tensor-parallel-size=2`, which spawns two CPU
  worker processes per pod (gloo). The plugin reads the TP size from vLLM's
  parallel config and divides per-GPU work in the physics model accordingly.
- Each sim pod requests 8 CPU / 16 GiB (4 CPU / 8 GiB per rank), with
  `VLLM_CPU_OMP_THREADS_BIND=0-3|4-7`, one range per rank.
- Each real pod requests 2 GPUs and 64 GiB.
- Resource names follow `vllm-qwen3-32b-pd-tp2-<hash>[-<role>]`.

## Deployment

```bash
export VLLM_SIM_NAMESPACE=default  # or your target namespace
K=deploy/qwen3-32b-h100-sxm5-pd-tp2/k8s
kubectl apply -n $VLLM_SIM_NAMESPACE \
  -f $K/configmap.yaml \
  -f $K/prefill-deployment.yaml -f $K/prefill-service.yaml \
  -f $K/decode-deployment.yaml  -f $K/decode-service.yaml
```

For local runs, follow the TP=1 instructions with this stack's
`k8s/sim-config.json` and add `--tensor-parallel-size 2` to both `vllm serve`
commands.

## Eval (real vs sim comparison)

```bash
bash evaluation/eval.sh deploy/qwen3-32b-h100-sxm5-pd-tp2 [physics|physics-beta-1.0|flat]
```

Reports land under `evaluation/results/qwen3-32b-h100-sxm5-pd-tp2/<variant>/`.
The sweep matrix is [`sweep.yaml`](sweep.yaml).
